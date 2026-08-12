"""Stages 7-9: hyperparameter optimisation, final fitting, calibration.

The tuning objective is **log loss**, a strictly proper scoring rule, rather
than AUROC. Calibration is a primary outcome of this study, and AUROC is
rank-based and wholly indifferent to calibration; optimising it would be
internally inconsistent with what the paper claims to measure. AUROC is
recorded as a secondary cross-validated metric.

Compute budgeting is handled by :func:`estimate_tuning_cost`, which times a
single fit before the full search is launched. Discovering that a search will
take fourteen hours is much cheaper before it starts than during it.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from .config import Config
from .models import ModelSpec, build_estimator, get_model_specs
from .preprocessing import TARGET
from .utils import RunManifest, gate, get_logger, write_json, write_table

log = get_logger("training")

__all__ = [
    "calibrate_model",
    "estimate_tuning_cost",
    "fit_final_models",
    "generate_predictions",
    "tune_all_models",
    "tune_model",
]


# ---------------------------------------------------------------------------
# Compute budgeting
# ---------------------------------------------------------------------------

def estimate_tuning_cost(
    spec: ModelSpec, dev: pd.DataFrame, cfg: Config
) -> dict[str, float]:
    """Time one fit and project the full search cost.

    Called before the search so that the budget can be reduced deliberately
    (repeats first, then trials, then early stopping) rather than discovered
    halfway through an overnight run.
    """
    X = dev.drop(columns=[TARGET])
    y = dev[TARGET].to_numpy()
    est = build_estimator(spec, dev, seed=cfg["seed"])

    skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                          random_state=cfg["seed"])
    train_idx, _ = next(skf.split(X, y))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X.iloc[train_idx], y[train_idx])
    single = time.perf_counter() - start

    folds = cfg["cv"]["n_splits"] * cfg["cv"]["tuning_n_repeats"]
    trials = cfg["tuning"]["n_trials"] if spec.tunable else 1
    projected = single * folds * trials
    log.info(
        "[%s] single fit %.2fs -> projected search %.1f min (%d folds x %d trials)",
        spec.key, single, projected / 60, folds, trials,
    )
    return {
        "model": spec.key,
        "single_fit_seconds": round(single, 3),
        "folds": folds,
        "trials": trials,
        "projected_seconds": round(projected, 1),
        "projected_minutes": round(projected / 60, 1),
    }


# ---------------------------------------------------------------------------
# Stage 7 - tuning
# ---------------------------------------------------------------------------

def tune_model(
    spec: ModelSpec, dev: pd.DataFrame, cfg: Config
) -> dict[str, Any]:
    """Optimise one model's hyperparameters by cross-validation.

    Preprocessing is inside the pipeline, so the encoder and scaler are
    refitted on the training portion of every fold.
    """
    X = dev.drop(columns=[TARGET])
    y = dev[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        n_repeats=cfg["cv"]["tuning_n_repeats"],
        random_state=cfg["seed"],
    )

    if not spec.tunable or spec.search_space is None:
        est = build_estimator(spec, dev, seed=cfg["seed"])
        scores = _cv_scores(est, X, y, cv, cfg)
        log.info("[%s] untuned baseline: %s", spec.key, _fmt(scores))
        return {"model": spec.key, "best_params": {}, "cv": scores, "n_trials": 0}

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = spec.search_space(trial)
        est = build_estimator(spec, dev, seed=cfg["seed"], params=params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(
                est, X, y, cv=cv,
                scoring=cfg["tuning"]["objective"],
                n_jobs=cfg["tuning"]["n_jobs"],
                error_score="raise",
            )
        return float(np.mean(res["test_score"]))

    sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"tune_{spec.key}")
    study.optimize(
        objective,
        n_trials=cfg["tuning"]["n_trials"],
        timeout=cfg["tuning"].get("timeout_seconds"),
        show_progress_bar=False,
    )

    best = study.best_params
    est = build_estimator(spec, dev, seed=cfg["seed"], params=best)
    scores = _cv_scores(est, X, y, cv, cfg)
    log.info("[%s] best params %s -> %s", spec.key, best, _fmt(scores))

    joblib.dump(study, cfg.path("models") / f"tuning_study_{spec.key}.pkl")
    return {
        "model": spec.key,
        "best_params": best,
        "cv": scores,
        "n_trials": len(study.trials),
    }


def _cv_scores(est, X, y, cv, cfg: Config) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(
            est, X, y, cv=cv,
            scoring=[cfg["tuning"]["objective"], cfg["tuning"]["secondary"]],
            n_jobs=cfg["tuning"]["n_jobs"],
            error_score="raise",
        )
    obj, sec = cfg["tuning"]["objective"], cfg["tuning"]["secondary"]
    return {
        f"{obj}_mean": float(np.mean(res[f"test_{obj}"])),
        f"{obj}_sd": float(np.std(res[f"test_{obj}"], ddof=1)),
        f"{sec}_mean": float(np.mean(res[f"test_{sec}"])),
        f"{sec}_sd": float(np.std(res[f"test_{sec}"], ddof=1)),
    }


def _fmt(scores: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in scores.items())


def tune_all_models(
    dev: pd.DataFrame, cfg: Config, manifest: RunManifest
) -> dict[str, dict]:
    """Run the search for every enabled model and emit Table 3."""
    specs = get_model_specs(cfg)

    budget = [estimate_tuning_cost(s, dev, cfg) for s in specs.values()]
    write_table(pd.DataFrame(budget), cfg.path("metrics") / "compute_budget.csv")
    total_min = sum(b["projected_minutes"] for b in budget)
    log.info("PROJECTED TOTAL TUNING TIME: %.1f minutes", total_min)

    results: dict[str, dict] = {}
    for key, spec in specs.items():
        results[key] = tune_model(spec, dev, cfg)

    rows = []
    for key, res in results.items():
        rows.append(
            {
                "model": specs[key].display_name,
                "n_trials": res["n_trials"],
                "selected_hyperparameters": (
                    "; ".join(f"{k}={_round(v)}" for k, v in res["best_params"].items())
                    or "defaults (untuned baseline)"
                ),
                **{k: round(v, 4) for k, v in res["cv"].items()},
            }
        )
    write_table(pd.DataFrame(rows), cfg.path("tables") / "table3_hyperparameters.csv")
    write_json(results, cfg.path("metrics") / "tuning_results.json")

    best_auc = max(r["cv"].get("roc_auc_mean", np.nan) for r in results.values())
    manifest.record_stage("stage07_tuning", best_cv_auroc=best_auc,
                          projected_minutes=total_min)
    gate(
        "cv_auroc_plausible",
        best_auc <= cfg["sanity"]["max_plausible_auroc"],
        f"best cross-validated AUROC = {best_auc:.4f} "
        f"(ceiling {cfg['sanity']['max_plausible_auroc']}); "
        "a higher value indicates leakage, not skill",
        manifest,
        fatal=cfg["sanity"]["abort_on_leakage"],
    )
    return results


def _round(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


# ---------------------------------------------------------------------------
# Stage 8 - final fitting and predictions
# ---------------------------------------------------------------------------

def fit_final_models(
    dev: pd.DataFrame, tuning: dict[str, dict], cfg: Config, manifest: RunManifest
) -> dict[str, Any]:
    """Refit each model on the full development set with selected parameters."""
    specs = get_model_specs(cfg)
    X, y = dev.drop(columns=[TARGET]), dev[TARGET].to_numpy()
    fitted: dict[str, Any] = {}

    for key, spec in specs.items():
        est = build_estimator(
            spec, dev, seed=cfg["seed"], params=tuning[key]["best_params"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y)
        path = cfg.path("models") / f"final_{key}.pkl"
        joblib.dump(est, path)
        manifest.record_file(f"model_{key}", path)
        fitted[key] = est
        log.info("[%s] fitted on %d development records", key, len(dev))

    manifest.record_stage("stage08_fit", models=list(fitted))
    return fitted


def generate_predictions(
    fitted: dict[str, Any],
    holdout: pd.DataFrame,
    dev: pd.DataFrame,
    cfg: Config,
    manifest: RunManifest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce and persist hold-out and development predicted probabilities.

    Every downstream stage reads these files rather than re-running models,
    which is what keeps figures and tables from silently disagreeing after a
    partial re-run.
    """
    from .preprocessing import FAIRNESS_COLUMNS

    out_hold = pd.DataFrame({"y_true": holdout[TARGET].to_numpy()})
    out_dev = pd.DataFrame({"y_true": dev[TARGET].to_numpy()})

    for col in FAIRNESS_COLUMNS:
        if col in holdout.columns:
            out_hold[f"subgroup_{col}"] = holdout[col].to_numpy()

    Xh = holdout.drop(columns=[TARGET])
    Xd = dev.drop(columns=[TARGET])
    for key, est in fitted.items():
        out_hold[key] = est.predict_proba(Xh)[:, 1]
        out_dev[key] = est.predict_proba(Xd)[:, 1]

    hp = cfg.path("predictions") / "holdout_probabilities.csv"
    dp = cfg.path("predictions") / "development_probabilities.csv"
    write_table(out_hold, hp)
    write_table(out_dev, dp)
    manifest.record_file("holdout_predictions", hp)

    prevalence = float(out_hold["y_true"].mean())
    for key in fitted:
        mean_p = float(out_hold[key].mean())
        gate(
            f"mean_prediction_{key}",
            abs(mean_p - prevalence) < 0.05,
            f"[{key}] mean predicted risk {mean_p:.4f} vs prevalence "
            f"{prevalence:.4f}; a value near 0.5 indicates an imbalance "
            "correction leaked into the primary analysis",
            manifest,
            fatal=False,
        )
    manifest.record_stage("stage08_predictions", holdout_n=len(out_hold))
    return out_hold, out_dev


# ---------------------------------------------------------------------------
# Stage 9 - recalibration
# ---------------------------------------------------------------------------

def calibrate_model(
    spec: ModelSpec,
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    params: dict,
    cfg: Config,
) -> np.ndarray:
    """Refit with cross-validated recalibration and return hold-out probabilities.

    The calibrator is fitted inside the development set via cross-validation.
    Fitting it on the hold-out would contaminate the final estimate.
    """
    X, y = dev.drop(columns=[TARGET]), dev[TARGET].to_numpy()
    base = build_estimator(spec, dev, seed=cfg["seed"], params=params)
    calibrated = CalibratedClassifierCV(
        base, method=cfg["calibration"]["method"], cv=cfg["calibration"]["cv"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(X, y)
    joblib.dump(calibrated, cfg.path("models") / f"calibrated_{spec.key}.pkl")
    return calibrated.predict_proba(holdout.drop(columns=[TARGET]))[:, 1]
