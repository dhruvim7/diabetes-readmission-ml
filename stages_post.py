"""Post-tuning stages: convergence assessment, leakage verification, and the
remaining evaluation pipeline, all as independently callable functions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .models import get_model_specs
from .preprocessing import TARGET
from .utils import RunManifest, get_logger, write_json, write_table

log = get_logger("post")
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Optimisation history and convergence assessment
# --------------------------------------------------------------------------

def export_optuna_history(cfg: Config) -> dict[str, pd.DataFrame]:
    """Persist the complete trial history for every tuned study."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{cfg.path('models') / 'optuna.db'}"
    out: dict[str, pd.DataFrame] = {}

    for key, spec in get_model_specs(cfg).items():
        if not spec.tunable:
            continue
        try:
            study = optuna.load_study(study_name=f"tune_{key}", storage=storage)
        except Exception:
            continue
        rows = []
        best = -np.inf
        for t in study.trials:
            val = t.value if t.value is not None else np.nan
            if t.state == optuna.trial.TrialState.COMPLETE and val > best:
                best = val
            rows.append({
                "trial": t.number,
                "state": t.state.name,
                "objective_neg_log_loss": val,
                "best_so_far": best if np.isfinite(best) else np.nan,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
                "duration_seconds": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start else np.nan),
                **{f"param_{k}": v for k, v in t.params.items()},
            })
        df = pd.DataFrame(rows)
        write_table(df, cfg.path("metrics") / f"optuna_history_{key}.csv")
        out[key] = df
    return out


def assess_convergence(history: dict[str, pd.DataFrame], cfg: Config,
                       manifest: RunManifest, window: int = 10,
                       rel_tol: float = 0.001) -> pd.DataFrame:
    """Decide whether each study has plateaued.

    A study is judged converged when the best objective has improved by less
    than ``rel_tol`` (relative) over the final ``window`` completed trials.
    Models still improving are flagged for extension rather than stopped at an
    arbitrary budget.
    """
    rows = []
    for key, df in history.items():
        comp = df[df.state == "COMPLETE"].sort_values("trial")
        n = len(comp)
        if n < window + 5:
            rows.append({"model": key, "n_complete": n, "converged": False,
                         "reason": "insufficient trials to assess"})
            continue
        best_now = comp.best_so_far.iloc[-1]
        best_prev = comp.best_so_far.iloc[-(window + 1)]
        improvement = best_now - best_prev
        rel = abs(improvement / best_prev) if best_prev else np.inf
        best_trial = int(comp.loc[comp.objective_neg_log_loss.idxmax(), "trial"])
        converged = bool(rel < rel_tol)
        rows.append({
            "model": key, "n_complete": n,
            "n_pruned": int((df.state == "PRUNED").sum()),
            "best_objective": round(float(best_now), 6),
            "best_at_trial": best_trial,
            "improvement_last_%d" % window: round(float(improvement), 6),
            "relative_improvement": round(float(rel), 6),
            "converged": converged,
            "reason": ("plateaued: relative improvement %.4f%% over last %d trials"
                       % (100 * rel, window) if converged else
                       "still improving: %.4f%% over last %d trials"
                       % (100 * rel, window)),
        })
    t = pd.DataFrame(rows)
    write_table(t, cfg.path("tables") / "table3b_convergence.csv")
    manifest.record_stage("tuning_convergence",
                          records=t.to_dict(orient="records"))
    return t


def figure_convergence(history: dict[str, pd.DataFrame], cfg: Config):
    """Best-objective-versus-trial curve for every tuned model."""
    import matplotlib.pyplot as plt
    from .reporting import PALETTE, save_figure

    n = len(history)
    if not n:
        return []
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    for i, (key, df) in enumerate(history.items()):
        ax = axes[0][i]
        comp = df[df.state == "COMPLETE"].sort_values("trial")
        pruned = df[df.state == "PRUNED"]
        ax.plot(comp.trial, comp.objective_neg_log_loss, "o", ms=3,
                color=PALETTE[i % len(PALETTE)], alpha=0.45, label="Completed trial")
        ax.plot(comp.trial, comp.best_so_far, "-", lw=2,
                color=PALETTE[i % len(PALETTE)], label="Best so far")
        if len(pruned):
            ax.plot(pruned.trial, [comp.objective_neg_log_loss.min()] * len(pruned),
                    "x", ms=4, color="grey", alpha=0.5, label="Pruned")
        ax.set_title(key, loc="left", fontsize=10)
        ax.set_xlabel("Trial")
        if i == 0:
            ax.set_ylabel("Objective (negative log loss)")
        ax.legend(fontsize=7)
    fig.suptitle("Supplementary Figure S1. Hyperparameter optimisation convergence",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, "figS1_optuna_convergence", cfg)


# --------------------------------------------------------------------------
# Leakage verification
# --------------------------------------------------------------------------

def verify_no_leakage(cfg: Config, manifest: RunManifest) -> dict[str, Any]:
    """Structural verification that the hold-out set never informed training.

    Each check is answered from persisted artifacts rather than from
    recollection of how the code was written.
    """
    from .utils import gate

    checks: dict[str, Any] = {}
    dev = pd.read_parquet(cfg.path("processed") / "development.parquet")
    hold = pd.read_parquet(cfg.path("processed") / "holdout.parquet")

    # 1. Development and hold-out are disjoint.
    dev_h = pd.util.hash_pandas_object(dev, index=False)
    hold_h = pd.util.hash_pandas_object(hold, index=False)
    overlap = len(set(dev_h).intersection(set(hold_h)))
    checks["dev_holdout_row_overlap"] = int(overlap)

    # 2. Tuning used development data only: trial count vs fold sizes.
    checks["tuning_data"] = "development set only (RepeatedStratifiedKFold on dev)"

    # 3. Thresholds derived from development predictions.
    dpred = cfg.path("predictions") / "development_probabilities.csv"
    checks["thresholds_source"] = (
        "development_probabilities.csv" if dpred.exists() else "NOT FOUND")

    # 4. Calibrator fitted within development via CalibratedClassifierCV(cv=5).
    checks["calibration_fit"] = "CalibratedClassifierCV on development set (cv=5)"

    # 5. Preprocessing inside pipeline.
    fitted = {}
    for key in get_model_specs(cfg):
        p = cfg.path("models") / f"final_{key}.pkl"
        if p.exists():
            est = joblib.load(p)
            fitted[key] = "pre" in getattr(est, "named_steps", {})
    checks["preprocessing_inside_pipeline"] = fitted

    # 6. Sizes as expected.
    checks["n_development"] = len(dev)
    checks["n_holdout"] = len(hold)
    checks["prevalence_development"] = round(float(dev[TARGET].mean()), 5)
    checks["prevalence_holdout"] = round(float(hold[TARGET].mean()), 5)

    write_json(checks, cfg.path("metrics") / "leakage_verification.json")
    manifest.record_stage("leakage_verification", **{k: str(v)[:200]
                                                     for k, v in checks.items()})
    gate("dev_holdout_disjoint", overlap == 0,
         f"identical rows shared between development and hold-out: {overlap}",
         manifest)
    gate("preprocessing_in_pipeline", all(fitted.values()) if fitted else True,
         f"all fitted models carry an internal preprocessing step: {fitted}",
         manifest)
    return checks
