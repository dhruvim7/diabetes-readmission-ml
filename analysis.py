"""Stages 13-15: explainability, fairness, and sensitivity analyses.

Explainability
    Three importance measures are computed and compared. Gain-based importance
    is included **only** as a comparator: impurity-based measures are
    systematically biased toward high-cardinality and continuous predictors
    (Strobl et al., BMC Bioinformatics 2007;8:25). Documenting where the three
    methods disagree - particularly for a low-cardinality demographic variable
    such as gender - is a reportable methodological result rather than an
    inconvenience.

Fairness
    Subgroup discrimination, calibration and operating-point behaviour across
    race, gender and age band. Subgroups below the configured minimum size are
    reported descriptively rather than suppressed.

Sensitivity
    Five pre-specified arms, each answering a predictable reviewer objection.
    The imbalance arm is the study's principal contribution: it quantifies the
    calibration damage caused by resampling corrections.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .config import Config
from .evaluation import (
    calibration_metrics,
    classification_metrics,
    discrimination_metrics,
)
from .models import ModelSpec, build_estimator, get_model_specs
from .preprocessing import TARGET
from .utils import RunManifest, get_logger, write_json, write_table

log = get_logger("analysis")

__all__ = [
    "compute_importances",
    "fairness_audit",
    "run_sensitivity_analyses",
]


# ---------------------------------------------------------------------------
# Stage 13 - explainability
# ---------------------------------------------------------------------------

def _parent_feature(encoded_name: str, originals: list[str]) -> str:
    """Map an encoded column back to its source variable.

    ``ColumnTransformer`` emits names like ``cat__race_Caucasian``. Aggregating
    SHAP values back to the parent variable is what makes the global plot
    readable; reporting 120 fragmented one-hot columns is not interpretable.
    """
    name = encoded_name.split("__", 1)[-1]
    matches = [c for c in originals if name == c or name.startswith(f"{c}_")]
    return max(matches, key=len) if matches else name


def compute_importances(
    model_key: str,
    fitted_pipeline: Any,
    holdout: pd.DataFrame,
    cfg: Config,
    manifest: RunManifest,
) -> dict[str, pd.DataFrame]:
    """Compute SHAP, permutation and gain-based importance and compare them."""
    X = holdout.drop(columns=[TARGET])
    y = holdout[TARGET].to_numpy()
    originals = list(X.columns)

    pre = fitted_pipeline.named_steps["pre"]
    model = fitted_pipeline.named_steps["model"]
    encoded_names = list(pre.get_feature_names_out())

    rng = np.random.default_rng(cfg["seed"])
    n_sample = min(cfg["explain"]["shap_sample_size"], len(X))
    idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X.iloc[idx]
    X_enc = pre.transform(X_sample)

    results: dict[str, pd.DataFrame] = {}

    # -- SHAP ---------------------------------------------------------------
    # The explainer must match the model family: TreeExplainer covers the
    # ensembles, LinearExplainer the logistic regressions. Falling back to
    # the model-agnostic explainer would be correct but far slower, so it is
    # used only when neither applies.
    shap_df = None
    try:
        import shap

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_enc, check_additivity=False)
        elif hasattr(model, "coef_"):
            background = shap.maskers.Independent(X_enc, max_samples=200)
            explainer = shap.LinearExplainer(model, background)
            values = explainer.shap_values(X_enc)
        else:  # pragma: no cover - defensive
            background = shap.sample(X_enc, 100, random_state=cfg["seed"])
            explainer = shap.KernelExplainer(model.predict_proba, background)
            values = explainer.shap_values(X_enc[:200])

        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        np.savez_compressed(
            cfg.path("metrics") / f"shap_values_{model_key}.npz",
            values=values, columns=np.array(encoded_names, dtype=object),
        )
        shap_df = pd.DataFrame(
            {
                "encoded_feature": encoded_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        if cfg["explain"]["aggregate_onehot"]:
            shap_df["feature"] = [
                _parent_feature(c, originals) for c in shap_df["encoded_feature"]
            ]
            shap_df = (
                shap_df.groupby("feature", as_index=False)["mean_abs_shap"]
                .sum()
                .sort_values("mean_abs_shap", ascending=False)
            )
        shap_df["rank_shap"] = range(1, len(shap_df) + 1)
        results["shap"] = shap_df
        log.info("[%s] SHAP computed on %d records", model_key, n_sample)
    except Exception as exc:  # pragma: no cover
        log.warning("[%s] SHAP unavailable: %s", model_key, exc)

    # -- permutation importance --------------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                fitted_pipeline, X_sample, y[idx],
                n_repeats=cfg["explain"]["permutation_repeats"],
                random_state=cfg["seed"],
                scoring="roc_auc",
                n_jobs=1,
            )
        perm_df = (
            pd.DataFrame(
                {"feature": originals, "permutation_importance": perm.importances_mean,
                 "permutation_sd": perm.importances_std}
            )
            .sort_values("permutation_importance", ascending=False)
            .reset_index(drop=True)
        )
        perm_df["rank_permutation"] = range(1, len(perm_df) + 1)
        results["permutation"] = perm_df
    except Exception as exc:  # pragma: no cover
        log.warning("[%s] permutation importance failed: %s", model_key, exc)

    # -- gain-based (comparator only) --------------------------------------
    try:
        gains = getattr(model, "feature_importances_", None)
        if gains is not None:
            gain_df = pd.DataFrame(
                {"encoded_feature": encoded_names, "gain_importance": gains}
            )
            gain_df["feature"] = [
                _parent_feature(c, originals) for c in gain_df["encoded_feature"]
            ]
            gain_df = (
                gain_df.groupby("feature", as_index=False)["gain_importance"]
                .sum()
                .sort_values("gain_importance", ascending=False)
            )
            gain_df["rank_gain"] = range(1, len(gain_df) + 1)
            results["gain"] = gain_df
    except Exception as exc:  # pragma: no cover
        log.warning("[%s] gain importance unavailable: %s", model_key, exc)

    # -- three-way comparison ----------------------------------------------
    if results:
        comparison = None
        for name in ("shap", "permutation", "gain"):
            if name not in results:
                continue
            cols = ["feature", f"rank_{name}"]
            part = results[name][cols]
            comparison = part if comparison is None else comparison.merge(
                part, on="feature", how="outer"
            )
        rank_cols = [c for c in comparison.columns if c.startswith("rank_")]
        comparison["rank_disagreement"] = comparison[rank_cols].max(axis=1) - (
            comparison[rank_cols].min(axis=1)
        )
        comparison = comparison.sort_values(
            rank_cols[0] if rank_cols else "feature"
        ).reset_index(drop=True)
        write_table(
            comparison,
            cfg.path("tables") / f"importance_comparison_{model_key}.csv",
        )
        results["comparison"] = comparison

        if "gender" in set(comparison["feature"]):
            row = comparison.loc[comparison["feature"] == "gender"].iloc[0]
            detail = {c: (None if pd.isna(row[c]) else int(row[c])) for c in rank_cols}
            log.info("gender ranking across methods: %s", detail)
            manifest.record_stage("stage13_gender_ranks", **detail)

    manifest.record_stage("stage13_explainability", model=model_key,
                          methods=list(results))
    return results


# ---------------------------------------------------------------------------
# Stage 14 - fairness
# ---------------------------------------------------------------------------

def fairness_audit(
    predictions: pd.DataFrame,
    model_key: str,
    threshold: float,
    cfg: Config,
    manifest: RunManifest,
) -> pd.DataFrame:
    """Subgroup discrimination, calibration and operating-point behaviour."""
    rows = []
    y_all = predictions["y_true"].to_numpy()
    p_all = predictions[model_key].to_numpy()
    overall_sens = classification_metrics(y_all, p_all, threshold)["sensitivity"]

    for attribute in cfg["fairness"]["subgroups"]:
        col = f"subgroup_{attribute}"
        if col not in predictions.columns:
            log.warning("subgroup column %s absent; skipped", col)
            continue

        for level, group in predictions.groupby(col, observed=True):
            y = group["y_true"].to_numpy()
            p = group[model_key].to_numpy()
            n, events = len(group), int(y.sum())
            adequate = (
                n >= cfg["fairness"]["min_subgroup_n"]
                and events >= cfg["fairness"]["min_subgroup_events"]
                and 0 < events < n
            )
            row: dict[str, Any] = {
                "attribute": attribute,
                "level": str(level),
                "n": n,
                "events": events,
                "prevalence": round(float(y.mean()), 4) if n else np.nan,
                "estimable": adequate,
            }
            if adequate:
                disc = discrimination_metrics(y, p)
                cal = calibration_metrics(y, p)
                clf = classification_metrics(y, p, threshold)
                row.update(
                    auroc=round(disc["auroc"], 4),
                    auprc=round(disc["auprc"], 4),
                    calibration_slope=round(cal["calibration_slope"], 3),
                    calibration_intercept=round(cal["calibration_intercept"], 3),
                    brier=round(cal["brier"], 4),
                    sensitivity=round(clf["sensitivity"], 4),
                    ppv=round(clf["ppv"], 4),
                    alert_rate=round(clf["alert_rate"], 4),
                    equal_opportunity_difference=round(
                        clf["sensitivity"] - overall_sens, 4
                    ),
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    write_table(table, cfg.path("tables") / f"table_fairness_{model_key}.csv")
    write_json(
        {"model": model_key, "threshold": threshold,
         "records": table.to_dict(orient="records")},
        cfg.path("metrics") / f"fairness_{model_key}.json",
    )
    manifest.record_stage(
        "stage14_fairness",
        model=model_key,
        subgroups_evaluated=int(table["estimable"].sum()),
        subgroups_total=len(table),
    )
    return table


# ---------------------------------------------------------------------------
# Stage 15 - sensitivity analyses
# ---------------------------------------------------------------------------

def _evaluate_arm(
    spec: ModelSpec,
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    cfg: Config,
    params: dict,
    class_weight: str | None = None,
    sampler: Any | None = None,
) -> dict[str, float]:
    est = build_estimator(
        spec, dev, seed=cfg["seed"], params=params,
        class_weight=class_weight, sampler=sampler,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(dev.drop(columns=[TARGET]), dev[TARGET].to_numpy())
    p = est.predict_proba(holdout.drop(columns=[TARGET]))[:, 1]
    y = holdout[TARGET].to_numpy()
    return {**discrimination_metrics(y, p), **calibration_metrics(y, p),
            "mean_predicted_risk": float(p.mean())}


def run_sensitivity_analyses(
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    full_matrix: pd.DataFrame,
    tuning: dict[str, dict],
    cfg: Config,
    manifest: RunManifest,
) -> pd.DataFrame:
    """Execute the five pre-specified sensitivity analyses."""
    specs = get_model_specs(cfg)
    primary_key = cfg["models"]["primary_comparison"][0]
    spec = specs[primary_key]
    params = tuning[primary_key]["best_params"]
    rows: list[dict] = []

    # -- 1. imbalance strategy (the study's principal contribution) --------
    if cfg["sensitivity"]["run"]["imbalance"]:
        for arm in cfg["sensitivity"]["imbalance_arms"]:
            sampler, weight = None, None
            try:
                if arm == "class_weight":
                    weight = "balanced"
                elif arm == "smote":
                    from imblearn.over_sampling import SMOTE

                    sampler = SMOTE(random_state=cfg["seed"])
                elif arm == "undersample":
                    from imblearn.under_sampling import RandomUnderSampler

                    sampler = RandomUnderSampler(random_state=cfg["seed"])
                metrics = _evaluate_arm(
                    spec, dev, holdout, cfg, params,
                    class_weight=weight, sampler=sampler,
                )
                rows.append({"analysis": "imbalance", "arm": arm, **metrics})
                log.info(
                    "[sensitivity/imbalance/%s] AUROC %.4f  slope %.3f  mean p %.4f",
                    arm, metrics["auroc"], metrics["calibration_slope"],
                    metrics["mean_predicted_risk"],
                )
            except Exception as exc:  # pragma: no cover
                log.warning("imbalance arm %s failed: %s", arm, exc)

    # -- 2. cohort: complete-case deletion (the original approach) ---------
    if cfg["sensitivity"]["run"]["missing_data"]:
        for arm in ("explicit_category", "complete_case"):
            try:
                d, h = dev, holdout
                if arm == "complete_case":
                    unknown_cols = [
                        c for c in cfg["features"]["unknown_category_columns"]
                        if c in dev.columns
                    ]
                    mask_d = ~(dev[unknown_cols] == "Unknown").any(axis=1)
                    mask_h = ~(holdout[unknown_cols] == "Unknown").any(axis=1)
                    d, h = dev.loc[mask_d], holdout.loc[mask_h]
                    log.info(
                        "complete-case arm retains %d/%d development records "
                        "(%.1f%%) - quantifies the original approach's cost",
                        len(d), len(dev), 100 * len(d) / len(dev),
                    )
                if len(h) < 200 or h[TARGET].nunique() < 2:
                    continue
                metrics = _evaluate_arm(spec, d, h, cfg, params)
                rows.append(
                    {"analysis": "missing_data", "arm": arm,
                     "n_development": len(d), "n_holdout": len(h), **metrics}
                )
            except Exception as exc:  # pragma: no cover
                log.warning("missing-data arm %s failed: %s", arm, exc)

    # -- 3. payer_code inclusion -------------------------------------------
    if cfg["sensitivity"]["run"]["payer_code"] and "payer_code" in dev.columns:
        for arm in ("included", "excluded"):
            d = dev if arm == "included" else dev.drop(columns=["payer_code"])
            h = holdout if arm == "included" else holdout.drop(columns=["payer_code"])
            try:
                metrics = _evaluate_arm(spec, d, h, cfg, params)
                rows.append({"analysis": "payer_code", "arm": arm, **metrics})
            except Exception as exc:  # pragma: no cover
                log.warning("payer_code arm %s failed: %s", arm, exc)

    # -- 4. glycemic variables (quantifies the original error) -------------
    if cfg["sensitivity"]["run"]["glycemic"]:
        glyc = [c for c in ["A1Cresult", "max_glu_serum", "A1C_tested",
                            "glucose_tested", "A1C_tested_x_change"]
                if c in dev.columns]
        for arm in ("included", "excluded"):
            d = dev if arm == "included" else dev.drop(columns=glyc)
            h = holdout if arm == "included" else holdout.drop(columns=glyc)
            try:
                metrics = _evaluate_arm(spec, d, h, cfg, params)
                rows.append({"analysis": "glycemic", "arm": arm, **metrics})
            except Exception as exc:  # pragma: no cover
                log.warning("glycemic arm %s failed: %s", arm, exc)

    table = pd.DataFrame(rows)
    if not table.empty:
        numeric = table.select_dtypes(include=[np.number]).columns
        table[numeric] = table[numeric].round(4)
    write_table(table, cfg.path("tables") / "table7_sensitivity.csv")
    write_json(table.to_dict(orient="records"),
               cfg.path("metrics") / "sensitivity.json")
    manifest.record_stage("stage15_sensitivity", arms=len(table))

    # The headline check: resampling should degrade calibration.
    if not table.empty and "arm" in table.columns:
        imb = table[table["analysis"] == "imbalance"].set_index("arm")
        if {"none", "smote"}.issubset(imb.index):
            log.info(
                "IMBALANCE CONTRAST | none: AUROC %.4f slope %.3f | "
                "smote: AUROC %.4f slope %.3f",
                imb.loc["none", "auroc"], imb.loc["none", "calibration_slope"],
                imb.loc["smote", "auroc"], imb.loc["smote", "calibration_slope"],
            )
    return table
