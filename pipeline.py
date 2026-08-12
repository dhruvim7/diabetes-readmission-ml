"""End-to-end orchestration of the twenty analysis stages.

Stages 1-8 are strictly sequential. Stages 9-14 all read the same persisted
prediction file and could be parallelised. Stages 16-20 assemble outputs.

Every stage writes its artifacts to disk before the next begins, so any stage
can be re-run in isolation and every manuscript number remains traceable to
the file that produced it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .analysis import compute_importances, fairness_audit, run_sensitivity_analyses
from .config import Config, load_config
from .data import build_analysis_matrix
from .evaluation import (
    bootstrap_ci,
    bootstrap_difference,
    calibration_metrics,
    classification_metrics,
    decision_curve,
    delong_test,
    discrimination_metrics,
    holm_correction,
    mcnemar_test,
    select_thresholds,
)
from .models import get_model_specs
from .preprocessing import TARGET, make_split, sample_size_report
from .reporting import (
    apply_style,
    figure_calibration,
    figure_decision_curve,
    figure_flow_diagram,
    figure_importance_comparison,
    figure_pr_curves,
    figure_roc_curves,
    figure_subgroup_forest,
    table_baseline_characteristics,
    table_literature_comparison,
)
from .training import (
    calibrate_model,
    fit_final_models,
    generate_predictions,
    tune_all_models,
)
from .utils import (
    RunManifest,
    gate,
    get_logger,
    seed_everything,
    setup_logging,
    stage_timer,
    write_json,
    write_table,
)

log = get_logger("pipeline")

__all__ = ["run_pipeline"]


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Execute the full analysis and return a summary of key results."""
    cfg = load_config(config_path)
    setup_logging(cfg.path("logs"))
    seed_everything(cfg["seed"])
    manifest = RunManifest(cfg.path("logs") / "run_manifest.json")
    apply_style(cfg)

    log.info("Readmission prediction pipeline | seed=%d", cfg["seed"])
    specs = get_model_specs(cfg)
    labels = {k: s.display_name for k, s in specs.items()}
    model_keys = list(specs)

    # ---------------------------------------------------------------- 1-5
    with stage_timer("stages 1-5: data preparation", manifest):
        matrix, flow = build_analysis_matrix(cfg, manifest)

    with stage_timer("stage 3b: baseline characteristics", manifest):
        table_baseline_characteristics(matrix, cfg)
        figure_flow_diagram(flow, cfg)

    # ------------------------------------------------------------------ 6
    with stage_timer("stage 6: split and sample size", manifest):
        dev, holdout = make_split(matrix, cfg, manifest)
        n_params = _estimate_parameters(matrix)
        sample_size_report(dev, n_params, cfg, manifest)

    # ------------------------------------------------------------------ 7
    with stage_timer("stage 7: hyperparameter tuning", manifest):
        tuning = tune_all_models(dev, cfg, manifest)

    # ------------------------------------------------------------------ 8
    with stage_timer("stage 8: final fitting and predictions", manifest):
        fitted = fit_final_models(dev, tuning, cfg, manifest)
        pred_hold, pred_dev = generate_predictions(
            fitted, holdout, dev, cfg, manifest
        )

    y_hold = pred_hold["y_true"].to_numpy()

    # ---------------------------------------------------------------- 9-11
    with stage_timer("stages 9-11: performance, calibration, inference", manifest):
        metrics, calibration = _evaluate_models(
            pred_hold, model_keys, cfg, manifest
        )
        thresholds = _threshold_analysis(
            pred_dev, pred_hold, model_keys, cfg, manifest
        )
        comparisons = _statistical_comparisons(
            pred_hold, model_keys, thresholds, cfg, manifest
        )

    # ----------------------------------------------------------------- 12
    with stage_timer("stage 12: decision curve analysis", manifest):
        grid = np.arange(
            cfg["dca"]["threshold_min"],
            cfg["dca"]["threshold_max"] + 1e-9,
            cfg["dca"]["threshold_step"],
        )
        curves = {
            k: decision_curve(y_hold, pred_hold[k].to_numpy(), grid)
            for k in model_keys
        }
        for k, df in curves.items():
            write_table(df, cfg.path("metrics") / f"dca_{k}.csv")
        figure_decision_curve(curves, labels, cfg)

    # ----------------------------------------------------------------- 9b
    best_key = max(
        model_keys, key=lambda k: metrics[k]["auroc"]["estimate"]
    )
    log.info("best model by hold-out AUROC: %s", labels[best_key])

    with stage_timer("stage 9b: recalibration of the best model", manifest):
        try:
            p_cal = calibrate_model(
                specs[best_key], dev, holdout, tuning[best_key]["best_params"], cfg
            )
            pred_hold[f"{best_key}_calibrated"] = p_cal
            calibration[f"{best_key}_calibrated"] = calibration_metrics(y_hold, p_cal)
            write_table(
                pred_hold, cfg.path("predictions") / "holdout_probabilities.csv"
            )
            figure_calibration(
                pred_hold, [best_key, f"{best_key}_calibrated"],
                {**labels, f"{best_key}_calibrated": f"{labels[best_key]} (recalibrated)"},
                calibration, cfg,
                name="fig4b_calibration_recalibrated",
                title="Figure 4b. Effect of recalibration",
            )
        except Exception as exc:  # pragma: no cover
            log.warning("recalibration failed: %s", exc)

    # ------------------------------------------------------------ figures
    with stage_timer("stage 16: discrimination and calibration figures", manifest):
        figure_roc_curves(pred_hold, model_keys, labels, metrics, cfg)
        figure_pr_curves(pred_hold, model_keys, labels, metrics, cfg)
        figure_calibration(pred_hold, model_keys, labels, calibration, cfg)

    # ----------------------------------------------------------------- 13
    with stage_timer("stage 13: explainability", manifest):
        importances = compute_importances(
            best_key, fitted[best_key], holdout, cfg, manifest
        )
        figure_importance_comparison(importances, cfg, best_key)

    # ----------------------------------------------------------------- 14
    with stage_timer("stage 14: fairness audit", manifest):
        fair = fairness_audit(
            pred_hold, best_key, thresholds[best_key]["capacity"], cfg, manifest
        )
        figure_subgroup_forest(fair, cfg)

    # ----------------------------------------------------------------- 15
    with stage_timer("stage 15: sensitivity analyses", manifest):
        sensitivity = run_sensitivity_analyses(
            dev, holdout, matrix, tuning, cfg, manifest
        )

    # -------------------------------------------------------------- 17-18
    with stage_timer("stage 17: manuscript tables", manifest):
        auc = metrics[best_key]["auroc"]
        table_literature_comparison(
            cfg,
            own={
                "n": f"{len(matrix):,} patients",
                "auc_summary": (
                    f"{labels[best_key]} {auc['estimate']:.3f} "
                    f"(95% CI {auc['ci_low']:.3f}-{auc['ci_high']:.3f})"
                ),
            },
        )

    summary = {
        "cohort_n": int(len(matrix)),
        "prevalence": float(matrix[TARGET].mean()),
        "development_n": int(len(dev)),
        "holdout_n": int(len(holdout)),
        "best_model": best_key,
        "best_auroc": metrics[best_key]["auroc"],
        "models": {k: metrics[k] for k in model_keys},
        "calibration": calibration,
        "thresholds": thresholds,
        "comparisons": comparisons,
        "row_counts": manifest.row_counts,
    }
    write_json(summary, cfg.path("metrics") / "summary.json")
    manifest.record_stage("pipeline_complete", best_model=best_key)
    log.info("PIPELINE COMPLETE - all artifacts written to outputs/")
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_parameters(df: pd.DataFrame) -> int:
    """Approximate the post-encoding parameter count for the sample-size report."""
    n = 0
    for col in df.columns:
        if col == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
            n += 1
        else:
            n += max(int(df[col].nunique()) - 1, 1)
    return n


def _evaluate_models(
    pred: pd.DataFrame, model_keys: list[str], cfg: Config, manifest: RunManifest
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Hold-out discrimination and calibration, every metric with a bootstrap CI."""
    y = pred["y_true"].to_numpy()
    n_boot = cfg["bootstrap"]["n_resamples"]
    alpha = cfg["bootstrap"]["alpha"]
    seed = cfg["seed"]

    metric_fns = {
        "auroc": lambda a, b: roc_auc_score(a, b),
        "auprc": lambda a, b: average_precision_score(a, b),
        "brier": lambda a, b: brier_score_loss(a, b),
    }

    metrics: dict[str, dict] = {}
    calibration: dict[str, dict] = {}
    rows: list[dict] = []

    for key in model_keys:
        p = pred[key].to_numpy()
        entry = {
            name: bootstrap_ci(y, p, fn, n_boot, alpha, seed,
                               cfg["bootstrap"]["stratified"])
            for name, fn in metric_fns.items()
        }
        metrics[key] = entry
        cal = calibration_metrics(y, p)
        calibration[key] = cal
        disc = discrimination_metrics(y, p)

        rows.append(
            {
                "model": key,
                "auroc": f"{entry['auroc']['estimate']:.3f} "
                         f"({entry['auroc']['ci_low']:.3f}-{entry['auroc']['ci_high']:.3f})",
                "auprc": f"{entry['auprc']['estimate']:.3f} "
                         f"({entry['auprc']['ci_low']:.3f}-{entry['auprc']['ci_high']:.3f})",
                "auprc_baseline": round(disc["auprc_baseline"], 4),
                "brier": f"{entry['brier']['estimate']:.4f} "
                         f"({entry['brier']['ci_low']:.4f}-{entry['brier']['ci_high']:.4f})",
                "brier_scaled": round(cal["brier_scaled"], 4),
                "calibration_slope": round(cal["calibration_slope"], 3),
                "calibration_intercept": round(cal["calibration_intercept"], 3),
                "ici": round(cal["ici"], 4),
                "log_loss": round(disc["log_loss"], 4),
            }
        )
        log.info(
            "[%s] AUROC %.4f (%.4f-%.4f)  AUPRC %.4f  slope %.3f  Brier %.4f",
            key, entry["auroc"]["estimate"], entry["auroc"]["ci_low"],
            entry["auroc"]["ci_high"], entry["auprc"]["estimate"],
            cal["calibration_slope"], cal["brier"],
        )

    write_table(pd.DataFrame(rows), cfg.path("tables") / "table4_performance.csv")
    write_json({"discrimination": metrics, "calibration": calibration},
               cfg.path("metrics") / "performance.json")

    best = max(m["auroc"]["estimate"] for m in metrics.values())
    gate(
        "holdout_auroc_plausible",
        best <= cfg["sanity"]["max_plausible_auroc"],
        f"best hold-out AUROC = {best:.4f} (ceiling "
        f"{cfg['sanity']['max_plausible_auroc']}); higher indicates leakage",
        manifest,
        fatal=cfg["sanity"]["abort_on_leakage"],
    )
    return metrics, calibration


def _threshold_analysis(
    pred_dev: pd.DataFrame,
    pred_hold: pd.DataFrame,
    model_keys: list[str],
    cfg: Config,
    manifest: RunManifest,
) -> dict[str, dict[str, float]]:
    """Derive operating points on development data, evaluate on hold-out."""
    y_dev = pred_dev["y_true"].to_numpy()
    y_hold = pred_hold["y_true"].to_numpy()
    thresholds: dict[str, dict[str, float]] = {}
    rows: list[dict] = []

    for key in model_keys:
        ts = select_thresholds(
            y_dev, pred_dev[key].to_numpy(),
            cfg["thresholds"]["capacity_fraction"],
            cfg["thresholds"]["target_sensitivity"],
        )
        thresholds[key] = ts.as_dict()
        for name, thr in ts.as_dict().items():
            m = classification_metrics(y_hold, pred_hold[key].to_numpy(), thr)
            rows.append({"model": key, "operating_point": name,
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in m.items()}})

    table = pd.DataFrame(rows)
    write_table(table, cfg.path("tables") / "table6_threshold_performance.csv")
    write_json(thresholds, cfg.path("metrics") / "thresholds.json")
    manifest.record_stage("stage10_thresholds", models=len(thresholds))
    return thresholds


def _statistical_comparisons(
    pred: pd.DataFrame,
    model_keys: list[str],
    thresholds: dict[str, dict[str, float]],
    cfg: Config,
    manifest: RunManifest,
) -> dict[str, Any]:
    """Pre-specified primary comparison, then Holm-corrected secondaries."""
    y = pred["y_true"].to_numpy()
    primary_a, primary_b = cfg["models"]["primary_comparison"]
    results: dict[str, Any] = {}
    rows: list[dict] = []

    if primary_a in pred.columns and primary_b in pred.columns:
        d = delong_test(y, pred[primary_a].to_numpy(), pred[primary_b].to_numpy())
        diff_ap = bootstrap_difference(
            y, pred[primary_a].to_numpy(), pred[primary_b].to_numpy(),
            lambda a, b: average_precision_score(a, b),
            cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["alpha"], cfg["seed"],
        )
        mc = mcnemar_test(
            y, pred[primary_a].to_numpy(), pred[primary_b].to_numpy(),
            thresholds[primary_a]["capacity"], thresholds[primary_b]["capacity"],
        )
        results["primary"] = {
            "comparison": f"{primary_a} vs {primary_b}",
            "delong": d, "auprc_difference": diff_ap, "mcnemar": mc,
        }
        rows.append(
            {
                "comparison": f"{primary_a} vs {primary_b}",
                "type": "primary (pre-specified)",
                "auc_a": round(d["auc_a"], 4), "auc_b": round(d["auc_b"], 4),
                "difference": round(d["difference"], 4),
                "ci": f"{d['ci_low']:.4f} to {d['ci_high']:.4f}",
                "p_value": d["p_value"], "p_adjusted": d["p_value"],
                "mcnemar_p": mc["p_value"],
            }
        )
        log.info(
            "PRIMARY COMPARISON %s vs %s: dAUC %.4f (95%% CI %.4f-%.4f), p=%.4g",
            primary_a, primary_b, d["difference"], d["ci_low"], d["ci_high"],
            d["p_value"],
        )

    secondary, raw_p = [], []
    for i, a in enumerate(model_keys):
        for b in model_keys[i + 1:]:
            if {a, b} == {primary_a, primary_b}:
                continue
            d = delong_test(y, pred[a].to_numpy(), pred[b].to_numpy())
            secondary.append((a, b, d))
            raw_p.append(d["p_value"])

    adjusted = holm_correction([p if np.isfinite(p) else 1.0 for p in raw_p])
    for (a, b, d), padj in zip(secondary, adjusted):
        rows.append(
            {
                "comparison": f"{a} vs {b}", "type": "secondary (Holm-corrected)",
                "auc_a": round(d["auc_a"], 4), "auc_b": round(d["auc_b"], 4),
                "difference": round(d["difference"], 4),
                "ci": f"{d['ci_low']:.4f} to {d['ci_high']:.4f}",
                "p_value": d["p_value"], "p_adjusted": padj, "mcnemar_p": np.nan,
            }
        )
    results["secondary"] = [
        {"comparison": f"{a} vs {b}", **d, "p_adjusted": padj}
        for (a, b, d), padj in zip(secondary, adjusted)
    ]

    table = pd.DataFrame(rows)
    write_table(table, cfg.path("tables") / "table5_statistical_comparisons.csv")
    write_json(results, cfg.path("metrics") / "statistical_tests.json")
    manifest.record_stage("stage11_comparisons", n_comparisons=len(rows))
    return results
