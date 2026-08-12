#!/usr/bin/env python3
"""Modular stage runner: execute one stage (or one model) per invocation.

Rationale
---------
The full pipeline is a multi-hour job, but each *stage* is short. Running
stages independently means:

* no single command needs to survive a long wall-clock window;
* any stage can be re-run in isolation after a fix, without recomputing
  everything upstream;
* completed work is never repeated, because each stage checkpoints its
  artifacts and skips if they already exist.

Every stage reads its inputs from disk and writes its outputs to disk. No
state is held in memory across stages.

Examples
--------
    python run_stage.py prep                      # stages 1-6
    python run_stage.py benchmark                 # time one fit per model
    python run_stage.py tune --model xgboost --trials 10   # resumable
    python run_stage.py fit                       # stage 8
    python run_stage.py evaluate                  # stages 9-12
    python run_stage.py explain                   # stages 13-14
    python run_stage.py sensitivity               # stage 15
    python run_stage.py status                    # what is done, what remains
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import build_analysis_matrix
from src.models import build_estimator, get_model_specs
from src.preprocessing import TARGET, make_split, sample_size_report
from src.reporting import apply_style, figure_flow_diagram, table_baseline_characteristics
from src.utils import RunManifest, get_logger, seed_everything, setup_logging, write_json

log = get_logger("stage")


def _setup(config_path: str):
    cfg = load_config(config_path)
    setup_logging(cfg.path("logs"))
    seed_everything(cfg["seed"])
    manifest = RunManifest(cfg.path("logs") / "run_manifest.json")
    apply_style(cfg)
    return cfg, manifest


def _load_split(cfg):
    dev = pd.read_parquet(cfg.path("processed") / "development.parquet")
    hold = pd.read_parquet(cfg.path("processed") / "holdout.parquet")
    return dev, hold


# ---------------------------------------------------------------------------
# prep: stages 1-6
# ---------------------------------------------------------------------------

def cmd_prep(cfg, manifest, args) -> None:
    out = cfg.path("processed") / "development.parquet"
    if out.exists() and not args.force:
        log.info("development set already exists; use --force to rebuild")
        return
    matrix, flow = build_analysis_matrix(cfg, manifest)
    table_baseline_characteristics(matrix, cfg)
    figure_flow_diagram(flow, cfg)
    dev, _ = make_split(matrix, cfg, manifest)
    n_params = sum(
        1 if pd.api.types.is_numeric_dtype(matrix[c]) and matrix[c].nunique() > 2
        else max(int(matrix[c].nunique()) - 1, 1)
        for c in matrix.columns if c != TARGET
    )
    sample_size_report(dev, n_params, cfg, manifest)
    log.info("prep complete: %d features -> %d encoded parameters",
             matrix.shape[1] - 1, n_params)


# ---------------------------------------------------------------------------
# benchmark: measure a single fit per model to budget the search
# ---------------------------------------------------------------------------

def cmd_benchmark(cfg, manifest, args) -> None:
    """Time one fit per model on a real CV training fold.

    Budgeting must be empirical. Projecting a search from an assumed fit time
    is how an overnight run becomes a three-day run.
    """
    from sklearn.model_selection import StratifiedKFold

    dev, _ = _load_split(cfg)
    X, y = dev.drop(columns=[TARGET]), dev[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                          random_state=cfg["seed"])
    tr, _ = next(skf.split(X, y))

    rows = []
    for key, spec in get_model_specs(cfg).items():
        est = build_estimator(spec, dev, seed=cfg["seed"])
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y[tr])
        secs = time.perf_counter() - t0
        rows.append({"model": key, "single_fit_seconds": round(secs, 2)})
        log.info("[%s] single fit: %.2fs", key, secs)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.path("metrics") / "benchmark_fits.csv", index=False)

    folds = cfg["cv"]["n_splits"] * cfg["cv"]["tuning_n_repeats"]
    print("\n  model                 1 fit    per trial   x10 trials   x30   x100")
    print("  " + "-" * 68)
    for r in rows:
        s = r["single_fit_seconds"]
        print(f"  {r['model']:<20} {s:>6.1f}s  {s*folds:>8.1f}s  "
              f"{s*folds*10/60:>9.1f}m {s*folds*30/60:>6.1f}m {s*folds*100/60:>6.1f}m")
    total = sum(r["single_fit_seconds"] for r in rows) * folds
    print(f"\n  All models, per trial: {total/60:.1f} min "
          f"| x30 trials: {total*30/3600:.1f} h | x100: {total*100/3600:.1f} h\n")


# ---------------------------------------------------------------------------
# tune: resumable Optuna study, N trials per invocation
# ---------------------------------------------------------------------------

def cmd_tune(cfg, manifest, args) -> None:
    """Append trials to a persistent Optuna study.

    The study lives in SQLite, so an invocation that is interrupted loses at
    most the trial in flight. Re-invoking resumes from the completed trials
    and continues toward the target, which makes a long search safe to run in
    short bounded slices.
    """
    import optuna
    from sklearn.metrics import log_loss
    from sklearn.model_selection import RepeatedStratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev, _ = _load_split(cfg)
    X, y = dev.drop(columns=[TARGET]), dev[TARGET].to_numpy()
    specs = get_model_specs(cfg)
    targets = [args.model] if args.model else list(specs)

    storage = f"sqlite:///{cfg.path('models') / 'optuna.db'}"
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        n_repeats=cfg["cv"]["tuning_n_repeats"],
        random_state=cfg["seed"],
    )

    for key in targets:
        spec = specs[key]
        if not spec.tunable or spec.search_space is None:
            log.info("[%s] untuned by design; skipping", key)
            continue

        study = optuna.create_study(
            study_name=f"tune_{key}",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8, n_warmup_steps=1
            ),
            load_if_exists=True,
        )
        # Budget counted in ATTEMPTED trials (Optuna convention): pruned
        # trials are partially evaluated and still inform the TPE sampler,
        # so they consume budget. Completed and pruned are reported separately.
        done = len([t for t in study.trials
                    if t.state in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED)])
        target = args.target or cfg["tuning"]["n_trials"]
        remaining = max(0, target - done)
        run_now = min(args.trials, remaining) if args.trials else remaining
        if run_now == 0:
            log.info("[%s] target of %d trials already reached", key, target)
            continue
        log.info("[%s] %d/%d complete; running %d more", key, done, target, run_now)

        folds = list(cv.split(X, y))

        def objective(trial):
            """Evaluate fold by fold so that hopeless configurations can be
            pruned before consuming the full cross-validation budget.

            Pruning discards only configurations already performing worse than
            the median of completed trials at the same fold; it does not
            restrict the search space, so the optimum remains reachable.
            """
            params = spec.search_space(trial)
            scores = []
            for step, (tr_idx, va_idx) in enumerate(folds):
                est = build_estimator(spec, dev, seed=cfg["seed"], params=params)
                est.fit(X.iloc[tr_idx], y[tr_idx])
                p = est.predict_proba(X.iloc[va_idx])[:, 1]
                scores.append(-log_loss(y[va_idx], np.clip(p, 1e-12, 1 - 1e-12)))
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study.optimize(objective, n_trials=run_now, show_progress_bar=False)
        n_pruned = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED])
        log.info("[%s] best value %.5f (%d pruned) params %s",
                 key, study.best_value, n_pruned, study.best_params)

    _write_tuning_results(cfg, manifest)


def _write_tuning_results(cfg, manifest) -> None:
    """Consolidate completed studies into tuning_results.json and Table 3."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{cfg.path('models') / 'optuna.db'}"
    specs = get_model_specs(cfg)
    results, rows = {}, []

    for key, spec in specs.items():
        if not spec.tunable or spec.search_space is None:
            results[key] = {"model": key, "best_params": {}, "n_trials": 0,
                            "cv": {}, "status": "untuned baseline"}
            rows.append({"model": spec.display_name, "n_trials": 0,
                         "selected_hyperparameters": "defaults (untuned baseline)",
                         "best_cv_objective": ""})
            continue
        try:
            study = optuna.load_study(study_name=f"tune_{key}", storage=storage)
            complete = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            results[key] = {
                "model": key,
                "best_params": study.best_params,
                "n_trials": len(complete),
                "best_value": study.best_value,
                "cv": {"neg_log_loss_mean": study.best_value},
            }
            rows.append({
                "model": spec.display_name,
                "n_trials": len(complete),
                "selected_hyperparameters": "; ".join(
                    f"{k}={round(v,4) if isinstance(v,float) else v}"
                    for k, v in study.best_params.items()),
                "best_cv_objective": round(study.best_value, 5),
            })
        except KeyError:
            log.warning("[%s] no study found yet", key)

    write_json(results, cfg.path("metrics") / "tuning_results.json")
    pd.DataFrame(rows).to_csv(
        cfg.path("tables") / "table3_hyperparameters.csv", index=False)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(cfg, manifest, args) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("\n=== PIPELINE STATUS ===\n")
    checks = [
        ("analysis matrix", cfg.path("processed") / "analysis_matrix.parquet"),
        ("development set", cfg.path("processed") / "development.parquet"),
        ("hold-out set", cfg.path("processed") / "holdout.parquet"),
        ("hold-out predictions",
         cfg.path("predictions") / "holdout_probabilities.csv"),
        ("performance metrics", cfg.path("metrics") / "performance.json"),
        ("sensitivity results", cfg.path("tables") / "table7_sensitivity.csv"),
    ]
    for label, path in checks:
        print(f"  [{'x' if path.exists() else ' '}] {label}")

    db = cfg.path("models") / "optuna.db"
    if db.exists():
        print("\n  Tuning progress:")
        storage = f"sqlite:///{db}"
        for key, spec in get_model_specs(cfg).items():
            if not spec.tunable:
                print(f"    {key:<20} untuned by design")
                continue
            try:
                st = optuna.load_study(study_name=f"tune_{key}", storage=storage)
                n = len([t for t in st.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
                print(f"    {key:<20} {n:>3}/{cfg['tuning']['n_trials']} trials"
                      f"   best={st.best_value:.5f}" if n else
                      f"    {key:<20} {n:>3}/{cfg['tuning']['n_trials']} trials")
            except Exception:
                print(f"    {key:<20}   0 trials")
    for key in get_model_specs(cfg):
        p = cfg.path("models") / f"final_{key}.pkl"
        if p.exists():
            print(f"    fitted model saved: {key}")
    print()



# ---------------------------------------------------------------------------
# convergence / verify / fit / evaluate / explain / sensitivity
# ---------------------------------------------------------------------------

def cmd_convergence(cfg, manifest, args) -> None:
    from src.stages_post import assess_convergence, export_optuna_history, figure_convergence
    hist = export_optuna_history(cfg)
    conv = assess_convergence(hist, cfg, manifest)
    figure_convergence(hist, cfg)
    print("\n=== CONVERGENCE ASSESSMENT ===")
    print(conv.to_string(index=False))
    still = conv[~conv.converged.astype(bool)]
    if len(still):
        print("\nMODELS REQUIRING EXTENSION:", ", ".join(still.model.tolist()))
    else:
        print("\nAll tuned studies have plateaued.")


def cmd_verify(cfg, manifest, args) -> None:
    from src.stages_post import verify_no_leakage
    checks = verify_no_leakage(cfg, manifest)
    print("\n=== LEAKAGE VERIFICATION ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")


def cmd_fit(cfg, manifest, args) -> None:
    from src.training import fit_final_models, generate_predictions
    dev, hold = _load_split(cfg)
    tuning = json.loads((cfg.path("metrics") / "tuning_results.json").read_text())
    fitted = fit_final_models(dev, tuning, cfg, manifest)
    generate_predictions(fitted, hold, dev, cfg, manifest)


def cmd_evaluate(cfg, manifest, args) -> None:
    """Stages 9-12: performance, calibration, thresholds, inference, DCA."""
    import numpy as np
    from src.pipeline import _evaluate_models, _statistical_comparisons, _threshold_analysis
    from src.evaluation import calibration_metrics, decision_curve
    from src.training import calibrate_model
    from src.reporting import (figure_calibration, figure_decision_curve,
                               figure_pr_curves, figure_roc_curves,
                               table_literature_comparison)
    dev, hold = _load_split(cfg)
    specs = get_model_specs(cfg)
    labels = {k: s.display_name for k, s in specs.items()}
    keys = list(specs)
    ph = pd.read_csv(cfg.path("predictions") / "holdout_probabilities.csv")
    pd_ = pd.read_csv(cfg.path("predictions") / "development_probabilities.csv")
    y = ph.y_true.to_numpy()

    metrics, calib = _evaluate_models(ph, keys, cfg, manifest)
    thr = _threshold_analysis(pd_, ph, keys, cfg, manifest)
    comp = _statistical_comparisons(ph, keys, thr, cfg, manifest)

    grid = np.arange(cfg["dca"]["threshold_min"],
                     cfg["dca"]["threshold_max"] + 1e-9, cfg["dca"]["threshold_step"])
    curves = {k: decision_curve(y, ph[k].to_numpy(), grid) for k in keys}
    for k, d in curves.items():
        d.to_csv(cfg.path("metrics") / f"dca_{k}.csv", index=False)
    figure_decision_curve(curves, labels, cfg)
    figure_roc_curves(ph, keys, labels, metrics, cfg)
    figure_pr_curves(ph, keys, labels, metrics, cfg)
    figure_calibration(ph, keys, labels, calib, cfg)

    best = max(keys, key=lambda k: metrics[k]["auroc"]["estimate"])
    log.info("best model: %s", best)
    tuning = json.loads((cfg.path("metrics") / "tuning_results.json").read_text())
    try:
        pc = calibrate_model(specs[best], dev, hold, tuning[best]["best_params"], cfg)
        ph[f"{best}_calibrated"] = pc
        calib[f"{best}_calibrated"] = calibration_metrics(y, pc)
        ph.to_csv(cfg.path("predictions") / "holdout_probabilities.csv", index=False)
        figure_calibration(ph, [best, f"{best}_calibrated"],
                           {**labels, f"{best}_calibrated": labels[best] + " (recalibrated)"},
                           calib, cfg, name="fig4b_calibration_recalibrated",
                           title="Figure 4b. Effect of recalibration")
    except Exception as exc:
        log.warning("recalibration failed: %s", exc)

    a = metrics[best]["auroc"]
    table_literature_comparison(cfg, own={
        "n": f"{len(dev)+len(hold):,} patients",
        "auc_summary": f"{labels[best]} {a['estimate']:.3f} (95% CI {a['ci_low']:.3f}-{a['ci_high']:.3f})"})
    write_json({"best_model": best, "models": metrics, "calibration": calib,
                "thresholds": thr, "comparisons": comp,
                "cohort_n": len(dev) + len(hold),
                "prevalence": float(y.mean())},
               cfg.path("metrics") / "summary.json")


def cmd_explain(cfg, manifest, args) -> None:
    from src.analysis import compute_importances, fairness_audit
    from src.reporting import figure_importance_comparison, figure_subgroup_forest
    _, hold = _load_split(cfg)
    summary = json.loads((cfg.path("metrics") / "summary.json").read_text())
    best = args.model or summary["best_model"]
    est = joblib.load(cfg.path("models") / f"final_{best}.pkl")
    imp = compute_importances(best, est, hold, cfg, manifest)
    figure_importance_comparison(imp, cfg, best)
    ph = pd.read_csv(cfg.path("predictions") / "holdout_probabilities.csv")
    fair = fairness_audit(ph, best, summary["thresholds"][best]["capacity"], cfg, manifest)
    figure_subgroup_forest(fair, cfg)


def cmd_sensitivity(cfg, manifest, args) -> None:
    from src.analysis import run_sensitivity_analyses
    dev, hold = _load_split(cfg)
    matrix = pd.read_parquet(cfg.path("processed") / "analysis_matrix.parquet")
    tuning = json.loads((cfg.path("metrics") / "tuning_results.json").read_text())
    run_sensitivity_analyses(dev, hold, matrix, tuning, cfg, manifest)

COMMANDS = {
    "prep": cmd_prep,
    "benchmark": cmd_benchmark,
    "tune": cmd_tune,
    "convergence": cmd_convergence,
    "verify": cmd_verify,
    "fit": cmd_fit,
    "evaluate": cmd_evaluate,
    "explain": cmd_explain,
    "sensitivity": cmd_sensitivity,
    "status": cmd_status,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="restrict to a single model")
    ap.add_argument("--trials", type=int, default=None,
                    help="trials to run in this invocation")
    ap.add_argument("--target", type=int, default=None,
                    help="override the total trial target")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    cfg, manifest = _setup(args.config)
    COMMANDS[args.command](cfg, manifest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
