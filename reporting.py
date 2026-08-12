"""Stages 16-17: publication figures and manuscript tables.

Every figure is rendered from persisted metric files, never from live model
objects. This is deliberate: regenerating a figure from an in-memory model
after a partial re-run is how a manuscript ends up with a figure that
contradicts its own table, a discrepancy invisible to the author and obvious
to a reviewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import Config
from .evaluation import calibration_curve_points
from .utils import get_logger, write_table

log = get_logger("reporting")

__all__ = [
    "apply_style",
    "figure_calibration",
    "figure_decision_curve",
    "figure_flow_diagram",
    "figure_importance_comparison",
    "figure_pr_curves",
    "figure_roc_curves",
    "figure_subgroup_forest",
    "save_figure",
    "table_baseline_characteristics",
    "table_literature_comparison",
]

# Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]


def apply_style(cfg: Config) -> None:
    """Consistent typography and axes for every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": cfg["figures"]["dpi"],
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "figure.autolayout": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, cfg: Config) -> list[Path]:
    """Write a figure in every configured format."""
    paths = []
    for fmt in cfg["figures"]["formats"]:
        p = cfg.path("figures") / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    log.info("figure saved: %s (%s)", name, ", ".join(cfg["figures"]["formats"]))
    return paths


# ---------------------------------------------------------------------------
# Figure 1 - participant flow
# ---------------------------------------------------------------------------

def figure_flow_diagram(flow: pd.DataFrame, cfg: Config) -> list[Path]:
    """CONSORT-style participant flow.

    This is the figure that resolves the credibility problem created by an
    unexplained cohort size: every exclusion is visible and the counts sum.
    """
    fig, ax = plt.subplots(figsize=(7.2, 1.75 * len(flow)))
    ax.set_axis_off()
    n_steps = len(flow)
    box_h, gap = 0.62, 0.38
    y = n_steps * (box_h + gap)

    for i, row in flow.iterrows():
        ax.add_patch(
            plt.Rectangle((0.06, y), 0.52, box_h, facecolor="#EEF3F8",
                          edgecolor="#33475B", linewidth=1.2)
        )
        ax.text(0.32, y + box_h / 2, f"{row['step']}\nn = {row['n']:,}",
                ha="center", va="center", fontsize=9.5)
        if i < n_steps - 1 and row_excluded(flow, i):
            nxt = flow.iloc[i + 1]
            ax.annotate("", xy=(0.32, y - gap), xytext=(0.32, y),
                        arrowprops=dict(arrowstyle="-|>", color="#33475B", lw=1.2))
            ax.add_patch(
                plt.Rectangle((0.63, y - gap - 0.12), 0.33, box_h * 0.72,
                              facecolor="#FBEFE7", edgecolor="#B45309",
                              linewidth=1.0, linestyle="--")
            )
            ax.text(
                0.795, y - gap - 0.12 + box_h * 0.36,
                f"Excluded n = {nxt['excluded']:,}\n{_wrap(nxt['reason'], 34)}",
                ha="center", va="center", fontsize=7.6, color="#7C2D12",
            )
        y -= box_h + gap

    ax.set_xlim(0, 1)
    ax.set_ylim(y, n_steps * (box_h + gap) + box_h + 0.15)
    ax.set_title("Figure 1. Participant flow", loc="left", fontsize=11, pad=8)
    return save_figure(fig, "fig1_participant_flow", cfg)


def row_excluded(flow: pd.DataFrame, i: int) -> bool:
    return i + 1 < len(flow) and int(flow.iloc[i + 1]["excluded"]) > 0


def _wrap(text: str, width: int) -> str:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures 2-3 - discrimination
# ---------------------------------------------------------------------------

def figure_roc_curves(
    predictions: pd.DataFrame,
    model_keys: Sequence[str],
    labels: dict[str, str],
    metrics: dict[str, dict],
    cfg: Config,
) -> list[Path]:
    y = predictions["y_true"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for i, key in enumerate(model_keys):
        fpr, tpr, _ = roc_curve(y, predictions[key].to_numpy())
        m = metrics.get(key, {}).get("auroc", {})
        legend = labels.get(key, key)
        if m:
            legend += f" (AUROC {m['estimate']:.3f}, 95% CI {m['ci_low']:.3f}-{m['ci_high']:.3f})"
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], label=legend)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Chance")
    ax.set_xlabel("1 - Specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.set_title("Figure 2. Receiver operating characteristic curves", loc="left")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return save_figure(fig, "fig2_roc_curves", cfg)


def figure_pr_curves(
    predictions: pd.DataFrame,
    model_keys: Sequence[str],
    labels: dict[str, str],
    metrics: dict[str, dict],
    cfg: Config,
) -> list[Path]:
    """Precision-recall curves. At ~9% prevalence these are more informative
    than ROC curves, and the prevalence baseline is drawn explicitly."""
    y = predictions["y_true"].to_numpy()
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for i, key in enumerate(model_keys):
        prec, rec, _ = precision_recall_curve(y, predictions[key].to_numpy())
        m = metrics.get(key, {}).get("auprc", {})
        legend = labels.get(key, key)
        if m:
            legend += f" (AUPRC {m['estimate']:.3f})"
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], label=legend)
    ax.axhline(prevalence, color="k", ls="--", lw=1.0, alpha=0.6,
               label=f"Prevalence baseline ({prevalence:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (positive predictive value)")
    ax.set_title("Figure 3. Precision-recall curves", loc="left")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    return save_figure(fig, "fig3_pr_curves", cfg)


# ---------------------------------------------------------------------------
# Figure 4 - calibration
# ---------------------------------------------------------------------------

def figure_calibration(
    predictions: pd.DataFrame,
    model_keys: Sequence[str],
    labels: dict[str, str],
    calibration: dict[str, dict],
    cfg: Config,
    name: str = "fig4_calibration",
    title: str = "Figure 4. Calibration",
) -> list[Path]:
    y = predictions["y_true"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    for i, key in enumerate(model_keys):
        pts = calibration_curve_points(
            y, predictions[key].to_numpy(),
            n_bins=cfg["calibration"]["n_bins"],
            strategy=cfg["calibration"]["binning"],
        )
        c = PALETTE[i % len(PALETTE)]
        cal = calibration[key] if key in calibration else {}
        legend = labels.get(key, key)
        if cal:
            legend += f" (slope {cal['calibration_slope']:.2f})"
        ax.plot(pts["mean_predicted"], pts["observed"], "o-", color=c,
                markersize=4, label=legend)
        ax.fill_between(pts["mean_predicted"], pts["ci_low"], pts["ci_high"],
                        color=c, alpha=0.12)
    lim = max(0.05, float(np.percentile(predictions[list(model_keys)].to_numpy(), 99.5)))
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration curves", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax2 = axes[1]
    for i, key in enumerate(model_keys):
        ax2.hist(predictions[key].to_numpy(), bins=50, alpha=0.45,
                 color=PALETTE[i % len(PALETTE)], label=labels.get(key, key))
    ax2.axvline(float(y.mean()), color="k", ls="--", lw=1.0,
                label=f"Prevalence ({y.mean():.3f})")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Predicted risk distribution", loc="left")
    ax2.legend()
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, name, cfg)


# ---------------------------------------------------------------------------
# Figure 5 - decision curve analysis
# ---------------------------------------------------------------------------

def figure_decision_curve(
    curves: dict[str, pd.DataFrame],
    labels: dict[str, str],
    cfg: Config,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    first = next(iter(curves.values()))
    ax.plot(first["threshold"], first["net_benefit_treat_all"], "k--",
            lw=1.2, label="Treat all")
    ax.plot(first["threshold"], first["net_benefit_treat_none"], "k:",
            lw=1.2, label="Treat none")
    for i, (key, df) in enumerate(curves.items()):
        ax.plot(df["threshold"], df["net_benefit_model"],
                color=PALETTE[i % len(PALETTE)], label=labels.get(key, key))
    lo = min(0.0, float(first["net_benefit_treat_all"].min()))
    ax.set_ylim(max(-0.05, lo), None)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Figure 5. Decision curve analysis", loc="left")
    ax.legend(loc="upper right")
    return save_figure(fig, "fig5_decision_curve", cfg)


# ---------------------------------------------------------------------------
# Figures 6-7 - explainability
# ---------------------------------------------------------------------------

def figure_importance_comparison(
    importances: dict[str, pd.DataFrame], cfg: Config, model_key: str
) -> list[Path]:
    """SHAP ranking alongside the rank assigned by each competing method.

    Divergence between methods is the point of this figure: gain-based
    importance is biased toward high-cardinality features, so a demographic
    variable ranked highly by gain but not by SHAP is an artifact worth
    reporting rather than a clinical finding.
    """
    top_n = cfg["explain"]["top_n_features"]
    paths: list[Path] = []

    if "shap" in importances:
        shap_df = importances["shap"].head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(shap_df) + 1.6))
        ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color=PALETTE[0])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Figure 6. SHAP feature importance ({model_key})", loc="left")
        paths += save_figure(fig, "fig6_shap_importance", cfg)

    comp = importances.get("comparison")
    if comp is not None:
        rank_cols = [c for c in comp.columns if c.startswith("rank_")]
        sort_col = "rank_shap" if "rank_shap" in rank_cols else rank_cols[0]
        sub = comp.nsmallest(top_n, sort_col).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(sub) + 1.8))
        width = 0.8 / max(len(rank_cols), 1)
        ypos = np.arange(len(sub))
        for i, col in enumerate(rank_cols):
            ax.barh(ypos + i * width, sub[col], height=width,
                    color=PALETTE[i % len(PALETTE)],
                    label=col.replace("rank_", "").title())
        ax.set_yticks(ypos + width * (len(rank_cols) - 1) / 2)
        ax.set_yticklabels(sub["feature"])
        ax.set_xlabel("Rank (lower = more important)")
        ax.set_title("Figure 7. Importance rank by method", loc="left")
        ax.legend()
        paths += save_figure(fig, "fig7_importance_comparison", cfg)
    return paths


# ---------------------------------------------------------------------------
# Figure 8 - fairness
# ---------------------------------------------------------------------------

def figure_subgroup_forest(
    fairness: pd.DataFrame, cfg: Config
) -> list[Path]:
    est = fairness[fairness["estimable"] & fairness["auroc"].notna()].copy()
    if est.empty:
        log.warning("no estimable subgroups; forest plot skipped")
        return []
    est["label"] = est["attribute"] + ": " + est["level"]
    est = est.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(est) + 1.8))
    ax.errorbar(est["auroc"], np.arange(len(est)), fmt="o",
                color=PALETTE[0], markersize=5)
    ax.axvline(0.5, color="k", ls=":", lw=1.0, label="Chance")
    ax.set_yticks(np.arange(len(est)))
    ax.set_yticklabels(est["label"], fontsize=8.5)
    ax.set_xlabel("Subgroup AUROC")
    ax.set_title("Figure 8. Discrimination by subgroup", loc="left")
    ax.legend()
    return save_figure(fig, "fig8_subgroup_forest", cfg)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_baseline_characteristics(
    df: pd.DataFrame, cfg: Config, target: str = "target", max_levels: int = 8
) -> pd.DataFrame:
    """Table 1 - baseline characteristics stratified by outcome.

    Standardised mean differences are reported rather than p-values: at
    ~70,000 patients every comparison is significant, so p-values carry no
    information about the magnitude of imbalance.
    """
    rows: list[dict] = []
    g1 = df[df[target] == 1]
    g0 = df[df[target] == 0]

    rows.append(
        {"variable": "N", "level": "", "overall": f"{len(df):,}",
         "not_readmitted": f"{len(g0):,}", "readmitted_lt30": f"{len(g1):,}", "smd": ""}
    )
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            m1, m0 = g1[col].mean(), g0[col].mean()
            s1, s0 = g1[col].std(), g0[col].std()
            pooled = np.sqrt((s1**2 + s0**2) / 2)
            smd = (m1 - m0) / pooled if pooled > 0 else np.nan
            rows.append(
                {
                    "variable": col, "level": "mean (SD)",
                    "overall": f"{s.mean():.2f} ({s.std():.2f})",
                    "not_readmitted": f"{m0:.2f} ({s0:.2f})",
                    "readmitted_lt30": f"{m1:.2f} ({s1:.2f})",
                    "smd": f"{smd:.3f}" if pd.notna(smd) else "",
                }
            )
        else:
            for level in s.value_counts().head(max_levels).index:
                p_all = float((s == level).mean())
                p1 = float((g1[col] == level).mean())
                p0 = float((g0[col] == level).mean())
                denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
                smd = (p1 - p0) / denom if denom > 0 else np.nan
                rows.append(
                    {
                        "variable": col, "level": str(level),
                        "overall": f"{int((s == level).sum()):,} ({100*p_all:.1f}%)",
                        "not_readmitted": f"{int((g0[col] == level).sum()):,} ({100*p0:.1f}%)",
                        "readmitted_lt30": f"{int((g1[col] == level).sum()):,} ({100*p1:.1f}%)",
                        "smd": f"{smd:.3f}" if pd.notna(smd) else "",
                    }
                )
    table = pd.DataFrame(rows)
    write_table(table, cfg.path("tables") / "table1_baseline_characteristics.csv")
    return table


def table_literature_comparison(cfg: Config, own: dict[str, Any] | None = None) -> pd.DataFrame:
    """Table 8 - benchmark against published work on this dataset.

    IMPORTANT: every value below must be verified against the source
    publication before submission. A wrong benchmark is a citation error and
    reviewers do check these.
    """
    rows = [
        {
            "study": "Strack et al. (2014)",
            "venue": "BioMed Research International",
            "dataset": "Diabetes 130-US (source publication)",
            "n": "~70,000 encounters",
            "models": "Logistic regression",
            "reported_auc": "not reported as AUC",
            "notes": "Established the HbA1c-measurement question; ICD grouping scheme",
        },
        {
            "study": "Emi-Johnson & Nkrumah (2025)",
            "venue": "Cureus 17(4):e82437",
            "dataset": "Diabetes 130-US, all encounters",
            "n": "101,766",
            "models": "LR / RF / XGBoost / DNN",
            "reported_auc": "XGB 0.667; LR 0.642; RF 0.630; DNN 0.579",
            "notes": "One-hot encoding + SHAP; closest published analogue",
        },
        {
            "study": "Salim & Ibrahim (2026)",
            "venue": "Healthcare (MDPI) 14(9):1185",
            "dataset": "Diabetes 130-US",
            "n": "not stated here",
            "models": "Nested CV, calibrated ensemble",
            "reported_auc": "XGB 0.664; 0.688 after calibration (Brier 0.094)",
            "notes": "Nested CV, Platt calibration, DCA, bootstrap CIs",
        },
        {
            "study": "Gandra (2024)",
            "venue": "Int. J. Health Sciences",
            "dataset": "Diabetes 130-US",
            "n": "not stated here",
            "models": "Six models incl. CatBoost/LightGBM",
            "reported_auc": "CatBoost 0.70",
            "notes": "Top predictors: number_inpatient, num_medications",
        },
        {
            "study": "LACE index (clinical comparator)",
            "venue": "Multiple validations",
            "dataset": "General medical cohorts",
            "n": "varies",
            "models": "Hand-built score",
            "reported_auc": "c-statistic ~0.55-0.78 across validations",
            "notes": "Not diabetes-specific; order-of-magnitude benchmark only",
        },
        {
            "study": "HOSPITAL score (clinical comparator)",
            "venue": "Multiple validations",
            "dataset": "General medical cohorts",
            "n": "varies",
            "models": "Hand-built score",
            "reported_auc": "c-statistic ~0.70-0.75",
            "notes": "Not diabetes-specific; order-of-magnitude benchmark only",
        },
    ]
    if own:
        rows.append(
            {
                "study": "Present study",
                "venue": "-",
                "dataset": "Diabetes 130-US, first encounter, death/hospice excluded",
                "n": f"{own.get('n', '')}",
                "models": "Parsimonious LR / penalised LR / RF / XGBoost / LightGBM",
                "reported_auc": own.get("auc_summary", ""),
                "notes": "No imbalance correction; calibration, DCA and fairness reported",
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, cfg.path("tables") / "table8_literature_comparison.csv")
    return table
