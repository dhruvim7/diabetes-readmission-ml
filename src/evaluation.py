"""Statistical evaluation: discrimination, calibration, utility, inference.

Design notes
------------
*Accuracy is not a headline metric.* At ~9% prevalence a model that predicts
"never readmitted" achieves ~91% accuracy. It is computed and reported once,
with that caveat attached.

*Every point estimate carries an interval.* Comparative claims additionally
require a test or an interval on the **difference**, which is what
:func:`delong_test` and :func:`bootstrap_difference` provide.

*The bootstrap is unclustered by design.* Because the cohort retains one
encounter per patient, rows are independent and the ordinary bootstrap is
valid. On all-encounter versions of this dataset it is not, and naive
row-level resampling produces intervals several times too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "ThresholdSet",
    "bootstrap_ci",
    "bootstrap_difference",
    "calibration_curve_points",
    "calibration_metrics",
    "classification_metrics",
    "decision_curve",
    "delong_test",
    "discrimination_metrics",
    "mcnemar_test",
    "select_thresholds",
]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

def discrimination_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUROC, average precision, log loss and the prevalence baseline."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "auprc_baseline": float(y.mean()),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration slope, intercept, Brier score, scaled Brier and ICI.

    The slope is the coefficient from a logistic regression of the outcome on
    the **logit** of the predicted probability (ideal value 1.0). The intercept
    (calibration-in-the-large) is fitted with the logit offset fixed at unit
    slope (ideal value 0.0). A slope well below 1 indicates systematic
    overestimation of risk - the signature of an imbalance-corrected model.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    lp = _logit(p)

    slope = np.nan
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(lp.reshape(-1, 1), y)
        slope = float(lr.coef_[0][0])
    except Exception:  # pragma: no cover - degenerate folds
        pass

    # Intercept with slope constrained to 1: minimise deviance over an offset.
    def _nll(a: float) -> float:
        q = 1.0 / (1.0 + np.exp(-(a + lp)))
        q = np.clip(q, EPS, 1 - EPS)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    from scipy.optimize import minimize_scalar

    intercept = float(minimize_scalar(_nll, bounds=(-10, 10), method="bounded").x)

    brier = float(brier_score_loss(y, p))
    prevalence = float(y.mean())
    brier_null = prevalence * (1 - prevalence)
    scaled_brier = float(1 - brier / brier_null) if brier_null > 0 else np.nan

    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "brier": brier,
        "brier_scaled": scaled_brier,
        "ici": _integrated_calibration_index(y, p),
    }


def _integrated_calibration_index(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute difference between smoothed observed risk and prediction."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(y, p, frac=0.6, it=0, return_sorted=False)
        return float(np.mean(np.abs(smoothed - p)))
    except Exception:  # pragma: no cover
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        fitted = iso.fit_transform(p, y)
        return float(np.mean(np.abs(fitted - p)))


def calibration_curve_points(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Binned calibration curve.

    Quantile binning is the default: predictions on this problem concentrate
    below 0.2, so equal-width bins leave the upper range almost empty and
    produce a misleading curve.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        obs = float(y[m].mean())
        n = int(m.sum())
        se = float(np.sqrt(max(obs * (1 - obs), EPS) / n))
        rows.append(
            {
                "bin": b,
                "n": n,
                "mean_predicted": float(p[m].mean()),
                "observed": obs,
                "ci_low": max(0.0, obs - 1.96 * se),
                "ci_high": min(1.0, obs + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class ThresholdSet:
    """Operating points, all derived from development-set predictions."""

    capacity: float
    youden: float
    fixed_sensitivity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "capacity": self.capacity,
            "youden": self.youden,
            "fixed_sensitivity": self.fixed_sensitivity,
        }


def select_thresholds(
    y_dev: np.ndarray,
    p_dev: np.ndarray,
    capacity_fraction: float = 0.10,
    target_sensitivity: float = 0.50,
) -> ThresholdSet:
    """Derive three operating points on development data only.

    ``capacity``
        Flag the highest-risk ``capacity_fraction`` of *patients*, reflecting
        finite care-management staffing. This is the primary operating point
        because it is the one a hospital can actually resource.
    ``youden``
        Sensitivity + specificity - 1 maximised. The statistically balanced point.
    ``fixed_sensitivity``
        The threshold achieving at least ``target_sensitivity``, answering
        "if the programme must catch half of early readmissions, what alert
        burden follows?"

    The default 0.5 threshold is never used: at this prevalence it yields
    almost no positive predictions.
    """
    y_dev = np.asarray(y_dev).astype(int)
    p_dev = np.asarray(p_dev, dtype=float)

    capacity = float(np.quantile(p_dev, 1 - capacity_fraction))

    fpr, tpr, thr = roc_curve(y_dev, p_dev)
    youden = float(thr[int(np.argmax(tpr - fpr))])

    order = np.argsort(-p_dev)
    y_sorted = y_dev[order]
    cum_tp = np.cumsum(y_sorted)
    total_pos = max(int(y_dev.sum()), 1)
    sens = cum_tp / total_pos
    hit = np.argmax(sens >= target_sensitivity)
    fixed = float(p_dev[order][hit]) if sens[hit] >= target_sensitivity else float(p_dev.min())

    return ThresholdSet(
        capacity=capacity,
        youden=float(np.clip(youden, 0.0, 1.0)),
        fixed_sensitivity=fixed,
    )


def classification_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float
) -> dict[str, float]:
    """Full classification profile at a stated threshold.

    ``nne`` (number needed to evaluate) is 1/PPV: how many flagged patients a
    care team must review to find one true early readmission. It is the metric
    an operations audience actually asks for.
    """
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "accuracy": float(accuracy_score(y, yhat)),
        "alert_rate": float(yhat.mean()),
        "nne": float(1.0 / ppv) if ppv and ppv > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ]
    )


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, float]:
    """Percentile bootstrap interval for an arbitrary metric.

    Resampling is stratified on the outcome so that no resample degenerates to
    too few events, which would otherwise destabilise AUROC estimates.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y, p))
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _stratified_indices(y, rng) if stratified else rng.integers(0, len(y), len(y))
        try:
            draws[i] = metric_fn(y[idx], p[idx])
        except Exception:  # pragma: no cover
            draws[i] = np.nan

    valid = draws[~np.isnan(draws)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.std(valid, ddof=1)),
        "n_resamples": int(len(valid)),
    }


def bootstrap_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap interval for the difference between two models.

    Both models are evaluated on the same resample, preserving the pairing.
    A comparative claim needs an interval on the difference, not two
    overlapping intervals on the individual estimates.
    """
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y, p_a) - metric_fn(y, p_b))
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _stratified_indices(y, rng)
        try:
            draws[i] = metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx])
        except Exception:  # pragma: no cover
            draws[i] = np.nan

    valid = draws[~np.isnan(draws)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "difference": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu fast algorithm)
# ---------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Return AUCs and the DeLong covariance matrix.

    ``predictions`` must be shape (k_models, n_samples) with positive cases
    occupying the first ``n_pos`` columns.
    """
    m, n = n_pos, predictions.shape[1] - n_pos
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(predictions[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, cov


def delong_test(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray
) -> dict[str, float]:
    """DeLong et al. (Biometrics 1988;44:837-45) test for two correlated AUCs."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    n_pos = int(y.sum())
    preds = np.vstack([np.asarray(p_a, float), np.asarray(p_b, float)])[:, order]

    aucs, cov = _fast_delong(preds, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {
            "auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "difference": diff, "z": np.nan, "p_value": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
        }
    se = float(np.sqrt(var))
    z = diff / se
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": diff,
        "se": se,
        "z": float(z),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))),
        "ci_low": float(diff - 1.96 * se),
        "ci_high": float(diff + 1.96 * se),
    }


def mcnemar_test(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, thr_a: float, thr_b: float
) -> dict[str, float]:
    """McNemar's test on paired classification errors at stated thresholds."""
    y = np.asarray(y).astype(int)
    a = (np.asarray(p_a, float) >= thr_a).astype(int)
    b = (np.asarray(p_b, float) >= thr_b).astype(int)
    a_correct, b_correct = (a == y), (b == y)

    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    # Exact binomial test; preferred over the chi-square approximation for
    # small discordant counts.
    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def decision_curve(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Net benefit across threshold probabilities (Vickers & Elkin 2006).

    Net benefit = TP/n - (FP/n) * pt/(1-pt). Both reference strategies are
    returned: treat-all crosses zero at the prevalence, which doubles as a
    correctness check on this implementation.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = float(y.mean())

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt) if pt < 1 else np.inf
        flagged = p >= pt
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        nb_model = tp / n - (fp / n) * w
        nb_all = prevalence - (1 - prevalence) * w
        rows.append(
            {
                "threshold": float(pt),
                "net_benefit_model": float(nb_model),
                "net_benefit_treat_all": float(nb_all),
                "net_benefit_treat_none": 0.0,
                "n_flagged": int(flagged.sum()),
            }
        )
    return pd.DataFrame(rows)
