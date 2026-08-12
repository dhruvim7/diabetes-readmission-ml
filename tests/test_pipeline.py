"""Unit tests for the statistical core.

These validate implementations against properties with known correct answers,
rather than against previously recorded output. A test that only asserts
"the number is what it was last time" cannot detect an error that was present
from the start.

Run with::

    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import _age_midpoint, _icd_group  # noqa: E402
from src.evaluation import (  # noqa: E402
    bootstrap_ci,
    calibration_metrics,
    classification_metrics,
    decision_curve,
    delong_test,
    holm_correction,
    mcnemar_test,
    select_thresholds,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.09, n)
    strong = np.clip(0.09 + 0.25 * y + rng.normal(0, 0.15, n), 1e-3, 1 - 1e-3)
    weak = np.clip(0.09 + 0.08 * y + rng.normal(0, 0.15, n), 1e-3, 1 - 1e-3)
    return y, strong, weak


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------

def test_delong_auc_matches_sklearn(synthetic):
    """The DeLong AUC must equal sklearn's to numerical precision."""
    y, a, b = synthetic
    r = delong_test(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a), abs=1e-9)
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, b), abs=1e-9)


def test_delong_detects_real_difference(synthetic):
    y, a, b = synthetic
    r = delong_test(y, a, b)
    assert r["difference"] > 0
    assert r["p_value"] < 0.01


def test_delong_identical_models_give_zero_difference(synthetic):
    """A model compared with itself has zero AUC difference."""
    y, a, _ = synthetic
    r = delong_test(y, a, a)
    assert r["difference"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibration_recovers_unit_slope():
    """Probabilities generated then sampled from are calibrated by construction:
    slope ~1, intercept ~0, ICI ~0."""
    rng = np.random.default_rng(1)
    n = 20000
    lp = rng.normal(-2.3, 1.0, n)
    p = 1 / (1 + np.exp(-lp))
    y = rng.binomial(1, p)
    m = calibration_metrics(y, p)
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.08)
    assert m["calibration_intercept"] == pytest.approx(0.0, abs=0.08)
    assert m["ici"] < 0.02


def test_calibration_detects_inflated_probabilities():
    """Systematically inflated risk - the signature of an imbalance-corrected
    model - must produce a strongly negative intercept and a large ICI."""
    rng = np.random.default_rng(1)
    n = 20000
    lp = rng.normal(-2.3, 1.0, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    inflated = 1 / (1 + np.exp(-(lp + 2.2)))
    m = calibration_metrics(y, inflated)
    assert m["calibration_intercept"] < -1.5
    assert m["ici"] > 0.1


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def test_treat_all_net_benefit_zero_at_prevalence(synthetic):
    """Treat-all net benefit crosses zero exactly at the prevalence. This is
    the standard correctness check on the net-benefit formula."""
    y, a, _ = synthetic
    prevalence = float(y.mean())
    df = decision_curve(y, a, np.array([prevalence]))
    assert df.iloc[0]["net_benefit_treat_all"] == pytest.approx(0.0, abs=1e-9)


def test_treat_none_net_benefit_is_zero(synthetic):
    y, a, _ = synthetic
    df = decision_curve(y, a, np.linspace(0.02, 0.3, 5))
    assert (df["net_benefit_treat_none"] == 0).all()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_capacity_threshold_flags_requested_fraction(synthetic):
    """The capacity operating point must flag approximately the requested
    fraction of patients."""
    y, a, _ = synthetic
    ts = select_thresholds(y, a, capacity_fraction=0.10)
    alert_rate = float((a >= ts.capacity).mean())
    assert alert_rate == pytest.approx(0.10, abs=0.02)


def test_default_threshold_is_useless_at_low_prevalence(synthetic):
    """Demonstrates why 0.5 is never used: at ~9% prevalence it flags almost
    nobody."""
    y, a, _ = synthetic
    m = classification_metrics(y, a, 0.5)
    assert m["alert_rate"] < 0.02


def test_classification_confusion_matrix_sums(synthetic):
    y, a, _ = synthetic
    m = classification_metrics(y, a, 0.15)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)


# ---------------------------------------------------------------------------
# Bootstrap and multiplicity
# ---------------------------------------------------------------------------

def test_bootstrap_interval_contains_point_estimate(synthetic):
    y, a, _ = synthetic
    r = bootstrap_ci(y, a, lambda u, v: roc_auc_score(u, v), n_resamples=300)
    assert r["ci_low"] < r["estimate"] < r["ci_high"]


def test_holm_correction_is_monotone_and_bounded():
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert all(0 <= p <= 1 for p in adjusted)
    assert adjusted[0] == pytest.approx(0.03)


def test_mcnemar_symmetric_case(synthetic):
    y, a, _ = synthetic
    r = mcnemar_test(y, a, a, 0.15, 0.15)
    assert r["n01"] == 0 and r["n10"] == 0
    assert r["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,expected",
    [
        ("250.83", "Diabetes"),
        ("250", "Diabetes"),
        ("428", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("786", "Respiratory"),
        ("577", "Digestive"),
        ("996", "Injury"),
        ("715", "Musculoskeletal"),
        ("584", "Genitourinary"),
        ("197", "Neoplasms"),
        ("V57", "Other"),
        ("E878", "Other"),
        ("?", "Other"),
        ("", "Other"),
    ],
)
def test_icd_grouping(code, expected):
    """ICD-9 grouping must follow Strack et al.'s scheme and must not crash on
    V codes, E codes or missing tokens."""
    assert _icd_group(code) == expected


@pytest.mark.parametrize(
    "bucket,expected", [("[0-10)", 5.0), ("[70-80)", 75.0), ("[90-100)", 95.0)]
)
def test_age_midpoint(bucket, expected):
    assert _age_midpoint(bucket) == expected
