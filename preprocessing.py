"""Stage 6: train/test split and the preprocessing architecture.

Every transform is defined inside a scikit-learn ``Pipeline`` so that it is
refitted on the training portion of each cross-validation fold. Fitting an
encoder or scaler on the full dataset before splitting leaks distributional
information from the test set into training and inflates performance.

One-hot encoding is used for all nominal variables. Label encoding assigns
arbitrary integers to unordered categories, asserting an ordering and an equal
spacing that do not exist. Tree ensembles are indifferent to this because they
split on thresholds, but linear models read the integer as a magnitude - which
is why the original analysis produced a logistic regression AUC of 0.536 where
one-hot encoded implementations on the same data reach ~0.64.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .utils import RunManifest, gate, get_logger, write_json

log = get_logger("preprocessing")

__all__ = [
    "TARGET",
    "build_preprocessor",
    "column_types",
    "make_split",
    "sample_size_report",
]

TARGET = "target"

# Retained outside the feature matrix for subgroup analysis but never used
# as predictors beyond their legitimate roles.
FAIRNESS_COLUMNS = ["race", "gender", "age_band"]


def column_types(df: pd.DataFrame, target: str = TARGET) -> tuple[list[str], list[str]]:
    """Split feature columns into continuous and categorical."""
    features = [c for c in df.columns if c != target]
    continuous = [
        c for c in features if pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 2
    ]
    categorical = [c for c in features if c not in continuous]
    return continuous, categorical


def build_preprocessor(
    df: pd.DataFrame,
    *,
    scale: bool,
    drop_first: bool,
    target: str = TARGET,
    subset: list[str] | None = None,
) -> ColumnTransformer:
    """Construct the preprocessing transformer.

    Parameters
    ----------
    scale
        Standardise continuous features. Enabled for penalised logistic
        regression only; tree ensembles receive unscaled inputs.
    drop_first
        Drop one level per categorical variable. Required for linear models to
        avoid collinearity; unnecessary (and mildly harmful to interpretation)
        for tree models.
    subset
        Restrict to a named subset of columns, used by the parsimonious
        baseline model.
    """
    continuous, categorical = column_types(df, target)
    if subset is not None:
        continuous = [c for c in continuous if c in subset]
        categorical = [c for c in categorical if c in subset]

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False,
        min_frequency=None,
    )
    numeric_steps: list = [("scaler", StandardScaler())] if scale else [
        ("passthrough", "passthrough")
    ]
    numeric = (
        Pipeline(numeric_steps) if scale else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, continuous),
            ("cat", encoder, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_split(
    df: pd.DataFrame, cfg: Config, manifest: RunManifest, target: str = TARGET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified development / hold-out split.

    The hold-out set is written to disk and must not be read again until the
    final evaluation stage. Every tuning decision, threshold and calibration
    parameter is derived from the development set alone.
    """
    dev, holdout = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        stratify=df[target] if cfg["split"]["stratify"] else None,
        random_state=cfg["seed"],
        shuffle=True,
    )
    dev = dev.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    dev.to_parquet(cfg.path("processed") / "development.parquet", index=False)
    holdout.to_parquet(cfg.path("processed") / "holdout.parquet", index=False)

    p_dev, p_hold = float(dev[target].mean()), float(holdout[target].mean())
    manifest.record_stage(
        "stage06_split",
        development_n=len(dev),
        holdout_n=len(holdout),
        development_events=int(dev[target].sum()),
        holdout_events=int(holdout[target].sum()),
        development_prevalence=p_dev,
        holdout_prevalence=p_hold,
    )
    log.info(
        "development %d (%.2f%% events) | hold-out %d (%.2f%% events)",
        len(dev), 100 * p_dev, len(holdout), 100 * p_hold,
    )
    gate(
        "split_prevalence_match",
        abs(p_dev - p_hold) < 0.002,
        f"prevalence difference {abs(p_dev - p_hold):.4f} (< 0.002 required)",
        manifest,
    )
    return dev, holdout


def sample_size_report(
    df: pd.DataFrame, n_parameters: int, cfg: Config, manifest: RunManifest
) -> dict:
    """Riley et al. (BMJ 2020;368:m441) minimum sample size for a binary
    prediction model, plus events-per-parameter.

    Criterion 1 targets a small optimism in apparent model fit
    (shrinkage factor 0.9); criterion 3 targets precise estimation of the
    overall risk. These are reported to satisfy the TRIPOD+AI sample size item.
    """
    n = len(df)
    events = int(df[TARGET].sum())
    prevalence = events / n

    # Criterion 1: expected shrinkage of 0.9 with a conservative Cox-Snell R2.
    shrinkage = 0.9
    max_r2_cs = 1 - (prevalence ** prevalence * (1 - prevalence) ** (1 - prevalence)) ** 2
    r2_cs = 0.15 * max_r2_cs  # conservative assumption
    n_criterion1 = n_parameters / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))

    # Criterion 3: risk estimated within +/- 0.05 with 95% confidence.
    n_criterion3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)

    required = float(max(n_criterion1, n_criterion3))
    report = {
        "n_available": n,
        "events": events,
        "prevalence": prevalence,
        "candidate_parameters": int(n_parameters),
        "events_per_parameter": round(events / max(n_parameters, 1), 2),
        "minimum_n_criterion1": round(float(n_criterion1), 1),
        "minimum_n_criterion3": round(float(n_criterion3), 1),
        "minimum_n_required": round(required, 1),
        "adequate": bool(n >= required),
    }
    write_json(report, cfg.path("metrics") / "sample_size.json")
    manifest.record_stage("stage06_sample_size", **report)
    gate(
        "sample_size_adequate",
        report["adequate"],
        f"N={n:,} against Riley minimum {required:,.0f}; "
        f"{report['events_per_parameter']} events per parameter",
        manifest,
        fatal=False,
    )
    return report
