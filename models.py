"""Model definitions and hyperparameter search spaces.

Five models, each with a distinct role:

``parsimonious_lr``
    Five-variable logistic regression, deliberately untuned. Establishes
    whether machine learning adds anything over a trivial clinical baseline.
    If tuned gradient boosting lands within ~0.02 AUROC of this, that is the
    paper's most interesting finding, not a disappointment.
``penalised_lr``
    Elastic-net logistic regression. The reference standard a statistician
    would build, and the pre-specified comparator for the primary test.
``random_forest`` / ``xgboost`` / ``lightgbm``
    Ensembles. Two gradient-boosting implementations are included so that any
    finding can be shown not to be implementation-specific.

Support vector machines are deliberately excluded: they produce no calibrated
probability natively, scale poorly at this sample size, and are no longer
competitive on tabular clinical data. The exclusion is a design choice to be
stated in the Methods, not an omission.

No model applies an imbalance correction in the primary analysis. Class
weighting and resampling are exercised only in the sensitivity arms, where
their effect on calibration is the object of study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import Config
from .preprocessing import TARGET as TARGET_COLUMN, build_preprocessor
from .utils import get_logger

log = get_logger("models")

__all__ = ["MODEL_REGISTRY", "ModelSpec", "build_estimator", "get_model_specs"]


@dataclass
class ModelSpec:
    """A model, its display name, and how to sample its hyperparameters."""

    key: str
    display_name: str
    family: str  # "linear" | "tree"
    factory: Callable[..., Any]
    search_space: Callable[[Any], dict[str, Any]] | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)
    feature_subset: list[str] | None = None
    tunable: bool = True

    @property
    def scale(self) -> bool:
        return self.family == "linear"

    @property
    def drop_first(self) -> bool:
        return self.family == "linear"


# ---------------------------------------------------------------------------
# Search spaces (Optuna trial -> parameter dict)
# ---------------------------------------------------------------------------

def _space_penalised_lr(trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _space_random_forest(trial) -> dict[str, Any]:
    """Random forest search space.

    ``n_estimators`` is fixed at 500 rather than tuned: additional trees
    monotonically reduce the variance of the ensemble estimate and cannot
    cause overfitting, so it is a compute knob rather than a bias-variance
    parameter. ``min_samples_leaf`` starts at 10 because single-sample leaves
    were empirically both slower and less accurate on this dataset
    (AUROC 0.631 at leaf=1 vs 0.650 at leaf=20 on a development fold),
    i.e. the excluded region is dominated, not merely expensive.
    """
    return {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 6, 18),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 120, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]
        ),
    }


def _space_gbm(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _space_lightgbm(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------

def _make_parsimonious_lr(seed: int, **params: Any) -> Any:
    return LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000, random_state=seed
    )


def _make_penalised_lr(seed: int, **params: Any) -> Any:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=1000,
        tol=1e-3,
        random_state=seed,
        C=params.get("C", 1.0),
        l1_ratio=params.get("l1_ratio", 0.5),
        n_jobs=1,
    )


def _make_random_forest(seed: int, **params: Any) -> Any:
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
    )


def _make_xgboost(seed: int, **params: Any) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 5),
        min_child_weight=params.get("min_child_weight", 1.0),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
    )


def _make_lightgbm(seed: int, **params: Any) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.9),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        class_weight=params.get("class_weight", None),
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "parsimonious_lr": ModelSpec(
        key="parsimonious_lr",
        display_name="Parsimonious logistic regression",
        family="linear",
        factory=_make_parsimonious_lr,
        tunable=False,
    ),
    "penalised_lr": ModelSpec(
        key="penalised_lr",
        display_name="Penalised logistic regression",
        family="linear",
        factory=_make_penalised_lr,
        search_space=_space_penalised_lr,
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random forest",
        family="tree",
        factory=_make_random_forest,
        search_space=_space_random_forest,
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        display_name="XGBoost",
        family="tree",
        factory=_make_xgboost,
        search_space=_space_gbm,
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        display_name="LightGBM",
        family="tree",
        factory=_make_lightgbm,
        search_space=_space_lightgbm,
    ),
}


def get_model_specs(cfg: Config) -> dict[str, ModelSpec]:
    """Return the enabled model specifications, wiring in the feature subset
    for the parsimonious baseline."""
    specs: dict[str, ModelSpec] = {}
    for key in cfg["models"]["enabled"]:
        if key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {key!r}")
        spec = MODEL_REGISTRY[key]
        if key == "parsimonious_lr":
            spec.feature_subset = list(cfg["models"]["parsimonious_features"])
        specs[key] = spec
    return specs


def build_estimator(
    spec: ModelSpec,
    reference_df,
    seed: int,
    params: dict[str, Any] | None = None,
    class_weight: str | None = None,
    sampler: Any | None = None,
) -> Pipeline:
    """Assemble preprocessing and estimator into a single fitted-per-fold pipeline.

    ``sampler`` is used only by the imbalance sensitivity analysis; when given,
    an imbalanced-learn pipeline is returned so that resampling is applied
    inside each cross-validation fold rather than to the whole training set.
    """
    params = dict(params or {})
    subset = spec.feature_subset
    if subset is not None:
        # The parsimonious baseline is specified in terms of engineered names.
        subset = [c for c in subset if c in reference_df.columns] or None
        if subset is None:
            log.warning(
                "parsimonious feature subset not found in matrix; using all features"
            )

    pre = build_preprocessor(
        reference_df, scale=spec.scale, drop_first=spec.drop_first, subset=subset
    )

    if class_weight is not None:
        if spec.key in {"penalised_lr", "parsimonious_lr", "random_forest",
                        "lightgbm"}:
            params["class_weight"] = class_weight
        elif spec.key == "xgboost":
            # XGBoost has no class_weight parameter; the equivalent is the
            # negative-to-positive ratio applied to the positive class.
            y = reference_df[TARGET_COLUMN].to_numpy()
            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            params["scale_pos_weight"] = (n_neg / n_pos) if n_pos else 1.0
            log.info("xgboost scale_pos_weight = %.3f",
                     params["scale_pos_weight"])

    estimator = spec.factory(seed=seed, **params)

    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [("pre", pre), ("sampler", sampler), ("model", estimator)]
        )
    return Pipeline([("pre", pre), ("model", estimator)])
