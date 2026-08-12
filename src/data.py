"""Stages 1-5: raw ingestion through to the analysis matrix.

The ordering in :func:`build_analysis_matrix` is load-bearing and must not be
rearranged:

1. ingest with ``?`` preserved as a token, never coerced to ``NaN`` blindly
2. eligibility exclusions (death / hospice) **before** deduplication
3. first encounter per patient
4. outcome derivation
5. feature engineering
6. explicit missing-category handling, with **zero row deletion**

Deduplicating before applying eligibility exclusions loses patients whose
first encounter was terminal, rather than advancing to their next eligible
encounter. Deleting rows for missingness reproduces the selection bias that
invalidated the original analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import RunManifest, gate, get_logger, sha256, write_table

log = get_logger("data")

__all__ = [
    "ICD_GROUPS",
    "build_analysis_matrix",
    "derive_cohort",
    "engineer_features",
    "handle_missing",
    "ingest",
    "leakage_audit",
    "variable_dictionary",
]

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

CONTINUOUS_COLUMNS = [
    "age_midpoint", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "prior_utilisation_total",
    "n_diabetes_meds", "n_meds_changed", "comorbidity_breadth",
]

ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

ICD_GROUPS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]

# Condensed label maps for the administrative ID columns. Codes absent from a
# map are routed to "Unknown", which is an informative level here rather than
# missing data.
ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 7: "Trauma",
    5: "Unknown", 6: "Unknown", 8: "Unknown",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician referral", 2: "Clinic referral", 3: "HMO referral",
    4: "Transfer from hospital", 5: "Transfer from SNF",
    6: "Transfer from other facility", 7: "Emergency room",
    8: "Court/law enforcement", 9: "Unknown", 15: "Unknown", 17: "Unknown",
    20: "Unknown", 21: "Unknown",
}
DISCHARGE_MAP = {
    1: "Home", 2: "Short-term hospital", 3: "SNF", 4: "ICF",
    5: "Other inpatient care", 6: "Home health service", 7: "Left AMA",
    8: "Home IV provider", 9: "Readmitted same institution",
    10: "Neonate discharge", 12: "Still patient", 15: "Swing bed",
    16: "Outpatient other facility", 17: "Outpatient this institution",
    22: "Rehab facility", 23: "Long-term care", 24: "Medicaid nursing",
    27: "Federal hospital", 28: "Psychiatric hospital", 29: "Critical access",
    30: "Other facility", 18: "Unknown", 25: "Unknown", 26: "Unknown",
}


# ---------------------------------------------------------------------------
# Stage 1 - ingestion
# ---------------------------------------------------------------------------

def ingest(cfg: Config, manifest: RunManifest) -> pd.DataFrame:
    """Load the raw dataset with provenance recording and a raw audit table.

    ``?`` is read as a plain string token, never as ``NaN``. Several columns
    (notably ``A1Cresult`` and ``max_glu_serum``) encode "test not ordered" as
    a legitimate category, and coercing those to missing at load time is the
    exact error that removed the clinically central variables previously.
    """
    raw_path = cfg.path("raw") / cfg["data"]["raw_file"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Download "
            "'Diabetes 130-US hospitals for years 1999-2008' from the UCI "
            "Machine Learning Repository (DOI 10.24432/C5230J) and place "
            "diabetic_data.csv there."
        )

    df = pd.read_csv(raw_path, dtype=str, keep_default_na=False, na_values=[])
    manifest.record_file("raw_dataset", raw_path)
    log.info("loaded raw dataset: %d rows x %d columns", *df.shape)

    numeric = [
        "encounter_id", "patient_nbr", "admission_type_id",
        "discharge_disposition_id", "admission_source_id", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    audit = _raw_audit(df, cfg)
    write_table(audit, cfg.path("tables") / "raw_audit.csv")

    manifest.record_rows("raw_encounters", len(df))
    manifest.record_stage(
        "stage01_ingest",
        rows=len(df),
        columns=int(df.shape[1]),
        sha256=sha256(raw_path),
        unique_patients=int(df["patient_nbr"].nunique()),
    )

    gate(
        "raw_row_count",
        len(df) == 101_766,
        f"expected 101,766 encounters, observed {len(df):,}",
        manifest,
        fatal=False,
    )
    gate(
        "raw_unique_patients",
        60_000 <= df["patient_nbr"].nunique() <= 80_000,
        f"unique patients = {df['patient_nbr'].nunique():,} (expected ~71,518)",
        manifest,
        fatal=False,
    )
    return df


def _raw_audit(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    tokens = set(cfg["data"]["missing_tokens"])
    rows = []
    for col in df.columns:
        s = df[col]
        as_str = s.astype(str)
        missing = int(as_str.isin(tokens).sum() + s.isna().sum())
        top = as_str.value_counts().head(5)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "n_missing": missing,
                "pct_missing": round(100 * missing / len(df), 2),
                "top_values": "; ".join(f"{k}={v}" for k, v in top.items()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stages 2-3 - cohort derivation and outcome
# ---------------------------------------------------------------------------

def derive_cohort(
    df: pd.DataFrame, cfg: Config, manifest: RunManifest
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply eligibility exclusions, deduplicate, and derive the outcome.

    Returns the cohort and the participant-flow table used for Figure 1.
    """
    flow: list[dict] = []
    n = len(df)
    flow.append({"step": "Raw encounters", "n": n, "excluded": 0, "reason": "-"})

    # -- eligibility: death and hospice ------------------------------------
    excluded_ids = set(cfg["cohort"]["excluded_discharge_ids"])
    mask = df["discharge_disposition_id"].isin(excluded_ids)
    n_death = int(mask.sum())
    df = df.loc[~mask].copy()
    flow.append(
        {
            "step": "After excluding death/hospice discharges",
            "n": len(df),
            "excluded": n_death,
            "reason": f"discharge_disposition_id in {sorted(excluded_ids)}",
        }
    )
    log.info("excluded %d death/hospice encounters -> %d", n_death, len(df))

    # -- eligibility: uninterpretable gender -------------------------------
    if cfg["cohort"]["drop_unknown_gender"]:
        gmask = df["gender"].isin(["Unknown/Invalid"])
        n_gender = int(gmask.sum())
        df = df.loc[~gmask].copy()
        flow.append(
            {
                "step": "After excluding unknown gender",
                "n": len(df),
                "excluded": n_gender,
                "reason": "gender = Unknown/Invalid",
            }
        )

    # -- one encounter per patient -----------------------------------------
    # Sorting is essential: drop_duplicates without an explicit order returns
    # whichever row happens to appear first in the file.
    before = len(df)
    df = df.sort_values("encounter_id", kind="mergesort")
    df = df.drop_duplicates(subset="patient_nbr", keep="first").copy()
    flow.append(
        {
            "step": "After retaining first encounter per patient",
            "n": len(df),
            "excluded": before - len(df),
            "reason": "repeat encounters removed (statistical independence)",
        }
    )
    log.info("deduplicated to first encounter -> %d patients", len(df))

    # -- outcome ------------------------------------------------------------
    counts = df["readmitted"].value_counts().to_dict()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    prevalence = float(df["target"].mean())
    flow.append(
        {
            "step": "Final analytical cohort",
            "n": len(df),
            "excluded": 0,
            "reason": f"{df['target'].sum():,} early readmissions "
            f"({prevalence:.1%})",
        }
    )

    flow_df = pd.DataFrame(flow)
    write_table(flow_df, cfg.path("tables") / "participant_flow.csv")

    manifest.record_rows("after_death_hospice_exclusion", int(flow[1]["n"]))
    manifest.record_rows("final_cohort", len(df))
    manifest.record_stage(
        "stage02_cohort",
        final_n=len(df),
        prevalence=prevalence,
        readmitted_levels=counts,
    )

    # -- gates --------------------------------------------------------------
    lo, hi = cfg["cohort"]["expected_n_min"], cfg["cohort"]["expected_n_max"]
    gate(
        "cohort_size",
        lo <= len(df) <= hi,
        f"final cohort N = {len(df):,} (expected {lo:,}-{hi:,}). "
        "A value near 22,000 indicates row deletion has reappeared.",
        manifest,
    )
    plo = cfg["cohort"]["expected_prevalence_min"]
    phi = cfg["cohort"]["expected_prevalence_max"]
    gate(
        "outcome_prevalence",
        plo <= prevalence <= phi,
        f"30-day readmission prevalence = {prevalence:.3f} "
        f"(expected {plo}-{phi})",
        manifest,
    )
    gate(
        "outcome_partition",
        sum(counts.values()) == len(df),
        f"readmitted levels sum to {sum(counts.values()):,} = cohort N",
        manifest,
    )
    return df, flow_df


# ---------------------------------------------------------------------------
# Stage 4 - feature engineering
# ---------------------------------------------------------------------------

def _icd_group(code: object) -> str:
    """Map an ICD-9 code to one of the nine Strack et al. (2014) categories."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Other"
    s = str(code).strip()
    if s in {"", "?", "nan", "None"}:
        return "Other"
    # V and E codes have no numeric range and are grouped as Other.
    if s[0] in {"V", "E", "v", "e"}:
        return "Other"
    try:
        value = float(s)
    except ValueError:
        return "Other"
    whole = int(np.floor(value))
    if 250 <= value < 251:
        return "Diabetes"
    if (390 <= value < 460) or whole == 785:
        return "Circulatory"
    if (460 <= value < 520) or whole == 786:
        return "Respiratory"
    if (520 <= value < 580) or whole == 787:
        return "Digestive"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if (580 <= value < 630) or whole == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"


def _age_midpoint(bucket: object) -> float:
    """Convert a ten-year age bracket such as ``[70-80)`` to its midpoint."""
    try:
        lo = int(str(bucket).strip("[)").split("-")[0])
    except (ValueError, IndexError):
        return np.nan
    return lo + 5.0


def _age_band(midpoint: float) -> str:
    if np.isnan(midpoint):
        return "Unknown"
    if midpoint < 40:
        return "<40"
    if midpoint < 65:
        return "40-64"
    if midpoint < 80:
        return "65-79"
    return ">=80"


def engineer_features(
    df: pd.DataFrame, cfg: Config, manifest: RunManifest
) -> pd.DataFrame:
    """Construct the modelling features.

    The glycemic block is the most consequential part: ``None`` in
    ``A1Cresult`` and ``max_glu_serum`` means *the test was not ordered*, a
    clinician behaviour, not a missing value. Strack et al. built this dataset
    specifically to study that behaviour.
    """
    out = df.copy()

    # -- ICD-9 diagnosis grouping ------------------------------------------
    for i in (1, 2, 3):
        col = f"diag_{i}"
        if col in out.columns:
            out[f"diag_{i}_group"] = out[col].map(_icd_group)
    group_cols = [c for c in out.columns if c.endswith("_group")]
    out["comorbidity_breadth"] = out[group_cols].nunique(axis=1)

    # -- glycemic variables -------------------------------------------------
    a1c_map = {
        "None": "Not measured", "Norm": "Norm", ">7": ">7", ">8": ">8",
    }
    out["A1Cresult"] = out["A1Cresult"].map(a1c_map).fillna("Not measured")
    out["A1C_tested"] = (out["A1Cresult"] != "Not measured").astype(int)

    glu_map = {
        "None": "Not measured", "Norm": "Normal",
        ">200": "Elevated", ">300": "Elevated",
    }
    out["max_glu_serum"] = out["max_glu_serum"].map(glu_map).fillna("Not measured")
    out["glucose_tested"] = (out["max_glu_serum"] != "Not measured").astype(int)

    # Pre-specified interaction: the modern restatement of Strack's question.
    out["A1C_tested_x_change"] = (
        out["A1C_tested"].astype(str) + "_" + out["change"].astype(str)
    )

    # -- age ----------------------------------------------------------------
    out["age_midpoint"] = out["age"].map(_age_midpoint)
    out["age_band"] = out["age_midpoint"].map(_age_band)

    # -- administrative identifiers ----------------------------------------
    out["admission_type"] = (
        out["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
    )
    out["admission_source"] = (
        out["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown")
    )
    out["discharge_disposition"] = (
        out["discharge_disposition_id"].map(DISCHARGE_MAP).fillna("Unknown")
    )

    # -- medications --------------------------------------------------------
    present = [c for c in MEDICATION_COLUMNS if c in out.columns]
    threshold = cfg["features"]["near_zero_variance_threshold"]
    kept, dropped = [], []
    for col in present:
        active = float((out[col] != "No").mean())
        (kept if active >= threshold else dropped).append(col)
    log.info(
        "medications retained: %d, dropped for near-zero variance: %d (%s)",
        len(kept), len(dropped), ", ".join(dropped) or "none",
    )

    out["n_diabetes_meds"] = (out[present] != "No").sum(axis=1)
    out["n_meds_changed"] = out[present].isin(["Up", "Down"]).sum(axis=1)
    out = out.drop(columns=dropped)

    # -- utilisation --------------------------------------------------------
    out["prior_utilisation_total"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )

    # -- rare-level collapsing ---------------------------------------------
    rare = cfg["features"]["rare_level_threshold"]
    for col in ["admission_type", "admission_source", "discharge_disposition",
                "medical_specialty"]:
        if col in out.columns:
            freq = out[col].value_counts(normalize=True)
            small = freq[freq < rare].index
            if len(small):
                out[col] = out[col].where(~out[col].isin(small), "Other")

    # -- drop unused --------------------------------------------------------
    drop = [c for c in cfg["features"]["drop_columns"] if c in out.columns]
    drop += [c for c in ["diag_1", "diag_2", "diag_3", "age", "readmitted",
                         *ID_COLUMNS] if c in out.columns]
    out = out.drop(columns=drop)

    a1c_rate = float(out["A1C_tested"].mean())
    manifest.record_stage(
        "stage04_features",
        n_features=int(out.shape[1]),
        a1c_tested_rate=a1c_rate,
        medications_kept=kept,
        medications_dropped=dropped,
    )
    lo = cfg["features"]["expected_a1c_tested_min"]
    hi = cfg["features"]["expected_a1c_tested_max"]
    gate(
        "a1c_tested_rate",
        lo <= a1c_rate <= hi,
        f"HbA1c tested in {a1c_rate:.1%} of encounters "
        f"(Strack et al. report 18.4%; expected {lo:.0%}-{hi:.0%}). "
        "This validates the glycemic recoding.",
        manifest,
        fatal=False,
    )
    return out


# ---------------------------------------------------------------------------
# Stage 5 - missing data and leakage audit
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame, cfg: Config, manifest: RunManifest
) -> pd.DataFrame:
    """Replace missing tokens with explicit categories. Deletes no rows.

    Missingness here is structurally informative (documentation and ordering
    behaviour) and therefore plausibly MNAR, which violates the assumption
    multiple imputation requires. Complete-case deletion is run only as a
    sensitivity analysis.
    """
    n_before = len(df)
    out = df.copy()
    tokens = cfg["data"]["missing_tokens"]

    for col in cfg["features"]["unknown_category_columns"]:
        if col in out.columns:
            out[col] = out[col].replace(tokens, "Unknown").fillna("Unknown")

    obj_cols = out.select_dtypes(include="object").columns
    for col in obj_cols:
        out[col] = out[col].replace(tokens, "Unknown").fillna("Unknown")

    num_cols = out.select_dtypes(include=[np.number]).columns
    residual = int(out[num_cols].isna().sum().sum())
    if residual:
        log.warning("median-imputing %d residual numeric missing values", residual)
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    assert len(out) == n_before, (
        f"Row count changed during missing-data handling "
        f"({n_before} -> {len(out)}). A dropna() has been reintroduced."
    )
    total_missing = int(out.isna().sum().sum())
    manifest.record_stage(
        "stage05_missing", rows=len(out), residual_missing=total_missing
    )
    gate(
        "no_row_deletion",
        len(out) == n_before,
        f"row count invariant at {len(out):,}",
        manifest,
    )
    gate(
        "zero_missing",
        total_missing == 0,
        f"residual missing values = {total_missing}",
        manifest,
    )
    return out


def leakage_audit(
    df: pd.DataFrame, cfg: Config, manifest: RunManifest, target: str = "target"
) -> dict:
    """Structural checks that no identifier or outcome proxy survives."""
    forbidden = {"encounter_id", "patient_nbr", "readmitted"}
    present = sorted(forbidden.intersection(df.columns))
    excluded = set(cfg["cohort"]["excluded_discharge_ids"])

    disposition_leak = False
    if "discharge_disposition" in df.columns:
        labels = {DISCHARGE_MAP.get(i, "Unknown") for i in excluded}
        disposition_leak = bool(
            set(df["discharge_disposition"].unique()).intersection(labels - {"Unknown"})
        )

    perfect: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        corr = df[col].corr(df[target])
        if pd.notna(corr) and abs(corr) > 0.95:
            perfect.append(col)

    report = {
        "forbidden_columns_present": present,
        "death_hospice_disposition_present": disposition_leak,
        "near_perfect_correlations": perfect,
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
    }
    manifest.record_stage("stage05_leakage_audit", **report)

    gate("no_identifier_columns", not present,
         f"forbidden columns present: {present or 'none'}", manifest)
    gate("no_death_hospice_rows", not disposition_leak,
         "no death/hospice discharge categories survive in the feature matrix",
         manifest)
    gate("no_perfect_predictors", not perfect,
         f"features correlated >0.95 with outcome: {perfect or 'none'}", manifest)
    return report


def variable_dictionary(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table 2 - every feature with its type, levels and handling decision."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "variable": col,
                "type": "continuous" if is_num else "categorical",
                "n_levels": "-" if is_num else int(s.nunique()),
                "levels": "-" if is_num else "; ".join(map(str, sorted(s.unique())[:12])),
                "missing_pct": round(100 * float(s.isna().mean()), 3),
                "summary": (
                    f"median {s.median():.1f} [IQR {s.quantile(.25):.1f}-"
                    f"{s.quantile(.75):.1f}]" if is_num else
                    f"mode {s.mode().iloc[0] if len(s.mode()) else '-'}"
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration for stages 1-5
# ---------------------------------------------------------------------------

def build_analysis_matrix(
    cfg: Config, manifest: RunManifest
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute stages 1-5 and persist the analysis matrix."""
    raw = ingest(cfg, manifest)
    cohort, flow = derive_cohort(raw, cfg, manifest)
    cohort.to_parquet(cfg.path("interim") / "cohort.parquet", index=False)

    features = engineer_features(cohort, cfg, manifest)
    features = handle_missing(features, cfg, manifest)
    leakage_audit(features, cfg, manifest)

    write_table(
        variable_dictionary(features, cfg),
        cfg.path("tables") / "table2_variable_dictionary.csv",
    )
    out_path = Path(cfg.path("processed") / "analysis_matrix.parquet")
    features.to_parquet(out_path, index=False)
    manifest.record_file("analysis_matrix", out_path)
    log.info("analysis matrix: %d rows x %d columns", *features.shape)
    return features, flow
