#!/usr/bin/env python3
"""Generate a synthetic dataset with the exact UCI Diabetes 130 schema.

This exists so the pipeline can be verified end-to-end without the real file:
identical column names, identical categorical levels, identical missingness
tokens, repeat encounters per patient, and death/hospice discharge codes.

It is a structural fixture, NOT a scientific substitute. Signal is injected
crudely, so metrics from a synthetic run carry no clinical meaning - the run
proves the machinery works, nothing more.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_BUCKETS = [f"[{i}-{i+10})" for i in range(0, 100, 10)]

SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Psychiatry", "Pulmonology", "Urology",
]


def generate(n_patients: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Repeat encounters, as in the real data (~1.4 encounters per patient).
    counts = rng.choice([1, 2, 3, 4], size=n_patients, p=[0.72, 0.19, 0.06, 0.03])
    patient_nbr = np.repeat(np.arange(1, n_patients + 1), counts)
    n = len(patient_nbr)

    df = pd.DataFrame(
        {
            "encounter_id": np.arange(1, n + 1),
            "patient_nbr": patient_nbr,
            "race": rng.choice(
                ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"],
                n, p=[0.72, 0.19, 0.02, 0.01, 0.04, 0.02],
            ),
            "gender": rng.choice(["Female", "Male", "Unknown/Invalid"],
                                 n, p=[0.537, 0.462, 0.001]),
            "age": rng.choice(AGE_BUCKETS, n,
                              p=[.002, .003, .008, .03, .095, .17, .22, .256, .17, .046]),
            "weight": np.where(rng.random(n) < 0.97, "?", "[75-100)"),
            "admission_type_id": rng.choice([1, 2, 3, 4, 5, 6, 7, 8], n,
                                            p=[.53, .18, .19, .001, .048, .045, .001, .005]),
            "discharge_disposition_id": rng.choice(
                [1, 2, 3, 4, 5, 6, 7, 11, 13, 14, 18, 22, 25],
                n, p=[.60, .02, .13, .01, .01, .12, .006, .016, .004, .004, .035, .04, .005],
            ),
            "admission_source_id": rng.choice([1, 2, 4, 6, 7, 9, 17, 20],
                                              n, p=[.29, .011, .03, .022, .565, .012, .06, .01]),
            "time_in_hospital": rng.integers(1, 15, n),
            "payer_code": rng.choice(["?", "MC", "HM", "SP", "BC", "UN"],
                                     n, p=[.40, .32, .07, .07, .08, .06]),
            "medical_specialty": rng.choice(["?"] + SPECIALTIES, n,
                                            p=[.49] + [0.051] * 10),
            "num_lab_procedures": rng.integers(1, 121, n),
            "num_procedures": rng.integers(0, 7, n),
            "num_medications": rng.integers(1, 60, n),
            "number_outpatient": rng.poisson(0.37, n),
            "number_emergency": rng.poisson(0.20, n),
            "number_inpatient": rng.poisson(0.64, n),
            "number_diagnoses": rng.integers(1, 17, n),
            "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"],
                                        n, p=[.947, .026, .015, .012]),
            "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"],
                                    n, p=[.833, .049, .037, .081]),
        }
    )

    # ICD-9 codes spanning the ranges the grouping function must handle,
    # including V and E codes.
    pools = ["250.83", "428", "414", "486", "577", "996", "715", "584",
             "197", "V57", "E878", "?"]
    for col in ("diag_1", "diag_2", "diag_3"):
        df[col] = rng.choice(pools, n,
                             p=[.19, .13, .13, .08, .06, .07, .06, .09, .04, .07, .05, .03])

    for med in MEDICATIONS:
        if med in {"examide", "citoglipton"}:
            df[med] = "No"                       # constant, must be dropped
        elif med in {"metformin", "insulin", "glipizide", "glyburide"}:
            df[med] = rng.choice(["No", "Steady", "Up", "Down"],
                                 n, p=[.60, .30, .06, .04])
        else:
            df[med] = rng.choice(["No", "Steady", "Up", "Down"],
                                 n, p=[.985, .011, .002, .002])

    df["change"] = rng.choice(["No", "Ch"], n, p=[.54, .46])
    df["diabetesMed"] = rng.choice(["Yes", "No"], n, p=[.77, .23])

    # Inject modest, realistic signal so models can learn something.
    logit = (
        -2.6
        + 0.22 * df["number_inpatient"]
        + 0.10 * df["number_emergency"]
        + 0.05 * df["number_diagnoses"]
        + 0.25 * (df["change"] == "Ch")
        + 0.20 * (df["A1Cresult"] != "None")
        + 0.02 * df["time_in_hospital"]
    )
    p = 1 / (1 + np.exp(-logit))
    early = rng.binomial(1, np.clip(p, 0.01, 0.6)).astype(bool)
    late = rng.random(n) < 0.35
    df["readmitted"] = np.where(early, "<30", np.where(late, ">30", "NO"))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12000, help="number of patients")
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    path = out / "diabetic_data.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df):,} encounters, {df['patient_nbr'].nunique():,} patients)")
    print(f"  <30 readmission rate: {(df['readmitted'] == '<30').mean():.3%}")
    print(f"  death/hospice codes  : {df['discharge_disposition_id'].isin([11,13,14,19,20,21]).sum():,}")


if __name__ == "__main__":
    main()
