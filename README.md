# diabetes-readmission-ml

Reproducible machine learning analysis of 30-day hospital readmission using the UCI Diabetes 130-US Hospitals dataset (1999–2008).

This repository contains the **exact code and configuration used to generate the results reported in the associated manuscript**. The manuscript-associated version is tagged **`v1.0-manuscript`**. It is not a simplified reimplementation. No methodology was altered during packaging, and the packaging exercise does not change any manuscript result.

---

## 1. Data — not included in this repository

**The raw dataset is not distributed here.** Download it directly from the UCI Machine Learning Repository:

- <https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008>
- DOI: `10.24432/C5230J`

Place `diabetic_data.csv` (and `IDS_mapping.csv`, if you wish to consult the code mappings) in `data/raw/`.

The analysis uses only this **publicly available, fully de-identified** secondary dataset. No new data were collected and no individual patient was contacted.

**Integrity check.** The file used for the locked analysis had SHA-256:

```
0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97
```

Verify with `sha256sum data/raw/diabetic_data.csv`. The pipeline records this hash automatically in `logs/run_manifest.json` on every run. A different hash means a different file version, and results may differ.

---

## 2. Installation

```bash
git clone https://github.com/dhruvim7/diabetes-readmission-ml.git
cd diabetes-readmission-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required. The locked analysis ran on **Python 3.12.3**, Linux x86-64.

Package versions used for the locked results (recorded in `logs/run_manifest.json`):

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| xgboost | 3.4.0 |
| lightgbm | 4.7.0 |
| optuna | 4.9.0 |
| shap | 0.52.0 |
| imbalanced-learn | 0.14.2 |
| statsmodels | 0.14.6 |

scikit-learn, XGBoost and SHAP have all had breaking API changes across recent releases; pinning matters.

---

## 3. Random seeds

A single global seed of **42** is set in `config.yaml` and applied to:

- the stratified development/hold-out split (`train_test_split`)
- all cross-validation fold generation (`RepeatedStratifiedKFold`)
- every estimator's `random_state`
- the Optuna TPE sampler
- bootstrap resampling
- SHAP background sampling and permutation importance

`seed_everything()` in `src/utils.py` additionally sets `random`, `numpy` and `PYTHONHASHSEED`.

**Caveat on exact reproduction.** The split, bootstrap intervals, statistical tests and all evaluation stages are deterministic under a fixed seed. The Optuna search is *not* bit-identical across runs, because trial pruning depends on wall-clock ordering of fold evaluations. Re-running the tuning stage may therefore select marginally different hyperparameters. To reproduce the manuscript's numbers exactly, use the hyperparameters recorded in `locked_results/metrics/tuning_results.json` rather than re-running the search (see §7).

---

## 4. Directory structure

```
diabetes-readmission-ml/
├── README.md
├── requirements.txt
├── config.yaml                  # every analysis constant; nothing hard-coded in src/
├── run_pipeline.py              # single-command end-to-end entry point
├── run_stage.py                 # modular, resumable stage-by-stage runner
├── run_all.sh                   # the exact sequence used for the locked analysis
├── run_downstream.sh            # post-tuning stages only (used after interruptions)
├── make_smoke_config.py         # generates a reduced config for environment testing
├── .gitignore
├── src/
│   ├── config.py                # configuration loading, path management
│   ├── utils.py                 # logging, run manifest, artifact IO, stage gates
│   ├── data.py                  # cohort derivation, features, missing data, leakage audit
│   ├── preprocessing.py         # split, ColumnTransformer, sample-size calculation
│   ├── models.py                # model zoo and hyperparameter search spaces
│   ├── training.py              # tuning, final fitting, predictions, calibration
│   ├── evaluation.py            # discrimination, calibration, thresholds, bootstrap,
│   │                            #   DeLong, McNemar, Holm, decision curve analysis
│   ├── analysis.py              # SHAP, permutation/gain importance, fairness, sensitivity
│   ├── reporting.py             # publication figures and manuscript tables
│   ├── stages_post.py           # convergence assessment, leakage verification
│   └── pipeline.py              # orchestration
├── tests/
│   ├── test_pipeline.py         # 31 unit tests of the statistical core
│   └── make_synthetic.py        # schema-faithful synthetic data generator
├── docs/
│   └── analysis_plan.md         # frozen methodology specification
├── data/raw/                    # place diabetic_data.csv here (not tracked)
└── locked_results/              # outputs from the locked run, for verification
    ├── tables/                  # all manuscript tables
    ├── metrics/                 # all computed statistics, incl. Optuna histories
    └── optuna.db                # SQLite study database from the locked run
```

`locked_results/` is provided so that a reader can verify every manuscript number without re-running anything. Delete it if you prefer a code-only repository; it is not required by the pipeline.

---

## 5. Running the pipeline

### Verify the environment first (recommended)

```bash
python make_smoke_config.py
python tests/make_synthetic.py --n 12000 --out data/raw
python run_pipeline.py --config config.smoke.yaml
python -m pytest tests/test_pipeline.py -q
```

The synthetic generator reproduces the real schema exactly (column names, categorical levels, missingness tokens, repeat encounters, death/hospice codes). Its metrics are meaningless — it verifies the machinery, nothing more. **Delete the synthetic `diabetic_data.csv` before running the real analysis.**

### Full analysis

```bash
bash run_all.sh
```

Or stage by stage (resumable; each stage checkpoints and skips completed work):

```bash
python run_stage.py prep          # stages 1–6: ingest → cohort → features → split
python run_stage.py benchmark     # times one fit per model; projects search cost
python run_stage.py tune --model lightgbm      --target 40
python run_stage.py tune --model xgboost       --target 40
python run_stage.py tune --model penalised_lr  --target 40
python run_stage.py tune --model random_forest --target 12
python run_stage.py convergence   # optimisation histories + Figure S1
python run_stage.py fit           # final models, hold-out predictions
python run_stage.py verify        # leakage verification
python run_stage.py evaluate      # performance, calibration, thresholds, DeLong, DCA
python run_stage.py explain       # SHAP, permutation, gain, fairness
python run_stage.py sensitivity   # five pre-specified sensitivity analyses
python run_stage.py status        # progress report at any point
```

Optuna studies persist in SQLite, so an interrupted tuning run resumes from its last completed trial without recomputation.

**Compute note.** The locked analysis ran single-core. `run_stage.py benchmark` times one fit per model and projects the total search cost before committing. Random forest configurations were substantially more expensive per fold than the gradient-boosting models.

---

## 6. Cohort derivation

Implemented in `src/data.py :: derive_cohort()`. Order is load-bearing and must not be rearranged.

| Step | N | Excluded |
|---|---|---|
| Raw encounters | 101,766 | — |
| After excluding death/hospice discharges (`discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}) | 99,343 | 2,423 |
| After excluding uninterpretable gender | 99,340 | 3 |
| After retaining first encounter per patient | **69,987** | 29,353 |

Outcome: `readmitted == '<30'` → 1; `>30` and `NO` → 0. **6,285 events (8.98%).**

Eligibility exclusions are applied **before** deduplication, so that a patient whose first encounter was terminal is not lost from the cohort entirely. Retaining one encounter per patient makes observations independent, which validates unclustered bootstrap intervals.

Automated gates (`src/utils.py :: gate()`) halt the run if cohort N falls outside 66,000–72,000 or prevalence outside 7–13%.

---

## 7. Development / hold-out split

`src/preprocessing.py :: make_split()`. Stratified 80/20, seed 42:

- Development: **55,989** patients (5,028 events, 8.980%)
- Hold-out: **13,998** patients (1,257 events, 8.980%)

Preprocessing parameters, feature engineering, hyperparameter optimisation, threshold derivation and calibration were carried out **exclusively within the development set**. The hold-out set was evaluated once. It was, however, also used to designate the primary model by highest AUROC point estimate — a rule fixed in advance, disclosed in the manuscript as a source of model-selection uncertainty.

A gate asserts development and hold-out prevalence match to within 0.002. `run_stage.py verify` re-checks partition disjointness, threshold provenance and calibration provenance.

---

## 8. Preprocessing

`src/data.py`, `src/preprocessing.py`.

- **Missing data:** no rows deleted. `weight` dropped (~97% missing). `race`, `medical_specialty`, `payer_code` receive an explicit `Unknown` category. An assertion enforces row-count invariance.
- **Glycaemic variables:** `None` in `A1Cresult`/`max_glu_serum` means *test not ordered* and is retained as an informative category, not treated as missing. A gate checks the HbA1c testing rate falls in 15–21% (observed 18.4%).
- **ICD-9 grouping:** `diag_1/2/3` → nine clinical categories; V and E codes → Other; `comorbidity_breadth` derived.
- **Encoding:** one-hot for all nominal variables (drop-first for linear models only). Label encoding is not used. Standardisation applied to continuous features for the penalised logistic regression only.
- **Leak prevention:** every transform lives inside a scikit-learn `Pipeline`/`ColumnTransformer` and is refitted on the training portion of each fold.
- **Near-zero variance:** medication columns with <0.5% non-"No" values dropped.

Resulting matrix: 132 encoded parameters; 38.09 events per parameter against a Riley minimum of 16,797.

---

## 9. Model development

`src/models.py`. Five models:

| Key | Role |
|---|---|
| `parsimonious_lr` | Five-variable clinical baseline, deliberately untuned |
| `penalised_lr` | Elastic-net logistic regression; pre-specified comparator |
| `random_forest` | Ensemble comparator |
| `xgboost` | Gradient boosting |
| `lightgbm` | Second gradient-boosting implementation |

Support vector machines were excluded by design (no native calibrated probabilities; poor scaling at this sample size).

**No class-imbalance correction in the primary analysis.** Corrections are exercised only in the sensitivity arms.

---

## 10. Hyperparameter search — settings and actual trial counts

Bayesian optimisation (Optuna TPE), objective **log loss** (a strictly proper scoring rule, chosen because calibration is a primary outcome), evaluated by repeated stratified 5-fold cross-validation within the development set. Median pruning discards individual under-performing configurations; it does not restrict the search space.

**Actual counts from the locked run** (`locked_results/metrics/optuna_history_*.csv`):

| Model | Recorded | Completed | Pruned | Interrupted | Converged? |
|---|---|---|---|---|---|
| XGBoost | 63 | 35 | 28 | 1 | Yes (≤0.0000% over final 10) |
| LightGBM | 54 | 33 | 21 | 1 | Yes (≤0.0000%) |
| Penalised LR | 53 | 32 | 21 | 1 | Yes (≤0.0001%) |
| **Random forest** | **10** | **8** | **0** | **2** | **No — not established** |

### ⚠ Two provenance discrepancies, documented rather than smoothed over

**(a) `run_all.sh` specifies `--target 40`, but three models recorded 53–63 trials.** The budget-counting semantics were changed part-way through the analysis, from counting *completed* trials only to counting *attempted* (completed + pruned) trials, following Optuna convention. Trials completed before that change were retained rather than discarded, so final counts exceed the nominal target. The counts in the table above and in the manuscript are the actual recorded counts, taken from the Optuna histories.

**(b) `run_all.sh` specifies `--target 12` for random forest, but only 10 trials were recorded.** The run was halted at 10 for computational reasons. **The random forest was not exhaustively optimised and its convergence was not established.** This is stated in the manuscript's Results and Limitations. The random forest was statistically indistinguishable from both fully optimised gradient-boosting models and was neither the primary model nor the pre-specified comparator.

The random forest search space was also narrowed once, mid-analysis, after benchmarking showed the excluded region was substantially more expensive per fold; its study was reset at that point so that no results from two different search spaces were mixed. `src/models.py` contains the final space (`max_depth` 6–18, `min_samples_leaf` 15–120, `n_estimators` fixed at 500), and the rationale is documented in the function docstring.

**Selected hyperparameters** are in `locked_results/metrics/tuning_results.json`. To reproduce the manuscript exactly, use these rather than re-running the search.

---

## 11. Evaluation

`src/pipeline.py :: _evaluate_models()`, `src/evaluation.py`.

- **Discrimination:** AUROC, AUPRC (reported against the prevalence baseline), log loss.
- **Calibration:** slope, intercept, Brier, scaled Brier, integrated calibration index, smoothed curves. Recalibration, where applied, is fitted by `CalibratedClassifierCV(cv=5)` within the development set only.
- **Accuracy** is computed once and reported with the caveat that a "never readmitted" model scores ~91% at this prevalence.
- A gate halts the run if any hold-out AUROC exceeds 0.75, the configured leakage ceiling.

---

## 12. Bootstrap settings

`src/evaluation.py :: bootstrap_ci()`, `bootstrap_difference()`.

- 2,000 resamples, percentile method, α = 0.05, seed 42
- **Stratified on the outcome**, so no resample degenerates to too few events
- **Unclustered** — valid because each patient contributes exactly one row. On all-encounter versions of this dataset it would not be.
- Paired resampling for between-model differences, preserving the pairing

---

## 13. Statistical comparisons

`src/pipeline.py :: _statistical_comparisons()`.

- **Primary, pre-specified:** best gradient-boosting model vs penalised logistic regression, by DeLong's test for correlated AUCs, plus a paired bootstrap interval on the AUPRC difference and McNemar's test at the primary threshold.
- **Secondary:** all remaining pairwise DeLong comparisons, Holm-adjusted.
- DeLong is implemented via the Sun & Xu fast algorithm and was validated to match `sklearn.roc_auc_score` to nine decimal places (`tests/test_pipeline.py`).

---

## 14. Threshold analysis

`src/evaluation.py :: select_thresholds()`. All thresholds derived from **development-set predictions only**, then applied unchanged to the hold-out set. Three pre-specified operating points:

1. **Capacity-constrained (primary):** top 10% of predicted risk
2. **Youden's J**
3. **Fixed 50% sensitivity target**

Reported per point: sensitivity, specificity, PPV, NPV, F1, balanced accuracy, alert rate, number needed to evaluate, full confusion matrix. The default 0.5 threshold is never used.

---

## 15. Decision curve analysis

`src/evaluation.py :: decision_curve()`. Net benefit across threshold probabilities 0.01–0.30 in 0.005 steps, against treat-all and treat-none. Implementation validated by the property that treat-all net benefit crosses zero exactly at prevalence.

---

## 16. Explainability

`src/analysis.py :: compute_importances()`. Three measures, deliberately compared:

1. **SHAP** (primary) — `TreeExplainer` for ensembles, `LinearExplainer` for linear models, aggregated from one-hot columns back to source variables
2. **Permutation importance** — 10 repeats on the hold-out set, model-agnostic cross-check
3. **Gain-based importance** — computed *only* as a comparator, being biased toward high-cardinality variables

A three-way ranking comparison is emitted so that method disagreement is visible.

---

## 17. Fairness / subgroup analysis

`src/analysis.py :: fairness_audit()`. Pre-specified subgroups: race, gender, age band. Per subgroup: AUROC, calibration slope and intercept, sensitivity, PPV, alert rate, equal-opportunity difference. Subgroups below 200 patients or 20 events are reported descriptively without estimated discrimination. Bootstrap confidence intervals accompany every estimable subgroup.

---

## 18. Sensitivity analyses

`src/analysis.py :: run_sensitivity_analyses()`. Five pre-specified arms:

1. **Class-imbalance strategy** — none / class weighting / SMOTE / random undersampling, compared on discrimination **and** calibration. Resampling is applied strictly inside cross-validation folds.
2. **Missing-data handling** — explicit categories vs complete-case deletion
3. **Payer code** — included vs excluded
4. **Glycaemic variables** — included vs excluded
5. **Cohort definition** — configurable in `config.yaml`

---

## 19. Configuration

Every analysis constant lives in `config.yaml`: seed, split ratio, CV folds and repeats, tuning engine and objective, trial budgets, calibration method, threshold definitions, bootstrap settings, DCA grid, SHAP sample size, fairness minimums, sensitivity arms, and the leakage ceiling. Nothing is hard-coded in `src/`. Changing this file changes the analysis; it should be version-controlled and tagged before any run.

---

## 20. Reproducibility statement

- The raw dataset is **not** included in this repository.
- Users should download it directly from the UCI Machine Learning Repository (§1).
- The analysis uses only the publicly available, de-identified secondary dataset.
- Random seeds are documented in §3 and set in `config.yaml`.
- This pipeline is intended to reproduce the analysis reported in the associated manuscript.
- **The random forest received a reduced search budget (10 recorded trials, 8 evaluated) and was not exhaustively optimised; its convergence was not established.**
- **The final manuscript results are not changed by this packaging exercise.** No methodology, parameter or result was modified in preparing this repository; `locked_results/` contains the outputs exactly as produced by the locked run.

---

## Author

Dhruvi Mewada, Ahmedabad University, Ahmedabad, Gujarat, India — dhruvi.mewada7@gmail.com

## Citation

If you use this code, please cite the associated manuscript:

> Mewada D. Realistic performance limits of machine learning for 30-day readmission prediction in diabetic inpatients: a calibration-focused analysis of 130 US hospitals. (Under review.)

Dataset citation:

> Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository; 2014. doi:10.24432/C5230J

## Ethics

This project analyses the publicly available, fully de-identified UCI Diabetes 130-US Hospitals dataset. It involves no direct participant contact, no individual-level intervention, and no access to identifiable patient information. No new data were collected.
