# Frozen Methodology Specification
## 30-Day Readmission Prediction in Diabetic Inpatients — UCI Diabetes 130-US Hospitals

**Status:** Analysis plan. To be frozen before any modelling is run.
**Research question (unchanged):** Can routinely collected structured hospitalisation data support a clinically useful model for predicting 30-day all-cause readmission in diabetic inpatients, and which predictors are most informative?

---

## PART 0 — Design philosophy and the single root cause of the previous failure

The previous pipeline was not a collection of unrelated errors. It was built as a **classification exercise** — maximise accuracy, balance the classes, rank the features — when the object of study is a **clinical risk prediction model**, whose purpose is to output a calibrated probability that supports a resource-allocation decision at discharge.

Every difference below follows from that single reframe:

| Classification framing (old) | Risk prediction framing (new) |
|---|---|
| Accuracy is the headline metric | Accuracy is not reported as a headline; it is uninformative at 9% prevalence |
| Class imbalance is a problem to "fix" with SMOTE | Imbalance is a property of the disease, not a defect; correcting it destroys calibration |
| Threshold 0.5 is assumed | Threshold is a clinical/operational decision, chosen explicitly and justified |
| Output = a label | Output = a probability, which must be calibrated to be usable |
| Feature importance = a ranking | Explainability = attribution with a stated method and known biases |
| One number per model | Every number carries a 95% confidence interval |

**Consequence:** this reframe converts your previous errors into the paper's contribution. Dozens of published papers on this dataset use the SMOTE-plus-accuracy recipe. By running that recipe as a *documented sensitivity analysis* and showing it produces miscalibrated, clinically unusable probabilities, you contribute something genuinely useful rather than being the 40th four-model bake-off. This is the novelty hook, and it costs almost nothing extra because you were going to run those models anyway.

**Second design principle — pre-specification.** Write this plan, commit it to a timestamped Git repository (or OSF), and do not alter it after seeing results. The single most common (and undetectable) source of inflated findings in ML papers is silently trying variations until the numbers look good. A pre-registered plan is a rigour signal that costs you nothing and that very few Cureus submissions have.

**Third design principle — every preprocessing step lives inside a cross-validation-aware pipeline.** Not one imputation value, scaling parameter, encoding category, or threshold may be learned from data the model will later be evaluated on.

---

## PART 1 — The chronological workflow

Each step below states: **What / Why / Why the previous approach was wrong / Expected reviewer comment if ignored / Does it change results?**

---

### STEP 1 — Pre-specify and freeze the analysis plan

**What.** Commit this document to version control before running any model. Record: primary outcome, primary metric, primary model comparison, all sensitivity analyses, and the fairness subgroups — all named in advance.

**Why.** Distinguishes confirmatory from exploratory findings. TRIPOD+AI (Collins et al., BMJ 2024;385:e078378) expects a pre-specified analysis approach.

**Why previous was wrong.** No plan existed; the analysis was exploratory throughout, so no result carries confirmatory weight.

**Reviewer comment if ignored.** *"The authors report the best-performing of four models without indicating whether the comparison was pre-specified. Selective reporting cannot be excluded."*

**Changes results?** No — but it changes their credibility, which is the binding constraint.

---

### STEP 2 — Load raw data and establish the audit trail

**What.** Load all 101,766 encounters × 50 features. Record: dataset version, download date, source URL (UCI Machine Learning Repository, `Diabetes 130-US hospitals for years 1999-2008`, DOI 10.24432/C5230J), file hash, package versions, all random seeds. Log row counts at every subsequent operation.

**Why.** Reproducibility is a scored TRIPOD+AI item; the file hash lets a reviewer verify you used the same file.

**Why previous was wrong.** No provenance recorded; the drop from 101,766 to 22,716 was untraceable — which is precisely why it read as a red flag rather than a choice.

**Reviewer comment if ignored.** *"The analysis cannot be reproduced from the information provided."*

**Changes results?** No.

---

### STEP 3 — Define the outcome variable

**What.** The `readmitted` field has three levels: `<30`, `>30`, `NO`. Define the binary outcome as:
- **Positive (1):** `readmitted == '<30'`
- **Negative (0):** `readmitted ∈ {'>30', 'NO'}`

State explicitly in Methods that patients readmitted after 30 days are treated as negatives, because the question is *readmission within 30 days*, and excluding them would create an artificial contrast between early readmitters and never-readmitters that does not exist at the point of discharge decision-making.

**Why.** The clinician at discharge does not know whether a patient will return later; the model must be trained on the population it will be applied to. Excluding `>30` inflates apparent discrimination — a subtle leakage of future information into cohort definition.

**Why previous was wrong.** The manuscript never defines the target construction. A reviewer cannot verify whether `>30` was dropped, and dropping it is a common silent inflator.

**Reviewer comment if ignored.** *"The construction of the binary outcome from the three-level readmission variable is not specified."*

**Changes results?** Potentially substantially, if the previous code excluded `>30`. Must be verified and disclosed either way.

---

### STEP 4 — Apply encounter-level eligibility exclusions (BEFORE deduplication)

**What.** Exclude encounters where the patient could not be readmitted. Using `discharge_disposition_id`:
- **11** — Expired
- **13** — Hospice / home
- **14** — Hospice / medical facility
- **19** — Expired at home (hospice)
- **20** — Expired in medical facility (hospice)
- **21** — Expired, place unknown (hospice)

Expected removal: roughly 2,400 encounters (≈2.4%), leaving ≈99,300.

**Order matters:** apply eligibility exclusions *first*, then select the first remaining encounter per patient. If you deduplicate first and a patient's first encounter was terminal, you lose that patient entirely rather than moving to their next eligible encounter.

**Why.** A patient who died or entered hospice has a structurally impossible outcome. Including them contaminates the negative class with non-events that the model cannot learn to distinguish, biasing the class distribution and attenuating discrimination. This is standard practice in the readmission literature and is the exclusion most reviewers check first.

**Why previous was wrong.** This exclusion was absent entirely. `discharge_disposition_id` appeared in the top-15 feature importance list — meaning the model was very likely learning "this disposition code means the patient cannot return," which is a form of outcome leakage disguised as a predictor. **This is an independent fatal flaw.**

**Reviewer comment if ignored.** *"Encounters ending in death or hospice discharge appear to have been retained. As these patients cannot be readmitted, their inclusion invalidates the outcome definition. The prominence of discharge_disposition_id among predictors is consistent with this error."*

**Changes results?** **Yes, materially.** Expect the feature importance profile to change and `discharge_disposition_id` to fall in rank.

---

### STEP 5 — Handle repeat encounters: one encounter per patient

**What.** From the eligible encounters, retain the **first** encounter per `patient_nbr` (by `encounter_id` order, which is chronological). Expected final cohort: **≈69,000–70,500 patients**, with a 30-day readmission rate of approximately **9–11%**.

Report the actual N. If your number differs materially from ~69,000–70,500, stop and debug before proceeding — that discrepancy is the exact failure mode of the previous version.

**Why.** Encounters from the same patient are statistically dependent. Retaining all of them (a) leaks patient-specific information across the train/test boundary, and (b) invalidates standard confidence intervals, which assume independent observations. Taking one encounter per patient makes every row independent — which, importantly, means **standard stratified cross-validation and standard bootstrap CIs become valid**, with no need for grouped/clustered variants.

**Why previous was wrong.** The manuscript claims first-encounter deduplication but reports 22,716 records, when that operation alone yields ~70,000. The missing ~47,000 came from undisclosed listwise deletion (Step 6). The stated cohort derivation did not match the actual one.

**Reviewer comment if ignored.** *"The reported sample size is inconsistent with the stated derivation. The authors must provide a complete participant flow diagram."*

**Changes results?** **Yes, dramatically.** Tripling the sample changes every estimate, narrows every confidence interval, and materially improves the stability of tuning.

---

### STEP 6 — Missing data: explicit categories, not row deletion

**What.** Handle each variable by its missingness mechanism. **Delete no rows for missingness.**

| Variable | Missing | Decision | Rationale |
|---|---|---|---|
| `weight` | ~97% | **Drop the column** | Unusable; no reasonable imputation |
| `medical_specialty` | ~49% | **Retain; explicit `Unknown` level**; collapse to top ~10 specialties + `Other` | Missingness reflects admitting-service documentation practice — informative, not random |
| `payer_code` | ~40–52% | **Retain; explicit `Unknown` level** | Only available proxy for insurance status / socioeconomic position; flagged for sensitivity analysis |
| `race` | ~2% | **Retain; explicit `Unknown` level. Do not impute.** | Required for the fairness audit; imputing race is methodologically and ethically inappropriate |
| `diag_1/2/3` | 0.02% / 0.35% / 1.4% | Map to `Other` after ICD grouping | Negligible |
| `gender` | 3 records `Unknown/Invalid` | **Drop those 3 records** | Genuinely uninterpretable; trivial N |
| `A1Cresult`, `max_glu_serum` | "None" is **not** missing | **Retain — see Step 8** | "None" means the test was not ordered |

**Primary approach = missing-indicator / explicit category, not multiple imputation.** Justification: missingness here is structurally informative and plausibly MNAR (documentation and ordering behaviour), violating the MAR assumption that multiple imputation requires. Multiple imputation of `race` only will be run as a **sensitivity analysis** to pre-empt the reviewer who expects MICE.

**Why previous was wrong.** Converting `?` to null and dropping those rows removed ~47,000 patients non-randomly — driven mainly by `medical_specialty`, whose missingness correlates with admission source and hospital. This is textbook selection bias (Sterne et al., BMJ 2009;338:b2393: complete-case analysis is valid only under MCAR). It was also entirely undisclosed.

**Reviewer comment if ignored.** *"Two-thirds of eligible patients were excluded by complete-case deletion on a variable missing for structural reasons. The analytic sample is unlikely to be representative and the results are at high risk of selection bias."*

**Changes results?** **Yes, dramatically** — this is the largest single change to the study.

---

### STEP 7 — ICD-9 diagnosis grouping

**What.** Map `diag_1`, `diag_2`, `diag_3` to the nine clinical categories used by Strack et al. (2014):

| Category | ICD-9 range |
|---|---|
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Diabetes | 250.xx |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Genitourinary | 580–629, 788 |
| Neoplasms | 140–239 |
| Other | everything else, incl. E/V codes and missing |

Additionally derive **`comorbidity_breadth`** = count of distinct categories across the three diagnosis fields (1–3).

**Why.** Raw ICD-9 codes have >700 levels — unusable for one-hot encoding, and they invite high-cardinality overfitting. Grouping preserves clinical meaning, matches the reference paper for this dataset (aiding comparability), and produces interpretable SHAP output.

**Why previous was wrong.** The manuscript does not mention `diag_1/2/3` at all. They were either dropped (discarding the primary clinical signal) or label-encoded into a meaningless 700-level integer scale. Neither is defensible.

**Reviewer comment if ignored.** *"The handling of the three diagnosis fields is not described. Given that primary diagnosis is among the strongest determinants of readmission, this omission is material."*

**Changes results?** **Yes** — adds genuine clinical signal and should improve discrimination.

---

### STEP 8 — Glycemic variables: `A1Cresult` and `max_glu_serum`

**What.**
- **`A1Cresult`** — retain as a 4-level categorical: `Not measured` (≈83%), `Norm`, `>7`, `>8`. Rename the "None" level to `Not measured` in all outputs so the semantics are unambiguous.
- **`max_glu_serum`** — collapse to 3 levels: `Not measured` (≈95%), `Normal`, `Elevated` (combining `>200` and `>300`). The >300 cell is too sparse to support a separate level.
- Derive two binary indicators: **`A1C_tested`** and **`glucose_tested`**.
- Pre-specify one interaction of clinical interest: **`A1C_tested × change`** (whether HbA1c testing is associated with readmission conditional on medication change) — the direct modern-ML restatement of Strack's original hypothesis.

**Why.** "None" does not mean missing. It means *the test was not ordered during that encounter* — a clinician behaviour that reflects intensity of diabetes management. This is not incidental: Strack et al. (2014) assembled this entire dataset to test whether HbA1c measurement relates to readmission, finding testing was performed in only 18.4% of encounters and that the relationship with readmission depended on primary diagnosis. These are the most clinically meaningful diabetes-specific variables in the dataset.

**Why previous was wrong.** Both variables were deleted as ">90% missing" — a category error that mistook an informative clinical signal for absent data, in a study specifically about diabetic patients. It also produced an internal contradiction (a ">90%" rule applied to a variable listed at 85%) and left the paper with no diabetes-specific glycemic predictor at all.

**Reviewer comment if ignored.** *"The authors removed HbA1c and serum glucose measurement from a study of readmission in diabetes, apparently misinterpreting 'None' (test not performed) as missing data. This removes the variables of greatest clinical relevance and contradicts the source publication for this dataset."* Expect **reject or major revision** on this point alone.

**Changes results?** **Yes**, and it restores the paper's clinical credibility.

---

### STEP 9 — Remaining feature engineering

**What.**

*Administrative ID variables* — map `admission_type_id`, `discharge_disposition_id`, `admission_source_id` from integers to their labelled categories using the dataset's `IDS_mapping.csv`; collapse the `NULL` / `Not Available` / `Not Mapped` codes into a single `Unknown` level; collapse levels with <1% frequency into `Other`.

*Age* — supplied as ten-year bins (`[0-10)` … `[90-100)`). Convert to the **numeric bin midpoint** (5, 15, … 95). This is a genuinely ordered variable, so ordinal treatment is legitimate; midpoints keep it interpretable in the logistic model.

*Medication variables (23)* — screen for near-zero variance and **drop any with <0.5% non-`No` values** (`examide` and `citoglipton` are 100% `No`; several others are near-constant). Retain the informative agents (typically `metformin`, `insulin`, `glipizide`, `glyburide`, `pioglitazone`, `rosiglitazone`, `glimepiride`, `glyburide-metformin`). Retain `change` and `diabetesMed` as binaries. Derive **`n_diabetes_meds`** (count prescribed) and **`n_meds_changed`** (count with `Up` or `Down`).

*Utilisation* — retain `number_outpatient`, `number_emergency`, `number_inpatient`; derive **`prior_utilisation_total`** = their sum.

*Continuous* — retain `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`.

*Removed* — `encounter_id`, `patient_nbr` (after deduplication), `weight`, near-zero-variance medications.

**Critically:** state the `change` variable's true meaning — *any change in diabetic medication during the encounter*. It does **not** mean "within 24 hours of admission."

**Why.** Near-constant features add dimensionality without signal and destabilise regularised models. Derived counts capture regimen complexity that individual binaries miss.

**Why previous was wrong.** No variance screening; the `change` variable was misinterpreted in the Conclusion, which is the kind of error that makes a reviewer doubt the author understands the data.

**Reviewer comment if ignored.** *"The definition of the 'change' variable given in the Conclusion is incorrect."*

**Changes results?** Modestly — mainly improves stability and interpretability.

---

### STEP 10 — Leakage audit (explicit, documented step)

**What.** Before splitting, verify and document:
1. No identifier columns remain in the feature matrix.
2. No feature encodes the outcome (re-examine every retained variable against the outcome definition).
3. Every patient appears exactly once (making rows independent).
4. All fitting operations — imputation, encoding, scaling, resampling, threshold selection, calibration — occur **inside** the cross-validation pipeline, fitted on training folds only.

**Why.** Published AUCs above ~0.85 on this dataset are almost certainly leakage artifacts, typically from retaining all encounters per patient or applying SMOTE before splitting. Documenting the audit pre-empts the suspicion.

**Why previous was wrong.** Scaling and encoding were fitted on the full dataset before splitting — a real, if modest, leak.

**Reviewer comment if ignored.** *"It is unclear whether preprocessing was fitted within cross-validation folds."*

**Changes results?** Slightly reduces apparent performance — correctly.

**Sanity rule to hold yourself to:** if any model exceeds **AUROC 0.75** on this dataset, treat it as a bug and find the leak. The honest ceiling is ~0.66–0.69.

---

### STEP 11 — Train/test strategy

**What.** A three-layer design:

1. **Stratified 80/20 split** into a **development set** (~55,000) and a **hold-out test set** (~14,000), stratified on the outcome. Fixed seed, reported.
2. Within development: **repeated stratified 5-fold cross-validation (5 folds × 5 repeats)** for hyperparameter tuning, model selection, threshold selection, and calibration fitting. All internal-validation estimates are reported as mean ± SD across folds.
3. **Final models refit on the full development set**, then evaluated **exactly once** on the hold-out test set.

The hold-out set is touched once, at the end. No decision — not a hyperparameter, not a threshold, not a calibration parameter — is informed by it.

**Why.** Repeated CV gives stable internal estimates and a defensible model-selection procedure; a genuinely untouched hold-out gives an unbiased final estimate and allows honest calibration curves and decision-curve analysis on unseen data.

*(Nested cross-validation is the alternative. It gives a less biased performance estimate but does not yield a single deployable model or a clean unseen-data calibration curve. If a reviewer requests it, it can be added as a sensitivity analysis. The chosen design is the more reviewer-legible of the two.)*

**Why previous was wrong.** A single 80/20 split with no cross-validation means every reported number carries large, unquantified sampling variance — on a test set of only 4,544 with ~418 events. Differences of 0.02 AUC between models were almost certainly noise.

**Reviewer comment if ignored.** *"Performance estimates derive from a single random split without cross-validation; the stability of the reported model ranking cannot be assessed."*

**Changes results?** **Yes** — some or all of the previous model ranking will likely prove non-significant.

---

### STEP 12 — Sample size justification

**What.** Report the minimum sample size required for a prediction model with the final number of parameters using the Riley criteria (Riley et al., BMJ 2020;368:m441; `pmsampsize` in R or its Python equivalent). Report events, events-per-candidate-parameter, and confirm the cohort exceeds the requirement.

**Why.** TRIPOD+AI expects a sample-size rationale. With ~6,000+ events you will comfortably pass — so this is a cheap paragraph that scores a checklist item.

**Why previous was wrong.** No justification given; at 22,716 records with ~2,100 events and ~43 label-encoded features it was probably adequate, but that was never demonstrated.

**Reviewer comment if ignored.** *"No sample size justification is provided."*

**Changes results?** No.

---

### STEP 13 — Categorical encoding and scaling

**What.**
- **One-hot encoding** for all nominal categoricals (race, gender, medical_specialty, payer_code, admission/discharge/source categories, diagnosis groups, medications, A1Cresult, max_glu_serum), dropping one reference level for the linear models to avoid collinearity.
- **Ordinal/numeric** treatment only for genuinely ordered variables (age midpoint).
- **Standardisation** (`StandardScaler`) applied **only to continuous features, only for the penalised logistic regression**, and fitted inside each CV fold. Tree ensembles receive unscaled features.
- All of it implemented via `ColumnTransformer` inside a `Pipeline` so it is refitted per fold.

**Why.** Label encoding assigns arbitrary integers to unordered categories and thereby asserts a false ordering and false equal spacing. Tree models are immune (they split on thresholds); **linear models are not** — logistic regression and SVM read the integer as a magnitude and are forced to fit a monotonic relationship that does not exist.

**Why previous was wrong.** Nominal variables were label-encoded and fed to logistic regression and SVM. This handicapped exactly those two models by design, so the four-model comparison was never apples-to-apples. The evidence is direct: the closest published analog on this dataset (Emi-Johnson & Nkrumah, *Cureus* 2025;17(4):e82437) used one-hot encoding and obtained **logistic regression AUC 0.642** against your 0.5360. Most of that gap is your encoding, not the algorithm.

**Reviewer comment if ignored.** *"Nominal categorical variables were label-encoded for logistic regression and SVM, imposing an arbitrary ordinal structure. The comparison between linear and tree-based models is therefore confounded by preprocessing, and the conclusion that tree-based models are superior is not supported."*

**Changes results?** **Yes, substantially** — expect the logistic regression AUC to rise from ~0.54 toward ~0.64, which may well overturn the paper's central claim.

---

### STEP 14 — Class imbalance strategy: no correction (and this is the contribution)

**What.**
- **Primary models: trained on the natural class distribution with no resampling and no class weighting.**
- Threshold is chosen separately and explicitly (Step 17); calibration is assessed and corrected if needed (Step 18).
- **Pre-specified sensitivity analysis:** train the best-performing model under four conditions — (a) no correction, (b) `class_weight='balanced'` / `scale_pos_weight`, (c) SMOTE on training folds, (d) random undersampling — and compare **discrimination (AUROC, AUPRC) *and* calibration (slope, intercept, Brier)** across all four.

**Why.** The methodological consensus is now clear. van den Goorbergh et al. (*JAMIA* 2022;29(9):1525–34) found that imbalance corrections "led to poor calibration (strong overestimation of the probability to belong to the minority class), but not to better discrimination," and that classification gains "were obtained by shifting the probability threshold instead" — concluding that "outcome imbalance is not a problem in itself, and… imbalance correction may even worsen model performance." Carriero et al. (*Statistics in Medicine* 2025, doi:10.1002/sim.10320) extended this to machine learning: "no noticeable benefit for model discrimination, and… worse model calibration."

**Why previous was wrong.** SMOTE was presented as a methodological strength. It was applied correctly with respect to the split (post-split, training only — genuinely the right call, and one of the few things the old pipeline got right), but the model's output probabilities are calibrated to an artificial 50% prevalence rather than the true ~9%. Every probability the model emits is inflated by roughly a factor of ten, which makes it unusable for the exact clinical purpose the Discussion claims.

**Reviewer comment if ignored.** *"The use of SMOTE is presented as a strength, but the current methodological literature indicates that imbalance correction degrades calibration without improving discrimination. No calibration assessment is presented, so the impact cannot be evaluated."*

**Changes results?** **Yes** — and the sensitivity analysis is a publishable finding in its own right. This is the single highest-value addition in the entire rebuild.

*Honest caveat to include in the Discussion:* Carriero et al. note the discrimination effect is "more nuanced" for some ML algorithms. Present the debate fairly rather than as settled dogma — that nuance is itself a marker of a competent author.

---

### STEP 15 — Model selection

**What.** Five models, each with a defined role:

| Model | Role |
|---|---|
| **Parsimonious logistic regression** (5 variables: `number_inpatient`, `number_emergency`, `number_diagnoses`, `time_in_hospital`, `age`) | Simple clinical baseline — does ML add anything at all? |
| **Penalised logistic regression** (elastic net) | Reference standard; the model a statistician would build |
| **Random Forest** | Standard ensemble comparator |
| **XGBoost** | Gradient boosting; expected best performer |
| **LightGBM** | Second gradient-boosting implementation; tests whether the finding is implementation-specific |

**Drop SVM.** Rationale to state explicitly: it does not produce calibrated probabilities natively (requiring Platt scaling that confounds the calibration comparison), scales poorly at ~55,000 training rows, and is no longer competitive for tabular clinical data. Removing it is a defensible design choice, not an omission.

**The parsimonious baseline is the most important addition here.** If a five-variable logistic regression achieves AUROC within ~0.02 of tuned XGBoost — which on this dataset is a realistic outcome — that is a genuinely valuable, honest, publishable finding: complexity buys little, and the ceiling is imposed by the data. Reviewers respect authors who test their own premise.

**Why previous was wrong.** Four models at library defaults, one crippled by encoding, with no simple baseline to establish whether any of it beat a trivial alternative.

**Reviewer comment if ignored.** *"No simple baseline is presented, so the incremental value of machine learning over standard regression cannot be assessed."*

**Changes results?** Adds interpretive context that may reframe the entire conclusion.

---

### STEP 16 — Hyperparameter tuning

**What.** For each model except the parsimonious baseline, tune via **randomised search or Bayesian optimisation (Optuna TPE), 100 iterations**, using the repeated stratified 5-fold CV within the development set.

**Optimisation objective: log loss** (a strictly proper scoring rule), with AUROC reported as a secondary CV metric.

Report every search space and every selected hyperparameter in a table.

Suggested spaces: RF — `n_estimators` 300–1000, `max_depth` 5–30/None, `min_samples_leaf` 1–50, `max_features` sqrt/log2/0.3–0.7. XGBoost/LightGBM — `learning_rate` 0.01–0.3 (log), `max_depth` 3–10, `min_child_weight` 1–20, `subsample` 0.6–1.0, `colsample_bytree` 0.6–1.0, `reg_lambda` 1e-3–10 (log), `n_estimators` 200–1500 with early stopping. Elastic net LR — `C` 1e-4–1e2 (log), `l1_ratio` 0–1.

**Why log loss rather than AUROC.** AUROC is rank-based and indifferent to calibration; optimising it selects models that order patients well but may emit poorly calibrated probabilities. Since calibration is a primary outcome of this study, tuning on a proper scoring rule is internally consistent. Say so in the Methods — it demonstrates you understand *why* you chose the objective, which is exactly the sophistication signal reviewers look for.

**Why previous was wrong.** No tuning at all. As noted in the earlier review, comparing library defaults is a defaults comparison, not a model comparison, and default XGBoost versus tuned Random Forest could easily reverse.

**Reviewer comment if ignored.** *"No hyperparameter optimisation was performed. The comparison reflects default settings rather than model capability."*

**Changes results?** **Yes** — expect AUROC to rise toward the ~0.66–0.69 literature range and the ranking to potentially shift.

---

### STEP 17 — Threshold selection (operational, pre-specified)

**What.** Do **not** use 0.5. Select thresholds on the **development set only**, and report performance at three pre-specified operating points:

1. **Capacity-constrained (primary):** flag the top **10%** of patients by predicted risk — reflecting realistic care-management staffing. Report the implied threshold, sensitivity, specificity, PPV, NPV, and **number needed to evaluate (NNE)** to identify one true early readmission.
2. **Youden's J** — the statistically balanced point.
3. **Fixed sensitivity of 50%** — "if the programme must catch half of all early readmissions, what alert burden does that impose?"

Report the full confusion matrix at each.

**Why.** A threshold is a resource-allocation decision, not a statistical constant. The capacity-constrained framing is what a hospital operations audience actually needs, and it maps directly onto the paper's translational claims. The corroborating evidence is stark: a 2026 preprint on this dataset found that the default 0.5 threshold produced sensitivity of **1.4%**, rising to 58.0% at a tuned threshold of 0.128. A model at 0.5 on this problem catches essentially nobody.

**Why previous was wrong.** F1 of 0.2074 was reported at an unstated threshold — almost certainly 0.5 applied to a SMOTE-trained model, which is close to uninterpretable. The paper claims discharge-planning utility while reporting metrics at an operating point no hospital would ever use.

**Reviewer comment if ignored.** *"Classification metrics are reported at an unspecified threshold. Given the outcome prevalence, the default 0.5 threshold is clinically meaningless."*

**Changes results?** **Yes** — all classification metrics change, and the translational argument becomes concrete rather than hand-waved.

---

### STEP 18 — Calibration

**What.**
1. Assess calibration on the hold-out set for every model: **calibration plot** (loess or spline-smoothed, not decile bins alone), **calibration slope**, **calibration intercept (calibration-in-the-large)**, **Brier score**, **scaled Brier**, and **Integrated Calibration Index (ICI)**.
2. If the best model is miscalibrated, apply recalibration (Platt scaling or isotonic regression) fitted **within the development set via cross-validation** (`CalibratedClassifierCV`, `cv=5`), never on the hold-out.
3. Report calibration **before and after** recalibration, and for every arm of the Step 14 imbalance sensitivity analysis.

**Why.** Calibration is what makes a probability clinically usable; discrimination alone tells you the ordering is right, not that the numbers mean anything (Van Calster et al., *BMC Medicine* 2019;17:230 — "the Achilles heel of predictive analytics"). TRIPOD+AI requires reporting both. For a model meant to inform discharge resourcing, a predicted 30% risk must actually mean 30%.

**Why previous was wrong.** Calibration was never assessed — yet the Discussion asserted clinical deployment value. That combination (deployment claims, no calibration) is precisely what a methodologically literate reviewer flags.

**Reviewer comment if ignored.** *"No calibration assessment is presented. Claims regarding clinical deployment cannot be supported by discrimination metrics alone."*

**Changes results?** Adds an entire results dimension — and is where the SMOTE contribution lands.

---

### STEP 19 — Performance metrics (final reporting set)

**What.** Report, for every model, on the hold-out set:

- **Discrimination:** AUROC (95% CI); AUPRC (95% CI), plotted against the prevalence baseline.
- **Calibration:** slope, intercept, Brier, scaled Brier, ICI, calibration plot.
- **Clinical utility:** **decision curve analysis / net benefit** across threshold probabilities 0–0.30, versus treat-all and treat-none (Vickers & Elkin, *Med Decis Making* 2006;26(6):565–74).
- **Classification** (at each of the three pre-specified thresholds): sensitivity, specificity, PPV, NPV, F1, balanced accuracy, alert rate, NNE.
- **Accuracy:** report it once, in a table, with an explicit note that a model predicting "never readmitted" achieves ~90% accuracy — and therefore that accuracy is uninformative here.

**Why.** AUPRC is the appropriate discrimination summary under 9% prevalence (Saito & Rehmsmeier, *PLOS ONE* 2015;10(3):e0118432). Decision curve analysis answers the question the paper actually poses — is this model useful for a decision? — and remains uncommon on this dataset, so it is cheap novelty.

**Why previous was wrong.** Accuracy, AUROC, and F1 only. Random Forest's 0.8554 accuracy alongside 0.5927 AUROC and 0.1157 F1 is the signature of a model predicting the majority class almost always. The paper identified this correctly in prose — genuinely good metric literacy — but still reported accuracy in the headline table without the caveat attached.

**Reviewer comment if ignored.** *"Accuracy is not an appropriate metric at this prevalence, and no measure of clinical utility is presented."*

**Changes results?** Reframes what "best model" means.

---

### STEP 20 — Confidence intervals and statistical testing

**What.**
- **CIs:** non-parametric **bootstrap, 2,000 resamples**, percentile method, on the hold-out set, for every reported metric. Because each patient contributes exactly one row (Step 5), standard bootstrap is valid — state this explicitly, since it is a known trap on this dataset (a 2026 preprint showed that naive row-level bootstrap CIs on all-encounter data are ~3.3× too narrow).
- **Primary comparison, pre-specified:** best gradient-boosting model vs penalised logistic regression, via **DeLong's test** for correlated AUROCs (DeLong et al., *Biometrics* 1988;44(3):837–45).
- **Secondary comparisons:** all remaining pairwise AUROC comparisons, with **Holm correction** for multiplicity.
- **McNemar's test** for paired classification agreement at the primary threshold.
- **Bootstrap CIs for differences** in AUPRC and Brier score between the primary pair.

**Why.** "XGBoost outperformed Random Forest" is a claim about a difference, and a claim about a difference requires an interval or a test. Pre-specifying one primary comparison controls the multiplicity that ten pairwise tests would otherwise introduce.

**Why previous was wrong.** No CIs, no tests. AUROC 0.6161 vs 0.5927 on 4,544 records with ~418 events is very plausibly within noise — meaning the paper's headline conclusion was unsupported as written.

**Reviewer comment if ignored.** *"No confidence intervals or statistical comparisons are provided. The claim that XGBoost outperformed the alternatives is not substantiated."*

**Changes results?** **Yes** — some previously claimed differences will likely prove non-significant, which is itself an honest and reportable finding.

---

### STEP 21 — Explainability

**What.** Three complementary methods, reported together:

1. **SHAP (TreeSHAP)** on the best model — primary. Global mean |SHAP| bar plot, beeswarm summary, dependence plots for the top five features, and two or three individual waterfall plots illustrating patient-level explanation.
2. **Permutation importance** on the hold-out set — model-agnostic cross-check.
3. **Gain-based importance** — reported *only* for comparison, to demonstrate divergence.

Include a table or figure comparing the three rankings and discuss where they disagree.

**Why.** Impurity/gain-based importance is systematically biased toward high-cardinality and continuous predictors (Strobl et al., *BMC Bioinformatics* 2007;8:25). SHAP (Lundberg & Lee, NeurIPS 2017) is locally accurate, additive, and now the field standard.

**Specifically on gender.** A low-cardinality binary ranking third — above established drivers such as prior inpatient utilisation and comorbidity burden — contradicts the entire readmission literature and is a signature of estimator bias, not a clinical discovery. Under the corrected pipeline (one-hot encoding, proper cohort, no SMOTE) and under SHAP, expect it to fall substantially. **Documenting that fall — showing gain-based importance elevated a demographic variable that SHAP and permutation importance do not — is a legitimate methodological contribution**, and it turns your most embarrassing previous finding into a teaching point.

**Why previous was wrong.** Default gain-based importance was reported as if it were established fact, and described as "statistically most important," which conflates variable importance with statistical significance. They are unrelated concepts.

**Reviewer comment if ignored.** *"Feature importance is derived from a biased impurity-based measure and described as statistical significance. The prominence of gender is not clinically plausible and likely reflects methodological artifact."*

**Changes results?** **Yes** — the feature narrative will change, and Figure 2 must be rebuilt.

---

### STEP 22 — Fairness / subgroup audit

**What.** Pre-specify subgroups: **race** (Caucasian, African American, Hispanic, Asian, Other, Unknown), **gender**, and **age band** (<40, 40–64, 65–79, ≥80). For each, report AUROC, calibration slope and intercept, sensitivity and PPV at the primary threshold, and alert rate. Compute **equal-opportunity difference** and **calibration-in-the-large by group**. Report negative findings honestly.

**Why.** Obermeyer et al. (*Science* 2019;366:447–53) showed a widely deployed health algorithm systematically underestimated Black patients' needs, and that correcting it would raise the share of Black patients receiving extra help from 17.7% to 46.5%. Any model proposed for resource allocation now invites this question. Given that race and gender are in your feature set and gender previously ranked third, pre-empting it is essential — and a subgroup fairness audit is currently uncommon on this dataset, so it doubles as novelty.

**Why previous was wrong.** Not considered, despite race and gender being included as predictors and gender being highlighted as a top finding.

**Reviewer comment if ignored.** *"Race and gender are included as predictors and gender is reported as a leading feature, yet no subgroup performance or fairness analysis is presented."*

**Changes results?** Adds a results section; may surface real disparities worth reporting.

---

### STEP 23 — Pre-specified sensitivity analyses

Run all five. Each pre-empts a specific, predictable reviewer objection.

| # | Analysis | Objection it answers |
|---|---|---|
| 1 | **Imbalance strategy:** none vs class-weight vs SMOTE vs undersampling — discrimination *and* calibration | "Why did you not use SMOTE?" — and this is the paper's contribution |
| 2 | **Cohort:** first encounter only (primary) vs all encounters with patient-level `GroupKFold` | "Why discard 30% of the data?" |
| 3 | **Missing data:** explicit-category (primary) vs MICE for race vs complete-case | "Why not multiple imputation?" |
| 4 | **`payer_code`:** included vs excluded | "Is payer an SDOH proxy or a site-confounding artifact?" |
| 5 | **Glycemic variables:** included (primary) vs excluded | Directly quantifies the cost of the previous version's error |

Sensitivity analysis 5 deserves emphasis: it lets you state, with a number, exactly how much discrimination the old approach discarded. That is a far stronger paper than one that quietly fixes the mistake.

---

### STEP 24 — Figures and tables

**Figures (7):**
1. **CONSORT-style participant flow diagram** — 101,766 → eligibility exclusions → deduplication → final cohort, with N at every node. *Non-negotiable; this is the figure that resolves the credibility problem.*
2. ROC curves, all models, with AUROC and 95% CI in the legend.
3. Precision-recall curves with the prevalence baseline.
4. Calibration plots, all models — before and after recalibration.
5. Decision curve analysis / net benefit.
6. SHAP beeswarm summary for the best model.
7. Importance-method comparison (gain vs permutation vs SHAP).

*Optional 8th:* subgroup performance forest plot for the fairness audit.

**Tables (8):**
1. Baseline characteristics of the cohort, stratified by outcome (readmitted <30 days vs not), with standardised mean differences. *Currently missing entirely — a basic expectation of any clinical paper.*
2. Variable dictionary: every feature, type, missingness, and handling decision.
3. Hyperparameter search spaces and selected values.
4. Model performance: all metrics with 95% CIs.
5. Statistical comparisons: DeLong and McNemar results with corrected p-values.
6. Classification performance at all three thresholds.
7. Sensitivity analysis results.
8. **Literature comparison** — your AUROC benchmarked against Strack et al. 2014, Emi-Johnson & Nkrumah 2025 (XGBoost 0.667, LR 0.642), Salim & Ibrahim 2026 (nested-CV XGBoost 0.664, 0.688 calibrated), Gandra 2024 (CatBoost 0.70), and the LACE (~0.55–0.78) and HOSPITAL (~0.70–0.75) clinical scores.

Table 8 is strategically important: it pre-frames a ~0.67 AUROC as *consistent with the field* rather than as weak, and the LACE/HOSPITAL comparison supports an honest, interesting conclusion — that modern ML on administrative data performs comparably to simple hand-built clinical scores and does not clearly surpass them.

---

### STEP 25 — Reporting and compliance

**What.**
- Complete the **TRIPOD+AI checklist** (Collins et al., *BMJ* 2024;385:e078378) and submit as supplementary material.
- Complete a **PROBAST+AI** self-assessment (Moons et al., *BMJ* 2025;388:e082505) and address each domain in the Limitations.
- **Ethics statement:** publicly available, fully de-identified secondary data; exempt from IRB review (45 CFR 46.104(d)(4), Exempt Category 4).
- **Data availability:** UCI Machine Learning Repository, with DOI.
- **Code availability:** public GitHub repository with the full pipeline, seeds, environment file, and the frozen analysis plan.
- **AI-use disclosure** per journal policy if applicable.
- **Reference list:** 25–30 references (Cureus caps original articles at 30). The earlier review's citation list supplies most of these.
- Report all seeds and package versions.

**Why previous was wrong.** Zero references, no ethics statement, no data-availability statement. Any one of these blocks acceptance outright and prevents PubMed Central indexing.

**Changes results?** No — but without them there is no publication.

---

## PART 2 — What we are deliberately NOT doing, and why

Scope discipline matters as much as rigour. Each of the following was considered and rejected; if a reviewer requests one, it can be added, but none should be built pre-emptively.

| Excluded | Reason |
|---|---|
| **Nested cross-validation** | Marginal gain over dev/hold-out + repeated CV; complicates reporting and yields no single deployable model. Offer if requested. |
| **CatBoost, ExtraTrees, AdaBoost, HistGradientBoosting** | Two gradient-boosting implementations already test robustness. More models add multiplicity and dilute the narrative without adding insight. |
| **Boruta, RFE, extensive feature selection** | Regularisation and gradient boosting handle this internally; separate selection adds a leakage surface and tuning burden for negligible gain. |
| **Deep learning / TabNet** | Consistently underperforms gradient boosting on small tabular clinical data (the Cureus comparator's DNN reached only 0.579). Would invite "why?" |
| **Temporal or external validation** | Impossible — the dataset has no admission dates or hospital identifiers. State as a limitation. |
| **Multiple imputation as the primary strategy** | Missingness here is structurally informative (MNAR); MI's assumptions are violated. Included as sensitivity analysis only. |
| **Charlson Comorbidity Index** | Only three diagnosis fields are available; a proper Charlson index needs the full code list. `comorbidity_breadth` is the honest substitute. |
| **Cost-effectiveness modelling** | Requires cost parameters the dataset does not contain. Decision curve analysis delivers the utility argument without inventing numbers. |

---

## PART 3 — The frozen specification

Locked. Implement exactly this.

```
COHORT
  Source: UCI Diabetes 130-US Hospitals (101,766 encounters, 50 features)
  Exclude: discharge_disposition_id ∈ {11,13,14,19,20,21}   → ~99,300
  Exclude: gender = Unknown/Invalid (3 records)
  Deduplicate: first encounter per patient_nbr               → ~69,000–70,500
  Outcome: readmitted == '<30' → 1; {'>30','NO'} → 0         (~9–11% positive)
  NO row deletion for missingness.

FEATURES
  Drop:      weight; encounter_id; patient_nbr; meds with <0.5% non-'No'
  Unknown category: medical_specialty, payer_code, race
  ICD:       diag_1/2/3 → 9 Strack categories; + comorbidity_breadth
  Glycemic:  A1Cresult 4-level (Not measured/Norm/>7/>8); max_glu_serum 3-level
             (Not measured/Normal/Elevated); + A1C_tested, glucose_tested
             + pre-specified interaction A1C_tested × change
  Derived:   prior_utilisation_total, n_diabetes_meds, n_meds_changed
  Age:       bin midpoint (numeric)
  IDs:       admission_type / discharge_disposition / admission_source → labels,
             rare levels → Other, unmapped → Unknown

ENCODING
  One-hot for all nominals (drop-first for linear models)
  StandardScaler on continuous, penalised LR only
  All transforms inside ColumnTransformer + Pipeline, fitted per fold

SPLIT
  Stratified 80/20 → development / hold-out
  Development: repeated stratified CV, 5 folds × 5 repeats
  Hold-out: evaluated exactly once, at the end

IMBALANCE
  Primary: no correction
  Sensitivity: none vs class-weight vs SMOTE vs undersampling
               (compare discrimination AND calibration)

MODELS
  Parsimonious LR (5 vars) | Penalised LR (elastic net)
  Random Forest | XGBoost | LightGBM
  SVM excluded (justified in Methods)

TUNING
  Optuna TPE or randomised search, 100 iterations
  Objective: log loss (proper scoring rule); AUROC secondary
  Search spaces and selected values reported in full

THRESHOLDS (selected on development set only)
  Primary:   top 10% predicted risk (capacity-constrained)
  Secondary: Youden's J; fixed sensitivity 50%

CALIBRATION
  Assess: slope, intercept, Brier, scaled Brier, ICI, calibration plot
  Correct if needed: Platt or isotonic via CalibratedClassifierCV (cv=5, dev only)
  Report before and after, and for every imbalance arm

METRICS
  AUROC, AUPRC, calibration suite, decision curve analysis / net benefit,
  sens/spec/PPV/NPV/F1/balanced accuracy/alert rate/NNE at each threshold
  Accuracy reported once, with the majority-class caveat attached

INFERENCE
  Bootstrap 2,000 resamples, percentile CIs, all metrics
  Primary comparison (pre-specified): best GBM vs penalised LR, DeLong
  Secondary pairwise: DeLong with Holm correction
  McNemar at primary threshold; bootstrap CIs for AUPRC and Brier differences

EXPLAINABILITY
  SHAP (TreeSHAP) primary; permutation importance cross-check;
  gain-based reported only for comparison; three-way ranking comparison shown

FAIRNESS
  Subgroups: race, gender, age band
  Per group: AUROC, calibration, sensitivity, PPV, alert rate
  + equal-opportunity difference, calibration-in-the-large by group

SENSITIVITY ANALYSES
  1. Imbalance strategy   2. Cohort (first vs all encounters, GroupKFold)
  3. Missing data (category vs MICE vs complete-case)
  4. payer_code in/out    5. Glycemic variables in/out

REPORTING
  TRIPOD+AI checklist | PROBAST+AI self-assessment
  CONSORT-style flow diagram | 7 figures | 8 tables
  Ethics exemption (45 CFR 46.104(d)(4)) | UCI data availability
  Public GitHub repo with seeds, environment, frozen plan | 25–30 references

SANITY CHECK
  Expected AUROC 0.64–0.69. Anything above 0.75 is a leak, not a result.
```

---

## PART 4 — What this changes about the paper

Under this methodology the study is no longer "we compared four models." It becomes:

> **A methodologically rigorous readmission risk model that (a) quantifies the realistic performance ceiling of structured administrative data, (b) demonstrates that widely used class-imbalance corrections degrade calibration without improving discrimination, (c) shows that biased importance measures can elevate clinically implausible predictors, and (d) evaluates clinical utility and subgroup fairness at operationally realistic decision thresholds.**

That is a defensible contribution at BMC Medical Informatics and Decision Making or JMIR Medical Informatics, not merely at Cureus — which matters, given that Cureus lost Web of Science indexing in October 2025 and its signalling value for graduate applications has correspondingly weakened.

Expect discrimination in the **0.64–0.69** range. Do not chase a higher number. A well-calibrated 0.67 with confidence intervals, decision curves, and a fairness audit is a substantially stronger paper than an unexplained 0.85 — and the 0.85 would be a bug.
