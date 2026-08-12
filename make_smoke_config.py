#!/usr/bin/env python3
"""Derive a reduced configuration for fast end-to-end verification.

Produces config.smoke.yaml: smaller cohort bounds, a minimal search budget and
fewer bootstrap resamples, so the whole pipeline runs in minutes on synthetic
data. Never use this configuration for the real analysis.
"""
import yaml

cfg = yaml.safe_load(open("config.yaml"))
cfg["cohort"].update(expected_n_min=5000, expected_n_max=20000,
                     expected_prevalence_min=0.03, expected_prevalence_max=0.30)
cfg["features"].update(expected_a1c_tested_min=0.10, expected_a1c_tested_max=0.30)
cfg["tuning"].update(n_trials=4)
cfg["cv"].update(n_splits=3, n_repeats=2, tuning_n_repeats=1)
cfg["bootstrap"].update(n_resamples=200)
cfg["explain"].update(shap_sample_size=800, permutation_repeats=3)
cfg["fairness"].update(min_subgroup_n=100, min_subgroup_events=10)
cfg["figures"].update(formats=["png"])
yaml.safe_dump(cfg, open("config.smoke.yaml", "w"), sort_keys=False)
print("wrote config.smoke.yaml")
