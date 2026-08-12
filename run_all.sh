#!/bin/bash
# Full analysis, sequential and resumable. Safe to re-run: every stage skips
# completed work (Optuna studies resume; artifacts are checkpointed).
set -x
cd "$(dirname "$0")"
python run_stage.py tune --model lightgbm     --target 40
python run_stage.py tune --model xgboost      --target 40
python run_stage.py tune --model penalised_lr --target 40
python run_stage.py tune --model random_forest --target 12   # NOTE: halted at 10 recorded trials; see README
python run_stage.py convergence
python run_stage.py fit
python run_stage.py verify
python run_stage.py evaluate
python run_stage.py explain
python run_stage.py sensitivity
python run_stage.py convergence
echo "PIPELINE_COMPLETE"
