#!/bin/bash
set -x
cd "$(dirname "$0")"
python run_stage.py fit
python run_stage.py verify
python run_stage.py evaluate
python run_stage.py explain
python run_stage.py sensitivity
python run_stage.py convergence
echo "DOWNSTREAM_COMPLETE"
