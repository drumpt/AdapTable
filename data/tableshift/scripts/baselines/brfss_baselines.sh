#!/bin/bash

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift

for EXPT in brfss_diabetes brfss_blood_pressure
do
  python scripts/ray_train.py \
    --experiment $EXPT \
    --num_samples 25 \
    --num_workers 1 \
    --cpu_per_worker 2 \
    --
    --cpu_models_only
done