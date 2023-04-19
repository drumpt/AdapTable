#!/bin/bash

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift

for EXPT in metamimic_alcohol metamimic_anemia metamimic_atrial \
      metamimic_diabetes metamimic_heart metamimic_hypertension \
      metamimic_hypotension metamimic_ischematic metamimic_lipoid \
      metamimic_overweight metamimic_purpura metamimic_respiratory
do
  python scripts/ray_train.py \
    --experiment $EXPT \
    --num_samples 25 \
    --num_workers 1 \
    --cpu_per_worker 2 \
    --cpu_models_only
done