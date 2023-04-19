#!/bin/bash

# Example script to run a ray training experiment.

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift

python scripts/ray_train.py \
	--experiment adult \
	--num_samples 2 \
	--num_workers 1 \
	--cpu_per_worker 4 \
	--use_cached \
	--models node