#!/bin/bash
# usage:
# sbatch --export=EXPERIMENT=$EXPERIMENT scripts/sbatch/run_expt.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-a40
#SBATCH --gpus-per-node=3
#SBATCH --mem-per-gpu=100000m
#SBATCH --time=15-00:00:00
#SBATCH --account=efml
#SBATCH --chdir=/gscratch/efml/jpgard/tableshift
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jpgard@cs.washington.edu
#SBATCH --output=/gscratch/efml/jpgard/tableshift/slurm-out/slurm-%j.out

date;hostname;id;pwd

echo 'experiment is: ' $EXPERIMENT

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift
which python

## experimental command to test on a single experiment iteration
python scripts/ray_train.py \
  --experiment $EXPERIMENT \
  --num_samples 100 \
  --gpu_models_only \
  --num_workers 3 \
  --cpu_per_worker 5 \
  --use_cached
