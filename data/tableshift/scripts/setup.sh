# m5a.16xlarge instance (12GB connection)
# m4.16xlarge instance (25GB connection)

sudo yum install git -y

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh && \
bash Miniconda3-py38_22.11.1-1-Linux-x86_64.sh

source ~/.bashrc && conda config --set auto_activate_base false

git clone https://github.com/jpgard/tableshift.git

conda env create -f tableshift/environment.yml

mkdir tableshift/tmp && \
scp -r jpgard@bam.cs.washington.edu:/homes/gws/jpgard/tablebench/tmp/mimic_extract* tableshift/tmp


# example command to run after setup:
#python experiments/domain_shift.py \
#  --num_samples 100 \
#  --num_workers 8 \
#  --cpu_per_worker 3 \
#  --use_cached \
#  --cpu_models_only \
#  --experiment mimic_extract_mort_hosp_ins
