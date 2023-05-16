
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="log_ablation_vary_train_ratio"
LOG_DIR="log_ablation_vary_train_ratio"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=20 #12
  local num_max_jobs=20
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

openml_mlpbase(){
  SEED="0 1 2 3 4"
#  DATASET="cmc"
#  METHOD="mae"
#  TRAIN_RATIO="0.1"
  echo "run"

#  DATASET="cmc"
  DATASET="cmc semeion mfeat-karhunen optdigits diabetes mfeat-pixel dna"
  METHOD="mae"
  TRAIN_RATIO="0.2 0.4 0.6 0.8 1"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi

  for dataset in $DATASET; do
    for method in $METHOD; do
      for train_ratio in $TRAIN_RATIO; do
        for seed in $SEED; do
          echo "run"
          if [ $method = "sar" ]; then
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=random_drop \
                shift_severity=0.6 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                train_ratio=${train_ratio} \
                --config-name config_sar.yaml \
                2>&1 &
            wait_n
            i=$((i + 1))
          else
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=random_drop \
                shift_severity=0.6 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                train_ratio=${train_ratio} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                2>&1 &
            wait_n
            i=$((i + 1))
          fi
        done
      done
    done
  done
}
echo "run"
openml_mlpbase

wait
python send_email.py