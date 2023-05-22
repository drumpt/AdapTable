
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="log_ablation_vary_severity_debugging"
LOG_DIR="log_ablation_vary_severity_debugging"

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

#  DATASET="cmc"/ru
  DATASET="cmc semeion"
  METHOD="sar mae"
  SHIFT_SEVERITY="0.2 0.4 0.6 0.8"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi

  for dataset in $DATASET; do
    for method in $METHOD; do
      for shift_severity in $SHIFT_SEVERITY; do
        for seed in $SEED; do
          if [ $method = "sar" ]; then
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=random_drop \
                shift_severity=$shift_severity \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                --config-name config_sar.yaml \
                2>&1 &
            wait_n
            i=$((i + 1))
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=column_drop \
                shift_severity=$shift_severity \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
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
                shift_severity=$shift_severity \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                2>&1 &
            wait_n
            i=$((i + 1))
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=column_drop \
                shift_severity=$shift_severity \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
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

openml_mlpbase
python send_email.py