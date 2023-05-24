
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="log_ablation_vary_batchsize"
LOG_DIR="log_ablation_vary_batchsize"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=60 #12
  local num_max_jobs=60
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
  DATASET="semeion"
  METHOD="sar mae"
  BATCH_SIZE="1 2 4 8 16 32 64"

  if [ $method = "mae" ] || [ $method = "mae_random_mask" ] || [ $method = "memo" ]; then
    episodic="true"
  else
    episodic="false"
  fi

  for dataset in $DATASET; do
    for method in $METHOD; do
      for test_batch_size in $BATCH_SIZE; do
        for seed in $SEED; do
          echo "run"
          if [ $method = "sar" ]; then
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=null \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_type=mean_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_type=std_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_type=mean_std_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_type=random_drop \
                shift_severity=0.6 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_severity=0.6 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
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
                shift_type=null \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                test_batch_size=${test_batch_size} \
                2>&1 &
            wait_n
            i=$((i + 1))
            python main.py \
                method=$method \
                log_dir=$LOG_DIR \
                episodic=$episodic \
                meta_dataset=openml-cc18 \
                dataset="${dataset}" \
                shift_type=std_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                test_batch_size=${test_batch_size} \
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
                shift_type=mean_std_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                test_batch_size=${test_batch_size} \
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
                shift_type=mean_shift \
                shift_severity=1 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                test_batch_size=${test_batch_size} \
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
                shift_severity=0.6 \
                retrain=true \
                seed=${seed} \
                log_prefix=${LOG_POSTFIX} \
                test_batch_size=${test_batch_size} \
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