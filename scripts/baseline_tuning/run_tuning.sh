
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="tuning"
LOG_DIR="tuning"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=72 #12
  local num_max_jobs=72
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

openml_mlpbase(){
  seed="0"
  method="em"
  dataset="cmc"
  benchmark="openml-cc18"

  EPISODIC="true false"
  NUM_STEPS="1 5 10 15 20 25 30"
  TEST_LR="1e-3 1e-4 1e-5 1e-6 1e-7"

  for episodic in $EPISODIC; do
    for num_steps in $NUM_STEPS; do
      for test_lr in $TEST_LR; do
        python main.py \
            method=$method \
            seed=${seed} \
            log_dir=$LOG_DIR \
            log_prefix=${LOG_POSTFIX} \
            episodic=$episodic \
            benchmark=$benchmark \
            dataset="${dataset}" \
            shift_type=None \
            shift_severity=1 \
            retrain=true \
            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
            --config-name config.yaml \
            2>&1 &
        wait_n
      done
    done
  done
}

openml_mlpbase
#python send_email.py