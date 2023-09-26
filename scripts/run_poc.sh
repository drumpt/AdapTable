############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
NUM_MAX_JOBS=20
i=0 # GPU index
##############################################
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=${NUM_MAX_JOBS}
  # echo $num_max_jobs
  if ((${#background[@]} >= ${NUM_MAX_JOBS})); then
    wait -n
  fi
}

poc_tableshift() {
    local MODEL="MLP"
    local DATASET="heloc diabetes_readmission anes"
    local LOG_DIR="log"
    local LOG_PREFIX="tableshift_poc"

    for model in $MODEL; do
        for dataset in $DATASET; do
            python main.py \
            model=${model} \
            benchmark=tableshift \
            dataset=${dataset} \
            shift_type=null \
            shift_severity=0 \
            seed=0 \
            method=[label_shift_handler] \
            log_dir=${LOG_DIR} \
            log_prefix=${LOG_PREFIX} \
            --config-name config.yaml \
        2>&1 &
        wait_n
        i=$((i + 1))
        done
    done
}

poc_openml_cc18(){
    local MODEL="MLP"
    local DATASET="diabetes"
    local SHIFT_TYPE=(null Gaussian uniform random_drop column_drop numerical categorical)
    local SHIFT_SEVERITY=(0 0.5 0.5 0.5 0.5 0.5 0.5)
    local LOG_DIR="log"
    local LOG_PREFIX="openml_cc18_poc"

    for model in $MODEL; do
        for dataset in $DATASET; do
            for ((shift_idx=0; shift_idx<${#SHIFT_TYPE[@]}; shift_idx++)); do
                python main.py \
                model=${model} \
                benchmark=openml-cc18 \
                dataset=${dataset} \
                shift_type=${SHIFT_TYPE[shift_idx]} \
                shift_severity=${SHIFT_SEVERITY[shift_idx]} \
                seed=0 \
                method=[label_shift_handler] \
                log_dir=${LOG_DIR} \
                log_prefix=${LOG_PREFIX} \
                --config-name config.yaml \
            2>&1 &
            wait_n
            i=$((i + 1))
            done
        done
    done
}

poc_openml_cc18
# poc_tableshift