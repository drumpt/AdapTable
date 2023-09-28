############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_DIR="231115_computation_time"
LOG_POSTFIX="231115_computation_time"
CONF_DIR="conf/baseline_config"


wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local num_max_jobs=4
  # echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}


openml_mlpbase(){
    SEEDS="0"
    MODELS="mlp tabnet fttransformer"
    METHODS="ours"
    DATASETS="cmc mfeat-pixel"
    benchmark="openml-cc18"

    for seed in $SEEDS; do
        for dataset in $DATASETS; do
            for model in $MODELS; do
                for method in $METHODS; do
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOG_POSTFIX} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=Gaussian \
                        shift_severity=0.1 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                done
            done
        done
    done
}


tableshift_mlpbase(){
    SEEDS="0"
    MODELS="mlp tabnet fttransformer"
    METHODS="ours"
    DATASETS="heloc"
    benchmark="tableshift"

    for seed in $SEEDS; do
        for dataset in $DATASETS; do
            for model in $MODELS; do
                for method in $METHODS; do
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOSG_LOG_POSTFIX}_${model}_${dataset}_${seed} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=None \
                        retrain=false \
                        shift_severity=1 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                done
            done
        done
    done
}


openml_mlpbase_baseline(){
  SEEDS="0"
  # MODELS="mlp fttransformer tabnet"
  MODELS="mlp"
  # METHODS="pl ttt++ em sam eata sar lame"
  METHODS="sar"
  # DATASETS="cmc mfeat-pixel"
  DATASETS="cmc"
  benchmark="openml-cc18"

  for seed in $SEEDS; do
    for dataset in $DATASETS; do
      for model in $MODELS; do
        for method in $METHODS; do
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_gaussian \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=Gaussian \
              shift_severity=0.1 \
              --config-dir ${CONF_DIR} \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}


tableshift_mlpbase_baseline(){
  SEEDS="0"
  MODELS="mlp fttransformer tabnet"
  METHODS="pl ttt++ em sam eata sar lame"
  DATASETS="heloc"
  benchmark="tableshift"

  for seed in $SEEDS; do
    for dataset in $DATASETS; do
      for model in $MODELS; do
        for method in $METHODS; do
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=None \
              retrain=False \
              shift_severity=1 \
              --config-dir ${CONF_DIR} \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}

# openml_mlpbase
# tableshift_mlpbase
openml_mlpbase_baseline
# tableshift_mlpbase_baseline