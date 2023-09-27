############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="230927_baseline"
LOG_DIR="230927_baseline"
CONF_DIR="conf/baseline_config"

wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=32 #12
  local num_max_jobs=32
  echo $num_max_jobs
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

openml_mlpbase(){
  SEEDS="0 1 2"
  MODELS="tabtransformer tabnet mlp"
  METHODS="ttt++ eata em lame memo pl sam sar"
  DATASETS="adult cmc mfeat-karhunen optdigits diabetes semeion mfeat-pixel dna"
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
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=numerical \
              shift_severity=0.5 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=categorical \
              shift_severity=0.5 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=Gaussian \
              shift_severity=0.1 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=uniform \
              shift_severity=0.1 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=random_drop \
              shift_severity=0.2 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
          python main.py \
              method=$method \
              seed=${seed} \
              log_dir=$LOG_DIR \
              log_prefix=${LOG_POSTFIX} \
              device=cuda:${GPUS[i % ${NUM_GPUS}]} \
              out_dir=${LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=column_drop \
              shift_severity=0.2 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}

tableshift_mlpbase(){
  SEEDS="0 1 2"
  MODELS="mlp tabtransformer tabnet"
  METHODS="eata em lame memo pl sam sar ttt++"
  DATASETS="heloc diabetes_readmission anes"
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
              out_dir=${LOSG_LOG_POSTFIX} \
              benchmark=$benchmark \
              dataset="${dataset}" \
              shift_type=None \
              shift_severity=1 \
              --config-dir $CONF_DIR \
              --config-name config_${method}_${model}.yaml \
              2>&1 &
          wait_n
          i=$((i + 1))
        done
      done
    done
  done
}

openml_mlpbase
tableshift_mlpbase
#python send_email.py