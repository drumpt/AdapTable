
############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="lrtune"

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

run_cmc(){
  MIXUP_SCALE="1 5 10"
  TEST_LR="3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5"
  for mixup_scale in $MIXUP_SCALE; do
    for test_lr in $TEST_LR; do
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=cmc \
          shift_type=null \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull} \
          log_dir=log/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=cmc \
          shift_type=random_drop \
          shift_severity=0.6 \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typerandom_drop \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          log_dir=log/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typerandom_drop \
          2>&1 &
      wait_n
      i=$((i + 1))
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=cmc \
          shift_type=column_drop \
          shift_severity=0.6 \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typecolumn_drop \
          log_dir=log/cmc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typecolumn_drop \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
    done
  done
}

run_semeion(){
  MIXUP_SCALE="1 5 10"
  TEST_LR="3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5"
  for mixup_scale in $MIXUP_SCALE; do
    for test_lr in $TEST_LR; do
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=semeion \
          shift_type=null \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull \
          log_dir=log/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=semeion \
          shift_type=random_drop \
          shift_severity=0.6 \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typerandom_drop \
          log_dir=log/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typerandom_drop \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
      python main.py \
          meta_dataset=openml-cc18 \
          dataset=semeion \
          shift_type=column_drop \
          shift_severity=0.6 \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typecolumn_drop \
          log_dir=log/semeion_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typecolumn_drop \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
    done
  done
}

run_heloc(){
  MIXUP_SCALE="1 5 10"
  TEST_LR="3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5"
  for mixup_scale in $MIXUP_SCALE; do
    for test_lr in $TEST_LR; do
      python main.py \
          meta_dataset=tableshift \
          dataset=heloc \
          retrain=true \
          seed=0 \
          mixup=true \
          mixup_scale=${mixup_scale} \
          test_lr=${test_lr} \
          out_dir=exps/heloc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull \
          log_dir=log/heloc_${LOG_POSTFIX}/test_lr${test_lr}_mixup_scale_${mixup_scale}_shfit_typenull \
          device=cuda:${GPUS[i % ${NUM_GPUS}]} \
          2>&1 &
      wait_n
      i=$((i + 1))
    done
  done
}


run_cmc
run_semeion
run_heloc
