############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_DIR="log/231116_new_hyperparameter_sensitivity"
LOG_POSTFIX="231115_hyperparameter_sensitivity"
CONF_DIR="conf/baseline_config"


wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local num_max_jobs=4
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

hyperparameter_ours() {
    SEEDS="0"
    DATASETS="heloc"
    MODELS="mlp"
    METHODS="ours"
    benchmark="tableshift"

    SMOOTHING_FACTOR_LIST="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8. 0.9 1"
    # SMOOTHING_FACTOR_LIST="0.8"
    UUP_PERCENTILE_LIST="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1"
    ULP_PERCENTILE_LIST="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"

    for seed in $SEEDS; do
        for dataset in $DATASETS; do
            for model in $MODELS; do
                for method in $METHODS; do
                    for smoothing_factor in ${SMOOTHING_FACTOR_LIST}; do
                        python main.py \
                            seed=${seed} \
                            log_dir=${LOG_DIR} \
                            log_prefix=${model}_${dataset}_${seed}_ours_smoothing_factor_${smoothing_factor} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=${model}_${dataset}_${seed}_ours_smoothing_factor_${smoothing_factor} \
                            benchmark=$benchmark \
                            dataset="${dataset}" \
                            shift_type=None \
                            retrain=false \
                            shift_severity=1 \
                            smoothing_factor=${smoothing_factor} \
                            --config-name ours_${model}.yaml \
                            2>&1 &
                        wait_n
                        i=$((i + 1))
                    done

                    for uncertainty_upper_percentile_threshod in ${UUP_PERCENTILE_LIST}; do
                        python main.py \
                            seed=${seed} \
                            log_dir=${LOG_DIR} \
                            log_prefix=${model}_${dataset}_${seed}_ours_uncertainty_upper_percentile_threshod_${uncertainty_upper_percentile_threshod} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=${model}_${dataset}_${seed}_ours_uncertainty_upper_percentile_threshod_${uncertainty_upper_percentile_threshod} \
                            benchmark=$benchmark \
                            dataset="${dataset}" \
                            shift_type=None \
                            retrain=false \
                            shift_severity=1 \
                            uncertainty_upper_percentile_threshod=${uncertainty_upper_percentile_threshod} \
                            --config-name ours_${model}.yaml \
                            2>&1 &
                        wait_n
                        i=$((i + 1))
                    done

                    for uncertainty_lower_percentile_threshod in ${ULP_PERCENTILE_LIST}; do
                        python main.py \
                            seed=${seed} \
                            log_dir=${LOG_DIR} \
                            log_prefix=${model}_${dataset}_${seed}_ours_uncertainty_lower_percentile_threshod_${uncertainty_lower_percentile_threshod} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=${model}_${dataset}_${seed}_ours_uncertainty_lower_percentile_threshod_${uncertainty_lower_percentile_threshod} \
                            benchmark=$benchmark \
                            dataset="${dataset}" \
                            shift_type=None \
                            retrain=false \
                            shift_severity=1 \
                            uncertainty_lower_percentile_threshod=${uncertainty_lower_percentile_threshod} \
                            --config-name ours_${model}.yaml \
                            2>&1 &
                        wait_n
                        i=$((i + 1))
                    done
                done
            done
        done
    done
}


hyperparameter_tent() {
  SEEDS="0"
  MODELS="mlp"
  METHODS="em"
  DATASETS="heloc"
  benchmark="tableshift"

  LR_LIST="1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2"
  NUM_STEPS_LIST="1 2 3 5 10 20 30"
  for seed in $SEEDS; do
        for dataset in $DATASETS; do
            for model in $MODELS; do
                for method in $METHODS; do
                #     for test_lr in ${LR_LIST}; do
                #         python main.py \
                #             method=$method \
                #             seed=${seed} \
                #             log_dir=${LOG_DIR} \
                #             log_prefix=seed_${seed}_dataset_${dataset}_model_${model}_method_${method}_lr_${test_lr} \
                #             device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                #             out_dir=seed_${seed}_dataset_${dataset}_model_${model}_method_${method}_lr_${test_lr} \
                #             benchmark=$benchmark \
                #             dataset=${dataset} \
                #             shift_type=None \
                #             retrain=False \
                #             shift_severity=1 \
                #             test_lr=${test_lr} \
                #             --config-dir ${CONF_DIR} \
                #             --config-name config_${method}_${model}.yaml \
                #             2>&1 &
                #         wait_n
                #         i=$((i + 1))
                #     done

                    for num_steps in ${NUM_STEPS_LIST}; do
                        python main.py \
                            method=$method \
                            seed=${seed} \
                            log_dir=${LOG_DIR} \
                            log_prefix=seed_${seed}_dataset_${dataset}_model_${model}_method_${method}_num_steps${num_steps} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=seed_${seed}_dataset_${dataset}_model_${model}_method_${method}_num_steps_${num_steps} \
                            benchmark=$benchmark \
                            dataset=${dataset} \
                            shift_type=None \
                            retrain=False \
                            shift_severity=1 \
                            num_steps=${num_steps} \
                            --config-dir ${CONF_DIR} \
                            --config-name config_${method}_${model}.yaml \
                            2>&1 &
                        wait_n
                        i=$((i + 1))
                    done

                done
            done
        done
  done
}

# hyperparameter_ours
hyperparameter_tent