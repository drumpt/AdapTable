############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
##############################################
i=0

LOG_POSTFIX="log/230927_baseline"
LOG_DIR="230927_baseline"
CONF_DIR="conf/baseline_config"

wait_n() {
    #limit the max number of jobs as NUM_MAX_JOB and wait
    background=($(jobs -p))
    local default_num_jobs=12 #12
    local num_max_jobs=12
    echo $num_max_jobs
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

openml_mlpbase() {
    SEEDS="0 1 2"
    MODELS="fttransformer"
    METHODS="ours"
    DATASETS="adult cmc mfeat-karhunen optdigits diabetes semeion mfeat-pixel dna"
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
                        shift_type=numerical \
                        shift_severity=0.5 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOG_POSTFIX} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=categorical \
                        shift_severity=0.5 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
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
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOG_POSTFIX} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=uniform \
                        shift_severity=0.1 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOG_POSTFIX} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=random_drop \
                        shift_severity=0.2 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                    python main.py \
                        seed=${seed} \
                        log_dir=$LOG_DIR \
                        log_prefix=${LOG_POSTFIX} \
                        device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                        out_dir=${LOG_POSTFIX} \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        shift_type=column_drop \
                        shift_severity=0.2 \
                        --config-name ours_${model}.yaml \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                done
            done
        done
    done
}

tableshift_mlpbase() {
    SEEDS="0 1 2"
    MODELS="mlp"
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
                        2>&1
                    wait_n
                    i=$((i + 1))
                done
            done
        done
    done
}

tableshift_shiftcheck() {
    SEEDS="0"
    MODELS="mlp"
    METHODS="ours"
    DATASETS="heloc"
    benchmark="tableshift"

    # SHIFT_TYPE_LIST="temp_corr imbalanced numerical categorical None Gaussian uniform random_drop column_drop"
    SHIFT_TYPE_LIST="temp_corr"
    for shift_type in ${SHIFT_TYPE_LIST}; do
        for seed in ${SEEDS}; do
            for dataset in ${DATASETS}; do
                for model in ${MODELS}; do
                    for method in ${METHODS}; do
                        python main.py \
                            model=${model} \
                            seed=${seed} \
                            train_lr=1e-4 \
                            method=[calibrator,label_distribution_handler] \
                            log_dir=$LOG_DIR \
                            log_prefix=${LOG_POSTFIX} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=checkpoints \
                            benchmark=${benchmark} \
                            dataset=${dataset} \
                            shift_type=${shift_type} \
                            retrain=false \
                            shift_severity=1 \
                            2>&1
                        wait_n
                        i=$((i + 1))
                    done
                done
            done
        done
    done
}

tableshift_modelcheck() {
    SEEDS="0"
    MODELS="ResNet AutoInt"
    METHODS="ours"
    DATASETS="heloc"
    benchmark="tableshift"

    SHIFT_TYPE_LIST="temp_corr"
    for shift_type in ${SHIFT_TYPE_LIST}; do
        for seed in ${SEEDS}; do
            for dataset in ${DATASETS}; do
                for model in ${MODELS}; do
                    for method in ${METHODS}; do
                        python main.py \
                            model=${model} \
                            seed=${seed} \
                            train_lr=1e-4 \
                            method=[calibrator,label_distribution_handler] \
                            log_dir=$LOG_DIR \
                            log_prefix=${LOG_POSTFIX} \
                            device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                            out_dir=checkpoints \
                            benchmark=${benchmark} \
                            dataset=${dataset} \
                            shift_type=${shift_type} \
                            retrain=false \
                            shift_severity=1 \
                            2>&1
                        wait_n
                        i=$((i + 1))
                    done
                done
            done
        done
    done
}

# openml_mlpbase
# tableshift_mlpbase
# tableshift_shiftcheck
tableshift_modelcheck
# python send_email.py
