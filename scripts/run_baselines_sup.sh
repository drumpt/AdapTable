############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=4
##############################################
i=0

LOG_POSTFIX="240815_baseline"
LOG_DIR="log"
CONF_DIR="conf/"

wait_n() {
    #limit the max number of jobs as NUM_MAX_JOB and wait
    background=($(jobs -p))
    local default_num_jobs=4 #12
    local num_max_jobs=4
    echo $num_max_jobs
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

openml_mlpbase() {
    SEEDS="0 1 2"
    MODELS="mlp"
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_numerical \
                        benchmark=$benchmark \
                        dataset_save_dir=data/raw_data \
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_categorical \
                        benchmark=$benchmark \
                        dataset_save_dir=data/raw_data \
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_gaussian \
                        benchmark=$benchmark \
                        dataset_save_dir=data/raw_data \
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_uniform \
                        benchmark=$benchmark \
                        dataset_save_dir=data/raw_data \
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_rd \
                        benchmark=$benchmark \
                        dataset_save_dir=data/raw_data \
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
                        out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model}_cd \
                        benchmark=$benchmark \
                        dataset="${dataset}" \
                        dataset_save_dir=data/raw_data \
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

tableshift() {
    SEEDS="0 1 2"
    MODELS="knn"
    DATASETS="nhanes_lead brfss_diabetes diabetes_readmission mimic_extract_mort_hosp assistments"
    benchmark="tableshift"

    for seed in $SEEDS; do
        for dataset in $DATASETS; do
            for model in $MODELS; do
                #        for method in $METHODS; do
                python supervised.py \
                    seed=${seed} \
                    log_dir=$LOG_DIR \
                    log_prefix=${LOG_POSTFIX} \
                    device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                    out_dir=${LOG_POSTFIX}_seed${seed}_dataset${dataset}_model${model} \
                    model="${model}" \
                    benchmark=$benchmark \
                    dataset_save_dir=data/raw_data \
                    dataset="${dataset}" \
                    shift_type=None \
                    retrain=False \
                    shift_severity=1 \
                    --config-dir $CONF_DIR \
                    --config-name ours_mlp.yaml \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done
    done
}

tableshift
#openml_mlpbase
#python send_email.py