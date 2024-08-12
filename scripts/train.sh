wait_n() {
    background=($(jobs -p))
    echo $num_max_jobs
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

train_architectures_main() {
    seed=0
    batch_size=64

    benchmark=tableshift
    DATASETS="heloc diabetes_readmission anes mimic_extract_mort_hosp assistments nhanes_lead"
    MODELS="ResNet AutoInt TabNet FTTransformer MLP"
    train_lr=1e-4
    train_batch_size=64
    method="[calibrator,label_distribution_handler]"

    for dataset in ${DATASETS}; do
        for model in ${MODELS}; do
            python train.py \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                seed=${seed} \
                benchmark=${benchmark} \
                dataset=${dataset} \
                model=${model} \
                train_lr=${train_lr} \
                train_batch_size=${train_batch_size} \
                method=${method} \
                out_dir=${OUT_DIR} \
                log_dir=${LOG_DIR} \
                log_prefix=${LOG_PREFIX} \
                2>&1 &
            wait_n
            i=$((i + 1))
        done
    done
}

train_architectures_sub() {
    seed=0
    batch_size=64

    benchmark=tableshift
    DATASETS="college_scorecard physionet brfss_diabetes mimic_extract_los_3"
    MODELS="ResNet AutoInt TabNet FTTransformer MLP"
    train_lr=1e-4
    train_batch_size=64
    method="[calibrator,label_distribution_handler]"

    for dataset in ${DATASETS}; do
        for model in ${MODELS}; do
            python train.py \
                device=cuda:${GPUS[i % ${NUM_GPUS}]} \
                seed=${seed} \
                benchmark=${benchmark} \
                dataset=${dataset} \
                model=${model} \
                train_lr=${train_lr} \
                train_batch_size=${train_batch_size} \
                method=${method} \
                out_dir=${OUT_DIR} \
                log_dir=${LOG_DIR} \
                log_prefix=${LOG_PREFIX} \
                2>&1 &
            wait_n
            i=$((i + 1))
        done
    done
}

############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=10
##############################################

OUT_DIR=checkpoints
LOG_DIR=log
LOG_PREFIX=train_architectures

train_architectures_main
python utils/send_email.py --message "finish running train_architectures_main"

train_architectures_sub
python utils/send_email.py --message "finish running train_architectures_sub"