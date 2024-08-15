run_baselines() {
    # baselines
    episodic=true
    num_steps=1
    test_lr=0.0001
    if [ "$method" == "pl" ]; then
        episodic=true
        num_steps=1
        test_lr=0.0001
    elif [ "$method" == "tent" ]; then
        episodic=true
        num_steps=1
        test_lr=0.0001
    elif [ "$method" == "eata" ]; then
        episodic=true
        num_steps=10
        test_lr=0.00001
    elif [ "$method" == "sar" ]; then
        episodic=true
        num_steps=1
        test_lr=0.001
    fi

    # ours
    smoothing_factor=0.1
    uncertainty_upper_percentile_threshod=0.75
    uncertainty_lower_percentile_threshod=0.25

    python main.py \
        seed=${seed} \
        benchmark=${benchmark} \
        dataset=${dataset} \
        shift_type=${shift_type} \
        shift_severity=${shift_severity} \
        model=${model} \
        method=${method} \
        episodic=${episodic} \
        num_steps=${num_steps} \
        test_lr=${test_lr} \
        smoothing_factor=${smoothing_factor} \
        uncertainty_upper_percentile_threshod=${uncertainty_upper_percentile_threshod} \
        uncertainty_lower_percentile_threshod=${uncertainty_lower_percentile_threshod} \
        log_dir=${LOG_DIR} \
        log_prefix=${LOG_PREFIX} \
        out_dir=${OUT_DIR} \
        device=cuda:${gpu_idx} \
        retrain=false \
        2>&1
}

common_corruption() {
    SEEDS="0 1 2"
    model=MLP
    benchmark="tableshift"

    DATASETS="nhanes_lead"
    # DATASETS="nhanes_lead diabetes_readmission anes heloc"
    # SHIFT_LIST=(Gaussian uniform random_drop column_drop numerical categorical)
    SHIFT_LIST=(categorical)
    SEVERITY_LIST=(0.5)
    # SEVERITY_LIST=(0.1 0.1 0.2 0.2 0.5 0.5)
    # METHOD_LIST="[calibrator,label_distribution_handler]"
    METHOD_LIST="[calibrator,label_distribution_handler]"
    for seed in ${SEEDS}; do
        for dataset in ${DATASETS}; do
            for shift_idx in "${!SHIFT_LIST[@]}"; do
                shift_type=${SHIFT_LIST[shift_idx]}
                shift_severity=${SEVERITY_LIST[shift_idx]}

                for method in ${METHOD_LIST}; do
                    run_baselines
                    wait_n
                    gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
                done
            done
        done
    done

    DATASETS="anes"
    SHIFT_LIST=(uniform categorical)
    SEVERITY_LIST=(0.1 0.5)
    METHOD_LIST="[calibrator,label_distribution_handler]"
    for seed in ${SEEDS}; do
        for dataset in ${DATASETS}; do
            for shift_idx in "${!SHIFT_LIST[@]}"; do
                shift_type=${SHIFT_LIST[shift_idx]}
                shift_severity=${SEVERITY_LIST[shift_idx]}

                for method in ${METHOD_LIST}; do
                    run_baselines
                    wait_n
                    gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
                done
            done
        done
    done

    # DATASETS="diabetes_readmission"
    # SHIFT_LIST=(Gaussian column_drop)
    # SEVERITY_LIST=(0.1 0.2)
    # METHOD_LIST="[calibrator,label_distribution_handler]"
    # for seed in ${SEEDS}; do
    #     for dataset in ${DATASETS}; do
    #         for shift_idx in "${!SHIFT_LIST[@]}"; do
    #             shift_type=${SHIFT_LIST[shift_idx]}
    #             shift_severity=${SEVERITY_LIST[shift_idx]}

    #             for method in ${METHOD_LIST}; do
    #                 run_baselines
    #                 wait_n
    #                 gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
    #             done
    #         done
    #     done
    # done
}

various_architectures() {
    SEEDS="0 1 2"

    # MODEL_LIST="ResNet AutoInt TabNet FTTransformer"
    MODEL_LIST="AutoInt"
    benchmark="tableshift"
    DATASETS="anes heloc nhanes_lead diabetes_readmission"
    shift_type=null
    shift_severity=0
    METHOD_LIST="[calibrator,label_distribution_handler]"
    for seed in ${SEEDS}; do
        for model in ${MODEL_LIST}; do
            for dataset in ${DATASETS}; do
                for method in ${METHOD_LIST}; do
                    run_baselines
                    wait_n
                    gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
                done
            done
        done
    done

    # MODEL_LIST="ResNet"
    # benchmark="tableshift"
    # DATASETS="anes heloc nhanes_lead diabetes_readmission"
    # shift_type=null
    # shift_severity=0
    # METHOD_LIST="sar"
    # for seed in ${SEEDS}; do
    #     for model in ${MODEL_LIST}; do
    #         for dataset in ${DATASETS}; do
    #             for method in ${METHOD_LIST}; do
    #                 run_baselines
    #                 wait_n
    #                 gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
    #             done
    #         done
    #     done
    # done
}

harsh_condition() {
    SEEDS="0 1 2"
    model=MLP
    benchmark="tableshift"
    DATASETS="anes heloc nhanes_lead diabetes_readmission"
    SHIFT_LIST=(temp_corr imbalanced)
    SEVERITY_LIST=(0.1 0.1)

    # anes: imbalanced / nhanes_lead: temp_corr, imbalanced
    # 우리꺼는 anes temp_corr 추가로

    # DATASETS="anes"
    # SHIFT_LIST=(temp_corr)
    # SEVERITY_LIST=(0.1 0.1)
    METHOD_LIST="[calibrator,label_distribution_handler] pl tent eata sar lame"
    for seed in ${SEEDS}; do
        for dataset in ${DATASETS}; do
            for shift_idx in "${!SHIFT_LIST[@]}"; do
                shift_type=${SHIFT_LIST[shift_idx]}
                shift_severity=${SEVERITY_LIST[shift_idx]}

                for method in ${METHOD_LIST}; do
                    run_baselines
                    wait_n
                    gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
                done
            done
        done
    done

    # DATASETS="anes"
    # SHIFT_LIST=(imbalanced)
    # SEVERITY_LIST=(0.1 0.1)
    # METHOD_LIST="[calibrator,label_distribution_handler] pl tent eata sar lame"
    # for seed in ${SEEDS}; do
    #     for dataset in ${DATASETS}; do
    #         for shift_idx in "${!SHIFT_LIST[@]}"; do
    #             shift_type=${SHIFT_LIST[shift_idx]}
    #             shift_severity=${SEVERITY_LIST[shift_idx]}

    #             for method in ${METHOD_LIST}; do
    #                 run_baselines
    #                 wait_n
    #                 gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
    #             done
    #         done
    #     done
    # done

    # DATASETS="nhanes_lead"
    # SHIFT_LIST=(temp_corr imbalanced)
    # SEVERITY_LIST=(0.1 0.1)
    # # METHOD_LIST="[calibrator,label_distribution_handler] pl tent eata sar lame"
    # METHOD_LIST="lame"
    # for seed in ${SEEDS}; do
    #     for dataset in ${DATASETS}; do
    #         for shift_idx in "${!SHIFT_LIST[@]}"; do
    #             shift_type=${SHIFT_LIST[shift_idx]}
    #             shift_severity=${SEVERITY_LIST[shift_idx]}

    #             for method in ${METHOD_LIST}; do
    #                 run_baselines
    #                 wait_n
    #                 gpu_idx=$(((gpu_idx + 1) % ${NUM_GPUS}))
    #             done
    #         done
    #     done
    # done
}

############# run in single GPU ##############
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
gpu_idx=0
num_max_jobs=5

wait_n() {
    #limit the max number of jobs as NUM_MAX_JOB and wait
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}
##############################################

OUT_DIR=checkpoints
LOG_DIR=log

# LOG_PREFIX=common_corruption
# common_corruption
# python utils/send_email.py --message "finish adapt to common corruptions"

LOG_PREFIX=various_architectures
various_architectures
python utils/send_email.py --message "finish adapt on various architectures"

# LOG_PREFIX=harsh_condition
# harsh_condition
# python utils/send_email.py --message "finish adapt on harsh condition"