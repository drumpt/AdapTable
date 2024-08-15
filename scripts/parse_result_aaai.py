# %%
import os
import re

import numpy as np
import pandas as pd


def parse(file_path):
    unparsed_result = {
        "test_acc_before": np.nan,
        "test_acc_after": np.nan,
        "test_bacc_before": np.nan,
        "test_bacc_after": np.nan,
        "test_f1_before": np.nan,
        "test_f1_after": np.nan,
    }
    if not os.path.exists(file_path):
        return unparsed_result

    with open(file_path, "r") as file:
        content = file.read()

    # define the regex patterns
    before_adaptation_pattern = r"before adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"
    after_adaptation_pattern = r"after adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"

    # perform the regex search on the entire file content
    before_adaptation_match = re.search(before_adaptation_pattern, content)
    after_adaptation_match = re.search(after_adaptation_pattern, content)

    if not before_adaptation_match or not after_adaptation_match:
        return unparsed_result

    (
        _,
        before_adaptation_test_acc,
        before_adaptation_test_bacc,
        before_adaptation_test_f1,
    ) = before_adaptation_match.groups()
    (
        _,
        after_adaptation_test_acc,
        after_adaptation_test_bacc,
        after_adaptation_test_f1,
    ) = after_adaptation_match.groups()

    return {
        "test_acc_before": before_adaptation_test_acc,
        "test_acc_after": after_adaptation_test_acc,
        "test_bacc_before": before_adaptation_test_bacc,
        "test_bacc_after": after_adaptation_test_bacc,
        "test_f1_before": before_adaptation_test_f1,
        "test_f1_after": after_adaptation_test_f1,
    }



def parse_knn_file(file_path):
    unparsed_result = {
        "test_acc_before": np.nan,
        "test_acc_after": np.nan,
        "test_bacc_before": np.nan,
        "test_bacc_after": np.nan,
        "test_f1_before": np.nan,
        "test_f1_after": np.nan,
    }
    if not os.path.exists(file_path):
        return unparsed_result

    with open(file_path, "r") as file:
        content = file.read()

    # define the regex patterns
    # before_adaptation_pattern = r"before adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"
    before_adaptation_pattern = r"before adaptation \| acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"
    # after_adaptation_pattern = r"after adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"
    after_adaptation_pattern = r"before adaptation \| acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)"

    # perform the regex search on the entire file content
    before_adaptation_match = re.search(before_adaptation_pattern, content)
    after_adaptation_match = re.search(after_adaptation_pattern, content)

    if not before_adaptation_match or not after_adaptation_match:
        return unparsed_result

    (
        before_adaptation_test_acc,
        before_adaptation_test_bacc,
        before_adaptation_test_f1,
    ) = before_adaptation_match.groups()
    (
        after_adaptation_test_acc,
        after_adaptation_test_bacc,
        after_adaptation_test_f1,
    ) = after_adaptation_match.groups()

    return {
        "test_acc_before": before_adaptation_test_acc,
        "test_acc_after": after_adaptation_test_acc,
        "test_bacc_before": before_adaptation_test_bacc,
        "test_bacc_after": after_adaptation_test_bacc,
        "test_f1_before": before_adaptation_test_f1,
        "test_f1_after": after_adaptation_test_f1,
    }


def parse_common_corruption():
    LOG_DIR = "log"
    LOG_PREFIX = "common_corruption"

    SEEDS = [0, 1, 2]
    MODEL_LIST = ["MLP"]
    benchmark = "tableshift"
    # DATASETS = ["anes", "heloc", "nhanes_lead", "diabetes_readmission"]
    DATASETS = ["anes", "heloc", "nhanes_lead"]
    SHIFT_LIST = [
        "Gaussian",
        "uniform",
        "random_drop",
        "column_drop",
        "numerical",
        "categorical",
    ]
    SEVERITY_LIST = [0.1, 0.1, 0.2, 0.2, 0.5, 0.5]
    METHOD_LIST = [
        "calibrator_label_distribution_handler",
        "pl",
        "tent",
        "eata",
        "sar",
        "lame",
    ]

    for model in MODEL_LIST:
        for dataset in DATASETS:
            for i in range(len(SHIFT_LIST)):
                shift_type = SHIFT_LIST[i]
                shift_severity = SEVERITY_LIST[i]
                result_list = []
                for method in METHOD_LIST:
                    stochastic_result_list = []
                    for seed in SEEDS:
                        log_dir = f"{LOG_DIR}/{LOG_PREFIX}/{benchmark}_{dataset}_shift_type_{shift_type}_shift_severity_{shift_severity}_{model}_{method}_seed_{seed}.txt"
                        result = parse(log_dir)

                        stochastic_result_list.append(
                            [
                                result["test_acc_before"],
                                result["test_bacc_before"],
                                result["test_f1_before"],
                                result["test_acc_after"],
                                result["test_bacc_after"],
                                result["test_f1_after"],
                            ]
                        )
                    stochastic_result = np.array(
                        stochastic_result_list, dtype=np.float32
                    )
                    try:
                        mean_list = np.mean(stochastic_result, axis=0)
                        sem_list = np.std(stochastic_result, axis=0, ddof=1) / np.sqrt(
                            stochastic_result.shape[-1]
                        )
                    except:
                        mean_list, sem_list = [np.nan] * 6, [np.nan] * 6
                    # temp_result_list = [f"{mean * 100:.1f} ± {sem * 100:.1f}" for mean, sem in zip(mean_list, sem_list)]
                    temp_result_list = [
                        f"{mean * 100:.1f}" for mean, sem in zip(mean_list, sem_list)
                    ]
                    result_list.append(temp_result_list)
                result_df = pd.DataFrame(
                    result_list,
                    columns=["Acc.", "bAcc.", "F1", "Acc.", "bAcc.", "F1"],
                    index=METHOD_LIST,
                )
                print(f"{model=} {dataset=} {shift_type}")
                print(f"{result_df=}\n")


def parse_various_architectures():
    LOG_DIR = "log"
    LOG_PREFIX = "various_architectures"

    SEEDS = [0, 1, 2]
    MODEL_LIST = ["ResNet", "AutoInt", "TabNet", "FTTransformer"]
    # MODEL_LIST = ["ResNet"]
    benchmark = "tableshift"
    DATASETS = ["anes", "heloc", "nhanes_lead", "diabetes_readmission"]
    # DATASETS = ["anes"]
    shift_type = "None"
    shift_severity = 0

    METHOD_LIST = [
        "calibrator_label_distribution_handler",
        "pl",
        "tent",
        "eata",
        "sar",
        "lame",
    ]

    for model in MODEL_LIST:
        result_list = []
        for method in METHOD_LIST:
            temp_result_list = []
            for dataset in DATASETS:
                stochastic_result_list = []
                for seed in SEEDS:
                    log_dir = f"{LOG_DIR}/{LOG_PREFIX}/{benchmark}_{dataset}_shift_type_{shift_type}_shift_severity_{shift_severity}_{model}_{method}_seed_{seed}.txt"

                    result = parse(log_dir)
                    stochastic_result_list.append(
                        [
                            result["test_acc_before"],
                            result["test_bacc_before"],
                            result["test_f1_before"],
                            result["test_acc_after"],
                            result["test_bacc_after"],
                            result["test_f1_after"],
                        ]
                    )
                stochastic_result = np.array(stochastic_result_list, dtype=np.float32)
                try:
                    mean_list = np.mean(stochastic_result, axis=0)
                    sem_list = np.std(stochastic_result, axis=0, ddof=1) / np.sqrt(
                        stochastic_result.shape[-1]
                    )
                except:
                    mean_list, sem_list = [np.nan] * 6, [np.nan] * 6
                temp_result = [
                    [mean * 100 for mean, sem in zip(mean_list, sem_list)][-4],
                    [mean * 100 for mean, sem in zip(mean_list, sem_list)][-1],
                ]
                temp_result_list.append(temp_result)
            print(f"{temp_result_list=}")
            try:
                result_list.append(np.mean(temp_result_list, axis=0))
            except:
                result_list.append([np.nan] * len(temp_result_list[0]))
            print(f"{result_list=}")
        result_df = pd.DataFrame(result_list, columns=["F1", "F1"], index=METHOD_LIST)
        print(f"{model=}")
        print(f"{result_df=}\n")


def parse_harsh_condition():
    LOG_DIR = "log"
    LOG_PREFIX = "harsh_condition"

    SEEDS = [0, 1, 2]
    MODEL_LIST = ["MLP"]
    benchmark = "tableshift"
    DATASETS = ["anes", "heloc", "nhanes_lead", "diabetes_readmission"]
    SHIFT_LIST = [
        "temp_corr",
        "imbalanced",
    ]
    SEVERITY_LIST = [0.1, 0.1]
    METHOD_LIST = [
        "calibrator_label_distribution_handler",
        "pl",
        "tent",
        "eata",
        "sar",
        "lame",
    ]

    for model in MODEL_LIST:
        for i in range(len(SHIFT_LIST)):
            shift_type = SHIFT_LIST[i]
            shift_severity = SEVERITY_LIST[i]
            for dataset in DATASETS:
                result_list = []
                for method in METHOD_LIST:
                    stochastic_result_list = []
                    for seed in SEEDS:
                        log_dir = f"{LOG_DIR}/{LOG_PREFIX}/{benchmark}_{dataset}_shift_type_{shift_type}_shift_severity_{shift_severity}_{model}_{method}_seed_{seed}.txt"
                        # print(f"{log_dir=} {os.path.exists(log_dir)=}")
                        result = parse(log_dir)

                        stochastic_result_list.append(
                            [
                                result["test_acc_before"],
                                result["test_bacc_before"],
                                result["test_f1_before"],
                                result["test_acc_after"],
                                result["test_bacc_after"],
                                result["test_f1_after"],
                            ]
                        )
                    stochastic_result = np.array(
                        stochastic_result_list, dtype=np.float32
                    )
                    try:
                        mean_list = np.mean(stochastic_result, axis=0)
                        sem_list = np.std(stochastic_result, axis=0, ddof=1) / np.sqrt(
                            stochastic_result.shape[-1]
                        )
                    except:
                        mean_list, sem_list = [np.nan] * 6, [np.nan] * 6
                    temp_result_list = [f"{mean * 100:.1f}±{sem * 100:.1f}" for mean, sem in zip(mean_list, sem_list)]
                    # temp_result_list = [
                    #     f"{mean * 100:.1f}" for mean, sem in zip(mean_list, sem_list)
                    # ]
                    result_list.append(temp_result_list)
                result_df = pd.DataFrame(
                    result_list,
                    columns=["Acc.", "bAcc.", "F1", "Acc.", "bAcc.", "F1"],
                    index=METHOD_LIST,
                )
                print(f"{model=} {shift_type=} {dataset=}")
                print(f"{result_df=}\n")


def parse_knn():
    LOG_DIR = "log"
    LOG_PREFIX = "240815_baseline"

    SEEDS = [0, 1, 2]
    MODEL_LIST = ["knn"]
    benchmark = "tableshift"
    DATASETS = [
        "nhanes_lead",
        "brfss_diabetes",
        "diabetes_readmission",
        "mimic_extract_mort_hosp",
        "assistments",
    ]
    shift_type = None
    shift_severity = 1
    method = "calibrator_label_distribution_handler"

    for model in MODEL_LIST:
        result_list = []
        for dataset in DATASETS:
            stochastic_result_list = []
            for seed in SEEDS:
                log_dir = f"{LOG_DIR}/{LOG_PREFIX}/{benchmark}_{dataset}_shift_type_{shift_type}_shift_severity_{shift_severity}_{model}_{method}_seed_{seed}.txt"
                # print(f"{log_dir=} {os.path.exists(log_dir)=}")
                result = parse_knn_file(log_dir)

                stochastic_result_list.append(
                    [
                        result["test_acc_before"],
                        result["test_bacc_before"],
                        result["test_f1_before"],
                        result["test_acc_after"],
                        result["test_bacc_after"],
                        result["test_f1_after"],
                    ]
                )
                # print(f"{stochastic_result_list=}")
            stochastic_result = np.array(stochastic_result_list, dtype=np.float32)
            # print(f"{stochastic_result=}")
            try:
                mean_list = np.mean(stochastic_result, axis=0)
                sem_list = np.std(stochastic_result, axis=0, ddof=1) / np.sqrt(
                    stochastic_result.shape[-1]
                )
            except:
                mean_list, sem_list = [np.nan] * 6, [np.nan] * 6
            temp_result_list = [f"{mean * 100:.1f}±{sem * 100:.1f}" for mean, sem in zip(mean_list, sem_list)]
            result_list.append(temp_result_list)
        result_df = pd.DataFrame(
            result_list,
            columns=["Acc.", "bAcc.", "F1", "Acc.", "bAcc.", "F1"],
            index=DATASETS,
        )
        print(f"{model=} {dataset=} {method=}")
        print(f"{result_df=}\n")


if __name__ == "__main__":
    # parse_common_corruption()
    # parse_various_architectures()
    parse_harsh_condition()
    # parse_knn()