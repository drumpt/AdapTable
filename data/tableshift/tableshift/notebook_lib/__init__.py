from typing import List
from dataclasses import dataclass
import glob
import os
import os.path as osp
import re
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from tableshift.models.compat import PYTORCH_MODEL_NAMES, \
    DOMAIN_GENERALIZATION_MODEL_NAMES

group_cols = ['task', 'estimator', 'domain_split_ood_values']

task_test_set_sizes = {
    'acsfoodstamps': {'id_test': 78628, 'ood_test': 48878},
    'acsincome': {'id_test': 158016, 'ood_test': 75911},
    'acspubcov': {'id_test': 500782, 'ood_test': 817877},
    'acsunemployment': {'id_test': 161365, 'ood_test': 163611},
    'anes': {'id_test': 520, 'ood_test': 2772},
    'brfss_blood_pressure': {'id_test': 27052, 'ood_test': 518622},
    'brfss_diabetes': {'id_test': 121154, 'ood_test': 209375},
    'diabetes': {'id_test': 121154, 'ood_test': 209375},
    'heloc': {'id_test': 278, 'ood_test': 6914},
    'mimic_extract_hosp_mort': {'id_test': 890, 'ood_test': 13544},
    'mimic_extract_los_3': {'id_test': 1080, 'ood_test': 11835},
    'nhanes_lead': {'id_test': 1476, 'ood_test': 11466},
    'physionet_los47': {'id_test': 140288, 'ood_test': 134402},
}


@dataclass
class ExperimentInfo:
    uid: str
    taskname_full: str
    taskname_short: str
    results_dir_name: str
    ood_vals: str
    domain_generalization: bool
    title_name: str


# UIDs for the benchmark tasks.
EXPERIMENTS_LIST = [
    ExperimentInfo(
        "acsfoodstampsdomain_split_varname_DIVISIONdomain_split_ood_value_06",
        'acsfoodstamps_region',
        'acsfoodstamps',
        'acsfoodstamps_region',
        "['06']",
        True,
        'Food Stamps'),
    ExperimentInfo(
        'acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_01',
        'acsincome_region',
        'acsincome',
        'acsincome_region',
        "['01']",
        True,
        'Income'),
    ExperimentInfo(
        'acspubcovdomain_split_varname_DISdomain_split_ood_value_1.0',
        'acspubcov_disability',
        'acspubcov',
        'acspubcov',
        "['1.0']",
        False,
        'Public Coverage'),
    ExperimentInfo(
        'acsunemploymentdomain_split_varname_SCHLdomain_split_ood_value_010203040506070809101112131415',
        'acsunemployment_edlvl',
        'acsunemployment',
        'acsunemployment',
        "['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']",
        True,
        'Unemployment'),
    ExperimentInfo(
        'anesdomain_split_varname_VCF0112domain_split_ood_value_3.0',
        'anes_region',
        'anes',
        'anes_region',
        "['3.0']",
        True,
        'Voting'),
    ExperimentInfo(
        'brfss_diabetesdomain_split_varname_PRACE1domain_split_ood_value_23456domain_split_id_values_1',
        'brfss_diabetes_race',
        'brfss_diabetes',
        'brfss_diabetes_race',
        '[2, 3, 4, 5, 6]',
        True,
        'Diabetes'),
    ExperimentInfo(
        'brfss_blood_pressuredomain_split_varname_BMI5CATdomain_split_ood_value_3.04.0',
        'brfss_blood_pressure_bmi',
        'brfss_blood_pressure',
        'brfss_blood_pressure',
        "['3.0', '4.0']",
        False,
        'Hypertension'),
    ExperimentInfo(
        'diabetes_readmissiondomain_split_varname_admission_source_iddomain_split_ood_value_7',
        'diabetes_admsrc',
        'diabetes',
        'diabetes_readmission',
        '(7,)',
        True,
        'Hospital Readmission'),
    ExperimentInfo(
        'helocdomain_split_varname_ExternalRiskEstimateLowdomain_split_ood_value_0',
        'heloc_externalrisk',
        'heloc',
        'heloc',
        '[0]',
        False,
        'HELOC'),
    ExperimentInfo(
        'mimic_extract_los_3domain_split_varname_insurancedomain_split_ood_value_Medicare',
        'mimic_extract_los_3_ins',
        'mimic_extract_los_3',
        'mimic_extract_los_3',
        "['Medicare']",
        True,
        'ICU Length of Stay'),
    ExperimentInfo(
        'mimic_extract_mort_hospdomain_split_varname_insurancedomain_split_ood_value_MedicareMedicaid',
        'mimic_extract_hosp_mort',
        'mimic_extract_hosp_mort',
        'mimic_extract_mort_hosp',
        "['1', '2']",
        True,
        'Hospital Mortality'),
    ExperimentInfo(
        'nhanes_leaddomain_split_varname_INDFMPIRBelowCutoffdomain_split_ood_value_1.0',
        'nhanes_lead_poverty',
        'nhanes_lead',
        'nhanes_lead',
        '[1.0]',
        False,
        'Childhood Lead'),
    ExperimentInfo(
        'physionetICULOSgt47.0',
        'physionet_los47',
        'physionet_los47',
        'physionet',
        "['all_ood_test_subdomains']",
        False,
        'Sepsis'),
]

UIDS_LIST = [e.uid for e in EXPERIMENTS_LIST]


def extract_name_from_uid(uid: str) -> str:
    return re.search('^(\w+)domain_split_varname.*', uid).group(1)


# various plot-related utilities
def values_to_colors(ser: pd.Series) -> Tuple[pd.Series, dict]:
    """Helper function for mapping a categorical variable to colors for matplotlib."""
    color_labels = ser.unique()
    rgb_values = sns.color_palette("colorblind", len(color_labels))
    color_map = dict(zip(color_labels, rgb_values))
    return ser.map(color_map), color_map


# a list of RGB values for deterministic categorical plotting
rgblist = list(sns.color_palette("colorblind", 12))


def best_results_by_metric(
        df,
        metric='validation_accuracy',
        group_cols=group_cols):
    for col in group_cols:
        assert col in df.columns, f"column {col} must be in dataframe."
    df.reset_index(inplace=True, drop=True)
    vals = df.dropna(subset=group_cols+[metric]).groupby(group_cols)[metric].idxmax().tolist()
    df_out = df.loc[vals]
    assert np.all(df_out.groupby(group_cols).size() == 1)
    return df_out


def _get_results_df(expt_files: List[str], model, expt) -> pd.DataFrame:
    files = [x for x in expt_files if re.search(f".*({model})\\.csv$", x)]
    if not len(files):
        return
    most_recent = sorted(files)[-1]
    df = pd.read_csv(most_recent)
    df['task'] = expt.taskname_short
    df['domain_split_ood_values'] = expt.ood_vals
    df['uid'] = expt.uid
    df['taskname_full'] = expt.taskname_full
    df['title_name'] = expt.title_name
    return df


def read_tableshift_results(tableshift_results_dir, baseline_results_dir):
    """Read the results."""
    dfs = []
    for expt in EXPERIMENTS_LIST:
        search_glob = os.path.join(tableshift_results_dir,
                                   expt.results_dir_name, '*', '*.csv')
        expt_files = glob.glob(search_glob)
        for model in PYTORCH_MODEL_NAMES:
            if model in DOMAIN_GENERALIZATION_MODEL_NAMES and not expt.domain_generalization:
                continue
            df = _get_results_df(expt_files, model, expt)
            if df is None:
                print(
                    f"[WARNING] missing results file for expt {expt.taskname_short} "
                    f"model {model} matching {search_glob}; skipping")
                continue

            dfs.append(df)

        # For the baseline methods, there are results from a full domain sweep
        # (with other tasks not in the final benchmark). So, we need to exclude these.
        baseline_search_glob = os.path.join(baseline_results_dir,
                                            expt.taskname_full, '*', '*.csv')
        expt_files = glob.glob(baseline_search_glob)
        if expt.taskname_short != 'physionet_los47':
            domain_shift_str = re.search(f".*(domain_split_varname.*)",
                                         expt.uid).group(1)
        else:
            domain_shift_str = 'physionet_los47'
        expt_files = [f for f in expt_files if domain_shift_str in f]
        assert len(
            expt_files), f"no files for {domain_shift_str} with uid {expt.uid}"
        for model in ('xgb', 'lightgbm'):

            df = _get_results_df(expt_files, model, expt)
            if df is None:
                print(
                    f"[WARNING] missing results file for expt {expt.taskname_short} "
                    f"model {model} matching {baseline_search_glob}; skipping")
                continue
            dfs.append(df)

    return dfs


def read_domain_shift_results(ds_results_dir):
    """Read the baseline results (CPU models: xgb and lightgbm only)."""
    df_list = []

    for shift in os.listdir(ds_results_dir):
        if shift.startswith('.'):
            continue
        runs = os.listdir(osp.join(ds_results_dir, shift))
        most_recent_run = sorted(runs)[-1]

        fileglob = os.path.join(ds_results_dir, shift, most_recent_run,
                                '*_full.csv')
        full_results = glob.glob(fileglob)
        if full_results:
            #         print(f'reading results for shift {shift}')
            filename = full_results[0]
            df = pd.read_csv(filename)
            df['task'] = shift
            df['filename'] = filename
            df_list.append(df)
        else:
            print(f'no results for shift {shift} matching {fileglob}')
    return pd.concat(df_list).reset_index()
