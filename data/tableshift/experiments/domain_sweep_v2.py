import argparse
import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score

from tableshift.configs.domain_shift import domain_shift_experiment_configs
from tableshift.core import TabularDataset, DatasetConfig, DomainSplitter
from tableshift.core.utils import timestamp_as_int
from tableshift.models.training import train
from tableshift.models.utils import get_estimator

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(model: str,
         experiment: str,
         results_dir: str,
         cache_dir: str):
    domain_shift_expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = DatasetConfig(cache_dir=cache_dir)
    start_time = timestamp_as_int()

    iterates = []

    for expt_config in domain_shift_expt_config.as_experiment_config_iterator():

        assert isinstance(expt_config.splitter, DomainSplitter)
        if expt_config.splitter.domain_split_id_values is not None:
            src = expt_config.splitter.domain_split_id_values
        else:
            src = None

        tgt = expt_config.splitter.domain_split_ood_values
        if not isinstance(tgt, tuple) and not isinstance(tgt, list):
            tgt = (tgt,)

        try:
            logging.info(f"fetchin dataset for src {src} tgt {tgt}")
            dset = TabularDataset(
                **expt_config.tabular_dataset_kwargs,
                config=dataset_config,
                splitter=expt_config.splitter,
                grouper=expt_config.grouper,
                preprocessor_config=expt_config.preprocessor_config)

            est = get_estimator(model)
            logging.info(f"training estimator {model} on src {src} tgt {tgt}")
            estimator = train(est, dset)

            X_te_ood, y_te_ood, _, _ = dset.get_pandas("ood_test")
            yhat_te_ood = estimator.predict(X_te_ood)
            acc_ood = accuracy_score(y_true=y_te_ood, y_pred=yhat_te_ood)

            X_te_id, y_te_id, _, _ = dset.get_pandas("id_test")
            yhat_te_id = estimator.predict(X_te_id)
            acc_id = accuracy_score(y_true=y_te_id, y_pred=yhat_te_id)
            iterates.append({"src": src, "tgt": tgt, "experiment": experiment,
                             "id_accuracy": acc_id,
                             "ood_accuracy": acc_ood})
        except Exception as e:
            logging.info(f"exception running experiment for "
                         f"src {src} tgt {tgt}: {e}; skipping")
    print(iterates)
    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))
    if not os.path.exists(expt_results_dir):
        os.makedirs(expt_results_dir)
    fp = os.path.join(expt_results_dir,
                      f"tune_results_{experiment}_{start_time}_full.csv")
    print(f"[INFO] writing results to {fp}")
    pd.DataFrame(iterates).to_csv(fp, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--model", type=str, default="xgb")
    parser.add_argument("--cache_dir", default="tmp")
    parser.add_argument("--results_dir", default="./domain_shift_results",
                        help="where to write results. CSVs will be written to "
                             "experiment-specific subdirectories within this "
                             "directory.")
    args = parser.parse_args()
    main(**vars(args))
