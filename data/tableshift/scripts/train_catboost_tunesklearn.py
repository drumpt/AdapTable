import argparse
import logging
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from ray import tune
from sklearn.metrics import accuracy_score, roc_auc_score, \
    average_precision_score
from tune_sklearn import TuneSearchCV

from tableshift.core import get_dataset
from tableshift.core.utils import timestamp_as_int
from tableshift.models.ray_utils import fetch_postprocessed_results_df


def evaluate(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series,
             split: str) -> dict:
    yhat_hard = model.predict(X)
    yhat_soft = model.predict_proba(X)
    metrics = {}
    metrics[f"{split}_accuracy"] = accuracy_score(y, yhat_hard)
    metrics[f"{split}_auc"] = roc_auc_score(y, yhat_soft)
    metrics[f"{split}_map"] = average_precision_score(y, yhat_soft)

    metrics[f"{split}_num_samples"] = len(y)
    metrics[f"{split}_ymean"] = np.mean(y).item()
    return metrics


def main(experiment: str, cache_dir: str, results_dir: str, num_samples: int,
         use_gpu: bool, use_cached: bool):
    start_time = timestamp_as_int()

    dset = get_dataset(experiment, cache_dir, use_cached=use_cached)
    uid = dset.uid

    X_tr, y_tr, _, _ = dset.get_pandas("train")

    # Since there is no native Ray Trainer for CatBoost, we use tune-sklearn.
    # See https://discuss.ray.io/t/how-can-i-use-catboostclassifier-with-ray-tune/304

    # Same tuning grid as https://arxiv.org/abs/2106.11959,
    # see supplementary section F.4.
    param_dists = {
        "learning_rate": tune.uniform(1e-3, 1.),
        "depth": tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
        "bagging_temperature": tune.uniform(0., 1.),
        "l2_leaf_reg": tune.loguniform(1., 10.),
        "leaf_estimation_iterations": tune.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    }

    clf = CatBoostClassifier(task_type="GPU" if use_gpu else None)
    hyperopt_tune_search = TuneSearchCV(
        clf,
        param_distributions=param_dists,
        n_trials=num_samples,
        # uses Async HyperBand if set to True
        early_stopping=True,
        max_iters=100,
        search_optimization="hyperopt",
        use_gpu=use_gpu
    )

    results = hyperopt_tune_search.fit(X_tr, y_tr)

    print("training completed!")

    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))
    df = fetch_postprocessed_results_df(results)
    model_name = "catboost"
    df["estimator"] = model_name
    df["domain_split_varname"] = dset.domain_split_varname
    df["domain_split_ood_values"] = str(dset.get_domains("ood_test"))
    df["domain_split_id_values"] = str(dset.get_domains("id_test"))

    for split in ('validation', 'id_test', 'ood_test', 'ood_validation'):
        X, y, _, _ = dset.get_pandas(split)
        metrics = evaluate(results.best_estimator, X, y, split)
        print(metrics)

    import ipdb;
    ipdb.set_trace()

    iter_fp = os.path.join(
        expt_results_dir,
        f"tune_results_{experiment}_{start_time}_{uid}_"
        f"{model_name}.csv")
    logging.info(f"writing results for {model_name} to {iter_fp}")
    df.to_csv(iter_fp, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--results_dir", default="./ray_train_results",
                        help="where to write results. CSVs will be written to "
                             "experiment-specific subdirectories within this "
                             "directory.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="whether to use GPU (if available)")
    args = parser.parse_args()
    main(**vars(args))
