"""
Run a domain sweep to explore a possible set of OOD values.

Usage:
python experiments/domain_shift.py \
    --experiment assistments \
    --no_tune \
    --model xgb
    
"""
import argparse
import os
from typing import Union, Optional

import pandas as pd
import torch

from tableshift.configs.domain_shift import domain_shift_experiment_configs
from tableshift.core import TabularDataset, DatasetConfig, \
    DomainSplitter, CachedDataset
from tableshift.models.ray_utils import RayExperimentConfig, \
    run_ray_tune_experiment, fetch_postprocessed_results_df, \
    accuracy_metric_name_and_mode_for_model
from tableshift.configs.benchmark_configs import ExperimentConfig
from tableshift.core.utils import make_uid, timestamp_as_int
from tableshift.configs.ray_configs import get_default_ray_tmp_dir, \
    get_default_ray_local_dir


def get_dataset(expt_config: ExperimentConfig,
                dataset_config: DatasetConfig,
                use_cached: bool) -> Union[TabularDataset, CachedDataset]:
    name = expt_config.tabular_dataset_kwargs["name"]
    if use_cached:
        uid = make_uid(name, expt_config.splitter)
        print(f"[INFO] using cached dataset for uid {uid} "
              f"at {dataset_config.cache_dir}")
        dset = CachedDataset(cache_dir=dataset_config.cache_dir, name=name,
                             uid=uid)
        return dset
    else:
        try:
            dset = TabularDataset(
                **expt_config.tabular_dataset_kwargs,
                config=dataset_config,
                splitter=expt_config.splitter,
                grouper=expt_config.grouper,
                preprocessor_config=expt_config.preprocessor_config)
            return dset
        except ValueError as ve:
            # Case: split is too small.
            print(f"[WARNING] error initializing dataset for expt {name} with "
                  f"{expt_config.splitter.domain_split_varname} {ve}")
            return None


def main(experiment: str, cache_dir: str,
         debug: bool,
         no_tune: bool,
         num_samples: int,
         results_dir: str,
         ray_tmp_dir: str,
         ray_local_dir: str,
         search_alg: str,
         model: str,
         max_concurrent_trials=2,
         num_workers=1,
         use_cached: bool = False,
         scheduler=None,
         gpu_per_worker: float = 1.0,
         cpu_per_worker: int = 1,
         time_budget_hrs: float = None):

    start_time = timestamp_as_int()

    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
        num_samples = 1

    if not ray_tmp_dir:
        ray_tmp_dir = get_default_ray_tmp_dir()
    if not ray_local_dir:
        ray_local_dir = get_default_ray_local_dir()

    print(f"DEBUG torch.cuda.is_available(): {torch.cuda.is_available()}")

    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))
    print(f"[INFO] results will be written to {expt_results_dir}")
    if not os.path.exists(expt_results_dir):
        os.makedirs(expt_results_dir)

    # List of dictionaries containing metrics and metadata for each
    # experimental iterate.
    iterates = []

    domain_shift_expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = DatasetConfig(cache_dir=cache_dir)

    ood_values = domain_shift_expt_config.domain_split_ood_values
    if debug:
        # Just test the first ood split values.
        ood_values = [ood_values[0]]

    tabular_dataset_kwargs = domain_shift_expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = experiment

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
            dset = get_dataset(expt_config=expt_config,
                               dataset_config=dataset_config,
                               use_cached=use_cached)
        except Exception as e:
            # This exception is raised e.g. when one or more of the target
            # domains is not cached, for example if it does not contain
            # multiple labels or there was another exception during caching;
            # we gracefully skip it.
            print(
                f"[WARNING] Exception fetching dataset with src={src}, "
                f"tgt={tgt}: {e}; this is probably due to one or more OOD "
                f"values that were excluded from the cache due to data "
                f"issues. Skipping.")
            continue

        uid = make_uid(experiment, expt_config.splitter)

        if use_cached and not dset.is_cached():
            print(
                f"[INFO] skipping dataset {dset.name}; not cached. This may "
                f"not be a problem and be due to an invalid domain split ("
                f"i.e. a domain split with only one target label).")
            continue


        print('#' * 100)
        print(f'training model {model} for experiment uid {uid}')
        print('#' * 100)

        metric_name, mode = accuracy_metric_name_and_mode_for_model(
            model)
        tune_config = RayExperimentConfig(
            max_concurrent_trials=max_concurrent_trials,
            ray_tmp_dir=ray_tmp_dir,
            ray_local_dir=ray_local_dir,
            num_workers=num_workers,
            num_samples=num_samples,
            tune_metric_name=metric_name,
            mode=mode,
            time_budget_hrs=time_budget_hrs,
            search_alg=search_alg,
            scheduler=scheduler,
            cpu_per_worker=cpu_per_worker,
            gpu_per_worker=gpu_per_worker,
        ) if not no_tune else None

        results = run_ray_tune_experiment(dset=dset, model_name=model,
                                          tune_config=tune_config,
                                          debug=debug)

        df = fetch_postprocessed_results_df(results)
        df["estimator"] = model
        df["domain_split_varname"] = \
            expt_config.splitter.domain_split_varname
        df["domain_split_ood_values"] = str(tgt)
        if src is not None:
            df["domain_split_id_values"] = str(src)

        print(df)
        try:
            # Case: We don't want the script to fail just if
            # .get_best_result() fails.
            best_result = results.get_best_result()
            print("Best trial config: {}".format(best_result.config))
            print("Best trial result: {}".format(best_result))
        except:
            pass

        df.to_csv(os.path.join(expt_results_dir,
                               f"tune_results_{uid}_{model}.csv"),
                  index=False)
        iterates.append(df)

    fp = os.path.join(expt_results_dir,
                      f"tune_results_{experiment}_{start_time}_full.csv")
    print(f"[INFO] writing results to {fp}")
    pd.concat(iterates).to_csv(fp, index=False)
    print(f"[INFO] completed domain shift experiment {experiment}!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")

    parser.add_argument("--cpu_per_worker", default=0, type=int,
                        help="Number of CPUs to provide per worker."
                             "If not set, Ray defaults to 1.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")

    parser.add_argument("--gpu_per_worker", default=1.0, type=float,
                        help="GPUs per worker. Use fractional values < 1. "
                             "(e.g. --gpu_per_worker=0.5) in order"
                             "to allow multiple workers to share GPU.")
    parser.add_argument("--model", default="xgb")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of hparam samples to take in tuning "
                             "sweep. Set to -1 and set time_budget_hrs to allow for"
                             "unlimited runs within the specified time budget.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to use.")
    parser.add_argument("--ray_local_dir", default=None, type=str,
                        help="""Set the local_dir argument to ray RunConfig. 
                            This is a local  directory where training results are 
                            saved to. If not specified, the script will first 
                            look for any of the dirs specified in ray_configs.py, 
                            and if none of those exist, it will use the Ray 
                            default.""")
    parser.add_argument("--ray_tmp_dir", default=None, type=str,
                        help="""Set the the root temporary path for ray. This 
                        is a local  directory where training results are 
                        saved to. If not specified, the script will first 
                        look for any of the dirs specified in ray_configs.py, 
                        and if none of those exist, it will use the Ray 
                        default of /tmp/ray. See 
                        https://docs.ray.io/en/latest/ray-core 
                        /configure.html#logging-and-debugging for more 
                        info.""")
    parser.add_argument("--results_dir", default="./domain_shift_results",
                        help="where to write results. CSVs will be written to "
                             "experiment-specific subdirectories within this "
                             "directory.")
    parser.add_argument("--search_alg", default="hyperopt",
                        choices=["hyperopt", "random"],
                        help="Ray search alg to use for hyperparameter tuning.")
    parser.add_argument("--scheduler", choices=(None, "asha", "median"),
                        default="asha",
                        help="Scheduler to use for hyperparameter optimization."
                             "See https://docs.ray.io/en/latest/tune/api_docs/schedulers.html .")
    parser.add_argument("--time_budget_hrs", type=float, default=None,
                        help="Time budget for each model tuning run, in hours. Fractional hours are ok."
                             "If this is set, num_samples has no effect.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--use_cached", action="store_true", default=False,
                        help="Whether to use cached data. If set to True,"
                             "and cached data does not exist, the job will fail.")
    args = parser.parse_args()
    main(**vars(args))
