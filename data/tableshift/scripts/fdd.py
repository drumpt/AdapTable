"""
Usage:
python scripts/fdd.py \
    --experiment anes \
    --uid anesdomain_split_varname_VCF0112domain_split_ood_value_3.0
"""
from collections import defaultdict

import torch.cuda
from tqdm import tqdm
import argparse
import logging
from tableshift.core import CachedDataset
import re
from scipy.linalg import sqrtm
from numpy import iscomplexobj, trace

import numpy as np
import pandas as pd

from tableshift.models.training import train
from tableshift.models.default_hparams import get_default_config
from tableshift.models.rtdl import MLPModelWithHook
from tableshift.models.compat import OPTIMIZER_ARGS

from tableshift.notebook_lib import read_tableshift_results, \
    best_results_by_metric

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


# See https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def fdd(id_activations, ood_activations):
    """Compute frechet dataset distance from a set of activations."""
    Sigma_id = np.cov(id_activations, rowvar=False)
    Sigma_ood = np.cov(ood_activations, rowvar=False)

    covmean = sqrtm(Sigma_id.dot(Sigma_ood))

    if iscomplexobj(covmean):
        covmean = covmean.real

    return np.linalg.norm(
        id_activations.mean(axis=0) - ood_activations.mean(axis=0),
        ord=2) ** 2 + \
           trace(Sigma_id + Sigma_ood - 2. * covmean)


def main(experiment, uid, cache_dir, model="mlp",
         tableshift_results_dir="./ray_train_results",
         baseline_results_dir='./domain_shift_results'):

    results = read_tableshift_results(tableshift_results_dir,
                                      baseline_results_dir)
    results = pd.concat(results)
    best_results = best_results_by_metric(results)
    assert experiment in best_results['task'].unique(), \
        f"experiment value {experiment} not in {best_results['task'].unique()}"

    best_results = best_results.query(
        f"estimator=='{model}' and task == '{experiment}'")
    assert len(best_results) == 1  # sanity check for single best result

    logging.info(f"experiment is {experiment}")
    logging.info(f"uid is {uid}")
    logging.info(f"constructing cached dataset from {cache_dir}")

    dset = CachedDataset(cache_dir=cache_dir, name=experiment, uid=uid,
                         skip_per_domain_eval=True)

    config = get_default_config(model, dset)

    best_config = {re.sub(".*/", '', c): best_results.loc[:, c].item()
                   for c in best_results.columns
                   if "config" in c}

    config.update(best_config)

    for k in ("n_epochs", "num_layers", "d_hidden"):
        config[k] = int(config[k])

    logging.info(f"config is {config}")

    estimator = MLPModelWithHook(
        d_in=config["d_in"],
        d_layers=[config["d_hidden"]] * config["num_layers"],
        d_out=1,
        dropouts=config["dropouts"],
        activation=config["activation"],
        **{k: config[k] for k in OPTIMIZER_ARGS})

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    estimator = train(
        estimator, dset,
        device=device,
        config=config)

    split_activations = defaultdict(list)

    for split in ('id_test', 'ood_test'):
        print(split)
        loader = dset.get_dataloader(split)
        for x, y, _, _ in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            activations = estimator.get_activations(x)
            split_activations[split].append(activations.detach().cpu().numpy())

    id_activations = np.row_stack(split_activations['id_test'])
    ood_activations = np.row_stack(split_activations['ood_test'])

    fdd_val = fdd(id_activations, ood_activations)
    print(f"fdd for uid {uid} is {fdd_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--uid", type=str)
    parser.add_argument("--cache_dir", type=str,
                        default="./tmp")
    args = parser.parse_args()
    main(**vars(args))
