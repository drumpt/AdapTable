import os
import math
import logging
import random
from datetime import datetime
from copy import deepcopy
import pickle
import omegaconf.listconfig
from slack_sdk import WebClient

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# logging.basicConfig(level = logging.INFO)


def set_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    method = (
        "_".join(args.method)
        if isinstance(args.method, omegaconf.listconfig.ListConfig)
        else args.method
    )
    log_path = os.path.join(args.log_dir, args.log_prefix)
    log_path = os.path.join(
        log_path,
        f"{args.benchmark}_{args.dataset}_shift_type_{args.shift_type}_shift_severity_{args.shift_severity}_{args.model}_{method}_seed_{args.seed}.txt",
    )

    print(f"log_path: {log_path}")
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def disable_logger():
    for logger_string in ["openml.datasets.dataset", "root"]:
        logger = logging.getLogger(logger_string)
        logger.propagate = False


def send_slack_message(message, token, channel):
    client = WebClient(token=token)
    response = client.chat_postMessage(channel="#" + channel, text=message)


##################################################################
# for test-time adaptation
def copy_model_and_optimizer(model, optimizer, scheduler):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None


def load_model_and_optimizer(
    model, optimizer, scheduler, model_state, optimizer_state, scheduler_state
):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else:
        return model, optimizer, None


def collect_params(model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        if "all" in train_params:
            for np, p in m.named_parameters():
                p.requires_grad = True
                if not f"{nm}.{np}" in names:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if "LN" in train_params:
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if "BN" in train_params:
            if isinstance(m, nn.BatchNorm1d):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if "GN" in train_params:
            if isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if "pretrain" in train_params:
            for np, p in m.named_parameters():
                if "main_head" in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
        if "downstream" in train_params:
            for np, p in m.named_parameters():
                if not "main_head" in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names

def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def generate_augmentation(x, args):  # MEMO with dropout
    dropout = nn.Dropout(p=0.1)
    x_aug = torch.stack([dropout(x) for _ in range(args.memo_aug_num - 1)]).to(
        args.device
    )
    return x_aug

