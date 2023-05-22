import os
import logging
import random
from datetime import datetime
from copy import deepcopy

import omegaconf.listconfig
from slack_sdk import WebClient
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import pickle


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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_logger(args):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if isinstance(args.method, omegaconf.listconfig.ListConfig):
        method = args.method[0]
    else:
        method = args.method
    log_path = f'{args.meta_dataset}_{args.dataset}/{method}/{args.model}/shift_type_{args.shift_type}_shift_severity_{args.shift_severity}/'

    # log path exists
    if not os.path.exists(os.path.join(args.log_dir, log_path)):
        os.makedirs(os.path.join(args.log_dir, log_path))

    # seed and dataset
    log_path += f'{args.log_prefix}_seed{args.seed}_dataset{args.dataset}_testlr{args.test_lr}_numstep{args.num_steps}'

    if float(args.train_ratio) != 1:
        log_path += f'_train_ratio_{args.train_ratio}'

    if args.test_batch_size != 64:
        log_path += f'_test_batch_size{args.test_batch_size}'

    if args.no_mae_based_imputation:
        log_path += f'_no_mae_based_imputation'
        log_path += f'_imputation_method_{args.imputation_method}'


    # txt addition
    json_path = log_path + '.json'
    log_path += '.txt'

    file_handler = logging.FileHandler(os.path.join(args.log_dir, log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, os.path.join(args.log_dir, json_path)


def disable_logger(args):
    for logger_string in ['openml.datasets.dataset', 'root']:
        logger = logging.getLogger(logger_string)
        logger.propagate = False


def send_message(message, token, channel):
    client = WebClient(token=token)
    response = client.chat_postMessage(channel="#"+channel, text=message)


def copy_model_and_optimizer(model, optimizer, scheduler):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None


def load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state):
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
        if 'all' in train_params:
            for np, p in m.named_parameters():
                p.requires_grad = True
                if not f"{nm}.{np}" in names:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if 'LN' in train_params: # TODO: change this (not working)
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if 'BN' in train_params:
            if isinstance(m, nn.BatchNorm1d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if 'GN' in train_params:
            if isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if "pretrain" in train_params:
            for np, p in m.named_parameters():
                if 'main_head' in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
        if "downstream" in train_params:
            for np, p in m.named_parameters():
                if not 'main_head' in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def safe_log(x, ver):
    if ver == 1:
        return torch.log(x + 1e-5)
    elif ver == 2:
        return torch.log(x + 1e-7)
    elif ver == 3:
        return torch.clamp(torch.log(x), min=-100)
    else:
        raise ValueError("safe_log version is not properly defined !!!")


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def renyi_entropy(x, alpha, dim=-1):
    if alpha == 1:
        return torch.mean(softmax_entropy(x, dim))
    if alpha == 'inf' or alpha == float('inf'):
        entropy, _ = torch.max(x, dim)
        return -torch.mean(torch.log(entropy))
    entropy = torch.log(torch.pow(x.softmax(dim), alpha).sum(dim))
    entropy = entropy / (1 - alpha)
    return torch.mean(entropy)


def softmax_diversity_regularizer(x):
    x2 = x.softmax(-1).mean(0)  # [b, c] -> [c]
    return (x2 * safe_log(x2, ver=3)).sum()


def mixup(data, targets, args, alpha=0.5):
    final_data, final_data2, final_target, final_target2 = [], [], [], []
    for _ in range(args.mixup_scale):
        indices = torch.randperm(data.size(0))
        data2 = data[indices]
        targets2 = targets[indices]

        final_data.append(data)
        final_data2.append(data2)
        final_target.append(targets)
        final_target2.append(targets2)

    final_data = torch.cat(final_data)
    final_data2 = torch.cat(final_data2)
    final_target = torch.cat(final_target)
    final_target2 = torch.cat(final_target2)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(args.device)
    data = final_data * lam + final_data2 * (1 - lam)
    targets = final_target * lam + final_target2 * (1 - lam)

    return data, targets


def draw_tsne(feats, cls, title, args):
    tsne = TSNE(n_components=2, verbose=1, random_state=args.seed)

    cls = np.array(cls).argmax(1)
    feats = np.array(feats)
    z = tsne.fit_transform(feats)

    df = pd.DataFrame()
    df["y"] = cls
    df["d1"] = z[:, 0]
    df["d2"] = z[:, 1]
    sns.scatterplot(x="d1", y="d2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", cls.max() + 1),
                    data=df)
    plt.title(title)
    plt.show()

def save_pickle(saving_object, title,args):
    if not os.path.exists(args.tsne_dir):
        os.makedirs(args.tsne_dir)

    file_name = f'{title}_model{args.model}_dataset{args.dataset}_shift_type{args.shift_type}_method{args.method}.pkl'
    with open(os.path.join(args.tsne_dir, file_name), 'wb') as f:
        pickle.dump(saving_object, f, pickle.HIGHEST_PROTOCOL)
