import os
import logging
import random
from datetime import datetime
from slack_sdk import WebClient

import numpy as np
import torch


def set_seed(seed):
    print(f'setting seed as : {seed}')
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


# def set_seed(seed):
#     random_seed = int(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # torch.use_deterministic_algorithms(True)
#     np.random.seed(random_seed)
#     random.seed(random_seed)
#
#     print('random output')
#     print(random.random())
#     # 8444218515250481
#     # 8444218515250481

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
    file_handler = logging.FileHandler(os.path.join(args.log_dir, f"log_{time_string}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def disable_logger(args):
    for logger_string in ['openml.datasets.dataset', 'root']:
        logger = logging.getLogger(logger_string)
        logger.propagate = False


def send_message(message, token, channel):
    client = WebClient(token=token)
    response = client.chat_postMessage(channel="#"+channel, text=message)


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

    final_data = []
    final_data2 = []
    final_target = []
    final_target2 = []

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