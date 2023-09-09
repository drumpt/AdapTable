import os
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
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if isinstance(args.method, omegaconf.listconfig.ListConfig):
        method = "_".join(args.method)
    else:
        method = args.method
    log_path = f'{args.benchmark}_{args.dataset}/{method}/{args.model}/shift_type_{args.shift_type}_shift_severity_{args.shift_severity}/'

    # log path exists
    if not os.path.exists(os.path.join(args.log_dir, log_path)):
        os.makedirs(os.path.join(args.log_dir, log_path))

    # seed and dataset
    log_path += f'{args.log_prefix}_seed_{args.seed}_dataset_{args.dataset}_testlr_{args.test_lr}_numstep_{args.num_steps}'
    if float(args.train_ratio) != 1:
        log_path += f'_train_ratio_{args.train_ratio}'
    if args.test_batch_size != 64:
        log_path += f'_test_batch_size{args.test_batch_size}'
    log_path += '.txt'

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(args.log_dir, log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def disable_logger(args):
    for logger_string in ['openml.datasets.dataset', 'root']:
        logger = logging.getLogger(logger_string)
        logger.propagate = False


def send_slack_message(message, token, channel):
    client = WebClient(token=token)
    response = client.chat_postMessage(channel="#"+channel, text=message)


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
        if 'LN' in train_params:
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
    print(f"names: {names}")
    return params, names


##################################################################
# for loss functions
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


def generate_augmentation(x, args): # MEMO with dropout
    dropout = nn.Dropout(p=0.1)
    x_aug = torch.stack([dropout(x) for _ in range(args.memo_aug_num - 1)]).to(args.device)
    return x_aug


def get_feature_importance(args, dataset, test_data, test_mask, source_model):
    if 'random_mask' in args.method:
        feature_importance = torch.ones_like(test_x[0])
    else:
        test_data.requires_grad = True
        test_data.grad = None
        estimated_test_x = source_model.get_recon_out(test_data)
        loss = F.mse_loss(estimated_test_x * test_mask, test_data * test_mask)
        loss.backward(retain_graph=True)
        feature_grads = torch.mean(test_data.grad, dim=0) # TODO: fix this (use instance-wise gradient)
        feature_importance = torch.reciprocal(torch.abs(feature_grads) + args.delta)
    feature_importance = feature_importance / torch.sum(feature_importance)
    return feature_importance


def get_mask_by_feature_importance(args, test_data, importance):
    mask = torch.ones_like(test_data, dtype=torch.float32)
    selected_rows = np.random.choice(test_data.shape[0], size=int(len(test_data.flatten()) * args.test_mask_ratio))
    selected_columns = np.random.choice(test_data.shape[-1], size=int(len(test_data.flatten()) * args.test_mask_ratio), p=importance.cpu().numpy())
    mask[selected_rows, selected_columns] = 0
    return mask


# def get_embedding_mask(args, mask, model): # TODO: implement this!
#     embedding_mask = []
#     if self.use_embedding:
#         inputs_cont = inputs[:, :self.cat_start_index]
#         inputs_cat = inputs[:, self.cat_start_index:]
#         inputs_cat_emb = []
#         for i, emb_layer in enumerate(self.emb_layers):
#             inputs_cat_emb.append(emb_layer(torch.argmax(inputs_cat[:, self.cat_start_indices[i]:self.cat_end_indices[i]], dim=-1)))
#         inputs_cat = torch.cat(inputs_cat_emb, dim=-1)
#         inputs = torch.cat([inputs_cont, inputs_cat], 1)
#     return mask


##################################################################
# for visualization
def draw_entropy_distribution(args, entropy_list, title):
    plt.clf()
    plt.hist(entropy_list, bins=50)
    plt.title(title)
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Number of Instances')
    plt.savefig(f"{args.img_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_entropy_gradient_plot(args, entropy_list, gradient_list, title):
    plt.clf()
    plt.scatter(entropy_list, gradient_list)
    plt.title(title)
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"{args.img_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_tsne(args, feats, cls, title):
    tsne = TSNE(n_components=2, verbose=1, random_state=args.seed)

    cls = np.array(cls).argmax(1)
    feats = np.array(feats)
    z = tsne.fit_transform(feats)

    df = pd.DataFrame()
    df["y"] = cls
    df["d1"] = z[:, 0]
    df["d2"] = z[:, 1]
    sns.scatterplot(x="d1", y="d2", hue=df.y.tolist(), palette=sns.color_palette("hls", cls.max() + 1), data=df)
    plt.title(title)
    plt.savefig(f"{args.img_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_calibration(args, pred, gt):
    from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    display = CalibrationDisplay.from_predictions(y_true=gt, y_prob=pred, n_bins=10)
    calibration_displays[f'{args.method}'] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    plt.show()


def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return (ece / sum(Bm), acc)


def save_pickle(saving_object, title,args):
    if not os.path.exists(args.tsne_dir):
        os.makedirs(args.tsne_dir)

    file_name = f'{title}_model{args.model}_dataset{args.dataset}_shift_type{args.shift_type}_method{args.method}.pkl'
    with open(os.path.join(args.tsne_dir, file_name), 'wb') as f:
        pickle.dump(saving_object, f, pickle.HIGHEST_PROTOCOL)
##################################################################