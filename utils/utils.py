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

    method = "_".join(args.method) if isinstance(args.method, omegaconf.listconfig.ListConfig) else args.method

    log_path = os.path.join(args.log_dir, args.log_prefix)
    if not os.path.exists(os.path.join(log_path)):
        os.makedirs(os.path.join(log_path))
    log_path = os.path.join(log_path, f"{args.benchmark}_{args.dataset}_shift_type_{args.shift_type}_shift_severity_{args.shift_severity}_seed_{args.seed}_{method}.txt")

    # time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # if isinstance(args.method, omegaconf.listconfig.ListConfig):
    #     method = "_".join(args.method) if isinstance(args.method, omegaconf.listconfig.ListConfig) else args.method
    # else:
    #     method = args.method
    # log_path = f'{args.benchmark}_{args.dataset}/{method}/{args.model}/shift_type_{args.shift_type}_shift_severity_{args.shift_severity}/'

    # seed and dataset
    # log_path += f'{args.log_prefix}_seed_{args.seed}_dataset_{args.dataset}_testlr_{args.test_lr}_numstep_{args.num_steps}'
    # if float(args.train_ratio) != 1:
    #     log_path += f'_train_ratio_{args.train_ratio}'
    # if args.test_batch_size != 64:
    #     log_path += f'_test_batch_size_{args.test_batch_size}'
    # log_path += '.txt'

    file_handler = logging.FileHandler(log_path)
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
    print(f"parameters to adapt: {names}")
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


def js_divergence(p1, p2):
    total_m = 0.5 * (p1 + p2)
    loss = 0.5 * F.kl_div(torch.log(p1), total_m, reduction="batchmean") + 0.5 * F.kl_div(torch.log(p2), total_m, reduction="batchmean")
    return loss


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


def get_feature_importance(args, dataset, test_data, test_mask, model):
    if 'gradient_mask' in args.method:
        test_data.requires_grad = True
        test_data.grad = None
        estimated_test_x = model.get_recon_out(test_data)
        loss = F.mse_loss(estimated_test_x * test_mask, test_data * test_mask)
        loss.backward(retain_graph=True)
        feature_importance = torch.reciprocal(torch.abs(test_data.grad) + args.delta)
    else:
        feature_importance = torch.ones_like(test_data)
    feature_importance = feature_importance / torch.sum(feature_importance, dim=-1, keepdim=True)
    return feature_importance


def get_mask_by_feature_importance(args, dataset, test_data, importance): # TODO: implement importance-aware masking and per-sample masking
    input_dim = dataset.cont_dim + len(dataset.cat_start_indices)
    len_keep = int(input_dim * (1 - args.test_mask_ratio))
    idx = np.random.randn(test_data.shape[0], input_dim).argsort(axis=1)
    mask = np.take_along_axis(np.concatenate([np.ones((test_data.shape[0], len_keep)), np.zeros((test_data.shape[0], input_dim - len_keep))], axis=1), idx, axis=1)   
    if dataset.cont_dim and len(dataset.cat_indices_groups):
        cont_mask = mask[:, :dataset.cont_dim]
        cat_mask = np.concatenate([np.repeat(mask[:, dataset.cont_dim:][:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(dataset.input_one_hot_encoder.categories_)], axis=1)
        mask = torch.FloatTensor(np.concatenate([cont_mask, cat_mask], axis=-1)).to(test_data.device)
    elif dataset.cont_dim:
        mask = torch.FloatTensor(mask).to(test_data.device)
    else:
        mask = torch.FloatTensor(np.concatenate([np.repeat(cat_mask[:, category_idx][:, None], len(category), axis=1) for category_idx, category in enumerate(dataset.input_one_hot_encoder.categories_)], axis=1)).to(test_data.device)
    return mask


##################################################################
# for visualization
def draw_entropy_distribution(args, entropy_list, title):
    plt.clf()
    plt.hist(entropy_list, bins=50)

    plt.title(title)
    plt.xlim([0, 1])
    plt.xlabel('Entropy')
    plt.ylabel('Number of Instances')
    plt.savefig(f"{args.vis_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_entropy_gradient_plot(args, entropy_list, gradient_list, title):
    plt.clf()
    plt.scatter(entropy_list, gradient_list)
    plt.title(title)

    plt.xlabel('Entropy')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"{args.vis_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_label_distribution_plot(args, label_list, title):
    plt.clf()
    uniq, count = np.unique(label_list, return_counts=True)
    ratio = count / np.sum(count)
    category = list(map(str, list(range(len(uniq)))))
    plt.bar(category, ratio, width=0.5)

    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Ratio')
    plt.savefig(f"{args.vis_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


def draw_tsne(args, feats, cls, title):
    tsne = TSNE(n_components=2, verbose=1, random_state=args.seed)
    feats = np.array(feats)
    z = tsne.fit_transform(feats)

    df = pd.DataFrame()
    df["y"] = cls
    df["d1"] = z[:, 0]
    df["d2"] = z[:, 1]
    plt.clf()
    sns.scatterplot(x="d1", y="d2", hue=df.y.tolist(), palette=sns.color_palette("hls", cls.max() + 1), data=df)

    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks([])
    plt.yticks([])

    plt.title(title)
    plt.savefig(f"{args.vis_dir}/{args.benchmark}_{args.dataset}_{args.shift_type}_{args.shift_severity}_{args.model}_{''.join(args.method)}_{title}.png")


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


def draw_feature_change(feat1, feat2):
    tsne = TSNE(n_components=2, verbose=1, random_state=0)

    if isinstance(feat1, torch.Tensor) or isinstance(feat2, torch.Tensor):
        feat1_np = feat1.detach().cpu().numpy()
        feat2_np = feat2.detach().cpu().numpy()
    else:
        feat1_np = feat1
        feat2_np = feat2


    X = tsne.fit_transform(np.concatenate([feat1_np, feat2_np], axis=0))
    plt.scatter(X[:len(feat1_np), 0], X[:len(feat1_np), 1], color='b')
    plt.scatter(X[len(feat1_np):, 0], X[len(feat1_np):, 1], color='r')
    plt.title('Feature Change')
    plt.legend(['Before', 'After'])
    plt.show()


def draw_input_change(input1, input2):
    if isinstance(input1, torch.Tensor) or isinstance(input2, torch.Tensor):
        input1_np = input1.detach().cpu().numpy()
        input2_np = input2.detach().cpu().numpy()
    else:
        input1_np = input1
        input2_np = input2

    plt.subplot(2, 1, 1)
    plt.gca().set_title('Original input')
    plt.bar(np.arange(len(input1_np[0])), input1_np[0], color='b')
    plt.subplot(2, 1, 2)
    plt.gca().set_title('Changed input')
    plt.bar(np.arange(len(input2_np[0])), input2_np[0], color='r')
    plt.tight_layout()
    plt.show()


def draw_feature_change(feat1, feat2):
    tsne = TSNE(n_components=2, verbose=1, random_state=0)

    if isinstance(feat1, torch.Tensor) or isinstance(feat2, torch.Tensor):
        feat1_np = feat1.detach().cpu().numpy()
        feat2_np = feat2.detach().cpu().numpy()
    else:
        feat1_np = feat1
        feat2_np = feat2


    X = tsne.fit_transform(np.concatenate([feat1_np, feat2_np], axis=0))
    plt.scatter(X[:len(feat1_np), 0], X[:len(feat1_np), 1], color='b')
    plt.scatter(X[len(feat1_np):, 0], X[len(feat1_np):, 1], color='r')
    plt.title('Feature Change')
    plt.legend(['Before', 'After'])
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


def visualize_dataset(X, y): # for scikit-learn benchmark
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=100, edgecolor="k", linewidth=2)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.show()


def save_pickle(saving_object, title,args):
    if not os.path.exists(args.tsne_dir):
        os.makedirs(args.tsne_dir)

    file_name = f'{title}_model{args.model}_dataset{args.dataset}_shift_type{args.shift_type}_method{args.method}.pkl'
    with open(os.path.join(args.tsne_dir, file_name), 'wb') as f:
        pickle.dump(saving_object, f, pickle.HIGHEST_PROTOCOL)