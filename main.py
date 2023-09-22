import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from functools import partial
from itertools import chain
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import hydra
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from data.dataset import *
from model.model import *
from utils.utils import *
from utils.sam import *
from utils.mae_util import *


def get_model(args, dataset):
    model = eval(args.model)(args, dataset)
    model = model.to(args.device)
    return model


def get_source_model(args, dataset):
    init_model = get_model(args, dataset)  # get initalized model architecture only
    print(init_model)
    if isinstance(args.method, str):
        args.method = [args.method]

    if os.path.exists(os.path.join(args.out_dir, "source_model.pth")) and not args.retrain:
        init_model.load_state_dict(torch.load(os.path.join(args.out_dir, "source_model.pth")))
        source_model = init_model
    elif set(args.method).intersection(['mae', 'ttt++']): # pretrain and train for masked autoencoder
        pretrain_optimizer = getattr(torch.optim, args.pretrain_optimizer)(collect_params(init_model, train_params="pretrain")[0], lr=args.pretrain_lr)
        pretrained_model = pretrain(args, init_model, pretrain_optimizer, dataset) # self-supervised learning (masking and reconstruction task)
        train_optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(pretrained_model, train_params="downstream")[0], lr=args.train_lr)
        source_model = train(args, pretrained_model, train_optimizer, dataset, with_mae=False) # supervised learning (main task)
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(init_model, train_params="all")[0], lr=args.train_lr)
        source_model = train(args, init_model, train_optimizer, dataset)
    return source_model


def get_column_shift_handler(args, dataset, source_model):
    column_shift_handler = ColumnShiftHandler(args, dataset).to(args.device)
    column_shift_handler_optimizer = getattr(torch.optim, "AdamW")(collect_params(column_shift_handler, train_params="all")[0], lr=1e-2)
    column_shift_handler = posttrain(args, source_model, column_shift_handler, column_shift_handler_optimizer, dataset)
    return column_shift_handler


def pretrain(args, model, optimizer, dataset):
    device = args.device
    loss_fn = partial(cat_aware_recon_loss, model=model)
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train()
        for train_x, _ in chain(dataset.train_loader, dataset.valid_loader):
            train_x = train_x.to(device)
            train_cor_x, _ = dataset.get_corrupted_data(train_x, dataset.train_x, shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)

            estimated_x = model.get_recon_out(train_cor_x)
            loss = loss_fn(estimated_x, train_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]
        logger.info(f"pretrain epoch {epoch} | train_loss {train_loss / train_len:.4f}")
    return model


def get_xgb_classifier(args, dataset):
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV

    if dataset.regression:
        objective = "reg:linear"
    elif dataset.train_y.argmax(1).max() == 1:
        objective = "binary:logistic"
    else:
        objective = "multi:softprob"
    param_grid = {
        'n_estimators': np.arange(50, 200, 5),
        'learning_rate': np.linspace(0.01, 1, 20),
        'max_depth': np.arange(2, 12, 1),
        'gamma': np.linspace(0, 0.5, 11)
    }
    tree_model = XGBClassifier(objective=objective, random_state=args.seed)
    rs = RandomizedSearchCV(tree_model, param_grid, n_iter=100, cv=5, verbose=1, n_jobs=-1)
    rs.fit(dataset.train_x, dataset.train_y.argmax(1))
    best_params = rs.best_params_
    tree_model = XGBClassifier(**best_params, random_state=args.seed)
    tree_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    return tree_model



def train(args, model, optimizer, dataset, with_mae=False):
    device = args.device
    source_model, best_loss, best_epoch = None, float('inf'), 0
    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        model = model.train()
        for train_x, train_y in dataset.train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            estimated_y = model(train_x)
            loss = loss_fn(estimated_y, train_y)

            # print(f"estimated_y: {estimated_y}")
            # print(f"train_y: {train_y}")

            # contrastive loss
            # train_x_1, _ = dataset.get_corrupted_data(train_x, dataset.train_x, shift_type="random_drop", shift_severity=0.75, imputation_method="mean")
            # train_x_2, _ = dataset.get_corrupted_data(train_x, dataset.train_x, shift_type="random_drop", shift_severity=0.75, imputation_method="mean")
            # train_x = torch.cat([train_x, train_x_1, train_x_2])
            # train_y = torch.cat([train_y, train_y, train_y])
            # estimated_feat = model.get_feature(train_x)
            # loss += 1 * torch.mean(torch.cdist(estimated_feat, estimated_feat, p=2) * (2 * (torch.cdist(train_y, train_y, p=2) == 0).float() - 1))
     
            # infonce_loss_fn = InfoNCE()
            # batch_size, embedding_size = 32, 128
            # query = torch.randn(batch_size, embedding_size)
            # positive_key = torch.randn(batch_size, embedding_size)
            # output = loss(query, positive_key)

            # print(f"torch.cdist(train_y, train_y, p=2).float(): {(torch.cdist(train_y, train_y, p=2) == 0).float()}")
            # print(f"estimated_feat: {estimated_feat}")
            # print(f"torch.cdist(estimated_feat, estimated_feat, p=2): {torch.cdist(estimated_feat, estimated_feat, p=2)}")
            # print(f"torch.cdist(estimated_feat, estimated_feat, p=2).shape: {torch.cdist(estimated_feat, estimated_feat, p=2).shape}")

            if with_mae:
                do = nn.Dropout(p=args.test_mask_ratio)
                estimated_x = model.get_recon_out(do(train_x))
                loss += 0.1 * cat_aware_recon_loss(estimated_x, train_x, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y[:len(estimated_y)], dim=-1)).sum().item()
            # train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1)).sum().item()
            train_len += train_x.shape[0]

        valid_loss, valid_acc, valid_len = 0, 0, 0
        model = model.eval()
        with torch.no_grad():
            for valid_x, valid_y in dataset.valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)

                estimated_y = model(valid_x)
                loss = loss_fn(estimated_y, valid_y)

                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1)).sum().item()
                valid_len += valid_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            source_model = deepcopy(model)
            torch.save(source_model.state_dict(), os.path.join(args.out_dir, "source_model.pth"))

        logger.info(f"train epoch {epoch} | train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}")
    logger.info(f"best epoch {best_epoch} | best_valid_loss {best_loss}")
    return source_model


def posttrain(args, model, column_shift_handler, column_shift_handler_optimizer, dataset):
    device = args.device
    source_model, best_loss, best_acc = None, float('inf'), 0
    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    source_mean_x = torch.zeros(1, dataset.in_dim)
    for train_x, train_y in dataset.train_loader:
        source_mean_x += torch.sum(train_x, dim=0, keepdim=True)
    source_mean_x /= len(dataset.train_x)
    source_mean_x = source_mean_x.to(args.device)

    model = model.train()
    for epoch in range(1, 10 + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        column_shift_handler = column_shift_handler.train()
        for train_x, train_y in dataset.posttrain_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            # print(f"train_x.shape: {train_x.shape}")
            # TODO 1: add column-wise shift-aware component
            # TODO 2: add regularization term
            estimated_y = model(train_x)
            estimated_y = column_shift_handler(train_x, estimated_y)
            loss = loss_fn(estimated_y, train_y)

            column_shift_handler_optimizer.zero_grad()
            loss.backward()
            column_shift_handler_optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1)).sum().item()
            train_len += train_x.shape[0]

        # for train_x, train_mask_x, train_y in dataset.test_loader:
        #     train_x, train_y = train_x.to(device), train_y.to(device)
        #     # print(f"train_y.shape: {train_y.shape}")
        #     # print(f"train_y: {train_y}")

        #     # train_x += 0.1 * torch.randn(train_x.shape, device=train_x.device)
        #     # print(f"train_x.shape: {train_x.shape}")
        #     # print(f"dataset.train_x.shape: {dataset.train_x.shape}")
        #     # train_x, _ = dataset.get_corrupted_data(train_x, dataset.train_x, shift_type="Gaussian", shift_severity=0.5, imputation_method="emd")
        #     # print(f"train_x cor.shape: {train_x.shape}")

        #     # dataset.get_corrupted_data(x, dataset.train_x, shift_type="random_drop", shift_severity=1, imputation_method=args.mae_imputation_method)
        #     # var_x, mean_x = torch.var_mean(train_x, dim=0, keepdim=True)
        #     mean_x = torch.mean(train_x, dim=0, keepdim=True)
        #     estimated_y = model(train_x)
        #     estimated_y = column_shift_handler(mean_x - source_mean_x, model(train_x))
 
        #     loss = loss_fn(estimated_y, train_y)

        #     column_shift_handler_optimizer.zero_grad()
        #     loss.backward()
        #     column_shift_handler_optimizer.step()

        #     train_loss += loss.item() * train_x.shape[0]
        #     train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1)).sum().item()
        #     train_len += train_x.shape[0]

        valid_loss, valid_acc, valid_len = 0, 0, 0
        column_shift_handler = column_shift_handler.eval()
        with torch.no_grad():
            for valid_x, valid_y in dataset.valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)

                estimated_y = model(valid_x)
                estimated_y = column_shift_handler(valid_x, estimated_y)
                loss = loss_fn(estimated_y, valid_y)

                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1)).sum().item()
                valid_len += valid_x.shape[0]

        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        if valid_loss < best_loss:
            best_loss = valid_loss
        # if train_loss < best_loss:
        #     best_loss = train_loss
            source_model = deepcopy(column_shift_handler)
            torch.save(source_model.state_dict(), os.path.join(args.out_dir, "column_shift_handler.pth"))

        logger.info(f"posttrain epoch {epoch} | train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}")
    return source_model



def forward_and_adapt(args, dataset, x, mask, model, optimizer):
    global EMA, original_source_model, eata_params, ttt_params
    optimizer.zero_grad()
    outputs = model(x)

    if 'mae' in args.method:
        cor_x, _ = dataset.get_corrupted_data(x, dataset.train_x, shift_type="random_drop", shift_severity=1, imputation_method=args.mae_imputation_method) # fully corrupted (masking is done below)
        feature_importance = get_feature_importance(args, dataset, x, mask, model)
        test_cor_mask_x = get_mask_by_feature_importance(args, dataset, x, feature_importance).to(x.device).detach()
        test_cor_x = test_cor_mask_x * x + (1 - test_cor_mask_x) * cor_x
        estimated_test_x = source_model.get_recon_out(test_cor_x)

        if 'threshold' in args.method:
            grad_list = []
            for idx, test_instance in enumerate(x):
                optimizer.zero_grad()
                outputs = source_model(test_instance.unsqueeze(0))
                recon_out = source_model.get_recon_out(test_instance.unsqueeze(0))
                loss = F.mse_loss(recon_out * mask[idx], test_instance.unsqueeze(0) * mask[idx]).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(np.sum(
                    [p.grad.detach().cpu().data.norm(2) ** 2 if p.grad != None else 0 for p in
                     source_model.parameters()]))
                grad_list.append(gradient_norm)
            grad_list = torch.tensor(grad_list).to(args.device)
            loss_idx = torch.where(grad_list < 1)
            optimizer.zero_grad()

            loss = F.mse_loss(estimated_test_x * mask, x * mask, reduction='none')  # l2 loss only on non-missing values
            loss = loss[loss_idx].mean()
        elif 'sam' in args.method:
            # if 'threshold' in args.method:
            optimizer.zero_grad()
            loss = F.mse_loss(estimated_test_x * mask, x * mask, reduction='none')
            loss_idx = torch.where(loss < 2 * np.log(outputs.shape[-1]))
            loss = loss[loss_idx].mean()
            loss.backward(retain_graph=True)
            optimizer.first_step()

            new_estimated_x = model.get_recon_out(x)
            loss_second = F.mse_loss(new_estimated_x * mask, x * mask, reduction='none')
            loss_idx = torch.where(loss_second < 2 * np.log(outputs.shape[-1]))
            loss_second = loss_second[loss_idx].mean()
            loss_second.backward(retain_graph=True)
            optimizer.second_step()
            return
        elif 'double_masking' in args.method:
            with torch.no_grad():
                recon_out = model.get_recon_out(x)
                recon_loss = F.mse_loss(recon_out, x, reduction='none').mean(0)
                recon_sorted_idx = torch.argsort(recon_loss, descending=True)
                recon_sorted_idx = recon_sorted_idx[:int(len(recon_sorted_idx) * 0.1)]
            mask = torch.zeros_like(x[0]).to(args.device)
            mask[recon_sorted_idx] = 1

            recon_x = model.get_recon_out(x * mask)  # ones with high reconstruction error
            recon_adverse_x = model.get_recon_out(x * (1 - mask))  # ones with low reconstruction error

            m = F.normalize(recon_x, dim=1) @ F.normalize(recon_adverse_x, dim=1).T
            id = torch.eye(m.shape[0]).to(args.device) - 1

            pos_loss = F.mse_loss(recon_x, recon_adverse_x, reduction='none').mean()
            neg_loss = F.mse_loss(m, id, reduction='none').mean()
            loss = pos_loss + neg_loss
        else:
            from utils.mae_util import cat_aware_recon_loss
            loss = cat_aware_recon_loss(estimated_test_x, x, model)
        loss.backward(retain_graph=True)
    if 'em' in args.method:
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
    if 'memo' in args.method:
        x = generate_augmentation(x, args)
        outputs = model(x)
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
    if 'sar' in args.method:
        entropy_first = softmax_entropy(outputs)
        filter_id = torch.where(entropy_first < 0.4 * np.log(outputs.shape[-1]))
        entropy_first = softmax_entropy(outputs)
        loss = entropy_first.mean()
        loss.backward(retain_graph=True)
        optimizer.first_step()

        new_outputs = model(x)
        entropy_second = softmax_entropy(new_outputs)
        entropy_second = entropy_second[filter_id]
        filter_id = torch.where(entropy_second < 0.4 * np.log(outputs.shape[-1]))
        loss_second = entropy_second[filter_id].mean()
        loss_second.backward(retain_graph=True)
        optimizer.second_step()

        EMA = 0.9 * EMA + (1 - 0.9) * loss_second.item() if EMA != None else loss_second.item()
        return
    if 'pl' in args.method:
        pseudo_label = torch.argmax(outputs, dim=-1)
        loss = F.cross_entropy(outputs, pseudo_label)
        loss.backward(retain_graph=True)

    if 'pl_csh' in args.method:
        global column_shift_handler
        outputs = column_shift_handler(x, outputs)
        pseudo_label = torch.argmax(outputs, dim=-1)
        # outputs = outputs/0.2

        threshold = 0.9
        filter = outputs.softmax(dim=-1).max(dim=-1)[0] > threshold
        # print(f"outputs.softmax(dim=-1).max(dim=-1)[0]: {outputs.softmax(dim=-1).max(dim=-1)[0]}")
        # print(f"outputs[filter]: {outputs[filter].shape}")
        # print(f"pseudo_label[filter]: {pseudo_label[filter].shape}")

        loss = F.cross_entropy(outputs[filter], pseudo_label[filter])
        # loss = softmax_entropy(outputs[filter]).mean()
        # print(f"loss: {loss}")
        loss.backward(retain_graph=True)        


    if 'ttt++' in args.method:
        # getting featuers
        z = model.get_feature(x)
        a = model.get_recon_out(x * mask)

        from utils.ttt import linear_mmd, coral, covariance
        criterion_ssl = nn.MSELoss()
        loss_ssl = criterion_ssl(x, a)

        # feature alignment
        loss_mean = linear_mmd(z.mean(axis=0), ttt_params['mean'])
        loss_coral = coral(covariance(z), ttt_params['sigma'])

        loss = loss_ssl * args.ttt_coef[0] + loss_mean * args.ttt_coef[1] + loss_coral * args.ttt_coef[2]
        loss.backward()
    if 'eata' in args.method:
        # version 1 implementation
        from utils.eata import update_model_probs
        # filter unreliable samples
        entropys = softmax_entropy(outputs / args.temp)
        filter_ids_1 = torch.where(entropys < args.eata_e_margin)

        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)

        # filtered entropy
        entropys = entropys[filter_ids_1]
        # filtered outputs
        if eata_params['current_model_probs'] is not None:
            cosine_similarities = F.cosine_similarity(eata_params['current_model_probs'].unsqueeze(dim=0),
                                                      outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < args.eata_d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(eata_params['current_model_probs'],
                                               outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(eata_params['current_model_probs'], outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - args.eata_e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        if x[ids1][ids2].size(0) != 0:
            loss.backward(retain_graph=True)

        # eata param update
        eata_params['current_model_probs'] = updated_probs
    if 'dem' in args.method:  # differential entropy minimization
        model.train()
        prediction_list = []
        for _ in range(args.dropout_steps):
            outputs = model(x)
            prediction_list.append(outputs)
        prediction_std = torch.std(torch.cat(prediction_list, dim=-1), dim=-1).mean()
        differential_entropy = - torch.log(2 * np.pi * np.e * prediction_std)
        differential_entropy.backward(retain_graph=True)
        model.eval()
    if 'gem' in args.method:  # generalized entropy minimization
        e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
        e_loss.backward(retain_graph=True)
    if 'ns' in args.method:  # generalized entropy minimization
        negative_outputs = outputs.clone()
        negative_loss = 0
        negative_mask = torch.where(
            torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
        negative_loss += torch.mean(
            -torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
        if torch.is_tensor(negative_loss):
            (args.ns_weight * negative_loss).backward(retain_graph=True)
    if 'dm' in args.method:  # diversity maximization
        mean_probs = torch.mean(outputs, dim=-1, keepdim=True)
        (- args.dm_weight * softmax_entropy(mean_probs / args.temp).mean()).backward(retain_graph=True)
    if 'kld' in args.method:  # kl-divergence loss
        original_outputs = original_source_model(x)
        probs = torch.softmax(outputs, dim=-1)
        original_probs = torch.softmax(original_outputs, dim=-1)
        kl_div_loss = F.kl_div(torch.log(probs), original_probs.detach(), reduction="batchmean")
        (args.kld_weight * kl_div_loss).backward(retain_graph=True)
    optimizer.step()


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    global logger, original_source_model, source_model, EMA, eata_params, ttt_params
    EMA, ENTROPY_LIST_BEFORE_ADAPTATION, ENTROPY_LIST_AFTER_ADAPTATION, GRADIENT_NORM_LIST, RECON_LOSS_LIST_BEFORE_ADAPTATION, RECON_LOSS_LIST_AFTER_ADAPTATION, FEATURE_LIST, LABEL_LIST = None, [], [], [], [], [], [], []
    SOURCE_LABEL_LIST, TARGET_PREDICTION_LIST = [], []
    SOURCE_INPUT_LIST, SOURCE_FEATURE_LIST, SOURCE_ENTROPY_LIST = [], [], []

    SOURCE_PREDICTION_LIST, SOURCE_CALIBRATED_PREDICTION_LIST, SOURCE_CALIBRATED_ENTROPY_LIST, SOURCE_ONE_HOT_LABEL_LIST = [], [], [], []
    SOURCE_PROB_LIST, PROB_LIST_BEFORE_ADAPTATION, PROB_LIST_AFTER_ADAPTATION, PROB_LIST_AFTER_CALIBRATION = [], [], [], []
    SOURCE_CALIBRATED_PROB_LIST = []

    eata_params = {'fishers': None, 'current_model_probs': None}
    ttt_params = {'mean': None, 'sigma': None}
    if hasattr(args, 'seed'):
        set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    disable_logger(args)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args, logger)

    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    # entropy_list = []
    # calibration_list = []
    # prediction_list = []

    # tree_model = get_xgb_classifier(args, dataset)
    # print(f"tree_model: {tree_model}")
    # tree_test_acc = 0

    # gt_prob_list = []

    # for test_x, test_mask_x, test_y in dataset.test_loader:
    #     estimated_y = tree_model.predict_proba(test_x)
    #     entropy_list.extend(softmax_entropy(torch.tensor(np.log(estimated_y))).tolist())

    #     gt_prob_list = []
    #     for est_inst_y, gt_class in zip(estimated_y, np.argmax(np.array(test_y), axis=-1)):
    #         gt_prob_list.append(est_inst_y[gt_class])
    #     calibration_list.extend(np.array(gt_prob_list)[np.argmax(estimated_y, axis=-1) == np.argmax(np.array(test_y), axis=-1)].tolist())
    #     prediction_list.extend(np.argmax(test_y, axis=-1)[np.argmax(estimated_y, axis=-1) == np.argmax(np.array(test_y), axis=-1)].tolist())

    #     tree_test_acc += (np.argmax(estimated_y, axis=-1) == np.argmax(np.array(test_y), axis=-1)).sum()
    # print(f"test_acc: {tree_test_acc / len(dataset.test_x)}")    
    # # draw_calibration(args, calibration_list, prediction_list)
    # draw_entropy_distribution(args, entropy_list, "Target Entropy Distribution using XGBoost")

    source_model = get_source_model(args, dataset)
    source_model.eval().requires_grad_(True)
    original_source_model = deepcopy(source_model)
    original_source_model.eval().requires_grad_(False)
    params, _ = collect_params(source_model, train_params=args.train_params)
    if 'sam' in args.method or 'sar' in args.method:
        test_optimizer = SAM(params, base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)

    if args.method == 'ttt++':  # for TTT++
        from utils.ttt import summarize  # offline summarization
        mean, sigma = summarize(args, dataset, source_model)
        # save to global variable
        ttt_params['mean'] = mean
        ttt_params['sigma'] = sigma

    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(source_model, test_optimizer, scheduler=None)
    test_loss_before, test_loss_after, ground_truth_label_list, estimated_before_label_list, estimated_after_label_list = 0, 0, [], [], []

    source_label_dist = F.normalize(torch.FloatTensor(np.unique(np.argmax(dataset.train_y, axis=-1), return_counts=True)[1]), p=1, dim=-1).to(args.device)
    target_label_dist = torch.full((1, dataset.out_dim), 1 / dataset.out_dim).to(args.device)
    # kl_divergence_dict = defaultdict(int) # TODO: remove (only for debugging)
    estimated_prob_list = []

    probs_per_label = defaultdict(list)
    for train_x, train_y in dataset.train_loader:
        train_x, train_y = train_x.to(args.device), train_y.to(args.device)
        estimated_y = source_model(train_x)
        probs = estimated_y.softmax(dim=-1)
        for class_idx in range(estimated_y.shape[-1]):
            probs_per_label[class_idx].extend(probs[:, class_idx].cpu().tolist())
            # print(f"probs[: class_idx].cpu().tolist(): {probs[:, class_idx].cpu().tolist()}")
    for label, probs in probs_per_label.items():
        # print(f"label -- probs: {label} {probs} ")
        draw_histogram(args, probs, f"class {label} probability distribution", "Confidence", "Number of Instances")
    # print(f"probs_per_label: {probs_per_label}")

    # if args.vis:
    for train_x, train_y in dataset.train_loader:
        SOURCE_INPUT_LIST.extend(original_source_model.get_embedding(train_x.to(args.device)).cpu().tolist())
        SOURCE_FEATURE_LIST.extend(original_source_model.get_feature(train_x.to(args.device)).cpu().tolist())
        SOURCE_ENTROPY_LIST.extend(softmax_entropy(original_source_model(train_x.to(args.device))).tolist())
        SOURCE_LABEL_LIST.extend(torch.argmax(train_y, dim=-1).tolist())
        SOURCE_PROB_LIST.extend(original_source_model(train_x.to(args.device)).softmax(dim=-1).max(dim=-1)[0].tolist())

    if 'use_graphnet' in args.method:
        from utils.graph import ColumnwiseGraphNet
        # from utils.graph import ColumnwiseGraphNet, RowwiseGraphNet, ColumnwiseGraphNet_rowfeat
        # gnn, graph_test_input = get_pretrained_graphnet(args, dataset, source_model)
        # graph_class = ColumnwiseGraphNet_rowfeat(args, dataset, source_model)
        graph_class = ColumnwiseGraphNet(args, dataset, source_model)
        gnn = graph_class.train_gnn()
        # gnn.eval().requires_grad_(False)

    if 'label_shift_handler' in args.method:
        global memory_queue_list
        memory_queue_list = []

    if 'column_shift_handler' in args.method:
        global column_shift_handler
        column_shift_handler = get_column_shift_handler(args, dataset, source_model)
        # source_mean_x = torch.zeros(1, dataset.in_dim)
        # for train_x, train_y in dataset.train_loader:
        #     source_mean_x += torch.sum(train_x, dim=0, keepdim=True)
        # source_mean_x /= len(dataset.train_x)
        # source_mean_x = source_mean_x.to(args.device)
        for train_x, train_y in dataset.train_loader:
            train_x, train_y = train_x.to(args.device), train_y.to(args.device)
            estimated_y = source_model(train_x)
            SOURCE_PREDICTION_LIST.extend(estimated_y.tolist())
            # SOURCE_CALIBRATED_PREDICTION_LIST.extend((estimated_y/0.2).tolist())
            # SOURCE_CALIBRATED_ENTROPY_LIST.extend(softmax_entropy(column_shift_handler(train_x, estimated_y)).tolist())
            SOURCE_CALIBRATED_PREDICTION_LIST.extend(column_shift_handler(train_x, estimated_y).softmax(dim=-1).tolist())
            SOURCE_CALIBRATED_ENTROPY_LIST.extend(softmax_entropy(column_shift_handler(train_x, estimated_y)).tolist())
            SOURCE_CALIBRATED_PROB_LIST.extend(column_shift_handler(train_x, estimated_y).softmax(dim=-1).max(dim=-1)[0].tolist())
            SOURCE_ONE_HOT_LABEL_LIST.extend(train_y.cpu().tolist())


    ece_loss_fn = ECELoss()
    # classwise_ece_loss_fn = ClasswiseECELoss()

    # ece_before = ece_score(np.array(SOURCE_PREDICTION_LIST), np.array(SOURCE_ONE_HOT_LABEL_LIST))
    # ece_after = ece_score(np.array(SOURCE_CALIBRATED_PREDICTION_LIST), np.array(SOURCE_ONE_HOT_LABEL_LIST))

    ece_before = ece_loss_fn(torch.tensor(SOURCE_PREDICTION_LIST), torch.tensor(SOURCE_LABEL_LIST)).item()
    ece_after = ece_loss_fn(torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST), torch.tensor(SOURCE_LABEL_LIST)).item()

    # print(f"ece_before: {ece_before}")
    # print(f"ece_after: {ece_after}")

    print(f"np.array(torch.tensor(SOURCE_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1): {np.array(torch.tensor(SOURCE_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1)}")
    print(f"torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1): {torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST).softmax(axis=-1).max(axis=-1)[0]}")
    reliability_plot(args, np.array(torch.tensor(SOURCE_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1), np.array(SOURCE_PREDICTION_LIST).argmax(axis=-1), np.array(SOURCE_LABEL_LIST), "calibration1.png")
    reliability_plot(args, np.array(torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1), np.array(SOURCE_CALIBRATED_PREDICTION_LIST).argmax(axis=-1), np.array(SOURCE_LABEL_LIST), "calibration2.png")

    loss_fn = nn.MSELoss() if dataset.regression else nn.CrossEntropyLoss()

    from sklearn.metrics import confusion_matrix

    for batch_idx, (test_x, test_mask_x, test_y) in enumerate(dataset.test_loader):
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            source_model, test_optimizer, _ = load_model_and_optimizer(source_model, test_optimizer, None,
                                                                       original_model_state, original_optimizer_state,
                                                                       None)
            print('reset model!')
        test_x, test_mask_x, test_y = test_x.to(device), test_mask_x.to(device), test_y.to(device)
        ground_truth_label_list.extend(torch.argmax(test_y, dim=-1).cpu().tolist())

        ori_estimated_y = original_source_model(test_x)
        loss = loss_fn(ori_estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        estimated_before_label_list.extend(torch.argmax(ori_estimated_y, dim=-1).cpu().tolist())
        estimated_prob_list.extend(torch.max(torch.softmax(ori_estimated_y, dim=-1), dim=-1)[0].cpu().tolist())

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, dataset, test_x, test_mask_x, source_model, test_optimizer)

        # threshold = 0.9
        # tree_estimated_y = tree_model.predict_proba(test_x.cpu().numpy())
        # above_threshold = np.max(tree_estimated_y, axis=-1) > threshold
        # tree_estimated_y_sub = tree_estimated_y[above_threshold]
        # print(f"tree_estimated_y: {tree_estimated_y}")
        # print(f"tree_estimated_y_sub: {tree_estimated_y_sub}")
        # print(f"np.max(tree_estimated_y, axis=-1) > threshold: {np.max(tree_estimated_y, axis=-1) > threshold}")
        # print(f"tree pseudo-label quality: {(torch.argmax(torch.tensor(tree_estimated_y_sub).to(args.device), dim=-1) == torch.argmax(test_y[above_threshold], dim=-1)).sum().item() / above_threshold.sum()}")
        # print(f"torch.argmax(torch.tensor(tree_estimated_y_sub).to(args.device), dim=-1): {torch.argmax(torch.tensor(tree_estimated_y_sub).to(args.device), dim=-1)}")
        # print(f"torch.argmax(test_y[above_threshold], dim=-1): {torch.argmax(test_y[above_threshold], dim=-1)}")
        # for _ in range(1, args.num_steps + 1):
        #     estimated_y = source_model(test_x)
        #     estimated_y_sub = estimated_y[torch.tensor(above_threshold)]
        #     loss = loss_fn(estimated_y_sub, torch.argmax(torch.tensor(tree_estimated_y_sub).to(args.device), dim=-1))

        #     test_optimizer.zero_grad()
        #     loss.backward()  
        #     test_optimizer.step()

        if "mae" in args.method:  # implement imputation with masked autoencoder
            estimated_x = source_model.get_recon_out(test_x)
            test_x = test_x * test_mask_x + estimated_x * (1 - test_mask_x)

        if "lame" in args.method:
            import utils.lame as lame
            estimated_y = lame.batch_evaluation(args, source_model, test_x)
        elif 'use_graphnet' in args.method and len(test_x) == args.test_batch_size:
            estimated_y = graph_class.get_gnn_out(test_x, ori_estimated_y)
            print(f"estimated_y dist: {np.unique(torch.argmax(estimated_y, dim=-1).detach().cpu().numpy(), return_counts=True)}")
            print(f"true dist : {np.unique(torch.argmax(test_y, dim=-1).detach().cpu().numpy(), return_counts=True)}")
        elif 'label_shift_gt' in args.method:
            estimated_y = source_model(test_x)
            if 'mae' in args.method:
                do = nn.Dropout(p=0.75)
                imputation_value_list = [-3, 0, 3]

                from utils.mae_util import cat_aware_recon_loss, expand_mask
                estimated_x = source_model.get_recon_out(do(test_x))
                recon_loss = cat_aware_recon_loss(estimated_x, test_x, source_model, reduction='none')
                sorted_recon_loss_idx = torch.argsort(recon_loss, dim=0, descending=True)

                mask = torch.zeros_like(recon_loss)
                mask[sorted_recon_loss_idx[int(len(sorted_recon_loss_idx) * 0.1):]] = 1
                mask = expand_mask(mask, source_model).detach()

                # test sample pulled to source
                imputed_recon = []
                for imputation_value in imputation_value_list:
                    corrected_test = test_x * mask + imputation_value * (1 - mask)
                    imputed_recon_test = source_model.get_recon_out(corrected_test)
                    imputed_recon.append(imputed_recon_test)
                imputed_recon = torch.stack(imputed_recon)
                imputed_recon = torch.mean(imputed_recon, dim=0)

                corrected_test = test_x * mask + imputed_recon * (1 - mask)

                # from utils.utils import draw_input_change
                # draw_input_change(test_x, corrected_test)

                # orig_feat_tsne = source_model.get_feature(test_x).detach().cpu().numpy()
                # imputed_feat_tsne = source_model.get_feature(corrected_test).detach().cpu().numpy()
                # from utils.utils import draw_feature_change
                # draw_feature_change(orig_feat_tsne, imputed_feat_tsne)

                estimated_y = source_model(corrected_test)

            # label_counts = np.unique(np.argmax(dataset.test_y.detach().cpu().numpy(), axis=-1), return_counts=True)[1]
            if args.smote:
                source_label_dist = torch.ones_like(torch.tensor(dataset.train_y[0])).to(args.device)
                source_label_dist = source_label_dist / torch.sum(source_label_dist)
            else:
                source_label_dist = np.unique(np.argmax(dataset.train_y, axis=-1), return_counts=True)[1]
                source_label_dist = torch.from_numpy(source_label_dist / np.sum(source_label_dist)).to(args.device)

            np_label_list = np.array(LABEL_LIST)
            target_label_dist = np.sum(np_label_list, axis=0) / np.sum(np_label_list)
            target_label_dist = torch.from_numpy(target_label_dist).to(args.device)
            # target_label_dist = torch.sum(test_y, dim=0)
            # target_label_dist = target_label_dist / torch.sum(target_label_dist)
            # target_label_dist = target_label_dist.to(args.device)

            # print(f'before calibration : {F.softmax(estimated_y / args.temp, dim=-1)[0]}')
            calibrated_probability = (F.softmax(estimated_y / args.temp,
                                                dim=-1) * target_label_dist / source_label_dist) / torch.sum(
                (F.softmax(estimated_y / args.temp, dim=-1) * target_label_dist / source_label_dist), dim=-1,
                keepdim=True)
            # print(f'after calibration : {calibrated_probability[0]}')
            estimated_y = calibrated_probability
        elif 'mae' in args.method:
            do = nn.Dropout(p=0.75)
            imputation_value_list = [-3, 0, 3]

            from utils.mae_util import cat_aware_recon_loss, expand_mask
            estimated_x = source_model.get_recon_out(do(test_x))
            recon_loss = cat_aware_recon_loss(estimated_x, test_x, source_model, reduction='none')
            sorted_recon_loss_idx = torch.argsort(recon_loss, dim=0, descending=True)

            mask = torch.zeros_like(recon_loss)
            mask[sorted_recon_loss_idx[int(len(sorted_recon_loss_idx) * 0.1):]] = 1
            mask = expand_mask(mask, source_model).detach()

            # test sample pulled to source
            imputed_recon = []
            for imputation_value in imputation_value_list:
                corrected_test = test_x * mask + imputation_value * (1 - mask)
                imputed_recon_test = source_model.get_recon_out(corrected_test)
                imputed_recon.append(imputed_recon_test)
            imputed_recon = torch.stack(imputed_recon)
            imputed_recon = torch.mean(imputed_recon, dim=0)

            corrected_test = test_x * mask + imputed_recon * (1 - mask)

            # from utils.utils import draw_input_change
            # draw_input_change(test_x, corrected_test)

            # orig_feat_tsne = source_model.get_feature(test_x).detach().cpu().numpy()
            # imputed_feat_tsne = source_model.get_feature(corrected_test).detach().cpu().numpy()
            # from utils.utils import draw_feature_change
            # draw_feature_change(orig_feat_tsne, imputed_feat_tsne)

            estimated_y = source_model(corrected_test)
        elif 'label_shift_handler' in args.method:
            estimated_y = source_model(test_x)
            # if 'column_shift_handler' in args.method:
            #     estimated_y = column_shift_handler(test_x, estimated_y)
            # before_div_source = torch.mean(F.normalize((F.softmax(estimated_y, dim=-1)), p=1, dim=-1), dim=0)
            # calibrated_probability = F.normalize((F.softmax(estimated_y, dim=-1) * before_div_source / source_label_dist), p=1, dim=-1)
            # target_label_dist = (1 - 0.5) * target_label_dist + 0.5 * torch.mean(calibrated_probability, dim=0, keepdim=True)
            # calibrated_probability = F.normalize((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist), p=1, dim=-1)
            # cal_use_ratio = F.tanh(F.kl_div(torch.log(target_label_dist), source_label_dist) * 100)
            # estimated_y = torch.log(cal_use_ratio * calibrated_probability + (1 - cal_use_ratio) * F.softmax(estimated_y, dim=-1))
            # calibrated_probability = F.normalize((F.softmax(estimated_y, dim=-1) * before_div_source / source_label_dist), p=1, dim=-1)

            # current_target_label_dist = torch.mean(test_y, dim=0, keepdim=True)
            # calibrated_probability = F.normalize((F.softmax(estimated_y, dim=-1) * current_target_label_dist / source_label_dist), p=1, dim=-1)
            probs = estimated_y.softmax(dim=-1)
            calibrated_probs = torch.zeros_like(probs, device=probs.device)
            for instance_idx in range(probs.shape[0]):
                for class_idx in range(probs.shape[-1]):
                    class_prob_tensor = torch.tensor(probs_per_label[class_idx]).to(args.device).unsqueeze(0)
                    calibrated_probs[instance_idx, class_idx] = (probs[instance_idx, class_idx] >= class_prob_tensor).float().sum().item() / class_prob_tensor.shape[-1]
            # print(f"calibrated_probs: {calibrated_probs}")
            calibrated_probs = torch.mean(F.normalize(calibrated_probs, p=1, dim=-1), dim=0, keepdim=True)
            calibrated_probability = F.normalize((F.softmax(estimated_y, dim=-1) * calibrated_probs), p=1, dim=-1)
            estimated_y = torch.log(calibrated_probability)

            # print(f"test_x: {test_x}")
            # print(f"estimated_x: {estimated_x}")
            # print(f"estimated_y: {torch.argmax(estimated_y, dim=-1)}")
            # print(f"estimated_cor_y: {torch.argmax(estimated_cor_y, dim=-1)}")
            # print(f"same ratio: {(torch.argmax(estimated_y, dim=-1) == torch.argmax(estimated_cor_y, dim=-1)).sum().item() / test_y.shape[0]}")

            # cor_test_x, _ = dataset.get_corrupted_data(test_x, dataset.train_x, shift_type="random_drop", shift_severity=0.05, imputation_method=args.mae_imputation_method) # fully corrupted (masking is done below)
            # cor_test_x = estimated_x
            # estimated_cor_y = source_model(cor_test_x)

            # # TODO: remove (only for debugging)
            # original_estimated_target_label_dist = torch.mean(F.softmax(ori_estimated_y, dim=-1), dim=0)
            # ada_pred = torch.mean(F.softmax(estimated_y, dim=-1), dim=0)
            # before_div_source = F.normalize(torch.mean((F.softmax(ori_estimated_y, dim=-1) / source_label_dist)), p=1, dim=-1)
            # gt_target_label_dist = torch.mean(test_y, dim=0).to(args.device)
            # kl_divergence_dict['ori'] += F.kl_div(torch.log(original_estimated_target_label_dist), gt_target_label_dist)
            # kl_divergence_dict['ori_div_src'] += F.kl_div(torch.log(before_div_source), gt_target_label_dist)
            # kl_divergence_dict['ada'] += F.kl_div(torch.log(ada_pred), gt_target_label_dist)
            # kl_divergence_dict['ada_div_src'] += F.kl_div(torch.log(after_div_source), gt_target_label_dist)
            # kl_divergence_dict['ma'] += F.kl_div(torch.log(target_label_dist), gt_target_label_dist)
        # elif 'column_shift_handler' in args.method:
        #     # mean_x = torch.mean(test_x, dim=0, keepdim=True)
        #     estimated_y = source_model(test_x)
        #     estimated_y = column_shift_handler(test_x, estimated_y)

        #     PROB_LIST_AFTER_CALIBRATION.extend(estimated_y.softmax(dim=-1).max(dim=-1)[0].cpu().detach())

        #     # print(f"estimated_y: {estimated_y}")
        #     # print(f"torch.argmax(estimated_y, dim=-1): {torch.argmax(estimated_y, dim=-1)}")
        # else:
        #     estimated_y = source_model(test_x)

        # shifted_column = dataset.get_shifted_column()
        # print(f"shifted_column:  {shifted_column}")
        # test_x[: shifted_column] = 0
        # print(f"test_x: {test_x}")
        # estimated_y = source_model(test_x)
        # print(f"estimated_y: {estimated_y}")

        # probs = source_model(test_x).softmax(dim=-1)
        # calibrated_probs = torch.zeros_like(probs, device=probs.device)
        # for instance_idx in range(probs.shape[0]):
        #     for class_idx in range(probs.shape[-1]):
        #         class_prob_tensor = torch.tensor(probs_per_label[class_idx]).to(args.device).unsqueeze(0)
        #         calibrated_probs[instance_idx, class_idx] = (probs[instance_idx, class_idx] >= class_prob_tensor).float().sum().item() / class_prob_tensor.shape[-1]
        # print(f"probs: {probs}")
        # # print(f"calibrated_probs: {calibrated_probs}")
        # print(f"F.normalize(calibrated_probs, p=1, dim=-1): {F.normalize(calibrated_probs, p=1, dim=-1)}")
        # calibrated_probs = F.normalize(calibrated_probs, p=1, dim=-1)
        # estimated_y = torch.log(calibrated_probs)

        # after_div_source = torch.mean(F.normalize((F.softmax(estimated_y, dim=-1) / source_label_dist), p=1, dim=-1), dim=0)

        loss = loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        estimated_after_label_list.extend(torch.argmax(estimated_y, dim=-1).cpu().tolist())
        logger.info(f'online batch [{batch_idx}]: cumulative acc before {accuracy_score(ground_truth_label_list, estimated_before_label_list):.4f}, cumulative acc after {accuracy_score(ground_truth_label_list, estimated_after_label_list):.4f}')

        logger.info(f"batch true distribution: {torch.mean(test_y, dim=0)}")
        logger.info(f'online batch [{batch_idx}]: current acc before {accuracy_score(torch.argmax(test_y, dim=-1).cpu().tolist(), torch.argmax(ori_estimated_y, dim=-1).cpu().tolist()):.4f}, current acc after {accuracy_score(torch.argmax(test_y, dim=-1).cpu().tolist(), torch.argmax(estimated_y, dim=-1).cpu().tolist()):.4f}')


        if args.vis: # for entropy and mae vs gradient norm visualization
            ENTROPY_LIST_BEFORE_ADAPTATION.extend(softmax_entropy(ori_estimated_y).tolist() / np.log(ori_estimated_y.shape[-1]))
            RECON_LOSS_LIST_BEFORE_ADAPTATION.extend(F.mse_loss(source_model.get_recon_out(test_x * test_mask_x), test_x, reduction='none').mean(dim=-1).cpu().tolist())
            FEATURE_LIST.extend(source_model.get_feature(test_x).cpu().tolist())
            LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())

            print(f"softmax_entropy(ori_estimated_y).max(dim=-1)[0]: {softmax_entropy(ori_estimated_y).max(dim=-1)[0]}")

            PROB_LIST_BEFORE_ADAPTATION.extend(ori_estimated_y.softmax(dim=-1).max(dim=-1)[0].tolist())
            TARGET_PREDICTION_LIST.extend(torch.argmax(ori_estimated_y, dim=-1).cpu().tolist())

            ENTROPY_LIST_AFTER_ADAPTATION.extend(softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1]))
            RECON_LOSS_LIST_AFTER_ADAPTATION.extend(F.mse_loss(source_model.get_recon_out(test_x * test_mask_x), test_x, reduction='none').mean(dim=-1).cpu().tolist())
            FEATURE_LIST.extend(source_model.get_feature(test_x).cpu().tolist())
            LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())
            PROB_LIST_AFTER_ADAPTATION.extend(estimated_y.softmax(dim=-1).max(dim=-1)[0].tolist())

        if args.entropy_gradient_vis:
            for test_instance in test_x:
                test_optimizer.zero_grad()
                outputs = original_source_model(test_instance.unsqueeze(0))
                loss = softmax_entropy(outputs / args.temp).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(np.sum([p.grad.detach().cpu().data.norm(2) ** 2 if p.grad != None else 0 for p in source_model.parameters()]))
                GRADIENT_NORM_LIST.append(gradient_norm)

    # print(f"final gt source dist: {source_label_dist}")
    # print(f"final gt target dist: {gt_target_label_dist}")
    # print(f"final pseudo target dist: {target_label_dist}")
    # for key, item in kl_divergence_dict.items():
    #     print(f"{key}: {item / (test_len / args.test_batch_size)}")

    # print(f"estimated_prob_list: {estimated_prob_list}")

    confusion_matrix_before = confusion_matrix(ground_truth_label_list, estimated_before_label_list)
    confusion_matrix_after = confusion_matrix(ground_truth_label_list, estimated_after_label_list)

    logger.info(f"before adaptation | loss {test_loss_before / len(ground_truth_label_list):.4f}, acc {accuracy_score(ground_truth_label_list, estimated_before_label_list):.4f}, bacc {balanced_accuracy_score(ground_truth_label_list, estimated_before_label_list):.4f}, macro f1-score {f1_score(ground_truth_label_list, estimated_before_label_list, average='macro'):.4f}")
    logger.info(f"after adaptation | loss {test_loss_before / len(ground_truth_label_list):.4f}, acc {accuracy_score(ground_truth_label_list, estimated_after_label_list):.4f}, bacc {balanced_accuracy_score(ground_truth_label_list, estimated_after_label_list):.4f}, macro f1-score {f1_score(ground_truth_label_list, estimated_after_label_list, average='macro'):.4f}")
    logger.info(f"before adaptation | confusion matrix\n{confusion_matrix_before}")
    logger.info(f"after adaptation | confusion matrix\n{confusion_matrix_after}")

    if args.vis:
        draw_histogram(args, np.array(SOURCE_ENTROPY_LIST), "Source Entropy Distribution", "Entropy", "Number of Instances")
        draw_histogram(args, np.array(SOURCE_PROB_LIST), "Source Confidence Distribution", "Confidence", "Number of Instances")
        if 'column_shift_handler' in args.method:
            draw_histogram(args, np.array(SOURCE_CALIBRATED_ENTROPY_LIST), "Source Entropy Distribution After Calibration", "Entropy", "Number of Instances")
            draw_histogram(args, np.array(SOURCE_CALIBRATED_PROB_LIST), "Source Confidence Distribution After Calibration", "Confidence", "Number of Instances")
            draw_histogram(args, np.array(PROB_LIST_AFTER_CALIBRATION), "Target Confidence Distribution After Calibration", "Confidence", "Number of Instances")

        draw_histogram(args, ENTROPY_LIST_BEFORE_ADAPTATION, "Entropy Distribution Before Adaptation", "Entropy", "Number of Instances")
        draw_histogram(args, ENTROPY_LIST_AFTER_ADAPTATION, "Entropy Distribution After Adaptation", "Entropy", "Number of Instances")
        draw_histogram(args, np.array(PROB_LIST_BEFORE_ADAPTATION), "Target Confidence Distribution Before Adaptation", "Confidence", "Number of Instances")
        draw_histogram(args, np.array(PROB_LIST_AFTER_ADAPTATION), "Target Confidence Distribution After Adaptation", "Confidence", "Number of Instances")

        draw_label_distribution_plot(args, SOURCE_LABEL_LIST, "Source Label Distribution")
        draw_label_distribution_plot(args, LABEL_LIST, "Target Label Distribution")
        draw_label_distribution_plot(args, TARGET_PREDICTION_LIST, "Pseudo Label Distribution")

        draw_tsne(args, np.array(FEATURE_LIST), np.array(LABEL_LIST), "Target Latent Space Visualization with t-SNE")
        draw_tsne(args, np.array(SOURCE_FEATURE_LIST), np.array(SOURCE_LABEL_LIST), "Source Latent Space Visualization with t-SNE")
        draw_tsne(args, np.array(SOURCE_INPUT_LIST), np.array(SOURCE_LABEL_LIST), "Source Input Space Visualization with t-SNE")



    if args.entropy_gradient_vis:
        draw_entropy_gradient_plot(args, ENTROPY_LIST_BEFORE_ADAPTATION, GRADIENT_NORM_LIST, "Entropy vs. Gradient Norm")


if __name__ == "__main__":
    main()