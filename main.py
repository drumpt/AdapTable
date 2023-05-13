import os
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import hydra
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import *
from model.mlp import MLP, MLP_MAE
from sam import SAM
from utils import utils
from utils.utils import *

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

def pretrain(args, model, optimizer, dataset, logger):
    device = args.device
    best_model, best_loss = None, float('inf')
    mse_loss_fn = nn.MSELoss(reduction='none')
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train().to(device)
        for train_x, _ in dataset.train_loader:
            train_x = train_x.to(device)
            train_cor_x, _ = get_corrupted_data(train_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")

            train_cor_x = torch.Tensor(train_cor_x).to(args.device)
            estimated_x = model(train_cor_x) if isinstance(model, MLP) else model(train_cor_x)[0]
            loss = mse_loss_fn(estimated_x, train_x).mean()

            # MAE reconstruction loss as cross-entropy loss for categorical variables
            # mse_loss, ce_loss, prev_cum_cat_dim = 0, 0, 0
            # if not len(dataset.dataset.cat_dim_list) or not dataset.dataset.cont_dim:
            #     categorical_weight = 0.5
            # else:
            #     categorical_weight = args.mae_cat_weight

            # mse_loss += mse_loss_fn(
            #     estimated_x[:, :dataset.dataset.cont_dim],
            #     train_x[:, :dataset.dataset.cont_dim]
            # ).mean()
            # for cum_cat_dim in utils.cumulative_sum(dataset.dataset.cat_dim_list):
            #     ce_loss += ce_loss_fn(
            #         estimated_x[:, dataset.dataset.cont_dim + prev_cum_cat_dim:dataset.dataset.cont_dim + cum_cat_dim],
            #         train_x[:, dataset.dataset.cont_dim + prev_cum_cat_dim:dataset.dataset.cont_dim + cum_cat_dim]
            #     ).mean()
            #     prev_cum_cat_dim = cum_cat_dim
            # loss = 2 * (1 - categorical_weight) * dataset.dataset.cont_dim / (dataset.dataset.cont_dim + len(dataset.dataset.cat_dim_list)) * mse_loss + 2 * categorical_weight * len(dataset.dataset.cat_dim_list) / (dataset.dataset.cont_dim + len(dataset.dataset.cat_dim_list)) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]

        valid_loss, valid_len = 0, 0
        model.eval().to(device)
        with torch.no_grad():
            for valid_x, _ in dataset.valid_loader:
                valid_x = valid_x.to(device)
                valid_cor_x, _ = get_corrupted_data(valid_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")

                valid_cor_x = torch.tensor(valid_cor_x).to(args.device).to(torch.float32)

                estimated_x = model(valid_cor_x) if isinstance(model, MLP) else model(valid_cor_x)[0]
                loss = mse_loss_fn(estimated_x, valid_x).mean()

                # MAE reconstruction loss as cross-entropy loss for categorical variables
                # mse_loss, ce_loss, prev_cum_cat_dim = 0, 0, 0
                # if not len(dataset.dataset.cat_dim_list) or not dataset.dataset.cont_dim:
                #     categorical_weight = 0.5
                # else:
                #     categorical_weight = args.mae_cat_weight
                # mse_loss += mse_loss_fn(
                #     estimated_x[:, :dataset.dataset.cont_dim],
                #     valid_x[:, :dataset.dataset.cont_dim]
                # ).mean()
                # for cum_cat_dim in utils.cumulative_sum(dataset.dataset.cat_dim_list):
                #     ce_loss += ce_loss_fn(
                #         estimated_x[:, dataset.dataset.cont_dim + prev_cum_cat_dim:dataset.dataset.cont_dim + cum_cat_dim],
                #         valid_x[:, dataset.dataset.cont_dim + prev_cum_cat_dim:dataset.dataset.cont_dim + cum_cat_dim]
                #     ).mean()
                #     prev_cum_cat_dim = cum_cat_dim
                # loss = 2 * (1 - categorical_weight) * dataset.dataset.cont_dim / (dataset.dataset.cont_dim + len(dataset.dataset.cat_dim_list)) * mse_loss + 2 * categorical_weight * len(dataset.dataset.cat_dim_list) / (dataset.dataset.cont_dim + len(dataset.dataset.cat_dim_list)) * ce_loss

                valid_loss += loss.item() * valid_cor_x.shape[0]
                valid_len += valid_cor_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(args.out_dir, "best_pretrained_model.pth"))

        logger.info(f"epoch {epoch}, train_loss {train_loss / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}")
    return best_model


def train(args, model, optimizer, dataset, logger):
    device = args.device
    best_model, best_loss = None, float('inf')
    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss() # for regression
    ce_loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        model.train().to(device)
        for train_x, train_y in dataset.train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            estimated_y = model(train_x) if isinstance(model, MLP) else model(train_x)[-1]
            loss = mse_loss_fn(estimated_y, train_y) if regression else ce_loss_fn(estimated_y, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1)).sum().item()
            train_len += train_x.shape[0]

        valid_loss, valid_acc, valid_len = 0, 0, 0
        model.eval().to(device)
        with torch.no_grad():
            for valid_x, valid_y in dataset.valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                estimated_y = model(valid_x) if isinstance(model, MLP) else model(valid_x)[-1]
                loss = mse_loss_fn(estimated_y, valid_y) if regression else ce_loss_fn(estimated_y, valid_y)

                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1)).sum().item()
                valid_len += valid_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

        logger.info(f"epoch {epoch}, train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}")
    return best_model



def joint_train(args, model, optimizer, dataset, logger):
    device = args.device
    best_model, best_loss = None, float('inf')
    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_len = 0, 0
        model.train().to(device)
        for train_x, train_y in dataset.train_loader:
            train_cor_x, _ = get_corrupted_data(train_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
            train_cor_x, train_x, train_y = train_cor_x.to(device), train_x.to(device), train_y.to(device)
            optimizer.zero_grad()

            estimated_x, _ = model(train_cor_x)
            recon_loss = mse_loss_fn(estimated_x, train_x)
            recon_loss.backward()

            _, estimated_y = model(train_x)
            main_loss = mse_loss_fn(estimated_x, train_x) if regression else ce_loss_fn(estimated_y, train_y)
            main_loss.backward()

            optimizer.step()

            train_loss += (recon_loss.item() + main_loss.item()) * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]

        valid_loss, valid_len = 0, 0
        model.eval().to(device)
        with torch.no_grad():
            for valid_x, valid_y in dataset.train_loader:
                valid_cor_x, _ = get_corrupted_data(valid_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
                valid_cor_x, valid_x, valid_y = valid_cor_x.to(device), valid_x.to(device), valid_y.to(device)

                estimated_x, _ = model(valid_cor_x)
                recon_loss = mse_loss_fn(estimated_x, valid_x)

                _, estimated_y = model(valid_x)
                main_loss = mse_loss_fn(estimated_x, valid_x) if regression else ce_loss_fn(estimated_y, valid_y)

                valid_loss += (recon_loss.item() + main_loss.item()) * valid_cor_x.shape[0]
                valid_len += valid_cor_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

        logger.info(f"epoch {epoch}, train_loss {train_loss / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}")
    return best_model


def forward_and_adapt(args, x, model, optimizer):
    global original_best_model, EMA
    optimizer.zero_grad()

    outputs = model(x) if isinstance(model, MLP) else model(x)[-1]

    if 'sar' in args.method:
        entropy_first = softmax_entropy(outputs)
        filter_id = torch.where(entropy_first < 0.4 * np.log(outputs.shape[-1]))
        entropy_first = softmax_entropy(outputs)
        loss = entropy_first.mean()
        loss.backward(retain_graph=True)

        optimizer.first_step()

        new_outputs = model(x) if isinstance(model, MLP) else model(x)[-1]
        entropy_second = softmax_entropy(new_outputs)
        entropy_second = entropy_second[filter_id]
        filter_id = torch.where(entropy_second < 0.4 * np.log(outputs.shape[-1]))
        loss_second = entropy_second[filter_id].mean()
        loss_second.backward(retain_graph=True)

        EMA = 0.9 * EMA + (1 - 0.9) * loss_second.item() if EMA != None else loss_second.item()

        optimizer.second_step()
        return
    if 'em' in args.method:
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
    if 'gem' in args.method:
        e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
        e_loss.backward(retain_graph=True)
    if 'ns' in args.method:
        negative_outputs = outputs.clone()
        negative_loss = 0
        negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
        negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
        if torch.is_tensor(negative_loss):
            negative_loss.backward(retain_graph=True)
    if 'dm' in args.method: # diversity maximization
        mean_probs = torch.mean(outputs, dim=-1, keepdim=True)
        (- args.dm_weight * softmax_entropy(mean_probs / args.temp).mean()).backward(retain_graph=True)
    if 'kld' in args.method:
        original_outputs = original_best_model(x)
        probs = torch.softmax(outputs, dim=-1)
        original_probs = torch.softmax(original_outputs, dim=-1)
        kl_div_loss = F.kl_div(torch.log(probs), original_probs.detach(), reduction="batchmean")
        (args.kld_weight * kl_div_loss).backward(retain_graph=True)

    optimizer.step()


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    if 'mae' in args.method and len(args.method) > 1:
        main_mae_method(args)
    elif 'mae' in args.method:
        main_mae(args)
    else:
        main_em(args)

def main_em(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    global original_best_model
    model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
    optimizer = getattr(torch.optim, args.train_optimizer)(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train_lr)
    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    # use pretrained model
    if os.path.exists(os.path.join(args.out_dir, "best_model.pth")) and not args.retrain:
        best_model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
        best_state_dict = torch.load(os.path.join(args.out_dir, "best_model.pth"))
        best_model.load_state_dict(best_state_dict)
        print(f"load pretrained model!")
    else:
        best_model = train(args, model, optimizer, dataset, logger)

    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0
    original_best_model = deepcopy(best_model)
    best_model.eval().requires_grad_(True).to(device)
    original_best_model.eval().requires_grad_(False).to(device)

    global EMA, ENTROPY_LIST, GRADIENT_NORM_LIST, ENTROPY_LIST_NEW, GRADIENT_NORM_LIST_NEW, PREDICTED_LABEL, BATCH_IDX
    EMA = None
    params, _ = collect_params(best_model, train_params=args.train_params)
    if "sar" in args.method:
        test_optimizer = SAM(params, base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)
    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(best_model, test_optimizer, scheduler=None)

    for test_x, test_mask_x, test_y in dataset.test_loader:
        if args.episodic or (EMA != None and EMA < 0.2):
            best_model, test_optimizer, _ = load_model_and_optimizer(best_model, test_optimizer, None, original_model_state, original_optimizer_state, None)

        test_x, test_y = test_x.to(device), test_y.to(device)
        test_len += test_x.shape[0]

        estimated_y = original_best_model(test_x) if isinstance(original_best_model, MLP) else original_best_model(test_x)[-1]
        loss = loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, test_x, best_model, test_optimizer)

        estimated_y = best_model(test_x) if isinstance(best_model, MLP) else best_model(test_x)[-1]
        loss = loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")


def main_mae(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.disable_logger(args)
    logger = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    if os.path.exists(os.path.join(args.out_dir, "best_model.pth")) and not args.retrain:
        best_model = MLP_MAE(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
        best_state_dict = torch.load(os.path.join(args.out_dir, "best_model.pth"))
        best_model.load_state_dict(best_state_dict)
        print(f"load pretrained model!")
    else:
        model = MLP_MAE(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4, dropout=0)

        # self-supervised learning (masking and reconstruction task)
        optimizer = getattr(torch.optim, args.pretrain_optimizer)(collect_params(model, train_params="all")[0], lr=args.pretrain_lr)
        model = pretrain(args, model, optimizer, dataset, logger)

        # supervised learning (main task)
        optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(model, train_params="downstream")[0], lr=args.train_lr)
        best_model = train(args, model, optimizer, dataset, logger)

        # we can use either joint training
        # optimizer = getattr(torch.optim, args.pretrain_optimizer)(collect_params(model, train_params="all")[0], lr=args.pretrain_lr)
        # best_model = joint_train(args, model, optimizer, dataset, logger)

    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0
    original_best_model = deepcopy(best_model)
    best_model.eval().requires_grad_(True).to(device)
    original_best_model = original_best_model.eval().requires_grad_(False).to(device)

    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # TODO: remove(only for debugging) - decision tree
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    forest = RandomForestRegressor(random_state=args.seed) if regression else RandomForestClassifier(random_state=args.seed)
    forest.fit(dataset.dataset.train_x, dataset.dataset.train_y)
    importances = forest.feature_importances_
    feature_importance = torch.tensor(importances)
    # feature_importance = torch.reciprocal(torch.tensor(importances))
    # feature_importance = feature_importance / torch.sum(feature_importance)

    global EMA
    EMA = None
    params, _ = collect_params(best_model, train_params="pretrain")
    test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)
    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(best_model, test_optimizer, scheduler=None)

    for test_x, test_mask_x, test_y in dataset.test_loader:
        if args.episodic or (EMA != None and EMA < 0.2):
            best_model, test_optimizer, _ = load_model_and_optimizer(best_model, test_optimizer, None, original_model_state, original_optimizer_state, None)
            best_model = best_model.eval().requires_grad_(True).to(device)

        test_x, test_mask_x, test_y = test_x.to(device), test_mask_x.to(device), test_y.to(device)
        test_cor_x, test_cor_mask_x = get_corrupted_data(test_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
        test_cor_x = torch.tensor(test_cor_x).to(torch.float32).to(args.device)
        test_cor_mask_x = torch.tensor(test_cor_mask_x).to(args.device)

        # test_cor_mask_x = get_mask_by_feature_importance(args, test_x, feature_importance).to(test_x.device)
        # test_cor_x = test_cor_mask_x * test_x + (1 - test_cor_mask_x) * torch.FloatTensor(get_imputed_data(test_x, dataset.dataset.train_x, data_type="numerical", imputation_method="emd")).to(test_x.device)

        _, estimated_y = original_best_model(test_x)
        loss = mse_loss_fn(estimated_y, test_y) if regression else ce_loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_len += test_x.shape[0]

        for _ in range(1, args.num_steps + 1):
            test_optimizer.zero_grad()
            test_cor_x, test_cor_mask_x = get_corrupted_data(test_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
            test_cor_x = torch.tensor(test_cor_x).to(torch.float32).to(args.device)
            test_cor_mask_x = torch.tensor(test_cor_mask_x).to(args.device)

            if args.mixup == "feature":
                print(f"feature mixup!")
                encoded_cor_x = best_model.encoder(test_cor_x)
                mixup_x, mixup_y = mixup(encoded_cor_x, test_x, args)

                tot_x = torch.cat((encoded_cor_x, mixup_x))
                tot_y = torch.cat((test_x, mixup_y))
                tmp_test_mask = test_mask_x.repeat(1+args.mixup_scale, 1)

                loss = mse_loss_fn(best_model.recon_head(tot_x) * tmp_test_mask, tot_y * tmp_test_mask)
            elif args.mixup == "input":
                mixup_x, mixup_y = mixup(test_cor_x, test_x, args)

                tot_x = torch.cat((test_cor_x, mixup_x))
                tot_y = torch.cat((test_x, mixup_y))
                tmp_test_mask = test_mask_x.repeat(1+args.mixup_scale, 1)

                loss = mse_loss_fn(best_model(tot_x)[0] * tmp_test_mask, tot_y * tmp_test_mask)
            else:
                estimated_test_x, _ = best_model(test_cor_x)
                loss = mse_loss_fn(estimated_test_x * test_mask_x, test_x * test_mask_x) # l2 loss only on non-missing values                

            # saliency-based masked autoencoder (다시 테스트)
            # test_cor_x.requires_grad = True # for acquisition
            # test_cor_x.grad = None
            # estimated_test_x, _ = best_model(test_cor_x)
            # loss = mse_loss_fn(estimated_test_x * test_mask_x, test_x * test_mask_x) # l2 loss only on non-missing values
            # loss.backward(retain_graph=True)

            # feature_grads = torch.mean(test_cor_x.grad, dim=0)
            # feature_importance = torch.reciprocal(torch.abs(feature_grads))
            # feature_importance = feature_importance / torch.sum(feature_importance)
            # test_cor_mask_x = get_mask_by_feature_importance(args, test_x, feature_importance).to(test_x.device)
            # test_cor_x = test_cor_mask_x * test_x + (1 - test_cor_mask_x) * torch.FloatTensor(get_imputed_data(test_x, dataset.dataset.train_x, data_type="numerical", imputation_method="emd")).to(test_x.device)

            # test_cor_mask_x = get_mask_by_feature_importance(args, test_x, feature_importance).to(test_x.device)
            # test_cor_x = test_cor_mask_x * test_x + (1 - test_cor_mask_x) * torch.FloatTensor(get_imputed_data(test_x, dataset.dataset.train_x, data_type="numerical", imputation_method="emd")).to(test_x.device)

            loss.backward()
            test_optimizer.step()

        # imputation with masked autoencoder
        estimated_x, _ = best_model(test_x)
        test_x = test_x * test_mask_x + estimated_x * (1 - test_mask_x)

        _, estimated_y = best_model(test_x)

        loss = mse_loss_fn(estimated_y, test_y) if regression else ce_loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        # naive input renormalization (not working)
        # test_x[:, :dataset.dataset.cont_dim] = (test_x[:, :dataset.dataset.cont_dim] - torch.mean(test_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True)) / torch.std(test_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True)
        # test_x = torch.nan_to_num(test_x, nan=0)

        # input renormalization with excluding missing ones (not working)
        # column_mean = torch.sum(test_x[:, :dataset.dataset.cont_dim] * test_mask_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True) / torch.sum(test_mask_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True)
        # column_std = torch.sum((test_x[:, :dataset.dataset.cont_dim] - column_mean) ** 2 * test_mask_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True) / (torch.sum(test_mask_x[:, :dataset.dataset.cont_dim], dim=0, keepdim=True) - 1)
        # test_x[:, :dataset.dataset.cont_dim] = (test_x[:, :dataset.dataset.cont_dim] - column_mean) / column_std
        # test_x = torch.nan_to_num(test_x, nan=0)

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")


def main_mae_method(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.disable_logger(args)
    logger = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    if os.path.exists(os.path.join(args.out_dir, "best_model.pth")) and not args.retrain:
        best_model = MLP_MAE(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
        best_state_dict = torch.load(os.path.join(args.out_dir, "best_model.pth"))
        best_model.load_state_dict(best_state_dict)
        print(f"load pretrained model!")
    else:
        model = MLP_MAE(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4, dropout=0)

        # self-supervised learning (masking and reconstruction task)
        optimizer = getattr(torch.optim, args.pretrain_optimizer)(collect_params(model, train_params="all")[0], lr=args.pretrain_lr)
        model = pretrain(args, model, optimizer, dataset, logger)

        # supervised learning (main task)
        optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(model, train_params="downstream")[0], lr=args.train_lr)
        best_model = train(args, model, optimizer, dataset, logger)

    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0
    original_best_model = deepcopy(best_model)
    best_model.eval().requires_grad_(True).to(device)
    original_best_model = original_best_model.eval().requires_grad_(False).to(device)

    params, _ = collect_params(best_model, train_params=args.train_params)
    if "sar" in args.method:
        test_optimizer = SAM(params, base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)
    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(best_model, test_optimizer, scheduler=None)

    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    global EMA
    EMA = None
    params, _ = collect_params(best_model, train_params="pretrain")
    mae_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.pretrain_lr)
    _, mae_original_optimizer_state, _ = copy_model_and_optimizer(best_model, mae_optimizer, scheduler=None)

    for test_x, test_y in dataset.test_loader:
        if args.episodic or (EMA != None and EMA < 0.2):
            best_model, test_optimizer, _ = load_model_and_optimizer(best_model, test_optimizer, None, original_model_state, original_optimizer_state, None)
            best_model, mae_optimizer, _ = load_model_and_optimizer(best_model, mae_optimizer, None, original_model_state, mae_original_optimizer_state, None)
            best_model = best_model.eval().requires_grad_(True).to(device)

        test_cor_x, test_cor_mask_x = get_corrupted_data(test_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
        test_cor_x, test_x, test_mask_x, test_cor_mask_x, test_y = test_cor_x.to(device), test_x.to(device), test_mask_x.to(device), test_cor_mask_x.to(device), test_y.to(device)

        _, estimated_y = original_best_model(test_x)
        loss = mse_loss_fn(estimated_y, test_y) if regresion else ce_loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_len += test_x.shape[0]

        for _ in range(1, args.num_steps + 1):
            mae_optimizer.zero_grad()

            estimated_test_x, _ = best_model(test_cor_x)
            loss = loss_fn(estimated_test_x * test_mask_x, test_x * test_mask_x) # l2 loss on non-missing values only

            loss.backward()
            mae_optimizer.step()

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, test_x, best_model, test_optimizer)

        _, estimated_y = best_model(test_x)

        loss = mse_loss_fn(estimated_y, test_y) if regresion else ce_loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")




if __name__ == "__main__":
    main()