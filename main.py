import os
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
    init_model = get_model(args, dataset) # get initalized model architecture only
    if os.path.exists(os.path.join(args.out_dir, "source_model.pth")) and not args.retrain:
        init_model.load_state_dict(torch.load(os.path.join(args.out_dir, "source_model.pth")))
        source_model = init_model
    elif len(set(args.method).intersection(['mae', 'ttt++'])): # pretrain and train for masked autoencoder
        pretrain_optimizer = getattr(torch.optim, args.pretrain_optimizer)(collect_params(init_model, train_params="pretrain")[0], lr=args.pretrain_lr)
        pretrained_model = pretrain(args, init_model, pretrain_optimizer, dataset) # self-supervised learning (masking and reconstruction task)
        train_optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(pretrained_model, train_params="downstream")[0], lr=args.train_lr)
        source_model = train(args, pretrained_model, train_optimizer, dataset, with_mae=False) # supervised learning (main task)
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(init_model, train_params="all")[0], lr=args.train_lr)
        source_model = train(args, init_model, train_optimizer, dataset)
    return source_model


def pretrain(args, model, optimizer, dataset):
    device = args.device
    loss_fn = partial(cat_aware_recon_loss, model=model)
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train()

        for train_x, _ in chain(dataset.train_loader, dataset.valid_loader):
            train_x = train_x.to(device)

            if len(dataset.cat_indices_groups):
                train_cont_cor_x, _ = Dataset.get_corrupted_data(train_x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
                train_cat_cor_x, _ = Dataset.get_corrupted_data(dataset.input_one_hot_encoder.inverse_transform(train_x[:, dataset.cont_dim:].cpu()), dataset.input_one_hot_encoder.inverse_transform(dataset.train_x[:, dataset.cont_dim:]), data_type="categorical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
                train_cor_x = torch.Tensor(np.concatenate([train_cont_cor_x, dataset.input_one_hot_encoder.transform(train_cat_cor_x)], axis=-1)).to(args.device)
            else:
                train_cont_cor_x, _ = Dataset.get_corrupted_data(train_x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
                train_cor_x = torch.Tensor(train_cont_cor_x).float().to(args.device)
            estimated_x = model.get_recon_out(train_cor_x)
            loss = loss_fn(estimated_x, train_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]
        logger.info(f"pretrain epoch {epoch} | train_loss {train_loss / train_len:.4f}")
    return model


def train(args, model, optimizer, dataset, with_mae=False):
    device = args.device
    source_model, best_loss = None, float('inf')
    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        model = model.train()
        for train_x, train_y in dataset.train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            estimated_y = model(train_x)
            loss = loss_fn(estimated_y, train_y)

            if with_mae:
                do = nn.Dropout(p=args.test_mask_ratio)
                estimated_x = model.get_recon_out(do(train_x))
                loss += 0.1 * cat_aware_recon_loss(estimated_x, train_x, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1)).sum().item()
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
            source_model = deepcopy(model)
            torch.save(source_model.state_dict(), os.path.join(args.out_dir, "source_model.pth"))

        logger.info(f"train epoch {epoch} | train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}")
    return source_model


def forward_and_adapt(args, dataset, x, mask, model, optimizer):
    global EMA, original_source_model, eata_params, ttt_params
    optimizer.zero_grad()
    outputs = model(x)

    if 'mae' in args.method:
        if len(dataset.cat_indices_groups):
            cont_cor_x, _ = Dataset.get_corrupted_data(x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
            cat_cor_x, _ = Dataset.get_corrupted_data(dataset.input_one_hot_encoder.inverse_transform(x[:, dataset.cont_dim:].detach().cpu()), dataset.input_one_hot_encoder.inverse_transform(dataset.train_x[:, dataset.cont_dim:]), data_type="categorical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
            cor_x = torch.Tensor(np.concatenate([cont_cor_x, dataset.input_one_hot_encoder.transform(cat_cor_x)], axis=-1)).to(args.device)
        else:
            cont_cor_x, _ = Dataset.get_corrupted_data(x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.mae_imputation_method)
            cor_x = torch.Tensor(cont_cor_x).to(args.device)
        feature_importance = get_feature_importance(args, dataset, x, mask, model)
        test_cor_mask_x = get_mask_by_feature_importance(args, x, feature_importance).to(x.device).detach()
        test_cor_x = test_cor_mask_x * x + (1 - test_cor_mask_x) * cor_x

        estimated_test_x = source_model.get_recon_out(test_cor_x)

        print(f"estimated_test_x: {estimated_test_x}")
        print(f"estimated_test_x: {test_cor_x}")


        if 'threshold' in args.method:
            grad_list = []
            for idx, test_instance in enumerate(x):
                optimizer.zero_grad()
                outputs = source_model(test_instance.unsqueeze(0))
                recon_out = source_model.get_recon_out(test_instance.unsqueeze(0))
                loss = F.mse_loss(recon_out * mask[idx], test_instance.unsqueeze(0) * mask[idx]).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(np.sum([p.grad.detach().cpu().data.norm(2) ** 2 if p.grad != None else 0 for p in source_model.parameters()]))
                grad_list.append(gradient_norm)
            grad_list = torch.Tensor(grad_list).to(args.device)
            loss_idx = torch.where(grad_list < 1)
            optimizer.zero_grad()

            loss = F.mse_loss(estimated_test_x * mask, x * mask, reduction='none') # l2 loss only on non-missing values
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
        ids2 = torch.where(ids1[0]>-0.1)

        # filtered entropy
        entropys = entropys[filter_ids_1]
        # filtered outputs
        if eata_params['current_model_probs'] is not None:
            cosine_similarities = F.cosine_similarity(eata_params['current_model_probs'].unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < args.eata_d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(eata_params['current_model_probs'], outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(eata_params['current_model_probs'], outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - args.eata_e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        if x[ids1][ids2].size(0) != 0:
            loss.backward(retain_graph=True)

        # eata param update
        eata_params['current_model_probs'] = updated_probs
    if 'dem' in args.method: # differential entropy minimization
        model.train()
        prediction_list = []
        for _ in range(args.dropout_steps):
            outputs = model(x)
            prediction_list.append(outputs)
        prediction_std = torch.std(torch.cat(prediction_list, dim=-1), dim=-1).mean()
        differential_entropy = - torch.log(2 * np.pi * np.e * prediction_std)
        differential_entropy.backward(retain_graph=True)
        model.eval()
    if 'gem' in args.method: # generalized entropy minimization
        e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
        e_loss.backward(retain_graph=True)
    if 'ns' in args.method: # generalized entropy minimization
        negative_outputs = outputs.clone()
        negative_loss = 0
        negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
        negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
        if torch.is_tensor(negative_loss):
            (args.ns_weight * negative_loss).backward(retain_graph=True)
    if 'dm' in args.method: # diversity maximization
        mean_probs = torch.mean(outputs, dim=-1, keepdim=True)
        (- args.dm_weight * softmax_entropy(mean_probs / args.temp).mean()).backward(retain_graph=True)
    if 'kld' in args.method: # kl-divergence loss
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

    source_model = get_source_model(args, dataset)
    source_model.eval().requires_grad_(True)
    original_source_model = deepcopy(source_model)
    original_source_model.eval().requires_grad_(False)
    params, _ = collect_params(source_model, train_params=args.train_params)
    if 'sam' in args.method or 'sar' in args.method:
        test_optimizer = SAM(params, base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)

    if args.method == 'ttt++': # for TTT++
        from utils.ttt import summarize # offline summarization
        mean, sigma = summarize(args, dataset, source_model)
        # save to global variable
        ttt_params['mean'] = mean
        ttt_params['sigma'] = sigma

    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(source_model, test_optimizer, scheduler=None)
    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0

    source_label_dist = torch.zeros(*(1, dataset.out_dim))
    for _, train_y in dataset.train_loader:
        source_label_dist += torch.sum(train_y, dim=0)
    source_label_dist /= torch.sum(source_label_dist)
    source_label_dist = source_label_dist.to(args.device)

    # target_label_dist = torch.zeros(*(1, dataset.out_dim))
    # for _, _, test_y in dataset.test_loader:
    #     target_label_dist += torch.sum(test_y, dim=0)
    # target_label_dist /= torch.sum(target_label_dist)
    # target_label_dist = target_label_dist.to(args.device)
    target_label_dist = torch.full((1, dataset.out_dim), 1 / dataset.out_dim).to(args.device)
    # target_label_dist = source_label_dist

    if args.vis:
        for train_x, train_y in dataset.train_loader:
            # SOURCE_INPUT_LIST.extend(train_x.tolist())
            SOURCE_INPUT_LIST.extend(original_source_model.get_embedding(train_x.to(args.device)).cpu().tolist())
            SOURCE_FEATURE_LIST.extend(original_source_model.get_feature(train_x.to(args.device)).cpu().tolist())
            SOURCE_ENTROPY_LIST.extend(softmax_entropy(original_source_model(train_x.to(args.device))).tolist())
            SOURCE_LABEL_LIST.extend(torch.argmax(train_y, dim=-1).tolist())

    kl_divergence_dict = defaultdict(int)
    kl_div_loss = nn.KLDivLoss()

    for batch_idx, (test_x, test_mask_x, test_y) in enumerate(dataset.test_loader):
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            source_model, test_optimizer, _ = load_model_and_optimizer(source_model, test_optimizer, None, original_model_state, original_optimizer_state, None)

        test_x, test_mask_x, test_y = test_x.to(device), test_mask_x.to(device), test_y.to(device)
        test_len += test_x.shape[0]

        estimated_y = source_model(test_x)

        gt_target_label_dist = torch.mean(test_y, dim=0)
        gt_target_label_dist = target_label_dist.to(args.device)
        # print(f"gt_target_label_dist: {gt_target_label_dist}")

        original_estimated_target_label_dist = torch.mean(F.softmax(estimated_y, dim=-1), dim=0)
        # print(f"original_estimated_target_label_dist: {original_estimated_target_label_dist}")
        before_div_source = torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True), dim=0)

        # calibrated_probability = (F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True)

        loss = loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        # test_acc_before += (torch.argmax(calibrated_probability, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        # for entropy and mae vs gradient norm visualization
        if args.entropy_gradient_vis:
            for test_instance in test_x:
                test_optimizer.zero_grad()
                outputs = source_model(test_instance.unsqueeze(0))
                if 'mae' in args.method:
                    recon_out = source_model.get_recon_out(test_instance.unsqueeze(0))
                    loss = F.mse_loss(recon_out * test_mask_x, test_instance.unsqueeze(0) * test_mask_x).mean()
                else:
                    loss = softmax_entropy(outputs / args.temp).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(np.sum([p.grad.detach().cpu().data.norm(2) ** 2 if p.grad != None else 0 for p in source_model.parameters()]))
                GRADIENT_NORM_LIST.append(gradient_norm)

        if args.vis:
            ENTROPY_LIST_BEFORE_ADAPTATION.extend(softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1])) # for
            RECON_LOSS_LIST_BEFORE_ADAPTATION.extend(F.mse_loss(source_model.get_recon_out(test_x * test_mask_x), test_x, reduction='none').mean(dim=-1).cpu().tolist())
            FEATURE_LIST.extend(source_model.get_feature(test_x).cpu().tolist())
            LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())
            TARGET_PREDICTION_LIST.extend(torch.argmax(estimated_y, dim=-1).cpu().tolist())

            # print(f"softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1]): {softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1])}")

        for step_idx in range(1, args.num_steps + 1):
            forward_and_adapt(args, dataset, test_x, test_mask_x, source_model, test_optimizer)

        if "mae" in args.method: # implement imputation with masked autoencoder
            estimated_x = source_model.get_recon_out(test_x)
            test_x = test_x * test_mask_x + estimated_x * (1 - test_mask_x)

        # import utils.lame as lame
        # estimated_y_lame = lame.batch_evaluation(args, original_source_model, test_x)
        # print(f"estimated_y_lame: {torch.mean(F.softmax(estimated_y_lame, dim=-1), dim=0)}")

        if 'lame' in args.method:
            import utils.lame as lame
            estimated_y = lame.batch_evaluation(args, source_model, test_x)
        else:
            estimated_y = source_model(test_x)

            ada_pred = torch.mean(F.softmax(estimated_y, dim=-1), dim=0)
            # print(f"mae_prediction: {original_estimated_target_label_dist}")

        after_div_source = torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True), dim=0)

        if args.vis:
            ENTROPY_LIST_AFTER_ADAPTATION.extend(softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1]))
            RECON_LOSS_LIST_AFTER_ADAPTATION.extend(F.mse_loss(source_model.get_recon_out(test_x * test_mask_x), test_x, reduction='none').mean(dim=-1).cpu().tolist())
            FEATURE_LIST.extend(source_model.get_feature(test_x).cpu().tolist())
            LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())
            ENTROPY_LIST_AFTER_ADAPTATION.extend(softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1]))

        # print(f"source_label_dist: {source_label_dist}")
        # print(f"F.softmax(estimated_y, dim=-1) / source_label_dist: {torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True), dim=0)}")

        # print(f"F.softmax(estimated_y, dim=-1): {F.softmax(estimated_y, dim=-1)}")
        # print(f"entropy: {softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1])}")

        # print(f"torch.mean((F.softmax(estimated_y, dim=-1): {torch.mean((F.softmax(estimated_y, dim=-1)), dim=0)}")
        # print(f"torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist)): {torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True), dim=0)}")

        # np_label_list = np.array(LABEL_LIST)
        # target_label_dist = np.sum(np_label_list, axis=0) / np.sum(np_label_list)
        # target_label_dist = torch.from_numpy(target_label_dist).to(args.device)
        # target_label_dist = torch.sum(test_y, dim=0)
        # target_label_dist = target_label_dist / torch.sum(target_label_dist)
        # target_label_dist = target_label_dist.to(args.device)
        # target_label_dist = 0 * target_label_dist + (1 - 0) * torch.mean((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist), dim=-1, keepdim=True), dim=0)
        # target_label_dist = 0.9 * target_label_dist + (1 - 0.9) * torch.mean((F.softmax(estimated_y, dim=-1) / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) / source_label_dist), dim=-1, keepdim=True), dim=0)
        # calibrated_probability = (F.softmax(estimated_y / args.temp, dim=-1) * target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y / args.temp, dim=-1) * target_label_dist / source_label_dist), dim=-1, keepdim=True)
        # calibrated_probability = torch.sqrt(F.softmax(estimated_y, dim=-1) * target_label_dist) / torch.sum(torch.sqrt(F.softmax(estimated_y, dim=-1) * target_label_dist), dim=-1, keepdim=True)
        # calibrated_probability = (F.softmax(estimated_y / 2.5, dim=-1) * target_label_dist / source_label_dist) / (torch.sum(F.softmax(estimated_y / 2.5, dim=-1) * target_label_dist / source_label_dist, dim=-1, keepdim=True))
        # calibrated_probability = (F.softmax(estimated_y, dim=-1) * source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * source_label_dist), dim=-1, keepdim=True)
        # calibrated_probability = (F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist), dim=-1, keepdim=True)
        # calibrated_probability = (F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist), dim=-1, keepdim=True)
        # calibrated_probability = estimated_y / source_label_dist + torch.log(target_label_dist)
        # print(f'after calibration : {calibrated_probability[0]}')
        # print(f"calibrated_probability: {calibrated_probability}")

        # calibrated_probability = (F.softmax(estimated_y, dim=-1) * gt_target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * gt_target_label_dist / source_label_dist), dim=-1, keepdim=True)
        calibrated_probability = (F.softmax(estimated_y, dim=-1) * before_div_source / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * before_div_source / source_label_dist), dim=-1, keepdim=True)

        target_label_dist = (1 - 0.5) * target_label_dist + 0.5 * torch.mean(calibrated_probability, dim=0, keepdim=True)

        cal_use_ratio = F.tanh(kl_div_loss(torch.log(target_label_dist), source_label_dist) * 100)
        # cal_use_ratio = 1 / (1 + 1000 * kl_div_loss(torch.log(target_label_dist), source_label_dist))
        # cal_use_ratio = torch.exp(- 100 * kl_div_loss(torch.log(target_label_dist), source_label_dist))
        # cal_use_ratio = 1 / (1 + torch.exp(kl_div_loss(torch.log(target_label_dist), source_label_dist)))
        # cal_use_ratio = 1
        print(f"cal_use_ratio: {cal_use_ratio}")
        print(f"kl_div: {kl_div_loss(torch.log(target_label_dist), source_label_dist)}")

        calibrated_probability = calibrated_probability * cal_use_ratio + F.softmax(estimated_y, dim=-1) * (1 - cal_use_ratio)

        # calibrated_probability = (F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist) / torch.sum((F.softmax(estimated_y, dim=-1) * target_label_dist / source_label_dist), dim=-1, keepdim=True)

        print(f"torch.mean(calibrated_probability, dim=0, keepdim=True): {torch.mean(calibrated_probability, dim=0, keepdim=True)}")
        print(f"target_label_dist: {target_label_dist}")

        kl_divergence_dict['ori'] += kl_div_loss(torch.log(original_estimated_target_label_dist), gt_target_label_dist)
        kl_divergence_dict['ori_div_src'] += kl_div_loss(torch.log(before_div_source), gt_target_label_dist)
        kl_divergence_dict['ada'] += kl_div_loss(torch.log(ada_pred), gt_target_label_dist)
        kl_divergence_dict['ada_div_src'] += kl_div_loss(torch.log(after_div_source), gt_target_label_dist)
        # kl_divergence_dict['lame'] += kl_div_loss(torch.log(estimated_y_lame), gt_target_label_dist)
        kl_divergence_dict['ma'] = kl_div_loss(torch.log(target_label_dist), gt_target_label_dist)

        loss = loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        # test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        # test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_acc_after += (torch.argmax(calibrated_probability, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        logger.info(f'online batch [{batch_idx}]: acc before {test_acc_before / test_len:.4f}, acc after {test_acc_after / test_len:.4f}')

    print(f"final pseudo target dist: {target_label_dist}")
    print(f"final gt target dist: {gt_target_label_dist}")
    for key, item in kl_divergence_dict.items():
        print(f"{key}: {item / (test_len / args.test_batch_size)}")

    logger.info(f"before adaptation | test loss {test_loss_before / test_len:.4f}, test acc {test_acc_before / test_len:.4f}")
    logger.info(f"after adaptation | test loss {test_loss_after / test_len:.4f}, test acc {test_acc_after / test_len:.4f}")

    if args.vis:
        draw_entropy_distribution(args, ENTROPY_LIST_BEFORE_ADAPTATION, "Entropy Distribution Before Adaptation")
        draw_entropy_distribution(args, ENTROPY_LIST_AFTER_ADAPTATION, "Entropy Distribution After Adaptation")
        draw_entropy_distribution(args, np.array(SOURCE_ENTROPY_LIST), "Source Entropy Distribution")

        draw_label_distribution_plot(args, SOURCE_LABEL_LIST, "Source Label Distribution")
        draw_label_distribution_plot(args, TARGET_PREDICTION_LIST, "Pseudo Label Distribution")
        draw_label_distribution_plot(args, LABEL_LIST, "Target Label Distribution")

        draw_tsne(args, np.array(FEATURE_LIST), np.array(LABEL_LIST), "Latent Space Visualization with t-SNE")
        draw_tsne(args, np.array(SOURCE_FEATURE_LIST), np.array(SOURCE_LABEL_LIST), "Source Latent Space Visualization with t-SNE")
        draw_tsne(args, np.array(SOURCE_INPUT_LIST), np.array(SOURCE_LABEL_LIST), "Source Input Space Visualization with t-SNE")
    if args.entropy_gradient_vis:
        draw_entropy_gradient_plot(args, ENTROPY_LIST_BEFORE_ADAPTATION, GRADIENT_NORM_LIST, "Entropy vs. Gradient Norm")



if __name__ == "__main__":
    main()