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
from model.model import *
from utils.utils import *
from utils.sam import *


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
        source_model = train(args, pretrained_model, train_optimizer, dataset) # supervised learning (main task)
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(collect_params(init_model, train_params="all")[0], lr=args.train_lr)
        source_model = train(args, init_model, train_optimizer, dataset)
    return source_model


def pretrain(args, model, optimizer, dataset):
    device = args.device
    source_model, best_loss = None, float('inf')
    loss_fn = nn.MSELoss(reduction='none')
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train()
        for train_x, _ in dataset.train_loader:
            train_x = train_x.to(device)
            train_cont_cor_x, _ = Dataset.get_corrupted_data(train_x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
            train_cat_cor_x, _ = Dataset.get_corrupted_data(dataset.input_one_hot_encoder.inverse_transform(train_x[:, dataset.cont_dim:].cpu()), dataset.input_one_hot_encoder.inverse_transform(dataset.train_x[:, dataset.cont_dim:]), data_type="categorical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
            train_cor_x = torch.Tensor(np.concatenate([train_cont_cor_x, dataset.input_one_hot_encoder.transform(train_cat_cor_x)], axis=-1)).to(args.device)

            estimated_x = model.get_recon_out(train_cor_x)
            loss = loss_fn(estimated_x, train_x if not model.use_embedding else model.get_embedding(train_x).detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]

        valid_loss, valid_len = 0, 0
        model.eval()
        with torch.no_grad():
            for valid_x, _ in dataset.valid_loader:
                valid_x = valid_x.to(device)
                valid_cont_cor_x, _ = Dataset.get_corrupted_data(valid_x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
                valid_cat_cor_x, _ = Dataset.get_corrupted_data(dataset.input_one_hot_encoder.inverse_transform(valid_x[:, dataset.cont_dim:].cpu()), dataset.input_one_hot_encoder.inverse_transform(dataset.train_x[:, dataset.cont_dim:]), data_type="categorical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
                valid_cor_x = torch.Tensor(np.concatenate([valid_cont_cor_x, dataset.input_one_hot_encoder.transform(valid_cat_cor_x)], axis=-1)).to(args.device)

                estimated_x = model.get_recon_out(valid_cor_x)
                loss = loss_fn(estimated_x, valid_x if not model.use_embedding else model.get_embedding(valid_x).detach()).mean()

                valid_loss += loss.item() * valid_cor_x.shape[0]
                valid_len += valid_cor_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            source_model = deepcopy(model)
            torch.save(source_model.state_dict(), os.path.join(args.out_dir, "best_pretrained_model.pth"))

        logger.info(f"pretrain epoch {epoch} | train_loss {train_loss / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}")
    return source_model


def train(args, model, optimizer, dataset):
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
        cont_cor_x, _ = Dataset.get_corrupted_data(x[:, :dataset.cont_dim], dataset.train_x[:, :dataset.cont_dim], data_type="numerical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
        cat_cor_x, _ = Dataset.get_corrupted_data(dataset.input_one_hot_encoder.inverse_transform(x[:, dataset.cont_dim:].detach().cpu()), dataset.input_one_hot_encoder.inverse_transform(dataset.train_x[:, dataset.cont_dim:]), data_type="categorical", shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method="emd")
        cor_x = torch.Tensor(np.concatenate([cont_cor_x, dataset.input_one_hot_encoder.transform(cat_cor_x)], axis=-1)).to(args.device)
        feature_importance = get_feature_importance(args, dataset, x, mask, model)
        test_cor_mask_x = get_mask_by_feature_importance(args, x, feature_importance).to(x.device)
        test_cor_x = test_cor_mask_x * x + (1 - test_cor_mask_x) * cor_x

        estimated_test_x = source_model.get_recon_out(test_cor_x)
        loss = F.mse_loss(estimated_test_x, x if not model.use_embedding else model.get_embedding(x).detach()) # TODO: l2 loss only on non-missing values
        # loss = F.mse_loss(estimated_test_x * mask, x if not model.use_embedding else model.get_embedding(x).detach() * mask) # l2 loss only on non-missing values
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
    EMA = None
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
    if "sar" in args.method:
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
    for test_x, test_mask_x, test_y in dataset.test_loader:
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            source_model, test_optimizer, _ = load_model_and_optimizer(source_model, test_optimizer, None, original_model_state, original_optimizer_state, None)

        test_x, test_mask_x, test_y = test_x.to(device), test_mask_x.to(device), test_y.to(device)
        test_len += test_x.shape[0]

        estimated_y = original_source_model(test_x)
        loss = loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        for step_idx in range(1, args.num_steps + 1):
            forward_and_adapt(args, dataset, test_x, test_mask_x, source_model, test_optimizer)

        if "mae" in args.method: # implement imputation with masked autoencoder
            estimated_x = source_model.get_recon_out(test_x)
            # test_x = test_x * test_mask_x + estimated_x * (1 - test_mask_x) # TODO: implement this

        estimated_y = source_model(test_x)
        loss = loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

    logger.info(f"before adaptation | test loss {test_loss_before / test_len:.4f}, test acc {test_acc_before / test_len:.4f}")
    logger.info(f"after adaptation | test loss {test_loss_after / test_len:.4f}, test acc {test_acc_after / test_len:.4f}")



if __name__ == "__main__":
    main()