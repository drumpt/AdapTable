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
from utils.sam import SAM
from utils import utils
from utils.utils import *


# for TSNE logging purpose
tsne_before_adaptation = [[], []] # feature_list, cls_list
tsne_after_adaptation = [[], []] # feature_list, cls_list
ece_list = [[], []]

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
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train().to(device)
        for train_x, _ in dataset.train_loader:
            train_x = train_x.to(device)
            train_cor_x, _ = get_corrupted_data(train_x, dataset.dataset.train_x, data_type="numerical", shift_type="random_drop", shift_severity=args.mask_ratio, imputation_method="emd")
            train_cor_x = torch.Tensor(train_cor_x).to(args.device)

            estimated_x = model(train_cor_x) if isinstance(model, MLP) else model(train_cor_x)[0]
            loss = mse_loss_fn(estimated_x, train_x).mean()

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
    mse_loss_fn = nn.MSELoss()
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


def forward_and_adapt(args, x, model, optimizer):
    global original_best_model, EMA
    optimizer.zero_grad()
    outputs = model(x) if isinstance(model, MLP) else model(x)[-1]

    print('before adapt, prediction becomes:')
    with torch.no_grad():
        print(model(x).argmax(1))
        confident_samples = torch.argsort(softmax_entropy(outputs / args.temp), dim=-1, descending=False)[:len(x) // 10]
        outputs.argmax(1)
        print()

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
    if 'memo' in args.method:
        from utils.memo_utils import generate_augmentation
        assert args.test_batch_size == 1
        x = generate_augmentation(x, args)
        outputs = model(x) if isinstance(model, MLP) else model(x)[-1]
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
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


    print(loss)
    print('after adapt, prediction becomes:')
    with torch.no_grad():
        print(model(x).argmax(1))
        print()
    optimizer.step()


def main_em(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger, json_path = utils.get_logger(args)
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

    for test_x, _, test_y in dataset.test_loader:
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

        with torch.no_grad():
            ece_list[0] += original_best_model(test_x).softmax(1).cpu().tolist()
            ece_list[1] += test_y.argmax(1).cpu().tolist()

        if args.tsne:
            with torch.no_grad():
                assert regression is False
                feature_original_model = original_best_model.get_feature(test_x).tolist()
                feature_best_model = best_model.get_feature(test_x).tolist()

                tsne_before_adaptation[0] += feature_original_model
                tsne_before_adaptation[1] += test_y.tolist()
                tsne_after_adaptation[0] += feature_best_model
                tsne_after_adaptation[1] += test_y.tolist()

    if args.tsne:
        from utils.utils import draw_tsne, save_pickle
        save_pickle(tsne_before_adaptation, 'before_adaptation', args)
        save_pickle(tsne_after_adaptation, 'after_adaptation', args)
        draw_tsne(tsne_before_adaptation[0], tsne_before_adaptation[1], 'before adaptation', args)
        draw_tsne(tsne_after_adaptation[0], tsne_after_adaptation[1], 'after adaptation', args)

    print('current ece score')
    from utils.utils import ece_score, draw_calibration
    positive_prob = np.array(ece_list[0]).max(axis=-1)
    # draw_calibration(positive_prob, ece_list[1], args)

    val, acc_ece = ece_score(ece_list[0], ece_list[1])

    n, bins, rects = plt.hist(x = np.arange(10)/10, bins=np.arange(11)/10, ec='k')

    # iterate through rectangles, change the height of each
    for idx in range(len(rects)):
        rects[idx].set_height(acc_ece[idx])

    # set the y limit
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], linestyle='dashed')

    plt.show()
    print(acc_ece)
    print(f'value is {val}')


    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")
    json_saving = {
        'test_acc_before': test_acc_before / test_len,
        'test_acc_after': test_acc_after / test_len
    }
    import json
    # Serializing json
    json_object = json.dumps(json_saving, indent=4)
    # Writing to sample.json
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

def main_mae(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.disable_logger(args)
    logger, json_path = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    logger.info(f'lenght of train dataset : {len(dataset.dataset.train_x)}')

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

    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

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

        _, estimated_y = original_best_model(test_x)
        loss = mse_loss_fn(estimated_y, test_y) if regression else ce_loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_len += test_x.shape[0]

        # saliency-based masked autoencoder (다시 테스트)



        if 'random_mask' in args.method:
            feature_importance = torch.ones_like(test_x[0])
        else:

            for _ in range(1, args.cumul_steps + 1):
                test_cor_x, test_cor_mask_x = get_corrupted_data(test_x, dataset.dataset.train_x, data_type="numerical",
                                                                 shift_type="random_drop",
                                                                 shift_severity=args.mask_ratio,
                                                                 imputation_method="emd")
                test_cor_x = torch.tensor(test_cor_x).to(torch.float32).to(args.device)
                test_cor_mask_x = torch.tensor(test_cor_mask_x).to(args.device)
                test_cor_x.requires_grad = True  # for acquisition
                test_cor_x.grad = None


                estimated_test_x, _ = best_model(test_cor_x)
                # loss = mse_loss_fn(estimated_test_x, test_x) # l2 loss only on all values
                loss = mse_loss_fn(estimated_test_x * test_mask_x,
                                   test_x * test_mask_x)  # l2 loss only on non-missing values
                loss = loss / args.cumul_steps
                loss.backward(retain_graph=True)

            feature_grads = torch.mean(test_cor_x.grad, dim=0)

            # feature_importance = torch.abs(feature_grads)
            # print('grad of feature')
            # print(feature_grads)
            # feature_grads = feature_grads()
            # feature_grads = torch.relu(feature_grads)
            # feature_grads = feature_grads - torch.min(feature_grads) + 1e-4
            feature_importance = torch.reciprocal(torch.abs(feature_grads) + torch.tensor(1e-6))


        feature_importance = feature_importance / torch.sum(feature_importance)
        print(feature_importance)

        with torch.no_grad():
            _, estim_y = best_model(test_x)
            acc = (torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
            bef_acc_log = acc / len(test_x)
            print(f'before adaptation, (num_step), acc is : {bef_acc_log}')

        for step_idx in range(1, args.num_steps + 1):
            test_optimizer.zero_grad()
            for _ in range(1, args.cumul_steps + 1):
                test_cor_mask_x = get_mask_by_feature_importance(args, test_x, feature_importance).to(test_x.device)

                test_cor_x = test_cor_mask_x * test_x + (1 - test_cor_mask_x) * torch.FloatTensor(get_imputed_data(test_x, dataset.dataset.train_x, data_type="numerical", imputation_method="emd")).to(test_x.device)
                estimated_test_x, _ = best_model(test_cor_x)

                if args.no_mask:
                    loss = mse_loss_fn(estimated_test_x, test_x)  # l2 loss only on non-missing values

                else:
                    loss = mse_loss_fn(estimated_test_x * test_mask_x, test_x * test_mask_x) # l2 loss only on non-missing values
                loss = loss / args.cumul_steps
                loss.backward()
                print(f'loss at step_idx {step_idx} : {loss}')
            test_optimizer.step()

            with torch.no_grad():
                _, estim_y = best_model(test_x)
                acc = (torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
                test_acc_log = acc / len(test_x)
                print(f'after adaptation, step : ({step_idx}), acc is : {test_acc_log - bef_acc_log}')



            # print(f'test loss is : {loss}')
            # loss.backward()
            # test_optimizer.step()

        if args.tsne:
            with torch.no_grad():
                assert regression is False

                # input-level
                tsne_before_adaptation[0] += test_x.tolist()
                tsne_before_adaptation[1] += test_y.tolist()
                tsne_after_adaptation[0] +=  test_x.tolist()
                tsne_after_adaptation[1] += test_y.tolist()

                # feature_original_model = original_best_model.get_feature(test_x).tolist()
                # feature_best_model = best_model.get_feature(test_x).tolist()
                #
                # tsne_before_adaptation[0] += feature_original_model
                # tsne_before_adaptation[1] += test_y.tolist()
                # tsne_after_adaptation[0] += feature_best_model
                # tsne_after_adaptation[1] += test_y.tolist()


        # imputation with masked autoencoder
        if not args.no_mae_based_imputation:
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

    if args.tsne:
        from utils.utils import draw_tsne
        draw_tsne(tsne_before_adaptation[0], tsne_before_adaptation[1], 'before adaptation', args)
        draw_tsne(tsne_after_adaptation[0], tsne_after_adaptation[1], 'after adaptation', args)

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")

    json_saving = {
        'test_acc_before': test_acc_before / test_len,
        'test_acc_after': test_acc_after / test_len
    }
    import json
    # Writing to sample.json
    with open(json_path, "w") as outfile:
        json.dump(json_saving, outfile)

def main_mae_with_em(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.disable_logger(args)
    logger, json_path = utils.get_logger(args)
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
        loss = mse_loss_fn(estimated_y, test_y) if regression else ce_loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_len += test_x.shape[0]

        for _ in range(1, args.num_steps + 1):
            mae_optimizer.zero_grad()

            estimated_test_x, _ = best_model(test_cor_x)
            loss = mse_loss_fn(estimated_test_x * test_mask_x, test_x * test_mask_x) # l2 loss on non-missing values only

            loss.backward()
            mae_optimizer.step()

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, test_x, best_model, test_optimizer)

        _, estimated_y = best_model(test_x)

        loss = mse_loss_fn(estimated_y, test_y) if regression else ce_loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")

    json_saving = {
        'test_acc_before': test_acc_before / test_len,
        'test_acc_after': test_acc_after / test_len
    }
    import json
    # Serializing json
    json_object = json.dumps(json_saving, indent=4)
    # Writing to sample.json
    with open(json_path, "w") as outfile:
        outfile.write(json_object)


def main_no_adapt(args):
    if hasattr(args, 'seed'):
        utils.set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.disable_logger(args)
    logger, json_path = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset = Dataset(args)
    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0

    regression = True if dataset.out_dim == 1 else False
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()


    if args.model == 'lr':
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression()
        best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        best_model = KNeighborsClassifier(n_neighbors=3)
        best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'xgboost':
        if regression:
            objective = "reg:linear"
        elif dataset.dataset.train_y.argmax(1).max() == 1:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"

        import xgboost as xgb

        if regression:
            best_model = xgb.XGBRegressor(objective=objective, random_state=args.seed)
            best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y)
        else:
            best_model = xgb.XGBClassifier(n_estimators=args.num_estimators, learning_rate=args.test_lr, max_depth=args.max_depth, random_state=args.seed)
            best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))
    elif args.model == 'rf':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if regression:
            best_model = RandomForestRegressor(n_estimators=args.num_estimators, max_depth=args.max_depth, random_state=args.seed)
            best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y)
        else:
            best_model = RandomForestClassifier(n_estimators=args.num_estimators, max_depth=args.max_depth, random_state=args.seed)
            best_model = best_model.fit(dataset.dataset.train_x, dataset.dataset.train_y.argmax(1))

    else:
        raise NotImplementedError

    for test_x, test_mask_x, test_y in dataset.test_loader:
        test_len += test_x.shape[0]
        estimated_y = best_model.predict(test_x)
        test_acc_after += (np.array(estimated_y)==np.argmax(np.array(test_y), axis=-1)).sum()

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")

    json_saving = {
        'test_acc_before': test_acc_after / test_len,
        'test_acc_after': test_acc_after / test_len
    }
    import json
    # Serializing json
    json_object = json.dumps(json_saving, indent=4)
    # Writing to sample.json
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    if 'memo' in args.method:
        args.test_batch_size = 1

    # if 'mae' in args.method and len(args.method) > 1:
    #     main_mae_with_em(args)
    if 'mae' in args.method:
        main_mae(args)
    elif 'no_adapt' in args.method:
        main_no_adapt(args)
    else:
        main_em(args)



if __name__ == "__main__":
    main()