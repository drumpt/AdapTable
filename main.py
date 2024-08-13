import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from data.dataset import *
from model.model import *
from utils.utils import *
from utils.sam import *
from utils.mae_util import *
from utils.calibrator import *


def get_model(args, dataset):
    if args.model == "tabnet":
        model = "TabNet"
    elif args.model == "tabtransformer":
        model = "TabTransformer"
    elif args.model == "mlp":
        model = "MLP"
    elif args.model == "fttransformer":
        model = "FTTransformer"
    elif args.model in ["MLP", "TabNet", "TabTransformer", "FTTransformer", "ResNet", "AutoInt", "NODE"]:
        model = args.model
    else:
        raise NotImplementedError

    model = eval(model)(args, dataset)
    model = model.to(args.device)
    return model


def get_source_model(args, dataset):
    init_model = get_model(args, dataset)  # get initalized model architecture only
    if isinstance(args.method, str):
        args.method = [args.method]

    if (
        os.path.exists(os.path.join(args.out_dir, f"{args.model}_{args.dataset}.pth"))
        and not args.retrain
    ):
        init_model.load_state_dict(
            torch.load(os.path.join(args.out_dir, f"{args.model}_{args.dataset}.pth"))
        )
        source_model = init_model
    elif set(args.method).intersection(
        ["mae", "ttt++"]
    ):  # pretrain and train for masked autoencoder
        pretrain_optimizer = getattr(torch.optim, args.pretrain_optimizer)(
            collect_params(init_model, train_params="pretrain")[0], lr=args.pretrain_lr
        )
        pretrained_model = pretrain(
            args, init_model, pretrain_optimizer, dataset
        )  # self-supervised learning (masking and reconstruction task)
        train_optimizer = getattr(torch.optim, args.train_optimizer)(
            collect_params(pretrained_model, train_params="downstream")[0],
            lr=args.train_lr,
        )
        source_model = train(
            args, pretrained_model, train_optimizer, dataset, with_mae=False
        )  # supervised learning (main task)
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(
            list(init_model.parameters()), lr=args.train_lr
        )
        # train_optimizer = getattr(torch.optim, args.train_optimizer)(
        #     collect_params(init_model, train_params="all")[0], lr=args.train_lr
        # )
        source_model = train(args, init_model, train_optimizer, dataset)
    return source_model


def get_column_distribution_handler(args, dataset, source_model):
    column_distribution_handler = ColumnShiftHandler(args, dataset).to(args.device)
    column_distribution_handler_optimizer = getattr(torch.optim, "AdamW")(
        collect_params(column_distribution_handler, train_params="all")[0],
        lr=args.posttrain_lr,
    )
    column_distribution_handler = posttrain(
        args,
        source_model,
        column_distribution_handler,
        column_distribution_handler_optimizer,
        dataset,
    )
    return column_distribution_handler


def pretrain(args, model, optimizer, dataset):
    device = args.device
    loss_fn = partial(cat_aware_recon_loss, model=model)
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train()
        for train_x, _ in chain(dataset.train_loader, dataset.valid_loader):
            train_x = train_x.to(device)
            train_cor_x, _ = dataset.get_corrupted_data(
                train_x,
                dataset.train_x,
                shift_type="random_drop",
                shift_severity=args.pretrain_mask_ratio,
                imputation_method=args.mae_imputation_method,
            )

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
        "n_estimators": np.arange(50, 200, 5),
        "learning_rate": np.linspace(0.01, 1, 20),
        "max_depth": np.arange(2, 12, 1),
        "gamma": np.linspace(0, 0.5, 11),
    }
    tree_model = XGBClassifier(objective=objective, random_state=args.seed)
    rs = RandomizedSearchCV(
        tree_model, param_grid, n_iter=100, cv=5, verbose=1, n_jobs=-1
    )
    rs.fit(dataset.train_x, dataset.train_y.argmax(1))
    best_params = rs.best_params_
    tree_model = XGBClassifier(**best_params, random_state=args.seed)
    tree_model.fit(dataset.train_x, dataset.train_y.argmax(1))
    return tree_model


def train(args, model, optimizer, dataset, with_mae=False):
    global TRAIN_GRADIENT_NORM_LIST, TRAIN_SMOOTHNESS_LIST
    TRAIN_GRADIENT_NORM_LIST, TRAIN_SMOOTHNESS_LIST = [], []
    device = args.device
    source_model, best_loss, best_epoch = None, float("inf"), 0
    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    patience = args.train_patience

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        # model = model.train().requires_grad_(True)
        model = model.train()
        for i, (train_x, train_y) in enumerate(dataset.train_loader):
            train_x, train_y = train_x.to(device), train_y.to(device).float()
            estimated_y = model(train_x)
            if regression:
                loss = loss_fn(estimated_y.squeeze(), train_y.squeeze().float())
            else:
                loss = loss_fn(estimated_y, train_y.argmax(1))

            if with_mae:
                do = nn.Dropout(p=args.test_mask_ratio)
                estimated_x = model.get_recon_out(do(train_x))
                loss += 0.1 * cat_aware_recon_loss(estimated_x, train_x, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (
                (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1))
                .sum()
                .item()
            )
            train_len += train_x.shape[0]

        valid_loss, valid_acc, valid_len = 0, 0, 0
        model = model.eval()
        with torch.no_grad():
            for valid_x, valid_y in dataset.valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)

                estimated_y = model(valid_x)
                if regression:
                    loss = loss_fn(estimated_y.squeeze(), valid_y.squeeze().float())
                else:
                    loss = loss_fn(estimated_y, valid_y.argmax(1))
                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (
                    (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1))
                    .sum()
                    .item()
                )
                valid_len += valid_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            patience = args.train_patience
            source_model = deepcopy(model)
            torch.save(
                source_model.state_dict(),
                os.path.join(args.out_dir, f"{args.model}_{args.dataset}.pth"),
            )
            dataset.best_valid_acc = valid_acc / valid_len
        else:
            patience -= 1
            if patience == 0:
                break

        logger.info(
            f"train epoch {epoch} | train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}"
        )
    logger.info(f"best epoch {best_epoch} | best_valid_loss {best_loss}")
    return source_model


def posttrain(
    args,
    model,
    column_distribution_handler,
    column_distribution_handler_optimizer,
    dataset,
):
    device = args.device
    source_handler, best_loss = None, float("inf")
    regression = True if dataset.out_dim == 1 else False
    from utils.calibration_loss_fn import Posttrain_loss

    loss_fn = Posttrain_loss(args.posttrain_shrinkage_factor)

    source_mean_x = torch.zeros(1, dataset.in_dim)
    for train_x, train_y in dataset.train_loader:
        source_mean_x += torch.sum(train_x, dim=0, keepdim=True)
    source_mean_x /= len(dataset.train_x)
    source_mean_x = source_mean_x.to(args.device)
    patience = args.posttrain_patience

    for epoch in range(1, args.posttrain_epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        calibrated_pred_list, label_list = [], []

        column_distribution_handler = column_distribution_handler.train()
        for train_x, train_y in dataset.posttrain_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)

            estimated_y = model(train_x).detach()
            estimated_y = column_distribution_handler(train_x, estimated_y)
            loss = loss_fn(estimated_y, train_y)

            column_distribution_handler_optimizer.zero_grad()
            loss.backward()
            column_distribution_handler_optimizer.step()

            train_loss += loss.item() * train_x.shape[0]
            train_acc += (
                (torch.argmax(estimated_y, dim=-1) == torch.argmax(train_y, dim=-1))
                .sum()
                .item()
            )
            train_len += train_x.shape[0]

            with torch.no_grad():
                estimated_y = column_distribution_handler(train_x, model(train_x))
                calibrated_pred_list.extend(estimated_y.detach().cpu().tolist())
                label_list.extend(train_y.detach().cpu().tolist())

        valid_loss, valid_acc, valid_len = 0, 0, 0
        column_distribution_handler = column_distribution_handler.eval()
        with torch.no_grad():
            for valid_x, valid_y in dataset.posttrain_validloader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)

                estimated_y = model(valid_x)
                estimated_y = column_distribution_handler(valid_x, estimated_y)
                loss = loss_fn(estimated_y, valid_y)

                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (
                    (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1))
                    .sum()
                    .item()
                )
                valid_len += valid_x.shape[0]

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = args.posttrain_patience
            source_handler = deepcopy(column_distribution_handler)
            torch.save(
                source_handler.state_dict(),
                os.path.join(args.out_dir, "column_distribution_handler.pth"),
            )
        else:
            patience -= 1
            if patience == 0:
                break

        logger.info(
            f"posttrain epoch {epoch} | train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}"
        )
    return source_handler


def forward_and_adapt(args, dataset, x, y, mask, model, optimizer):
    if not set(args.method).intersection(["mae", "em", "sam", "memo", "sar", "pl", "ttt++", "eata", "dem", "gem", "ns", "dm", "kld"]):
        return

    global EMA, original_source_model, eata_params, ttt_params
    optimizer.zero_grad()
    outputs = model(x)

    if "mae" in args.method:
        cor_x, _ = dataset.get_corrupted_data(
            x,
            dataset.train_x,
            shift_type="random_drop",
            shift_severity=1,
            imputation_method=args.mae_imputation_method,
        )  # fully corrupted (masking is done below)
        feature_importance = get_feature_importance(args, dataset, x, mask, model)
        test_cor_mask_x = (
            get_mask_by_feature_importance(args, dataset, x, feature_importance)
            .to(x.device)
            .detach()
        )
        test_cor_x = test_cor_mask_x * x + (1 - test_cor_mask_x) * cor_x
        estimated_test_x = source_model.get_recon_out(test_cor_x)

        if "threshold" in args.method:
            grad_list = []
            for idx, test_instance in enumerate(x):
                optimizer.zero_grad()
                outputs = source_model(test_instance.unsqueeze(0))
                recon_out = source_model.get_recon_out(test_instance.unsqueeze(0))
                loss = F.mse_loss(
                    recon_out * mask[idx], test_instance.unsqueeze(0) * mask[idx]
                ).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(
                    np.sum(
                        [
                            (
                                p.grad.detach().cpu().data.norm(2) ** 2
                                if p.grad != None
                                else 0
                            )
                            for p in source_model.parameters()
                        ]
                    )
                )
                grad_list.append(gradient_norm)
            grad_list = torch.tensor(grad_list).to(args.device)
            loss_idx = torch.where(grad_list < 1)
            optimizer.zero_grad()

            loss = F.mse_loss(
                estimated_test_x * mask, x * mask, reduction="none"
            )  # l2 loss only on non-missing values
            loss = loss[loss_idx].mean()
        elif "sam" in args.method:
            optimizer.zero_grad()
            loss = F.mse_loss(estimated_test_x * mask, x * mask, reduction="none")
            loss.backward(retain_graph=True)
            optimizer.step()
            return
        elif "double_masking" in args.method:
            with torch.no_grad():
                recon_out = model.get_recon_out(x)
                recon_loss = F.mse_loss(recon_out, x, reduction="none").mean(0)
                recon_sorted_idx = torch.argsort(recon_loss, descending=True)
                recon_sorted_idx = recon_sorted_idx[: int(len(recon_sorted_idx) * 0.1)]
            mask = torch.zeros_like(x[0]).to(args.device)
            mask[recon_sorted_idx] = 1

            recon_x = model.get_recon_out(
                x * mask
            )  # ones with high reconstruction error
            recon_adverse_x = model.get_recon_out(
                x * (1 - mask)
            )  # ones with low reconstruction error

            m = F.normalize(recon_x, dim=1) @ F.normalize(recon_adverse_x, dim=1).T
            id = torch.eye(m.shape[0]).to(args.device) - 1

            pos_loss = F.mse_loss(recon_x, recon_adverse_x, reduction="none").mean()
            neg_loss = F.mse_loss(m, id, reduction="none").mean()
            loss = pos_loss + neg_loss
        else:
            from utils.mae_util import cat_aware_recon_loss

            loss = cat_aware_recon_loss(estimated_test_x, x, model)
        loss.backward(retain_graph=True)
    if "em" in args.method:
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
    if "sam" in args.method:
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
        optimizer.first_step()
        outputs = model(x)
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
        optimizer.second_step()
        return
    if "memo" in args.method:
        x = generate_augmentation(x, args)
        outputs = model(x)
        loss = softmax_entropy(outputs / args.temp).mean()
        loss.backward(retain_graph=True)
    if "sar" in args.method:
        entropy_first = softmax_entropy(outputs)
        filter_id1 = torch.where(entropy_first < 0.4 * np.log(outputs.shape[-1]))
        entropy_first = entropy_first[filter_id1]
        loss = entropy_first.mean()
        loss.backward(retain_graph=True)

        optimizer.first_step(zero_grad=True)
        new_outputs = model(x)
        entropy_second = softmax_entropy(new_outputs)
        entropy_second = entropy_second[filter_id1]
        filter_id2 = torch.where(entropy_second < 0.4 * np.log(outputs.shape[-1]))
        loss_second = entropy_second[filter_id2].mean()

        loss_second.backward(retain_graph=True)
        optimizer.second_step()

        EMA = (
            0.9 * EMA + (1 - 0.9) * loss_second.item()
            if EMA != None
            else loss_second.item()
        )
        return
    if "pl" in args.method:
        pseudo_label = torch.argmax(outputs, dim=-1)
        loss = F.cross_entropy(outputs, pseudo_label)
        loss.backward(retain_graph=True)
    if "ttt++" in args.method:
        # getting featuers
        z = model.get_feature(x)
        a = model.get_recon_out(x * mask)

        from utils.ttt import linear_mmd, coral, covariance

        criterion_ssl = nn.MSELoss()
        loss_ssl = criterion_ssl(x, a)

        loss_mean = linear_mmd(z.mean(axis=0), ttt_params["mean"])
        loss_coral = coral(covariance(z), ttt_params["sigma"])

        loss = (
            loss_ssl * args.ttt_coef[0]
            + loss_mean * args.ttt_coef[1]
            + loss_coral * args.ttt_coef[2]
        )
        loss.backward()
    if "eata" in args.method:
        from utils.eata import update_model_probs

        entropys = softmax_entropy(outputs / args.temp)
        filter_ids_1 = torch.where(entropys < args.eata_e_margin)

        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)

        entropys = entropys[filter_ids_1]
        if eata_params["current_model_probs"] is not None:
            cosine_similarities = F.cosine_similarity(
                eata_params["current_model_probs"].unsqueeze(dim=0),
                outputs[filter_ids_1].softmax(1),
                dim=1,
            )
            filter_ids_2 = torch.where(
                torch.abs(cosine_similarities) < args.eata_d_margin
            )
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(
                eata_params["current_model_probs"],
                outputs[filter_ids_1][filter_ids_2].softmax(1),
            )
        else:
            updated_probs = update_model_probs(
                eata_params["current_model_probs"], outputs[filter_ids_1].softmax(1)
            )
        coeff = 1 / (torch.exp(entropys.clone().detach() - args.eata_e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        if x[ids1][ids2].size(0) != 0:
            loss.backward(retain_graph=True)

        # eata param update
        eata_params["current_model_probs"] = updated_probs
    if "dem" in args.method:  # differential entropy minimization
        model.train()
        prediction_list = []
        for _ in range(args.dropout_steps):
            outputs = model(x)
            prediction_list.append(outputs)
        prediction_std = torch.std(torch.cat(prediction_list, dim=-1), dim=-1).mean()
        differential_entropy = -torch.log(2 * np.pi * np.e * prediction_std)
        differential_entropy.backward(retain_graph=True)
        model.eval()
    if "gem" in args.method:  # generalized entropy minimization
        e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
        e_loss.backward(retain_graph=True)
    if "ns" in args.method:  # generalized entropy minimization
        negative_outputs = outputs.clone()
        negative_loss = 0
        negative_mask = torch.where(
            torch.softmax(negative_outputs, dim=-1)
            < args.ns_threshold * (10 / negative_outputs.shape[-1]),
            1,
            0,
        )
        negative_loss += torch.mean(
            -torch.log(
                1
                - torch.sum(
                    negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1),
                    dim=-1,
                )
            )
        )
        if torch.is_tensor(negative_loss):
            (args.ns_weight * negative_loss).backward(retain_graph=True)
    if "dm" in args.method:  # diversity maximization
        mean_probs = torch.mean(outputs, dim=-1, keepdim=True)
        (-args.dm_weight * softmax_entropy(mean_probs / args.temp).mean()).backward(
            retain_graph=True
        )
    if "kld" in args.method:  # kl-divergence loss
        original_outputs = original_source_model(x)
        probs = torch.softmax(outputs, dim=-1)
        original_probs = torch.softmax(original_outputs, dim=-1)
        kl_div_loss = F.kl_div(
            torch.log(probs), original_probs.detach(), reduction="batchmean"
        )
        (args.kld_weight * kl_div_loss).backward(retain_graph=True)
    optimizer.step()


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    global logger, original_source_model, source_model, EMA, eata_params, ttt_params
    (
        EMA,
        ENTROPY_LIST_BEFORE_ADAPTATION,
        ENTROPY_LIST_AFTER_ADAPTATION,
        GRADIENT_NORM_LIST,
        RECON_LOSS_LIST_BEFORE_ADAPTATION,
        RECON_LOSS_LIST_AFTER_ADAPTATION,
        FEATURE_LIST,
        LABEL_LIST,
    ) = (None, [], [], [], [], [], [], [])
    SOURCE_LABEL_LIST, TARGET_PREDICTION_LIST = [], []
    SOURCE_INPUT_LIST, SOURCE_FEATURE_LIST, SOURCE_ENTROPY_LIST = [], [], []
    (
        SOURCE_PREDICTION_LIST,
        SOURCE_CALIBRATED_PREDICTION_LIST,
        SOURCE_CALIBRATED_ENTROPY_LIST,
        SOURCE_CALIBRATED_PROB_LIST,
        SOURCE_ONE_HOT_LABEL_LIST,
    ) = ([], [], [], [], [])
    (
        SOURCE_PROB_LIST,
        PROB_LIST_BEFORE_ADAPTATION,
        PROB_LIST_AFTER_ADAPTATION,
        PROB_LIST_AFTER_CALIBRATION,
    ) = ([], [], [], [])
    TARGET_PREDICTION_LIST, TARGET_CALIBRATED_PREDICTION_LIST = [], []
    GROUND_TRUTH_LABEL_LIST, ESTIMATED_BEFORE_LABEL_LIST, ESTIMATED_AFTER_LABEL_LIST = (
        [],
        [],
        [],
    )

    global TRAIN_GRADIENT_NORM_LIST, TRAIN_SMOOTHNESS_LIST

    eata_params = {"fishers": None, "current_model_probs": None}
    ttt_params = {"mean": None, "sigma": None}
    if hasattr(args, "seed"):
        set_seed(args.seed)
        print(f"set seed as {args.seed}")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    disable_logger()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args, logger)

    regression = True if dataset.out_dim == 1 else False
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    ece_loss_fn = ECELoss()

    import time

    before_source_model_training = time.time()

    source_model = get_source_model(args, dataset)
    source_model.eval().requires_grad_(True)

    # with open(
    #     file=f"pickle/{args.dataset}_{args.model}_{args.seed}_grad_norm.pickle",
    #     mode="wb",
    # ) as f:
    #     pickle.dump(TRAIN_GRADIENT_NORM_LIST, f)
    # with open(
    #     file=f"pickle/{args.dataset}_{args.model}_{args.seed}_smoothness.pickle",
    #     mode="wb",
    # ) as f:
    #     pickle.dump(TRAIN_SMOOTHNESS_LIST, f)

    source_model_training_time = time.time() - before_source_model_training
    logger.info(f"source_model_training_time: {source_model_training_time}")

    original_source_model = deepcopy(source_model)
    original_source_model.eval().requires_grad_(False)
    params, _ = collect_params(source_model, train_params=args.train_params)
    if "sam" in args.method or "sar" in args.method:
        test_optimizer = SAM(
            params,
            base_optimizer=getattr(torch.optim, args.test_optimizer),
            lr=args.test_lr,
        )
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(
            params, lr=args.test_lr
        )

    print(f"{1}")

    if "ttt++" in args.method:  # for TTT++
        from utils.ttt import summarize  # offline summarization

        mean, sigma = summarize(args, dataset, source_model)
        # save to global variable
        ttt_params["mean"] = mean
        ttt_params["sigma"] = sigma

    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(
        source_model, test_optimizer, scheduler=None
    )
    test_loss_before, test_loss_after = 0, 0

    before_gnn_training = time.time()

    print(f"{2}")

    if "calibrator" in args.method:
        print(f"2-1")
        calibrator = Calibrator(args, dataset, source_model)
        print(f"2-2")
        if os.path.exists(os.path.join(args.out_dir, f"calibrator_{args.model}_{args.dataset}.pth")):
            calibrator.gnn.load_state_dict(
                torch.load(os.path.join(args.out_dir, f"calibrator_{args.model}_{args.dataset}.pth"))
            )
        else:
            print(f"2-3")
            calibrator.train_gnn()
            print(f"2-4")
            torch.save(
                calibrator.gnn.state_dict(),
                os.path.join(args.out_dir, f"calibrator_{args.model}_{args.dataset}.pth"),
            )
            print(f"2-5")
        # with torch.no_grad():
        #     for train_x, train_y in dataset.train_loader:
        #         train_x, train_y = train_x.to(args.device), train_y.to(args.device)
        #         estimated_y = source_model(train_x).detach().cpu()
        #         calibrated_y = (
        #             calibrator.get_gnn_out(source_model, train_x).detach().cpu()
        #         )

        #         SOURCE_PREDICTION_LIST.extend(estimated_y.tolist())
        #         SOURCE_CALIBRATED_PREDICTION_LIST.extend(calibrated_y.tolist())
        #         SOURCE_CALIBRATED_ENTROPY_LIST.extend(
        #             softmax_entropy(calibrated_y).tolist()
        #         )
        #         SOURCE_CALIBRATED_PROB_LIST.extend(
        #             calibrated_y.softmax(dim=-1).max(dim=-1)[0].tolist()
        #         )
        #         SOURCE_ONE_HOT_LABEL_LIST.extend(train_y.cpu().tolist())

    print(f"{3}")

    gnn_training_time = time.time() - before_gnn_training
    logger.info(f"gnn_training_time: {gnn_training_time}")

    source_label_dist = F.normalize(
        torch.FloatTensor(
            np.unique(
                np.argmax(
                    np.concatenate([dataset.train_y, dataset.valid_y], axis=0), axis=-1
                ),
                return_counts=True,
            )[1]
        ),
        p=1,
        dim=-1,
    ).to(args.device)
    target_label_dist = torch.full((1, dataset.out_dim), 1 / dataset.out_dim).to(
        args.device
    )

    import time

    avg_inference_time = 0
    avg_adaptation_time = 0

    for batch_idx, (test_x, test_mask_x, test_y) in enumerate(dataset.test_loader):
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            source_model, test_optimizer, _ = load_model_and_optimizer(
                source_model,
                test_optimizer,
                None,
                original_model_state,
                original_optimizer_state,
                None,
            )
        test_x, test_mask_x, test_y = (
            test_x.to(device),
            test_mask_x.to(device),
            test_y.to(device),
        )
        GROUND_TRUTH_LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())

        before_inference = time.time()
        ori_estimated_y = original_source_model(test_x)
        avg_inference_time += time.time() - before_inference

        if regression:
            loss = loss_fn(ori_estimated_y.squeeze(), test_y.squeeze().float())
        else:
            loss = loss_fn(ori_estimated_y, test_y.argmax(1))
        test_loss_before += loss.item() * test_x.shape[0]
        ESTIMATED_BEFORE_LABEL_LIST.extend(
            torch.argmax(ori_estimated_y, dim=-1).cpu().tolist()
        )
        TARGET_PREDICTION_LIST.extend(ori_estimated_y.detach().cpu().tolist())

        before_adaptation = time.time()

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(
                args, dataset, test_x, test_y, test_mask_x, source_model, test_optimizer
            )

        if "mae" in args.method:  # implement imputation with masked autoencoder
            estimated_x = source_model.get_recon_out(test_x)
            test_x = test_x * test_mask_x + estimated_x * (1 - test_mask_x)
        if "lame" in args.method:
            import utils.lame as lame

            estimated_y = lame.batch_evaluation(args, source_model, test_x)
        elif "label_distribution_handler" in args.method:
            estimated_y = source_model(test_x)
            calibrated_probability = F.normalize(
                (F.softmax(estimated_y, dim=-1) / source_label_dist), p=1, dim=-1
            )
            cur_target_label_dist = (1 - float(args.smoothing_factor)) * torch.mean(
                calibrated_probability, dim=0, keepdim=True
            ) + float(args.smoothing_factor) * target_label_dist

            if "column_distribution_handler" in args.method:
                column_distribution_handler = get_column_distribution_handler(
                    args, dataset, original_source_model
                )
                calibrated_estimated_y = column_distribution_handler(
                    test_x, estimated_y
                )
                TARGET_CALIBRATED_PREDICTION_LIST.extend(
                    calibrated_estimated_y.detach().cpu().tolist()
                )
            elif "calibrator" in args.method:
                calibrated_estimated_y = calibrator.get_gnn_out(
                    source_model, test_x, wo_softmax=True
                )
                TARGET_CALIBRATED_PREDICTION_LIST.extend(
                    calibrated_estimated_y.detach().cpu().tolist()
                )
            else:
                calibrated_estimated_y = estimated_y

            probs, _ = torch.topk(calibrated_estimated_y.softmax(dim=-1), k=2, dim=1)
            uncertainty = 1 / (probs[:, 0] - probs[:, 1])
            uncertainty_lower_threshold = torch.quantile(
                uncertainty, args.uncertainty_lower_percentile_threshod
            )
            uncertainty_upper_threshold = torch.quantile(
                uncertainty, args.uncertainty_upper_percentile_threshod
            )
            pos_mask = (uncertainty <= uncertainty_lower_threshold).long()
            neg_mask = (uncertainty >= uncertainty_upper_threshold).long()
            imb_ratio = np.max(dataset.train_counts[1]) / np.min(
                dataset.train_counts[1]
            )
            temperature = 1.5 * (imb_ratio) / (imb_ratio - 1 + 1e-6)
            for i in range(len(estimated_y)):
                if pos_mask[i]:
                    estimated_y[i] = estimated_y[i] * temperature
                elif neg_mask[i]:
                    estimated_y[i] = estimated_y[i] / temperature

            calibrated_probability = F.normalize(
                (
                    F.softmax(estimated_y, dim=-1)
                    * cur_target_label_dist
                    / source_label_dist
                ),
                p=1,
                dim=-1,
            )
            estimated_y = (
                estimated_y.softmax(dim=-1) / 2 + calibrated_probability / 2
            ).log()
            target_label_dist = (1 - float(args.smoothing_factor)) * torch.mean(
                estimated_y.softmax(dim=-1), dim=0, keepdim=True
            ) + float(args.smoothing_factor) * target_label_dist
        else:
            estimated_y = source_model(test_x)

        avg_adaptation_time += time.time() - before_adaptation

        if regression:
            loss = loss_fn(estimated_y.squeeze(), test_y.squeeze().float())
        else:
            loss = loss_fn(estimated_y, test_y.argmax(1))
        test_loss_after += loss.item() * test_x.shape[0]
        ESTIMATED_AFTER_LABEL_LIST.extend(
            torch.argmax(estimated_y, dim=-1).cpu().tolist()
        )
        logger.info(
            f"online batch [{batch_idx}]: current acc before {accuracy_score(torch.argmax(test_y, dim=-1).cpu().tolist(), torch.argmax(ori_estimated_y, dim=-1).cpu().tolist()):.4f}, current acc after {accuracy_score(torch.argmax(test_y, dim=-1).cpu().tolist(), torch.argmax(estimated_y, dim=-1).cpu().tolist()):.4f}"
        )

        if args.vis:  # for entropy and mae vs gradient norm visualization
            FEATURE_LIST.extend(source_model.get_feature(test_x).cpu().tolist())
            LABEL_LIST.extend(torch.argmax(test_y, dim=-1).cpu().tolist())
            ENTROPY_LIST_BEFORE_ADAPTATION.extend(
                softmax_entropy(ori_estimated_y).tolist()
                / np.log(ori_estimated_y.shape[-1])
            )
            RECON_LOSS_LIST_BEFORE_ADAPTATION.extend(
                F.mse_loss(
                    source_model.get_recon_out(test_x * test_mask_x),
                    test_x,
                    reduction="none",
                )
                .mean(dim=-1)
                .cpu()
                .tolist()
            )
            PROB_LIST_BEFORE_ADAPTATION.extend(
                ori_estimated_y.softmax(dim=-1).max(dim=-1)[0].tolist()
            )
            ENTROPY_LIST_AFTER_ADAPTATION.extend(
                softmax_entropy(estimated_y).tolist() / np.log(estimated_y.shape[-1])
            )
            RECON_LOSS_LIST_AFTER_ADAPTATION.extend(
                F.mse_loss(
                    source_model.get_recon_out(test_x * test_mask_x),
                    test_x,
                    reduction="none",
                )
                .mean(dim=-1)
                .cpu()
                .tolist()
            )
            PROB_LIST_AFTER_ADAPTATION.extend(
                estimated_y.softmax(dim=-1).max(dim=-1)[0].tolist()
            )

        if args.entropy_gradient_vis:
            for test_instance in test_x:
                test_optimizer.zero_grad()
                outputs = original_source_model(test_instance.unsqueeze(0))
                loss = softmax_entropy(outputs / args.temp).mean()
                loss.backward(retain_graph=True)
                gradient_norm = np.sqrt(
                    np.sum(
                        [
                            (
                                p.grad.detach().cpu().data.norm(2) ** 2
                                if p.grad != None
                                else 0
                            )
                            for p in source_model.parameters()
                        ]
                    )
                )
                GRADIENT_NORM_LIST.append(gradient_norm)

    logger.info(f"total_inference_time: {avg_inference_time}")
    logger.info(f"total_adaptation_time: {avg_adaptation_time}")

    # avg_inference_time  /= len(dataset.test_x)

    logger.info(f"avg_inference_time: {avg_inference_time / len(dataset.test_x)}")
    logger.info(f"avg_adaptation_time: {avg_adaptation_time / len(dataset.test_x)}")

    logger.info(
        f"before adaptation | loss {test_loss_before / len(GROUND_TRUTH_LABEL_LIST):.4f}, acc {accuracy_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_BEFORE_LABEL_LIST):.4f}, bacc {balanced_accuracy_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_BEFORE_LABEL_LIST):.4f}, macro f1-score {f1_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_BEFORE_LABEL_LIST, average='macro'):.4f}"
    )
    logger.info(
        f"after adaptation | loss {test_loss_after / len(GROUND_TRUTH_LABEL_LIST):.4f}, acc {accuracy_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_AFTER_LABEL_LIST):.4f}, bacc {balanced_accuracy_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_AFTER_LABEL_LIST):.4f}, macro f1-score {f1_score(GROUND_TRUTH_LABEL_LIST, ESTIMATED_AFTER_LABEL_LIST, average='macro'):.4f}"
    )

    confusion_matrix_before = confusion_matrix(
        GROUND_TRUTH_LABEL_LIST, ESTIMATED_BEFORE_LABEL_LIST
    )
    confusion_matrix_after = confusion_matrix(
        GROUND_TRUTH_LABEL_LIST, ESTIMATED_AFTER_LABEL_LIST
    )
    logger.info(f"before adaptation | confusion matrix\n{confusion_matrix_before}")
    logger.info(f"after adaptation | confusion matrix\n{confusion_matrix_after}")

    # if args.vis:
        # probs_per_label = defaultdict(list)
        # for train_x, train_y in dataset.train_loader:
        #     train_x, train_y = train_x.to(args.device), train_y.to(args.device)
        #     estimated_y = source_model(train_x)
        #     SOURCE_INPUT_LIST.extend(
        #         original_source_model.get_embedding(train_x.to(args.device))
        #         .cpu()
        #         .tolist()
        #     )
        #     SOURCE_FEATURE_LIST.extend(
        #         original_source_model.get_feature(train_x.to(args.device))
        #         .cpu()
        #         .tolist()
        #     )
        #     SOURCE_ENTROPY_LIST.extend(
        #         softmax_entropy(original_source_model(train_x.to(args.device))).tolist()
        #     )
        #     SOURCE_LABEL_LIST.extend(torch.argmax(train_y, dim=-1).tolist())
        #     SOURCE_PROB_LIST.extend(
        #         original_source_model(train_x.to(args.device))
        #         .softmax(dim=-1)
        #         .max(dim=-1)[0]
        #         .tolist()
        #     )

        #     probs = estimated_y.softmax(dim=-1)
        #     for class_idx in range(estimated_y.shape[-1]):
        #         probs_per_label[class_idx].extend(probs[:, class_idx].cpu().tolist())
        # for label, probs in probs_per_label.items():
        #     draw_histogram(
        #         args,
        #         probs,
        #         f"class {label} probability distribution",
        #         "Confidence",
        #         "Number of Instances",
        #     )

        # draw_histogram(args, np.array(SOURCE_ENTROPY_LIST), "Source Entropy Distribution", "Entropy", "Number of Instances")
        # draw_histogram(args, np.array(SOURCE_PROB_LIST), "Source Confidence Distribution", "Confidence", "Number of Instances")

        # draw_histogram(
        #     args,
        #     ENTROPY_LIST_BEFORE_ADAPTATION,
        #     "Entropy Distribution Before Adaptation",
        #     "Entropy",
        #     "Number of Instances",
        # )
        # # draw_histogram(args, ENTROPY_LIST_AFTER_ADAPTATION, "Entropy Distribution After Adaptation", "Entropy", "Number of Instances")
        # # draw_histogram(args, np.array(PROB_LIST_BEFORE_ADAPTATION), "Target Confidence Distribution Before Adaptation", "Confidence", "Number of Instances")
        # # draw_histogram(args, np.array(PROB_LIST_AFTER_ADAPTATION), "Target Confidence Distribution After Adaptation", "Confidence", "Number of Instances")

        # # draw_label_distribution_plot(args, SOURCE_LABEL_LIST, "Source Label Distribution")
        # # draw_label_distribution_plot(args, LABEL_LIST, "Target Label Distribution")
        # # draw_label_distribution_plot(args, TARGET_PREDICTION_LIST, "Pseudo Label Distribution")

        # draw_tsne(
        #     args,
        #     np.array(FEATURE_LIST),
        #     np.array(LABEL_LIST),
        #     "Target Latent Space Visualization with t-SNE",
        # )
        # # draw_tsne(args, np.array(SOURCE_FEATURE_LIST), np.array(SOURCE_LABEL_LIST), "Source Latent Space Visualization with t-SNE")
        # # draw_tsne(args, np.array(SOURCE_INPUT_LIST), np.array(SOURCE_LABEL_LIST), "Source Input Space Visualization with t-SNE")

        # if "calibrator" in args.method or "column_distribution_handler" in args.method:
            # train_ece_before = ece_loss_fn(torch.tensor(TARGET_PREDICTION_LIST), torch.tensor(LABEL_LIST)).item()
            # train_ece_after = ece_loss_fn(torch.tensor(TARGET_CALIBRATED_PREDICTION_LIST), torch.tensor(LABEL_LIST)).item()
            # logger.info(f"test ece before: {train_ece_before}")
            # logger.info(f"test ece after: {train_ece_after}")

            # test_ece_before = ece_loss_fn(torch.tensor(SOURCE_PREDICTION_LIST), torch.tensor(SOURCE_LABEL_LIST)).item()
            # test_ece_after = ece_loss_fn(torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST), torch.tensor(SOURCE_LABEL_LIST)).item()
            # logger.info(f"train_ece_before: {test_ece_before}")
            # logger.info(f"train_ece_after: {test_ece_after}")

            # draw_histogram(args, np.array(SOURCE_CALIBRATED_ENTROPY_LIST), "Source Entropy Distribution After Calibration", "Entropy", "Number of Instances")
            # draw_histogram(args, np.array(SOURCE_CALIBRATED_PROB_LIST), "Source Confidence Distribution After Calibration", "Confidence", "Number of Instances")
            # draw_histogram(args, np.array(PROB_LIST_AFTER_CALIBRATION), "Target Confidence Distribution After Calibration", "Confidence", "Number of Instances")

            # draw_reliability_plot(
            #     args,
            #     np.array(torch.tensor(SOURCE_PREDICTION_LIST).softmax(axis=-1)).max(
            #         axis=-1
            #     ),
            #     np.array(SOURCE_PREDICTION_LIST).argmax(axis=-1),
            #     np.array(SOURCE_LABEL_LIST),
            #     "calibration_before_calibration.png",
            # )
            # # draw_reliability_plot(args, np.array(torch.tensor(SOURCE_CALIBRATED_PREDICTION_LIST).softmax(axis=-1)).max(axis=-1), np.array(SOURCE_CALIBRATED_PREDICTION_LIST).argmax(axis=-1), np.array(SOURCE_LABEL_LIST), "calibration_after_calibration.png")

    # if args.entropy_gradient_vis:
    #     draw_entropy_gradient_plot(args, ENTROPY_LIST_BEFORE_ADAPTATION, GRADIENT_NORM_LIST, "Entropy vs. Gradient Norm")


if __name__ == "__main__":
    main()
