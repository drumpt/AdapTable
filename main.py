import os
import time

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from data.dataset import *
from model.model import *
from utils.utils import *
from utils.sam import *
from utils.mae_util import *
from utils.calibrator import *
from utils.calibration_loss_fn import Posttrain_loss


def get_model(args, dataset):
    if args.model in [
        "MLP",
        "TabNet",
        "TabTransformer",
        "FTTransformer",
        "ResNet",
        "AutoInt",
        "NODE",
    ]:
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
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(
            list(init_model.parameters()), lr=args.train_lr
        )
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


def train(args, model, optimizer, dataset):
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
    if not set(args.method).intersection(
        [
            "em",
            "sam",
            "sar",
            "pl",
            "eata"
        ]
    ):
        return

    global EMA, original_source_model, eata_params
    optimizer.zero_grad()
    outputs = model(x)

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
    optimizer.step()


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    global logger, original_source_model, source_model, EMA, eata_params
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

    import time

    before_source_model_training = time.time()

    source_model = get_source_model(args, dataset)
    source_model.eval().requires_grad_(True)

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

    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(
        source_model, test_optimizer, scheduler=None
    )
    test_loss_before, test_loss_after = 0, 0

    before_gnn_training = time.time()

    if "calibrator" in args.method:
        calibrator = Calibrator(args, dataset, source_model)
        calibrator.train_gnn()

        with torch.no_grad():
            for train_x, train_y in dataset.train_loader:
                train_x, train_y = train_x.to(args.device), train_y.to(args.device)
                estimated_y = source_model(train_x).detach().cpu()
                calibrated_y = (
                    calibrator.get_gnn_out(source_model, train_x).detach().cpu()
                )

                SOURCE_PREDICTION_LIST.extend(estimated_y.tolist())
                SOURCE_CALIBRATED_PREDICTION_LIST.extend(calibrated_y.tolist())
                SOURCE_CALIBRATED_ENTROPY_LIST.extend(
                    softmax_entropy(calibrated_y).tolist()
                )
                SOURCE_CALIBRATED_PROB_LIST.extend(
                    calibrated_y.softmax(dim=-1).max(dim=-1)[0].tolist()
                )
                SOURCE_ONE_HOT_LABEL_LIST.extend(train_y.cpu().tolist())


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


        if "lame" in args.method:
            import utils.lame as lame
            estimated_y = lame.batch_evaluation(args, source_model, test_x)
        elif "ours" in args.method:
            estimated_y = source_model(test_x)

            calibrated_probability = F.normalize(
                (F.softmax(estimated_y, dim=-1) / source_label_dist), p=1, dim=-1
            )
            cur_target_label_dist = (1 - float(args.smoothing_factor)) * torch.mean(
                calibrated_probability, dim=0, keepdim=True
            ) + float(args.smoothing_factor) * target_label_dist


            calibrated_estimated_y = calibrator.get_gnn_out(
                source_model, test_x, wo_softmax=True
            )
            TARGET_CALIBRATED_PREDICTION_LIST.extend(
                calibrated_estimated_y.detach().cpu().tolist()
            )

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


    logger.info(f"total_inference_time: {avg_inference_time}")
    logger.info(f"total_adaptation_time: {avg_adaptation_time}")

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

    return 0

if __name__ == "__main__":
    main()
