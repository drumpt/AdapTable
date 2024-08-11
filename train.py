import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from copy import deepcopy
import traceback
from functools import partial
from itertools import chain
import warnings
warnings.filterwarnings("ignore")
import hydra

import numpy as np
import torch
import torch.nn as nn
import optuna

from data.dataset import *
from model.model import *
from tta.tta import *
from tta.ttt_mae import *
from tta.calibrator import *
from utils.utils import *


class HOP(): # hyperparameter optimization
    def __init__(self, args, dataset, mode, source_model=None):
        self.args = args
        self.dataset = dataset
        self.mode = mode
        self.source_model = source_model


    def __call__(self, trial):
        # set hyperparameters
        if self.mode == "train":
            setattr(self.args, 'train_batch_size', trial.suggest_categorical("train_batch_size", [32, 64, 128, 256]))
            setattr(self.args, 'train_lr', trial.suggest_categorical("train_lr", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]))
            _, valid_loss = get_pretrained_model(self.args, self.dataset)
            return valid_loss
        elif self.mode == "posttrain":
            setattr(self.args, 'posttrain_lr', trial.suggest_categorical("posttrain_lr", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]))
            setattr(self.args, 'posttrain_shrinkage_factor', trial.suggest_categorical("posttrain_shrinkage_factor", [0, 0.1, 0.5, 1]))
            _, valid_loss = get_calibrator(self.args, self.dataset, self.source_model)
            return valid_loss


def get_architecture(args, dataset, topic_num=0, cluster_centers_=None):
    if args.model == 'mlp':
        model = 'MLP'
    elif args.model == 'fttransformer':
        model = 'FTTransformer'
    elif args.model in ['MLP', 'TabNet', 'TabTransformer', 'FTTransformer', "ResNet"]:
        model = args.model
    else:
        raise NotImplementedError

    if model == "MLP":
        model = eval(model)(args, dataset, topic_num, cluster_centers_)
    else:
        model = eval(model)(args, dataset)
    model = model.to(args.device)
    return model


def get_pretrained_model(args, dataset):
    source_model = get_architecture(args, dataset)

    if isinstance(args.method, str):
        args.method = [args.method]

    if set(args.method).intersection(['ttt_mae', 'ttt++']): # pretrain for test-time training methods
        pretrain_optimizer = getattr(torch.optim, args.pretrain_optimizer)(TTA.collect_params(source_model, train_params="pretrain")[0], lr=args.pretrain_lr)
        pretrain(args, source_model, pretrain_optimizer, dataset) # self-supervised learning (masking and reconstruction task)
        train_optimizer = getattr(torch.optim, args.train_optimizer)(TTA.collect_params(source_model, train_params="downstream")[0], lr=args.train_lr)
        valid_loss = train(args, source_model, train_optimizer, dataset) # supervised learning (main task)
    else:
        train_optimizer = getattr(torch.optim, args.train_optimizer)(TTA.collect_params(source_model, train_params="all")[0], lr=args.train_lr)
        valid_loss = train(args, source_model, train_optimizer, dataset)
    return source_model, valid_loss


def get_calibrator(args, dataset, source_model):
    calibrator = Calibrator(args, dataset, source_model)
    valid_loss = calibrator.train_gnn()
    return calibrator, valid_loss


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


def pretrain(args, model, optimizer, dataset):
    global logger
    device = args.device
    loss_fn = partial(cat_aware_recon_loss, model=model)
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_len = 0, 0
        model.train()
        for train_x, _ in chain(dataset.train_loader, dataset.valid_loader):
            train_x = train_x.to(device)
            train_cor_x, _ = dataset.get_corrupted_data(train_x, dataset.train_x, shift_type="random_drop", shift_severity=args.pretrain_mask_ratio, imputation_method=args.ttt_mae.imputation_method)

            estimated_x = model.get_recon_out(train_cor_x)
            loss = loss_fn(estimated_x, train_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_cor_x.shape[0]
            train_len += train_cor_x.shape[0]
        train_loss /= train_len
        logger.info(f"pretrain epoch {epoch} | train_loss {train_loss:.4f}")
    return train_loss


def train(args, model, optimizer, dataset):
    global logger
    device = args.device
    source_model, best_loss, best_acc, best_bacc, best_f1, best_epoch = None, float('inf'), 0, 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    patience = args.train_patience

    for epoch in range(1, args.train_epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        model = model.train().requires_grad_(True)
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
        train_loss /= train_len
        train_acc /= train_len

        valid_loss, valid_acc, valid_len = 0, 0, 0
        all_valid_gt_list, all_valid_pred_list = [], []
        model = model.eval()
        with torch.no_grad():
            for valid_x, valid_y in dataset.valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                estimated_y = model(valid_x)
                loss = loss_fn(estimated_y, valid_y)

                valid_loss += loss.item() * valid_x.shape[0]
                valid_acc += (torch.argmax(estimated_y, dim=-1) == torch.argmax(valid_y, dim=-1)).sum().item()
                valid_len += valid_x.shape[0]
                all_valid_gt_list.extend(valid_y.tolist())
                all_valid_pred_list.extend(estimated_y.cpu().detach().tolist())
        valid_loss /= valid_len
        valid_acc /= valid_len
        _, valid_bacc, valid_f1 = get_metrics(all_valid_gt_list, all_valid_pred_list)
        if valid_bacc > best_bacc:
            best_bacc = valid_bacc
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch = epoch
            patience = args.train_patience

            source_model = deepcopy(model)
            torch.save(source_model.state_dict(), os.path.join(args.out_dir, f"{args.model}_{args.dataset}_best_model_{args.mlp.norm_layer}_{args.mlp.moe}.pth"))
            print(f"save best model!")
        else:
            patience -= 1

        if patience == 0:
            break

        logger.info(f"train epoch {epoch} | train_loss {train_loss:.4f}, train_acc {train_acc:.4f}, valid_loss {valid_loss:.4f}, valid_acc {valid_acc:.4f} valid_bacc {valid_bacc:.4f} valid_f1 {valid_f1:.4f}")
    logger.info(f"best epoch {best_epoch} | best_valid_loss {best_loss:.4f} best_valid_acc {best_acc:.4f} best_valid_bacc {best_bacc:.4f} best_valid_f1 {best_f1:.4f}")
    return best_loss


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    # set seed
    if hasattr(args, 'seed'):
        set_seed(args.seed)
        print(f"set seed as {args.seed}")

    # save checkpoint
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # set logger
    global logger
    logger = get_logger(args)
    logger.info(args)
    disable_logger()

    dataset = Dataset(args, logger)

    if args.train_hparam_tune: # hyperparameter optimization
        objective = HOP(args, dataset, mode='train')
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30)
        logger.info(study.best_trial)
        for key, value in study.best_trial.params.items():
            setattr(args, key, value)
    source_model, _ = get_pretrained_model(args, dataset)

    if set(args.method).intersection(['calibrator']):
        if args.train_hparam_tune: # hyperparameter optimization
            objective = HOP(args, dataset, mode='posttrain', source_model=source_model)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=30)
            logger.info(study.best_trial)
            print(f"{study.best_trial.params=}")
            for key, value in study.best_trial.params.items():
                setattr(args, key, value)
        calibrator, _ = get_calibrator(args, dataset, source_model)



if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error(traceback.format_exc())