import os
from copy import deepcopy
import hydra
from omegaconf import OmegaConf
# from tqdm import tqdm
# import wandb

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from data.dataset import *
from model.mlp import MLP
from sam import SAM
from utils import utils
from utils.utils import softmax_entropy


def train(args, model, optimizer, dataset, loss_fn, logger):
    device = args.device

    best_model, best_loss = None, float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_len = 0, 0, 0
        model.train().to(device)
        for train_x, train_y in dataset.train_loader:
            train_x, train_y = train_x.float().to(device), train_y.float().to(device)
            estimated_y = model(train_x)
            loss = loss_fn(estimated_y, train_y)

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
                valid_x, valid_y = valid_x.float().to(device), valid_y.float().to(device)
                estimated_y = model(valid_x)
                loss = loss_fn(estimated_y, valid_y)

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
    global EMA, ENTROPY_LIST, GRADIENT_NORM_LIST, ENTROPY_LIST_NEW, GRADIENT_NORM_LIST_NEW
    optimizer.zero_grad()

    if "sar" in args.method:
        estimated_y = model(x)
        entropy_first = softmax_entropy(estimated_y)
        filter_id = torch.where(entropy_first < 0.4 * np.log(estimated_y.shape[-1]))
        entropy_first = softmax_entropy(estimated_y)
        loss = entropy_first.mean()
        loss.backward()

        # for visualization
        # ENTROPY_LIST.append(loss.item())
        # total_norm = 0
        # for p in [p for p in model.parameters() if p.grad is not None and p.requires_grad]:
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # GRADIENT_NORM_LIST.append(total_norm) 

        optimizer.first_step(zero_grad=True)

        new_estimated_y = model(x)
        entropy_second = softmax_entropy(new_estimated_y)
        entropy_second = entropy_second[filter_id]
        filter_id = torch.where(entropy_second < 0.4 * np.log(estimated_y.shape[-1]))
        loss_second = entropy_second[filter_id].mean()
        loss_second.backward()

        EMA = 0.9 * EMA + (1 - 0.9) * loss_second.item() if EMA != None else loss_second.item()

        # for visualization
        ENTROPY_LIST_NEW.append(loss_second.item())
        total_norm = 0
        for p in [p for p in model.parameters() if p.grad is not None and p.requires_grad]:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        GRADIENT_NORM_LIST_NEW.append(total_norm)

        optimizer.second_step(zero_grad=True)
    if "em" in args.method:
        estimated_y = model(x)
        loss = softmax_entropy(estimated_y / args.temp).mean()
        loss.backward()
        optimizer.step(zero_grad=True)


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        # if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
        #     for np, p in m.named_parameters():
        #         if np in ['weight', 'bias']:
        #             params.append(p)
        #             names.append(f"{nm}.{np}")
        for np, p in m.named_parameters():
            params.append(p)
            names.append(f"{nm}.{np}")
    print(f"names: {names}")
    return params, names


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    if args.seed:
        utils.set_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = utils.get_logger(args)
    logger.info(OmegaConf.to_yaml(args))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
    optimizer = getattr(torch.optim, args.train_optimizer)(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train_lr)
    loss_fn = nn.CrossEntropyLoss()

    if os.path.exists(os.path.join(args.out_dir, "best_model.pth")) and not args.retrain:
        best_model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
        best_state_dict = torch.load(os.path.join(args.out_dir, "best_model.pth"))
        best_model.load_state_dict(best_state_dict)
        print(f"load pretrained model!")
    else:
        best_model = train(args, model, optimizer, dataset, loss_fn, logger)

    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0
    original_best_model = deepcopy(best_model)
    best_model.eval().requires_grad_(True).to(device)
    original_best_model.eval().requires_grad_(False).to(device)

    global EMA, ENTROPY_LIST, GRADIENT_NORM_LIST, ENTROPY_LIST_NEW, GRADIENT_NORM_LIST_NEW, PREDICTED_LABEL, BATCH_IDX
    EMA = None
    ENTROPY_LIST, ENTROPY_LIST_NEW, GRADIENT_NORM_LIST, GRADIENT_NORM_LIST_NEW, PREDICTED_LABEL, BATCH_IDX = [], [], [], [], [], []
    params, _ = collect_params(best_model)
    if "sar" in args.method:
        test_optimizer = SAM(params, base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        test_optimizer = getattr(torch.optim, args.test_optimizer)(params, lr=args.test_lr)
    # if "sar" in args.method:
    #     test_optimizer = SAM(filter(lambda p: p.requires_grad, best_model.parameters()), base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    # else:
    #     test_optimizer = getattr(torch.optim, args.test_optimizer)(filter(lambda p: p.requires_grad, best_model.parameters()), lr=args.test_lr)
    original_model_state, original_optimizer_state, _ = copy_model_and_optimizer(best_model, test_optimizer, scheduler=None)

    for batch_idx, (test_x, test_y) in enumerate(dataset.test_loader):
        if args.episodic or (EMA != None and EMA < 0.2):
            best_model, test_optimizer, _ = load_model_and_optimizer(best_model, test_optimizer, None, original_model_state, original_optimizer_state, None)
            print("reset model!")

        test_x, test_y = test_x.float().to(device), test_y.float().to(device)
        test_len += test_x.shape[0]

        estimated_y = original_best_model(test_x)
        loss = loss_fn(estimated_y, test_y)
        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, test_x, best_model, test_optimizer)

        estimated_y = best_model(test_x)
        loss = loss_fn(estimated_y, test_y)
        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        PREDICTED_LABEL.extend(torch.argmax(estimated_y, dim=-1).tolist())
        BATCH_IDX.extend([batch_idx for _ in range(estimated_y.shape[0])])

    # gradient norm vs entropy
    # plt.figure(dpi=1000)
    # plt.scatter(GRADIENT_NORM_LIST, ENTROPY_LIST, c=["#ff7f0e"], s=1.5)
    # plt.xlabel("Gradient Norm")
    # plt.ylabel("Entropy")
    # plt.savefig(f"gn_vs_ent_em_{args.dataset}.png")

    # plt.figure(dpi=1000)
    # plt.scatter(GRADIENT_NORM_LIST_NEW, ENTROPY_LIST_NEW, c=["#ff7f0e"], s=1.5)
    # plt.xlabel("Gradient Norm")
    # plt.ylabel("Entropy")
    # plt.savefig(f"gn_vs_ent_sar_{args.dataset}.png")

    # print(f"len(list(range(batch_idx))): {len(list(range(batch_idx)))}")
    # print(f"len(GRADIENT_NORM_LIST_NEW): {len(GRADIENT_NORM_LIST_NEW)}")

    # visualize gradient norm and predicted label distribution
    plt.figure(dpi=1000)
    plt.scatter(list(range(batch_idx + 1)), GRADIENT_NORM_LIST_NEW, c=["#ff7f0e"], s=1.5)
    plt.xlabel("Online Batch")
    plt.ylabel("Gradient Norm (Avg.)")
    plt.savefig(f"batch_idx_vs_gradient_norm.png")

    print(f"len(BATCH_IDX): {len(BATCH_IDX)}")
    print(f"len(PREDICTED_LABEL): {len(PREDICTED_LABEL)}")

    plt.figure(dpi=1000)
    plt.scatter(BATCH_IDX, PREDICTED_LABEL, c=["#ff7f0e"], s=1.5)
    plt.xlabel("Online Batch")
    plt.ylabel("Predicted Label")
    plt.savefig(f"batch_idx_vs_predicted_label.png")

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")



if __name__ == "__main__":
    main()