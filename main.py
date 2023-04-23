import os
import copy
import hydra
# from tqdm import tqdm
# import wandb

import torch
import torch.nn as nn

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
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

        logger.info(f"epoch {epoch}, train_loss {train_loss / train_len:.4f}, train_acc {train_acc / train_len:.4f}, valid_loss {valid_loss / valid_len:.4f}, valid_acc {valid_acc / valid_len:.4f}")
    return best_model


def forward_and_adapt(args, x, model, optimizer):
    global EMA
    optimizer.zero_grad()

    if "sar" in args.method:
        estimated_y = model(x)
        entropy_first = softmax_entropy(estimated_y)
        filter_id = torch.where(entropy_first < 0.4 * estimated_y.shape[-1])
        entropy_first = softmax_entropy(estimated_y)
        loss = entropy_first.mean()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        new_estimated_y = model(x)
        entropy_second = softmax_entropy(new_estimated_y)
        entropy_second = entropy_second[filter_id]
        filter_id = torch.where(entropy_second < 0.4 * estimated_y.shape[-1])
        loss_second = entropy_second[filter_id].mean()
        loss_second.backward()
        optimizer.second_step(zero_grad=True)

        EMA = 0.9 * EMA + (1 - 0.9) * loss_second.item() if EMA != None else loss_second.item()

    if "em" in args.method:
        estimated_y = model(x)
        loss = softmax_entropy(estimated_y / args.temp).mean()
        loss.backward()
        optimizer.step()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    if args.seed:
        utils.set_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = utils.get_logger(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = args.device
    dataset = Dataset(args)

    model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
    optimizer = getattr(torch.optim, args.train_optimizer)(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train_lr)
    loss_fn = nn.CrossEntropyLoss()

    if os.path.exists(os.path.join(args.out_dir, "best_model.pth")):
        best_model = MLP(input_dim=dataset.in_dim, output_dim=dataset.out_dim, hidden_dim=256, n_layers=4)
        best_state_dict = torch.load(os.path.join(args.out_dir, "best_model.pth"))
        best_model.load_state_dict(best_state_dict)
    else:
        best_model = train(args, model, optimizer, dataset, loss_fn, logger)

    test_loss_before, test_acc_before, test_loss_after, test_acc_after, test_len = 0, 0, 0, 0, 0
    original_best_model = copy.deepcopy(best_model)
    original_best_model.eval().requires_grad_(True).to(device)
    best_model.eval().requires_grad_(True).to(device)

    if "sar" in args.method:
        global EMA
        EMA = None
        optimizer = SAM(filter(lambda p: p.requires_grad, best_model.parameters()), base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
    else:
        optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, best_model.parameters()), lr=args.tta_lr)

    for test_x, test_y in dataset.test_loader:
        if args.episodic or (EMA != None and EMA < 0.2):
            best_model = copy.deepcopy(original_best_model)
            optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, best_model.parameters()), lr=args.tta_lr)

        test_x, test_y = test_x.float().to(device), test_y.float().to(device)

        estimated_y = original_best_model(test_x)
        loss = loss_fn(estimated_y, test_y)

        test_loss_before += loss.item() * test_x.shape[0]
        test_acc_before += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()

        if "sar" in args.method:
            optimizer = SAM(filter(lambda p: p.requires_grad, best_model.parameters()), base_optimizer=getattr(torch.optim, args.test_optimizer), lr=args.test_lr)
        else:
            optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, best_model.parameters()), lr=args.tta_lr)

        for _ in range(1, args.num_steps + 1):
            forward_and_adapt(args, test_x, best_model, optimizer)

        estimated_y = best_model(test_x)
        loss = loss_fn(estimated_y, test_y)

        test_loss_after += loss.item() * test_x.shape[0]
        test_acc_after += (torch.argmax(estimated_y, dim=-1) == torch.argmax(test_y, dim=-1)).sum().item()
        test_len += test_x.shape[0]

    logger.info(f"test_loss before adaptation {test_loss_before / test_len:.4f}, test_acc {test_acc_before / test_len:.4f}")
    logger.info(f"test_loss after adaptation {test_loss_after / test_len:.4f}, test_acc {test_acc_after / test_len:.4f}")



if __name__ == "__main__":
    main()