import os
import random
from copy import deepcopy
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsolutePercentageError


def get_logger(log_dir):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"log_{time_string}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def normalize(input, normalizer='StandardScaler'): # StandardScaler, MinMaxScaler
    normalizer = getattr(sklearn.preprocessing, normalizer)()
    normalized_input = normalizer.fit_transform(X=input)
    return normalized_input, normalizer


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.kaiming_normal_(m.weight.data)



class INPUT(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X.index)

    def __getitem__(self, index):
        cur = self.X.iloc[index, :].values
        return cur, self.y.iloc[index]


# TODO: use different gen_data functions

class gen_data0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 1)
        )

        for layer in self.layers:
            layer.apply(weight_init)

    def forward(self, x):
        out = self.layers(x.float())
        return out



class gen_data1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        for layer in self.layers:
            layer.apply(weight_init)

    def forward(self, x):
        out = self.layers(x.float())
        return out



class gen_data2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        for layer in self.layers:
            layer.apply(weight_init)

    def forward(self, x):
        out = self.layers(x.float())
        return out



class gen_data3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        for layer in self.layers:
            layer.apply(weight_init)

    def forward(self, x):
        out = self.layers(x.float())
        return out



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

        for layer in self.layers:
            layer.apply(weight_init)

    def forward(self, x):
        out = self.layers(x.float())
        return out

################################################################################
### Hyperparameters
# default setting
tasks = ['regression', 'classification']
gen_models = [gen_data0(), gen_data1(), gen_data2(), gen_data3()]
use_cutoffs = [False, True]
num_repeats = 3
num_data = 100000 # 1000, 10000, 100000
log_dir = "experiments/test"
cutoff  = None # ReLU, None
device = 'cuda'
epochs = 50
use_explicit_input = True

# experiment 1 (use ReLU)
# tasks = ['regression']
# cutoff  = 'ReLU' # ReLU, None
# log_dir = "experiments/ReLU"

# experiment 2 (low-resource setting)
# num_data = 1000
# log_dir = "experiments/low-resource"

# num_data = 10000
# log_dir = "experiments/low-resource"

# experiment 3 (3-layer)
# gen_models = [gen_data2()]
# log_dir = "experiments/specific_generator"

# experiment 4 (explicit_input)
use_explicit_input = True
log_dir = "experiments/explicit_input"
################################################################################

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = get_logger(log_dir)

for task in tasks:
    loss_fn = nn.MSELoss() if task == 'regression' else nn.BCELoss()

    for gen_model in gen_models:
        for num_repeat in range(num_repeats):
            input_feat = np.random.normal(size=(num_data, 10))
            input_df = pd.DataFrame(input_feat)
            input_torch = torch.from_numpy(input_feat)

            for use_cutoff in use_cutoffs:
                input_T1 = input_feat.copy()
                if use_cutoff:
                    input_T1 = input_feat.copy()
                    T1_co = random.sample(range(10), k=5)
                    for c in T1_co:
                        cur = input_T1[:,c]
                        mean = np.mean(cur)
                        if cutoff == 'ReLU':
                            input_T1[:,c] = [0 if v < mean else v for v in cur]
                        else:
                            input_T1[:,c] = [0 if v < mean else 1 for v in cur]

                logger.info(f"model, num_repeat, use_cutoff: {gen_model}, {num_repeat}, {use_cutoff}")

                T1_torch = torch.from_numpy(input_T1)

                if task == 'regression':
                    s1 = gen_model(T1_torch)
                    s1, normalizer = normalize(s1.detach().numpy(), normalizer='StandardScaler')
                else:
                    s1 = torch.round(torch.sigmoid(gen_model(T1_torch))).detach()

                if use_explicit_input:
                    input_feat = input_T1

                T1 = pd.DataFrame(np.column_stack((input_feat, s1)))

                T1_train, T1_test, T1_train_y, T1_test_y = train_test_split(T1.iloc[:, :-1], T1.iloc[:, -1], test_size=0.2, random_state=42)
                T1_valid, T1_test, T1_valid_y, T1_test_y = train_test_split(T1_test, T1_test_y, test_size=0.5, random_state=42)
                train_dataset = INPUT(T1_train, T1_train_y)
                valid_dataset = INPUT(T1_valid, T1_valid_y)
                test_dataset = INPUT(T1_test, T1_test_y)
                T1_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                T1_valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
                T1_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

                model = MLP()
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                best_valid_loss = float('inf')
                best_model = None

                train_losses, valid_losses = [], []
                for epoch in range(1, epochs + 1):
                    train_loss, train_len = 0, 0
                    model.train()
                    for i, (input, target) in enumerate(T1_train_loader):
                        input, target = input.to(device), target.to(device).unsqueeze(1)
                        optimizer.zero_grad()
                        outputs = model(input)
                        if task == 'classification':
                            outputs = torch.sigmoid(outputs)

                        loss = loss_fn(outputs.float(), target.float())
                        loss.backward()
                        optimizer.step()
                        train_loss += outputs.shape[0] * loss.item()
                        train_len += outputs.shape[0]
                    train_loss /= train_len
                    train_losses.append(train_loss)

                    valid_loss, valid_len = 0, 0
                    model.eval()
                    for i, (input, target) in enumerate(T1_valid_loader):
                        input, target = input.to(device), target.to(device).unsqueeze(1)
                        outputs = model(input)
                        if task == 'classification':
                            outputs = torch.sigmoid(outputs)

                        loss = loss_fn(outputs.float(), target.float())
                        valid_loss += outputs.shape[0] * loss.item()
                        valid_len += outputs.shape[0]
                    valid_loss /= valid_len
                    valid_losses.append(valid_losses)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_model = deepcopy(model)

                    logger.info('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'.format(epoch, train_loss, valid_loss))

                full_target, full_output = [], []
                test_corrected, test_len = 0, 0
                best_model.eval()
                with torch.no_grad():
                    for i, (input, target) in enumerate(T1_test_loader):
                        input, target = input.to(device), target.to(device).unsqueeze(1)
                        outputs = best_model(input)
                        if task == 'classification':
                            predicted_label = torch.round(torch.sigmoid(outputs))
                            test_corrected += predicted_label.eq(target.data).sum()
                            test_len += outputs.shape[0]
                        else:
                            full_target.append(target.reshape(-1))
                            full_output.append(outputs.reshape(-1))

                if task == 'regression':
                    full_target = normalizer.inverse_transform(torch.cat(full_target).detach().cpu().numpy().reshape(-1, 1))
                    full_output = normalizer.inverse_transform(torch.cat(full_output).detach().cpu().numpy().reshape(-1, 1))
                    mae = mean_absolute_error(full_target, full_output)
                    mean_abs_percentage_error = MeanAbsolutePercentageError()
                    mape = mean_abs_percentage_error(torch.tensor(full_output), torch.tensor(full_target))

                    logger.info(f'test MAE: {mae}')
                    logger.info(f'test MAPE: {mape}')
                else:
                    logger.info(f"test Accuracy: {test_corrected / test_len}")