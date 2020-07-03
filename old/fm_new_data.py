import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
import os
np.random.seed(1024)
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# To compute probalities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getDataLoader(data_path, device=torch.device("cpu"), batch_size=32):
    # load train data
    train_df = pd.read_csv(data_path + 'dota_train_binary_heroes.csv', index_col='match_id_hash')
    # test_df = pd.read_csv('dota_train_binary_heroes.csv', index_col='match_id_hash')
    target = pd.read_csv(data_path + 'train_targets.csv', index_col='match_id_hash')
    y = target['radiant_win'].values.astype(np.float32)
    y = y.reshape(-1, 1)

    # convert to 32-bit numbers to send to GPU
    X_train = train_df.values.astype(np.float32)
    # X_test = test_df.values.astype(np.float32)

    # train_preds = np.zeros(y.shape)
    # test_preds = np.zeros((X_test.shape[0], 1))

    # X_tensor, X_test, y_tensor = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(
    #     y).to(device)

    X_tensor,  y_tensor = torch.from_numpy(X_train).to(device), torch.from_numpy(
        y).to(device)

    train_num = int(X_tensor.shape[0] * 0.9)
    train_set = TensorDataset(X_tensor[:train_num], y_tensor[:train_num])
    valid_set = TensorDataset(X_tensor[train_num:], y_tensor[train_num:])

    loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
               'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=False)}

    input_size = X_tensor.shape[1]
    return input_size, loaders

class FM(torch.nn.Module):
    def __init__(self, input_size=2, n_factors=10, lr=0.01, weight_decay=0.):
        super(FM, self).__init__()

        # get factor number
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(input_size, n_factors),requires_grad=True)
        self.lin = nn.Linear(input_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out

    def fit(self, loaders, epochs=5):
        # training cycle
        best_score = 0.
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}

            for phase in ['train', 'valid']:

                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()
                    out = self.forward(batch_x)
                    loss = nn.BCEWithLogitsLoss()(out, batch_y)
                    losses[phase] += loss.item() * batch_x.size(0)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()

                y_pred,y_true = [],[]
                for batch_x, batch_y in loaders['valid']:
                    out = self.forward(batch_x)
                    preds = sigmoid(out.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += batch_y.tolist()
                epoch_score = roc_auc_score(np.array(y_true), np.array(y_pred))


            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} valid roc auc {epoch_score:.3f}')
        return

if __name__ == '__main__':
    input_size, loader=getDataLoader('../data/fm_kaggle/')
    model = FM(input_size)
    model.fit(loader,5)
