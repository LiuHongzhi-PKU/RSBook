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
    train_df=train_df[:500]
    y=y[:500]

    y = y.reshape(-1, 1)

    # convert to 32-bit numbers to send to GPU
    X_train = train_df.values.astype(np.float32)
    # X_test = test_df.values.astype(np.float32)

    # train_preds = np.zeros(y.shape)
    # test_preds = np.zeros((X_test.shape[0], 1))

    # X_tensor, X_test, y_tensor = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(
    #     y).to(device)
    # print(train_df.shape,train_df.columns)
    # print(train_df.columns[:115])
    # print(train_df.columns[115:])
    # import sys
    # sys.exit()

    X_tensor,  y_tensor = torch.from_numpy(X_train).to(device), torch.from_numpy(
        y).to(device)

    train_num = int(X_tensor.shape[0] * 0.9)
    train_set = TensorDataset(X_tensor[:train_num], y_tensor[:train_num])
    valid_set = TensorDataset(X_tensor[train_num:], y_tensor[train_num:])

    loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
               'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=False)}

    input_size = X_tensor.shape[1]
    feature_field=[0 if i<115 else 1 for i in range(230)]
    field_size=2
    return (input_size,field_size,feature_field), loaders

class FFM(torch.nn.Module):
    def __init__(self, input_size=2, field_size=2, feature_field=[], n_factors=10, lr=0.01, weight_decay=0.):
        super(FFM, self).__init__()

        # get factor number
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.feature_size = input_size
        self.feature_embeddings = nn.Parameter(torch.randn(input_size, field_size, n_factors),requires_grad=True)
        self.lin = nn.Linear(input_size, 1)
        self.feature_field = feature_field

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        self.device=torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x=x.to(self.device)
        first_order = self.lin(x)

        second_order = torch.tensor([[0]] * x.shape[0], dtype=torch.float32, device=self.device)
        for i in range(self.feature_size):
            for j in range(i + 1, self.feature_size):
                vifj = self.feature_embeddings[torch.tensor([i]), torch.tensor([self.feature_field[j]]), :]
                vjfi = self.feature_embeddings[torch.tensor([j]), torch.tensor([self.feature_field[i]]), :]
                second_order += torch.sum(torch.mul(vifj, vjfi), dim=1, keepdim=True) * \
                                x[:, i][:,np.newaxis] * \
                                x[:,j][:, np.newaxis]

        out = first_order + second_order

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
                pbar = tqdm(enumerate(loaders[phase]),
                            total=len(loaders[phase]),
                            desc='({0:^3})'.format(epoch))
                for batch_id,(batch_x, batch_y) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                #     print('---------')
                #     print(batch_x)
                #     import sys
                #     sys.exit()
                    self.optimizer.zero_grad()
                    out = self.forward(batch_x)
                    batch_y = batch_y.to(self.device)
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
    model = FFM(input_size[0],input_size[1],input_size[2])
    model.fit(loader,5)


# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python ffm_new_data.py
# ( 0 ): 100%|########################################################################################################################################################################| 15/15 [01:42<00:00,  6.80s/it]
# ( 0 ): 100%|##########################################################################################################################################################################| 2/2 [00:06<00:00,  3.44s/it]
# epoch 1 train loss: 8.484 valid loss 8.797 valid roc auc 0.540
# ( 1 ): 100%|########################################################################################################################################################################| 15/15 [01:42<00:00,  6.80s/it]
# ( 1 ): 100%|##########################################################################################################################################################################| 2/2 [00:06<00:00,  3.47s/it]
# epoch 2 train loss: 5.739 valid loss 8.634 valid roc auc 0.548
# ( 2 ): 100%|########################################################################################################################################################################| 15/15 [01:42<00:00,  6.79s/it]
# ( 2 ): 100%|##########################################################################################################################################################################| 2/2 [00:06<00:00,  3.39s/it]
# epoch 3 train loss: 4.012 valid loss 8.494 valid roc auc 0.542
# ( 3 ):  13%|######################5