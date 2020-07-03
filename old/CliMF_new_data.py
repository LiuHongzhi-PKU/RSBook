import random
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

IMPLICT=True
SMALL=False

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

def getDataLoader(data_path, batch_size=32):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)

    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 4).astype(np.float32)

    # ua_base = allData.sample(n=90570, replace=False)
    df_train = data_df.sample(n=int(len(data_df) * 0.9), replace=False)


    df_test = data_df.drop(df_train.index, axis=0)

    user_item = defaultdict(dict)
    for (user, item, record, timestamp) in df_train.itertuples(index=False):
        user_item.setdefault(user - 1, {})
        user_item[user - 1][item - 1] = record

    # get user number
    n_users = max(set(data_df['user_id'].values))
    # get item number
    n_items = max(set(data_df['item_id'].values))

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items,user_item ), loaders

class LFM(torch.nn.Module):
    def __init__(self, n_users, n_items,user_item, n_factors=10, dropout_p=0.02, lr=0.01, weight_decay=0., sparse=False):
        super(LFM, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.user_item=user_item

        # get factor number
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sparse = sparse

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)


    def forward(self, users, items):
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()

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
                for batch_idx, ((row, col), val) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    row = row.long()
                    col = col.long()
                    val = val.float()
                    preds = self.forward(row, col)
                    loss = torch.log(torch.sigmoid(preds)).sum()

                    for i in range(row.shape[0]):
                        user=row[i]
                        for item in self.user_item[user]:
                            loss+=torch.log(1-torch.sigmoid(self.forward(user,item)-preds[i]))
                    loss*=-1
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / row.size()[0]
                    pbar.set_postfix(train_loss=batch_loss)

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

                for ((row, col), val) in loaders['valid']:
                    row = row.long()
                    col = col.long()
                    val = val.float()
                    preds = self.forward(row, col)
                    if IMPLICT:
                        preds = sigmoid(preds.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += val.tolist()
                y_true,y_pred=np.array(y_true), np.array(y_pred)
                if IMPLICT:
                    epoch_score = roc_auc_score(y_true,y_pred)
                    score='auc'
                else:
                    epoch_score=sum([(y - x) ** 2 for x, y in zip(y_true, y_pred)]) / len(y_pred)
                    score='mse'

            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')


            # if ((epoch + 1) % 1) == 0:
            #     print(
            #         f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = LFM(input_size[0],input_size[1],input_size[2],weight_decay=0.001)
    model.fit(loader,5)

# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python CliMF_new_data.py
# Initialize end.The user number is:943,item number is:1682
# ( 0 ): 100%|#################################################################################################################################################| 2813/2813 [00:37<00:00, 75.48it/s, train_loss=0.0376]
# ( 0 ): 100%|###################################################################################################################################################| 313/313 [00:03<00:00, 84.17it/s, train_loss=0.0356]
# epoch 1 train loss: 0.246 valid loss 0.044 mse 1.819
# ( 1 ): 100%|#################################################################################################################################################| 2813/2813 [00:35<00:00, 79.12it/s, train_loss=0.0346]
# ( 1 ): 100%|###################################################################################################################################################| 313/313 [00:03<00:00, 85.64it/s, train_loss=0.0285]
# epoch 2 train loss: 0.038 valid loss 0.035 mse 1.912
# ( 2 ): 100%|#################################################################################################################################################| 2813/2813 [00:35<00:00, 79.41it/s, train_loss=0.0147]
# ( 2 ): 100%|###################################################################################################################################################| 313/313 [00:03<00:00, 88.02it/s, train_loss=0.0264]
# epoch 3 train loss: 0.034 valid loss 0.034 mse 2.079
# ( 3 ): 100%|#################################################################################################################################################| 2813/2813 [00:36<00:00, 77.84it/s, train_loss=0.0237]
# ( 3 ): 100%|###################################################################################################################################################| 313/313 [00:03<00:00, 86.21it/s, train_loss=0.0255]
# epoch 4 train loss: 0.034 valid loss 0.034 mse 2.061
# ( 4 ): 100%|#################################################################################################################################################| 2813/2813 [00:36<00:00, 77.11it/s, train_loss=0.0296]
# ( 4 ): 100%|###################################################################################################################################################| 313/313 [00:03<00:00, 88.33it/s, train_loss=0.0289]
# epoch 5 train loss: 0.034 valid loss 0.034 mse 2.069



# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python CliMF_new_data.py
# Initialize end.The user number is:943,item number is:1682
# ( 0 ): 100%|#################################################################################################################################################| 2813/2813 [00:43<00:00, 64.70it/s, train_loss=0.0376]
# ( 0 ): 100%|###################################################################################################################################################| 313/313 [00:04<00:00, 69.43it/s, train_loss=0.0356]
# epoch 1 train loss: 0.246 valid loss 0.044 auc 0.544
# ( 1 ): 100%|#################################################################################################################################################| 2813/2813 [00:44<00:00, 63.40it/s, train_loss=0.0346]
# ( 1 ): 100%|###################################################################################################################################################| 313/313 [00:04<00:00, 68.08it/s, train_loss=0.0285]
# epoch 2 train loss: 0.038 valid loss 0.035 auc 0.547
# ( 2 ): 100%|#################################################################################################################################################| 2813/2813 [00:44<00:00, 62.93it/s, train_loss=0.0147]
# ( 2 ): 100%|###################################################################################################################################################| 313/313 [00:04<00:00, 67.65it/s, train_loss=0.0264]
# epoch 3 train loss: 0.034 valid loss 0.034 auc 0.545
# ( 3 ): 100%|#################################################################################################################################################| 2813/2813 [00:44<00:00, 63.00it/s, train_loss=0.0237]
# ( 3 ): 100%|###################################################################################################################################################| 313/313 [00:04<00:00, 66.80it/s, train_loss=0.0255]
# epoch 4 train loss: 0.034 valid loss 0.034 auc 0.549
# ( 4 ): 100%|#################################################################################################################################################| 2813/2813 [00:44<00:00, 65.30it/s, train_loss=0.0296]
# ( 4 ): 100%|###################################################################################################################################################| 313/313 [00:04<00:00, 69.79it/s, train_loss=0.0289]
# epoch 5 train loss: 0.034 valid loss 0.034 auc 0.550
