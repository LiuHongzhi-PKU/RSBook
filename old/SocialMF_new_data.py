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
    data_fields = ['user_id', 'item_id', 'rating']
    # all data file
    data_df = pd.read_table(data_path+'ratings_data.txt', names=data_fields,sep=' ')
    social_df = pd.read_table(data_path+'trust_data.txt', names=['user_id','user_id2','trust'],sep=' ')
    social_df.index=range(social_df.shape[0])

    data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)

    social_user = defaultdict(set)
    for (user, user2, record) in social_df.itertuples(index=False):
        social_user.setdefault(user - 1, set())
        social_user[user - 1].add(user2 - 1)

    # ua_base = allData.sample(n=90570, replace=False)
    df_train = data_df.sample(n=int(len(data_df) * 0.9), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    # get user number
    n_users = len(set(data_df['user_id'].values))
    # get item number
    n_items = len(set(data_df['item_id'].values))
    n_users = max(set(data_df['user_id'].values))
    n_items = max(set(data_df['item_id'].values))


    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items, social_user), loaders

class LFM(torch.nn.Module):
    def __init__(self, n_users, n_items, social_user, n_factors=10, dropout_p=0.02, lr=0.01, weight_decay=0., sparse=False):
        super(LFM, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.social_user = social_user

        # get factor number
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        self.social_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.social_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)


        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sparse = sparse

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)


    def forward(self, users, items):
        # print(users)
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def social_forward(self, users, sim_embs):
        user_embs = self.user_embeddings(users)
        return ((user_embs-sim_embs)*(user_embs-sim_embs)).sum(dim=1).sum()


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
                    sim_emb = []
                    row_2=[]
                    for i in range(row.shape[0]):
                        # print(row[i], self.social_user[int(row[i])])
                        if len(self.social_user[int(row[i])])>0:
                            tmp_emb = torch.zeros(self.n_factors, dtype=torch.float32)
                            for user_2 in self.social_user[int(row[i])]:
                                user_2=torch.tensor(user_2).long()
                                # tmp_emb += self.social_embeddings(user_2) / len(self.social_user[int(row[i])])
                                tmp_emb += self.user_embeddings(user_2) / len(self.social_user[int(row[i])])
                            sim_emb.append(tmp_emb)
                            row_2.append(row[i])
                    row_2 = torch.tensor(row_2).long()
                    sim_emb=torch.stack(sim_emb)
                    # sim_emb = torch.tensor(sim_emb)#.float()
                    # sim_emb = sim_emb.float()
                    # sim_loss = self.social_forward(row_2, sim_emb)
                    user_embs = self.user_embeddings(row_2)
                    sim_loss=((user_embs - sim_emb)*(user_embs - sim_emb)).sum()

                    loss = nn.MSELoss(reduction='sum')(preds, val)


                    loss += sim_loss

                    losses[phase] += loss.item()
                    batch_loss = loss.item() / row.size()[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
        return

if __name__ == '__main__':

    # data_path='../data/epinions/'
    # # load train data
    # data_fields = ['user_id', 'item_id', 'rating']
    # # all data file
    #
    # data_df = pd.read_table(data_path+'ratings_data.txt', names=data_fields,sep=' ')
    # print(data_df[:5])
    # social_df = pd.read_table(data_path+'trust_data.txt', names=['user_id','user_id2','trust'],sep=' ')
    # social_df.index=range(social_df.shape[0])
    # print(social_df)
    # print(1)
    input_size, loader=getDataLoader("../data/epinions/")

    model = LFM(input_size[0],input_size[1], input_size[2])
    model.fit(loader,5)

# Initialize end.The user number is:49289,item number is:139738
# ( 0 ): 100%|##########| 1870/1870 [01:11<00:00, 23.63it/s, train_loss=29.4]
# ( 0 ): 100%|##########| 208/208 [00:03<00:00, 68.16it/s, train_loss=17.3]
# ( 1 ):   0%|          | 0/1870 [00:00<?, ?it/s]epoch 1 train loss: 33.139 valid loss 24.301
# ( 1 ): 100%|##########| 1870/1870 [01:26<00:00, 21.68it/s, train_loss=14]
# ( 1 ):  99%|#########8| 205/208 [00:02<00:00, 69.78it/s, train_loss=20.3]epoch 2 train loss: 11.465 valid loss 20.062
# ( 1 ): 100%|##########| 208/208 [00:02<00:00, 70.82it/s, train_loss=15.4]
# ( 2 ): 100%|##########| 1870/1870 [01:28<00:00, 21.63it/s, train_loss=8.54]
# ( 2 ): 100%|##########| 208/208 [00:02<00:00, 71.57it/s, train_loss=14.4]
# ( 3 ):   0%|          | 0/1870 [00:00<?, ?it/s]epoch 3 train loss: 5.339 valid loss 17.935


# Initialize end.The user number is:49289,item number is:139738
# ( 0 ):   1%|          | 16/1870 [00:30<56:20,  1.82s/it, train_loss=39.5]

# 速度太慢了