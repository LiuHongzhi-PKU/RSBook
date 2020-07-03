import random
import math
import pandas as pd
import numpy as np
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

def drop_df(df):
    pos_cnt = df.groupby('user_id', as_index=False)['rating'].agg({"pos_cnt": 'sum'})
    tot_cnt = df.groupby('user_id', as_index=False)['rating'].agg({"tot_cnt": 'count'})
    df = pd.merge(df, pos_cnt, on=['user_id'], how='left')
    df = pd.merge(df, tot_cnt, on=['user_id'], how='left')
    df = df[(df.pos_cnt > 0) & (df.tot_cnt > df.pos_cnt)]
    df = df.drop(['pos_cnt', 'tot_cnt'], axis=1)
    return df

def getDataLoader(data_path, batch_size=32):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 5).astype(np.float32)

    for i in ['user_id', 'item_id']:
        data_df[i] = data_df[i].map(dict(zip(data_df[i].unique(), range(1,data_df[i].nunique()+1))))

    # print(data_df.describe())

    df_train = data_df.sample(n=int(len(data_df) * 0.9), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    df_train=drop_df(df_train)
    df_test = drop_df(df_test)
    # get user number
    n_users = max(data_df['user_id'].values)
    # get item number
    n_items = max(data_df['item_id'].values)

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    # 硬编码测试
    hinSim={'II':torch.FloatTensor([0.1,0.1,0.1,0.1,0.1]*n_items).reshape(-1,5)} #  前5个
    metaPath = {'PI': ['II']}  # 也有可能是 IUI
    hinSimI={'II':torch.LongTensor([1,2,3,4,5]*n_items).reshape(-1,5)}
    return (n_users,n_items, hinSim, hinSimI,metaPath), loaders

class SemRec(torch.nn.Module):
    def __init__(self, n_users, n_items, hinSim, hinSimI, metaPath, n_factors=10, lr=0.01, lambda_0=0.001,lambda_I=1, device=torch.device("cpu")):
        super(SemRec, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors # 隐特征维度
        self.hinSim = hinSim # (元路径，前k相似度值)
        self.hinSimI = hinSimI  # (元路径，前k相似度下标)
        self.metaPath = metaPath # (UU or II, 元路径)
        self.lambda_I = lambda_I
        self.device = device


        # 模型参数
        self.U = {}#nn.Embedding(self.n_users, self.n_factors)
        self.V = {}#nn.Embedding(self.n_items, self.n_factors)

        self.path2id = {}
        for pp in self.metaPath['UI']:
            self.path2id[pp]=torch.LongTensor([len(self.path2id)])
            self.U[pp] = nn.Embedding(self.n_users, self.n_factors)
            self.V[pp] = nn.Embedding(self.n_items, self.n_factors)

        self.W_U=nn.Embedding(self.n_users, len(self.path2id)) # 可能需要注册参数

        self.optimizer = torch.optim.SGD(self.parameters(),
                                   lr=lr, weight_decay=lambda_0)
        self=self.to(self.device)


    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        # print(items)
        preds=torch.zeros(users.shape,dtype=torch.float32).to(self.device)
        for p in self.metaPath['UI']:
            ues = self.U[p](users)
            uis = self.V[p](items)
            preds = preds + self.W_U(users)[self.path2id[p]](ues * uis).sum(dim=-1)
        return preds

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
                            desc='({0}:{1:^3})'.format(phase, epoch+1))
                for batch_idx, ((batch_U, batch_I), batch_y) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    batch_U = batch_U.long()
                    batch_I = batch_I.long()
                    batch_y = batch_y.float().to(self.device)
                    y_pred = self.forward(batch_U, batch_I)
                    loss = nn.MSELoss(reduction='sum')(y_pred, batch_y)

                    WI_sum=0
                    loss_PI=0
                    for p in self.metaPath['UU']:
                        WI_sum = WI_sum + torch.exp(self.W_I(self.path2id[p])).squeeze()
                    for p in self.metaPath['UU']:
                        hin_value = self.hinSim[p][batch_I] # I*K->B*K  前K个的取值
                        hin_index = self.hinSimI[p][batch_I] # I*K -> B*K 前K个的下标
                        w_embeddings = self.W_U(hin_index.reshape(-1,1)).reshape(hin_index.shape[0],hin_index.shape[1],-1) # B*K*1

                        hin_reg = (torch.exp(self.W_U[batch_U].reshape(batch_U.shape[0],1,-1))/WI_sum-hin_value*w_embeddings).pow(2).sum(-1)

                        # hin_pow = (self.V(batch_I).reshape(batch_I.shape[0],1,-1)-hin_embeddings).pow(2).sum(-1) # B*K*1
                        # print(hin_pow.shape)
                        loss_PI = loss_PI + (self.lambda_I  * hin_reg).sum()

                        # loss_PI += (self.lambda_I * 1 * hin_value * hin_pow).sum()
                    # print(loss)
                    # print(loss_PI)
                    # print(WI_sum)
                    # print(loss_PI/WI_sum)
                    loss+=loss_PI/WI_sum
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / batch_y.shape[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)
            # print('epoch done')
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
    model = SemRec(input_size[0],input_size[1], input_size[2], input_size[3],input_size[4])
    model.fit(loader,10)


