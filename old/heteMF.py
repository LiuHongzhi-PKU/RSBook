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

K=10
def calSim(path, M, hinSim, hinSimI):
    hinSim[path]=np.empty((M.shape[0],K))
    hinSimI[path]=np.empty((M.shape[0],K),dtype=np.int)
    M=M.tocsc()
    col_sum={}
    for i in range(M.shape[1]):
        col_sum[i]=M.getcol(i).toarray().sum()
    M=M.tocsr()
    row_sum={}
    for i in range(M.shape[0]):
        row_sum[i]=M.getrow(i).toarray().sum()
    for i in range(M.shape[0]):
        M_i=M.getrow(i).toarray()
        sim=[]
        for j in range(M_i.shape[1]):
            M_ij=M_i[0][j]
            M_i_=row_sum[i]
            M_j_=col_sum[j]

            S_ij=2*M_ij/(M_i_+M_j_)
            sim.append(S_ij)
        sim=np.array(sim)
        ids=np.argsort(-sim)
        hinSimI[path][i]=ids[:K]
        hinSim[path][i]=sim[ids[:K]]
    hinSim[path]=torch.from_numpy(hinSim[path]).float()
    hinSimI[path] = torch.from_numpy(hinSimI[path]).long()
    # print(hinSim[path].dtype, hinSimI[path].dtype)
    # xxx


# 使用每次getrow(i) 需要10min 速度太慢了
# 可以优化 预处理每个的row sum与col sum

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
    train_loader = torch.utils.data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    from scipy.sparse import coo_matrix
    # S = dok_matrix((5, 5), dtype=np.float32)
    data = np.ones((data_df.shape[0]))
    row = data_df.user_id - 1
    col = data_df.item_id - 1
    UI = coo_matrix((data, (row, col)), shape=(n_users, n_items))
    UIU = UI.dot(UI.transpose())
    IUI = UI.transpose().dot(UI)
    hinSim = {}
    hinSimI = {}
    calSim('UIU', UIU, hinSim, hinSimI)
    calSim('IUI', IUI, hinSim, hinSimI)
    calSim('UI', UI, hinSim, hinSimI)
    # 硬编码测试
    # hinSim={'II':torch.FloatTensor([0.1,0.1,0.1,0.1,0.1]*n_items).reshape(-1,5)} #  前5个
    metaPath = {'II': ['IUI'],'UU':['UIU'],'UI':'UI'}  # 也有可能是 IUI
    # hinSimI={'II':torch.LongTensor([1,2,3,4,5]*n_items).reshape(-1,5)}
    return (n_users,n_items, hinSim, hinSimI,metaPath), loaders

class heteMF(torch.nn.Module):
    def __init__(self, n_users, n_items, hinSim,
                 hinSimI, metaPath, n_factors=10, lr=0.01,
                 lambda_0=0.001,lambda_I=1, device=torch.device("cpu")):
        super(heteMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors # 隐特征维度
        self.hinSim = hinSim # (元路径->前k相似度值)
        self.hinSimI = hinSimI  # (元路径->前k相似度下标)
        self.metaPath = metaPath # (UU or II or UI, 元路径的起点与终点划分)
        self.lambda_I = lambda_I
        self.device = device


        # 模型参数
        self.U = nn.Embedding(self.n_users, self.n_factors)
        self.V = nn.Embedding(self.n_items, self.n_factors)

        self.path2id = {} # 路径编码
        for p in self.metaPath:
            for pp in self.metaPath[p]:
                self.path2id[pp]=torch.LongTensor([len(self.path2id)])
        self.W_I=nn.Embedding(len(self.path2id), 1) # 路径参数

        self.optimizer = torch.optim.SGD(self.parameters(),
                                   lr=lr, weight_decay=lambda_0)
        self=self.to(self.device)


    # 预测评分
    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        ues = self.U(users)
        uis = self.V(items)
        preds = (ues * uis).sum(dim=-1)
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
                    for p in self.metaPath['II']:
                        hin_value = self.hinSim[p][batch_I] # I*K->B*K  前K个的取值
                        hin_index = self.hinSimI[p][batch_I] # I*K -> B*K 前K个的下标
                        hin_embeddings = self.V(hin_index.reshape(-1,1)).reshape(
                            hin_index.shape[0],hin_index.shape[1],-1) # B*K*H

                        hin_reg = (self.V(batch_I).reshape(batch_I.shape[0], 1, -1) - hin_embeddings)\
                            .pow(2).sum(-1)  # B*K*1
                        loss_PI = loss_PI + (self.lambda_I *
                                    torch.exp(self.W_I(self.path2id[p])) * hin_value * hin_reg).sum()
                        WI_sum = WI_sum + torch.exp(self.W_I(self.path2id[p])).squeeze()
                    # loss += loss_PI / WI_sum


                        # self.hinSim[p] # I*I
                        # 一个V_i复制的矩阵，与V做差再平方和 弄成I*1
                        # 只要self.hinSim[p][i] 即1*I
                        # 因此 实际上是self.hinSim[p][col]  B*I
                        # torch.pow(V_i-V)(自动填充)  如果是B*I*1 那就可以利用dim来做
                        # B*H-I*H  -> B*I*H-I*H （这个操作开销太大了） I实际只存前K个的标号，然后用embedding会更好？
                        # B*H - B*K*H 这样就ok了 最终搞完是B*K*H

                        # hin_value = self.hinSim[p][batch_I] # I*K->B*K  前K个的取值
                        # hin_index = self.hinSimI[p][batch_I] # I*K -> B*K 前K个的下标
                        # hin_embeddings = self.V[hin_index.reshape(-1,1)].reshape(hin_index.shape[0],hin_index.shape[1],-1) # B*K*H
                        # hin_pow = (self.V[batch_I]-hin_embeddings).pow(2).sum(-1) # B*K*1
                        # loss += self.lambda_I * self.W_I[p] * hin_value * hin_pow



                        # print(hin_value.shape)
                        # print(hin_index.shape)
                        # print(hin_embeddings.shape)
                        # print(self.V(batch_I).shape)
                        # print(self.V(batch_I).reshape(batch_I.shape[0],1,-1).shape)


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
    model = heteMF(input_size[0],input_size[1],
                   input_size[2], input_size[3],input_size[4])
    model.fit(loader,10)


# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python heteMF.py
# Initialize end.The user number is:943,item number is:1682
# (train: 1 ): 100%|############################################################################################################################################| 2790/2790 [00:44<00:00, 62.28it/s, train_loss=0.331]
# (valid: 1 ): 100%|###############################################################################################################################################| 272/272 [00:03<00:00, 69.17it/s, train_loss=0.36]
# epoch 1 train loss: 2.006 valid loss 0.600 auc 0.500
# (train: 2 ): 100%|#############################################################################################################################################| 2790/2790 [00:44<00:00, 62.24it/s, train_loss=0.27]
# (valid: 2 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 69.48it/s, train_loss=0.307]
# epoch 2 train loss: 0.392 valid loss 0.428 auc 0.507
# (train: 3 ): 100%|############################################################################################################################################| 2790/2790 [00:44<00:00, 62.46it/s, train_loss=0.428]
# (valid: 3 ): 100%|###############################################################################################################################################| 272/272 [00:03<00:00, 69.31it/s, train_loss=0.26]
# epoch 3 train loss: 0.297 valid loss 0.369 auc 0.527
# (train: 4 ): 100%|############################################################################################################################################| 2790/2790 [00:44<00:00, 62.34it/s, train_loss=0.191]
# (valid: 4 ): 100%|##############################################################################################################################################| 272/272 [00:04<00:00, 67.98it/s, train_loss=0.258]
# epoch 4 train loss: 0.260 valid loss 0.335 auc 0.554
# (train: 5 ): 100%|############################################################################################################################################| 2790/2790 [00:44<00:00, 62.34it/s, train_loss=0.203]
# (valid: 5 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 68.98it/s, train_loss=0.251]
# epoch 5 train loss: 0.237 valid loss 0.306 auc 0.587
# (train: 6 ): 100%|############################################################################################################################################| 2790/2790 [00:45<00:00, 61.88it/s, train_loss=0.229]
# (valid: 6 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 69.15it/s, train_loss=0.232]
# epoch 6 train loss: 0.217 valid loss 0.280 auc 0.618
# (train: 7 ): 100%|############################################################################################################################################| 2790/2790 [00:47<00:00, 58.89it/s, train_loss=0.193]
# (valid: 7 ): 100%|##############################################################################################################################################| 272/272 [00:04<00:00, 64.49it/s, train_loss=0.217]
# epoch 7 train loss: 0.199 valid loss 0.259 auc 0.643
# (train: 8 ): 100%|############################################################################################################################################| 2790/2790 [00:46<00:00, 59.92it/s, train_loss=0.148]
# (valid: 8 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 68.99it/s, train_loss=0.206]
# epoch 8 train loss: 0.185 valid loss 0.242 auc 0.662
# (train: 9 ): 100%|#############################################################################################################################################| 2790/2790 [00:44<00:00, 62.19it/s, train_loss=0.22]
# (valid: 9 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 69.02it/s, train_loss=0.199]
# epoch 9 train loss: 0.175 valid loss 0.228 auc 0.675
# (train:10 ): 100%|############################################################################################################################################| 2790/2790 [00:44<00:00, 62.50it/s, train_loss=0.112]
# (valid:10 ): 100%|##############################################################################################################################################| 272/272 [00:03<00:00, 68.87it/s, train_loss=0.192]
# epoch 10 train loss: 0.166 valid loss 0.218 auc 0.687
