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
    UIUI = UI.dot(UI.transpose()).dot(UI)
    return (n_users,n_items), loaders

class FMG(torch.nn.Module):
    def __init__(self, n_users, n_items, input_size, L=2, n_factors=10, lr=0.01,
                 lambda_0=0.001,lambda_w=0.5, lambda_v=0.5,
                 device=torch.device("cpu")):
        super(FMG, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors # 隐特征维度
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.L=L
        self.device = device


        # 模型参数
        self.w = nn.Parameter(torch.randn(input_size, 1), requires_grad=True)
        self.v = nn.Parameter(torch.randn(input_size, self.n_factors), requires_grad=True)
        self.w0 = nn.Parameter(torch.randn(1), requires_grad=True)

        self.optimizer = torch.optim.SGD(self.parameters(),
                                   lr=lr)
        self=self.to(self.device)


    # 预测评分
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2
        out_inter = 0.5 * (out_1 - out_2)

        out_lin = torch.matmul(x, self.w)+self.w0
        preds = out_inter + out_lin

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
                for batch_idx, (batch_X, batch_y) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    batch_X = batch_X.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    y_pred = self.forward(batch_X)
                    loss = nn.MSELoss(reduction='sum')(y_pred, batch_y)



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
    model = FMG(input_size[0],input_size[1],
                   input_size[2], input_size[3],input_size[4])
    model.fit(loader,10)

