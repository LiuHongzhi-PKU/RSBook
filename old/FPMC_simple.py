import random
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

# 控制显式评分或者隐式反馈
IMPLICT=True
# 控制是否使用小数据集（测试能够跑通）
SMALL=True

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
def bpr_loss(preds):
    sig = nn.Sigmoid()
    return -torch.log(sig(preds)).sum()

# 删除没有正负例的样本，保证bpr有正负例
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
        # data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
        data_df = data_df[:100]
    if IMPLICT:
        data_df.rating = (data_df.rating >= 3).astype(np.float32)


    df = data_df

    # get user number
    n_users = max(set(data_df['user_id'].values))
    # get item number
    n_items = max(set(data_df['item_id'].values))

    df.index = range(len(df))  # 重设index
    df = df.sort_values(['user_id', 'timestamp'])

    # 获取每个user对应的basket序列
    user_basket = {}
    basket = {0: [df.iloc[0].item_id]}
    last_user = df.iloc[0].user_id
    last_t = df.iloc[0].timestamp
    for i in range(df.shape[0]):
        if i == 0 or df.iloc[i].rating <= 0: continue

        user = df.iloc[i].user_id
        if user != last_user:
            last_user = user
            basket = {}
        t = df.iloc[i].timestamp
        if t == last_t:
            basket[len(basket) - 1] += [df.iloc[i].item_id]
        else:
            basket[len(basket)] = [df.iloc[i].item_id]
            last_t = t
        user_basket[user] = basket

    res = []
    for i in user_basket:
        res += [(i, j) for j in range(len(user_basket[i]))]
    res = pd.DataFrame(res, columns=['user_id', 't'])

    df_train = res.sample(n=int(len(res) * 0.9), replace=False)
    df_test = res.drop(df_train.index, axis=0)
    loaders = {'train': df_train,
               'valid': df_test,
               }

    return (n_users,n_items, user_basket), loaders

class FPMC(torch.nn.Module):
    def __init__(self, n_users, n_items,user_basket, batch_size=32,n_factors=10, dropout_p=0.02, lr=0.01, weight_decay=0.001, sparse=False, device=torch.device("cpu")):
        super(FPMC, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.user_basket=user_basket
        self.batch_size=batch_size

        # get factor number
        self.n_factors = n_factors
        self.V_IL = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.V_LI = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.V_UL = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.V_LU = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.V_UI = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.V_IU = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        self=self.to(self.device)


    def forward(self, u, i,t):

        i = torch.tensor(i).long()
        u = torch.tensor(u).long()
        # print(i.dtype)
        # print(u.dtype)
        # print(u,i,self.n_users,self.n_items)

        last_basket = []
        if t > 0: last_basket = self.user_basket[u.item()][t - 1]
        u-=1  # 序号从1开始，所以要减1
        i-=1
        res=torch.tensor(0.)

        # 枚举basket转移
        for l in last_basket:
            l = torch.tensor(l).long()-1
            # print('L:',l)
            # self.V_IL(i)
            # self.V_LI(l)
            res += (self.V_IL(i)*self.V_LI(l)).sum(-1)
            res += (self.V_UL(u) * self.V_LU(l)).sum(-1)
        if len(last_basket)>0:res/=len(last_basket)
        res += (self.V_UI(u) * self.V_IU(i)).sum(-1)
        return res

    def get_loss(self, u,i,j,t):
        pos_v = self.forward(u, i, t)
        neg_v = self.forward(u, j, t)

        return bpr_loss(pos_v-neg_v)

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

                pbar = tqdm(loaders[phase].iterrows(),
                            total=loaders[phase].shape[0],
                            desc='({0}:{1:^3})'.format(phase, epoch+1))
                # for batch_idx, ((row, col), val) in pbar:
                batch_loss=[]

                for idx, row in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()
                    u = row.user_id
                    t=row.t
                    i = random.choice(self.user_basket[u][t])
                    found = False
                    while not found:
                        j = np.random.randint(self.n_items)
                        # print(self.user_item.shape,user)
                        if j not in self.user_basket[u][t]:
                            found = True

                    loss = self.get_loss(u,i,j,t)

                    # loss = bpr_loss(preds)

                    # losses[phase] += loss.item()
                    batch_loss += [loss.item()]
                    if len(batch_loss)>=self.batch_size:
                        batch_loss=np.mean(batch_loss)
                        pbar.set_postfix(train_loss=batch_loss)
                        losses[phase] += batch_loss


                        with torch.set_grad_enabled(phase == 'train'):
                            if phase == 'train':
                                loss.backward()
                                #                             scheduler.step()
                                self.optimizer.step()

                losses[phase] /= loaders[phase].shape[0]
            # print('epoch done')
            # after each epoch check if we improved roc auc and if yes - save model
            # with torch.no_grad():
            #     model.eval()
            #
            #     y_pred,y_true = [],[]
            #
            #     for ((row, col), val) in loaders['valid']:
            #         row = row.long()
            #         col = col.long()
            #         val = val.float()
            #         preds = self.forward(row, col)
            #         if IMPLICT:
            #             preds = sigmoid(preds.cpu().numpy())
            #         y_pred += preds.tolist()
            #         y_true += val.tolist()
            #     y_true,y_pred=np.array(y_true), np.array(y_pred)
            #     if IMPLICT:
            #         epoch_score = roc_auc_score(y_true,y_pred)
            #         score='auc'
            #     else:
            #         epoch_score=sum([(y - x) ** 2 for x, y in zip(y_true, y_pred)]) / len(y_pred)
            #         score='mse'

            # if ((epoch + 1) % 1) == 0:
            #     print(f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')

            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
        return

if __name__ == '__main__':
    # (n_users, n_items, user_basket), loaders
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = FPMC(input_size[0],input_size[1],input_size[2])
    model.fit(loader,10)

# 没有评估，计算recall等


