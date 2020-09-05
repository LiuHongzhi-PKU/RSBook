# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin


import math
import random
from collections import defaultdict
from operator import itemgetter
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions,PairwiseInteractions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from pandas.core.frame import DataFrame


EPS=1e-12

# 固定随机种子，用于复现
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


def bpr_loss(diff):
    sig = nn.Sigmoid()
    return -torch.log(sig(diff)).sum()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FPMC(torch.nn.Module):
    # 类变量：
    # n_users:用户数目
    # n_items：物品数目
    # device:设备
    # item_bias：物品偏置
    # item_embeddings：物品向量
    # loader：加载的数据
    # n_factors：隐空间维度
    # optimizer：优化器
    # topn：物品推荐数目
    # V_IL V_LI V_UI V_IU  ： 四个向量矩阵
    # user_basket :用户到session的映射

    def __init__(self, data_file, n_factors=20, lr=0.1, weight_decay=0.005,batch_size=2048,
                 device=torch.device("cpu"),
                 sparse=False, topn=10):
        super(FPMC, self).__init__()

        self.loadData2(transPath="../data/trans.txt",usersPath="../data/users.txt",itemsPath="../data/items.txt")

        # self.loadData(data_file)  # 读取数据

        self.n_factors = n_factors
        self.topn = topn  # 推荐物品的topn
        self.device = device
        self.batch_size=batch_size

        self.V_IL = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.V_LI = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        # self.V_UL = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        # self.V_LU = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.V_UI = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.V_IU = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        # weight_decay相当于L2正则化，因此Loss中不用考虑正则化项
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self = self.to(self.device)

    def loadUI(self,fname, delimiter=','):
        idx_set = set()
        with open(fname, 'r') as f:
            # dicard header
            f.readline()

            for l in csv.reader(f, delimiter=delimiter, quotechar='"'):
                idx = int(l[0])
                idx_set.add(idx)
        return idx_set

    def loadData2(self,transPath,usersPath,itemsPath):
        user_set = self.loadUI(usersPath)
        item_set = self.loadUI(itemsPath)

        user_basket = {}
        basket={}
        last_user=-1
        with open(transPath, 'r') as f:
            for l in f:
                l = [int(s) for s in l.strip().split()]
                user = l[0]
                b_tm1 = list(set(l[1:]))
                if last_user==-1 or user!=last_user:
                    last_user=user
                    basket = {}

                basket[len(basket)]=b_tm1
                user_basket[user] = basket
        self.n_users = max(user_set) + 1
        self.n_items = max(item_set) + 1
        for i in range(10):
            print(user_basket[i])
        print(user_basket.keys())

        user_basket_test = defaultdict(list)

        print("去掉长度<3的session,去掉session数目<4的用户，然后每个用户拿2个session作为测试集")
        for i in range(self.n_users):
            # print(i)
            for key in list(user_basket[i].keys()):
                if len(user_basket[i][key]) < 3:
                    user_basket[i].pop(key)
            count = 0
            for key in list(user_basket[i].keys()):
                user_basket[i][count] = user_basket[i].pop(key)
                count += 1
        for i in range(self.n_users):
            length = len(list(user_basket[i].keys()))
            if length < 4:
                user_basket.pop(i)

        for i in list(user_basket.keys()):
            length = len(list(user_basket[i].keys()))
            if length <= 2:
                continue
            user_basket_test[i].append(user_basket[i].pop(length - 2))
            user_basket_test[i].append(user_basket[i].pop(length - 1))

        for i in list(user_basket_test.keys()):
            print(user_basket_test[i])

        for i in list(user_basket.keys()):
            print(user_basket[i])

        self.user_basket_test = user_basket_test

        res = []
        for i in user_basket:
            res += [(i, j) for j in list(user_basket[i].keys())]

        # print("res", len(res))

        res = pd.DataFrame(res, columns=['user_id', 't'])
        print(res)

        df_train = res.sample(n=int(len(res) * 0.9), replace=False)
        df_test = res.drop(df_train.index, axis=0)
        self.loaders = {'train': df_train,
                        'valid': df_test,
                        }
        self.user_basket = user_basket


    def loadData(self, data_path, batch_size=2048):

        print("开始读数据")
        # load train data
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        data_df = pd.read_table(data_path, names=data_fields)
        # data_df=data_df[:100]
        data_df.rating = (data_df.rating >= 5).astype(np.float32)
        le = preprocessing.LabelEncoder()
        le.fit(data_df['user_id'])
        data_df['user_id'] = le.transform(data_df['user_id'])
        le.fit(data_df['item_id'])
        data_df['item_id'] = le.transform(data_df['item_id'])

        self.n_users = max(data_df['user_id'].values) + 1
        # get item number
        self.n_items = max(data_df['item_id'].values) + 1


        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

        df = data_df
        df.index = range(len(df))  # 重设index
        df = df.sort_values(['user_id', 'timestamp'])

        # 获取每个user对应的basket序列
        user_basket = {}
        basket = {0: [int(df.iloc[0].item_id)]}
        last_user = int(df.iloc[0].user_id)
        last_t = df.iloc[0].timestamp
        print(df.shape[0])
        for i in range(df.shape[0]):
            print(i)
            if i == 0 : continue

            user = int(df.iloc[i].user_id)
            if user != last_user:
                last_user = user
                basket = {}
            t = df.iloc[i].timestamp
            if t == last_t:
                basket[len(basket) - 1] += [int(df.iloc[i].item_id)]
            else:
                basket[len(basket)] = [int(df.iloc[i].item_id)]
                last_t = t
            user_basket[user] = basket


        user_basket_test=defaultdict(list)

        print("去掉长度<3的session,去掉session数目<4的用户，然后每个用户拿2个session作为测试集")
        for i in range(self.n_users):
            print(i)
            for key in list(user_basket[i].keys()):
                if len(user_basket[i][key])<3:
                    user_basket[i].pop(key)
            count=0
            for key in list(user_basket[i].keys()):
                user_basket[i][count]=user_basket[i].pop(key)
                count+=1
        for i in range(self.n_users):
            length = len(list(user_basket[i].keys()))
            if length < 4:
                user_basket.pop(i)

        for i in list(user_basket.keys()):
            length = len(list(user_basket[i].keys()))
            if length<=2:
                continue
            user_basket_test[i].append(user_basket[i].pop(length-2))
            user_basket_test[i].append(user_basket[i].pop(length - 1))

        for i in list(user_basket_test.keys()):
            print(user_basket_test[i])

        for i in list(user_basket.keys()):
            print(user_basket[i])

        self.user_basket_test=user_basket_test



        res = []
        for i in user_basket:
            res += [(i, j) for j in list(user_basket[i].keys())]

        # print("res", len(res))

        res = pd.DataFrame(res, columns=['user_id', 't'])
        print(res)

        df_train = res.sample(n=int(len(res) * 0.9), replace=False)
        df_test = res.drop(df_train.index, axis=0)
        self.loaders = {'train': df_train,
                   'valid': df_test,
                   }
        self.user_basket=user_basket


    # 预测结果
    def predict(self, u):
        # for i in range(self.n_items):
        #     i = torch.tensor(i).long()
        #     u = torch.tensor(u).long()
        #     last_basket = self.user_basket_test[u.item()][0]
        #     res = torch.tensor(0.)
        #     for l in last_basket:
        #         l = torch.tensor(l).long()
        #         res += (self.V_IL(i) * self.V_LI(l)).sum(-1)
        #     if len(last_basket) > 0: res /= len(last_basket)
        #     res += (self.V_UI(u) * self.V_IU(i)).sum(-1)
        #     score.append(res)
        # return np.array(score)

        last_basket = self.user_basket_test[u][0]
        length=len(last_basket)
        last_items=torch.from_numpy(np.array(last_basket)).long()
        # print("last",last_items)
        l_emb=self.V_LI(last_items)

        items = torch.arange(self.n_items).long().to(self.device)
        item_emb=self.V_IL(items)

        # print("lll",l_emb.shape)
        # print("iii",item_emb.shape)
        # lll
        # torch.Size([3, 20])
        # iii
        # torch.Size([1682, 20])

        score2=l_emb.mm(item_emb.t())
        score2=score2/length
        # print(score2.shape)
        # torch.Size([3, 1682])
        score2=score2.sum(0)
        # print(score2.shape)
        # torch.Size([1682])

        users = [int(u)] * self.n_items
        users = torch.Tensor(users).long().to(self.device)

        u_UI=self.V_UI(users)
        i_UI=self.V_IU(items)

        score1=u_UI*i_UI
        # torch.Size([1682, 20])
        score1=score1.sum(1)

        # print(score1.shape)

        return score1+score2





    # 前向函数，用于计算loss
    # 用户u，第t个session中买i的概率
    def forward(self, u, i,t):

        i = torch.tensor(i).long()
        u = torch.tensor(u).long()
        # print(i.dtype)
        # print(u.dtype)
        # print(u,i,self.n_users,self.n_items)

        last_basket = []
        if t > 0: last_basket = self.user_basket[u.item()][t - 1]

        res=torch.tensor(0.)

        # 枚举basket转移
        for l in last_basket:
            l = torch.tensor(l).long()
            res += (self.V_IL(i)*self.V_LI(l)).sum(-1)
        if len(last_basket)>0:res/=len(last_basket)
        res += (self.V_UI(u) * self.V_IU(i)).sum(-1)
        return res

    def get_loss(self, u,i,j,t):
        pos_v = self.forward(u, i, t)
        neg_v = self.forward(u, j, t)


        return bpr_loss(pos_v-neg_v)

    def fit(self, epochs=5):
        # training cycle
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}

            for phase in ['train', 'valid']:

                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                pbar = tqdm(self.loaders[phase].iterrows(),
                            total=self.loaders[phase].shape[0],
                            desc='({0}:{1:^3})'.format(phase, epoch + 1))
                # for batch_idx, ((row, col), val) in pbar:
                batch_loss = []
                all_loss=[]

                for idx, row in pbar:
                    # print("hereh::",idx,row)
                    # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()
                    # user_id的编号为t的session
                    u = row.user_id
                    t = row.t
                    if t==0:
                        continue;
                    i = random.choice(self.user_basket[u][t])
                    found = False
                    while not found:
                        j = np.random.randint(self.n_items)
                        # print(self.user_item.shape,user)
                        if j not in self.user_basket[u][t]:
                            found = True
                    # 用户u，第t个session，买过i，没买j
                    loss = self.get_loss(u, i, j, t)
                    # print("loss:",loss)
                    # print("loss:", loss.item())

                    # loss = bpr_loss(preds)

                    # losses[phase] += loss.item()
                    all_loss.append(loss)
                    batch_loss += [loss.item()]
                    if len(batch_loss) >= self.batch_size:
                        batch_loss = np.sum(batch_loss)
                        pbar.set_postfix(train_loss=batch_loss)
                        losses[phase] += batch_loss

                        batch_loss = []
                        the_loss=np.sum(all_loss)
                        print(the_loss)

                        with torch.set_grad_enabled(phase == 'train'):
                            if phase == 'train':
                                the_loss.backward()
                                self.optimizer.step()
                        all_loss=[]

                losses[phase] /= self.loaders[phase].shape[0]

            with torch.no_grad():
                # # 计算top10的recall、precision、推荐物品覆盖率
                # # 计算top10的recall、precision、推荐物品覆盖率
                user_item = self.user_basket_test

                hit, rec_count, test_count, all_rec_items = 0, 0, 0, set()

                for u in user_item:
                    # print("给",u,"推荐")
                    # 测试集有两个session，第二个session中的是目标物品
                    target_items = user_item[u][1]
                    # users = [int(u)] * self.n_items
                    # users = torch.Tensor(users).long().to(self.device)
                    scores = self.predict(u)


                    recs = np.argsort(scores)[-self.topn:].tolist()

                    for item in recs:  # 遍历给user推荐的物品
                        if item in target_items:  # 测试集中有该物品
                            hit += 1  # 推荐命中+1
                        all_rec_items.add(item)
                    rec_count += self.topn
                    test_count += len(target_items)
                precision = hit / (1.0 * rec_count)
                recall = hit / (1.0 * test_count)
                coverage = len(all_rec_items) / (1.0 * self.n_items)


            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
                print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
                print(hit, len(all_rec_items), len(user_item))
        return

if __name__ == '__main__':
    model = FPMC("../data/ml-100k/u.data")
    model.fit(1000)
# precisioin=0.0202	recall=0.0480	coverage=0.5172
# precisioin=0.2654	recall=0.5638	coverage=0.2407