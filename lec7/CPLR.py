# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import math
import random
from collections import defaultdict
from operator import itemgetter
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions, CPLR_Interactions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

EPS = 1e-12


# 固定随机种子，用于复现
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def cplr_loss(diff):
    sig = nn.Sigmoid()
    return -torch.log(sig(diff)).sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def getUsrSim(train_data):
    # 建立item_user倒排表
    item_users = dict()
    for u, items in train_data.items():
        for i in items:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    # 计算用户之间共同评分的物品数
    C = dict()
    for i, users in item_users.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                C.setdefault(u, {})
                C[u].setdefault(v, 0)
                C[u][v] += 1
    # 计算最终的用户相似度矩阵
    user_sim = dict()
    for u, related_users in C.items():
        user_sim[u] = {}
        for v, cuv in related_users.items():
            user_sim[u][v] = cuv / math.sqrt(len(train_data[u]) * len(train_data[v]))
    return user_sim


# 计算领域支持系数，用于训练时计算置信系数
def getSupport(user_sim, user_item, K=10):
    support = {}
    for user in user_item:
        support.setdefault(user, defaultdict(int))
        for similar_user, similarity_factor in sorted(user_sim[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for item in user_item[similar_user]:
                support[user][item] += similarity_factor
    return support


class CPLR(torch.nn.Module):
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
    # user_bias：用户偏置
    # user_embeddings：用户向量
    # support:领域支持系数
    # alpha/beta/gama：三个控制系数
    def __init__(self, data_file, n_factors=20, alpha=1, beta=1, gama=1, lr=0.1, weight_decay=0.005,
                 device=torch.device("cpu"), batch_size=2048,
                 sparse=False, topn=10):
        super(CPLR, self).__init__()

        self.loadData(data_file, batch_size=batch_size)  # 读取数据

        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.n_factors = n_factors
        self.topn = topn  # 推荐物品的topn
        self.device = device

        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        # weight_decay相当于L2正则化，因此Loss中不用考虑正则化项
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self = self.to(self.device)

    def loadData(self, data_path, batch_size=2048, K=10):
        # load train data
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        data_df = pd.read_table(data_path, names=data_fields)
        data_df.rating = (data_df.rating >= 5).astype(np.float32)
        le = preprocessing.LabelEncoder()
        le.fit(data_df['user_id'])
        data_df['user_id'] = le.transform(data_df['user_id'])
        le.fit(data_df['item_id'])
        data_df['item_id'] = le.transform(data_df['item_id'])
        #     data_df['user_id']-=1
        #     data_df['item_id'] -= 1
        #

        # get user number
        self.n_users = max(data_df['user_id'].values) + 1
        # get item number
        self.n_items = max(data_df['item_id'].values) + 1

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))
        df = {}
        df['train'] = data_df.sample(n=int(len(data_df) * 0.8), replace=False)
        df['valid'] = data_df.drop(df['train'].index, axis=0)
        self.support = {}
        self.loader = {}
        for phase in ['train', 'valid']:
            user_item = {}
            for (user, item, record, timestamp) in df[phase].itertuples(index=False):
                user_item.setdefault(user, {})
                user_item[user][item] = record
            user_sim = getUsrSim(user_item)
            self.support[phase] = getSupport(user_sim, user_item, K)

            self.loader[phase] = data.DataLoader(
                CPLR_Interactions(df[phase], self.n_items, user_sim, K), batch_size=batch_size,
                shuffle=(phase == 'train'))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

        self.loader['valid_simple'] = data.DataLoader(
            Interactions(df['valid']), batch_size=batch_size, shuffle=False)

    # 预测评分结果
    def predict(self, users, items):
        ues = self.user_embeddings(users)  # B,F
        uis = self.item_embeddings(items)  # B,F

        preds = self.user_biases(users)  # B,1
        preds += self.item_biases(items)  # B,1
        preds += (ues * uis).sum(dim=-1, keepdim=True)  # B,1

        return preds.squeeze(-1)  # B

    # 前向函数，用于计算loss
    def forward(self, users, items):
        (pos_items, neg_items) = items
        pos_preds = self.predict(users, pos_items)
        neg_preds = self.predict(users, neg_items)
        return pos_preds - neg_preds

    def fit(self, epochs=5):
        # training cycle
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}

            for phase in ['train', 'valid']:

                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                pbar = tqdm(enumerate(self.loader[phase]),
                            total=len(self.loader[phase]),
                            desc='({0}:{1:^3})'.format(phase, epoch + 1))
                for batch_idx, ((users, items), ratings) in pbar:
                    self.optimizer.zero_grad()

                    users = users.long()
                    items = tuple(c.long() for c in items)
                    c_uit, c_utj, c_uij = [], [], []
                    for id in range(users.size(0)):
                        u = users[id]
                        i, t, j = items[0][id], items[1][id], items[2][id]
                        c_uit += [(1 + self.support[phase][int(u)][int(i)]) / (1 + self.support[phase][int(u)][int(t)])]
                        c_utj += [1 + self.support[phase][int(u)][int(t)]]
                        c_uij += [1 + self.support[phase][int(u)][int(i)]]

                    c_uit = torch.Tensor(c_uit).float()
                    c_utj = torch.Tensor(c_utj).float()
                    c_uij = torch.Tensor(c_uij).float()

                    r_uit = c_uit * self.forward(users, (items[0], items[1]))
                    r_utj = c_utj * self.forward(users, (items[1], items[2]))
                    r_uij = c_uij * self.forward(users, (items[0], items[2]))
                    loss = -self.alpha * torch.log(torch.sigmoid(r_uit) + EPS) + \
                           -self.beta * torch.log(torch.sigmoid(r_utj) + EPS) + \
                           -self.gama * torch.log(torch.sigmoid(r_uij) + EPS)
                    loss = loss.sum()
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / users.size()[0]
                    pbar.set_postfix(loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                losses[phase] /= len(self.loader[phase].dataset)

            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()

                y_pred, y_true = [], []

                for ((row, col), val) in self.loader['valid_simple']:
                    row = row.long()
                    col = col.long()
                    val = val.float()
                    preds = self.predict(row, col)
                    # if IMPLICT:
                    #     preds = sigmoid(preds.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += val.tolist()
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                epoch_score = roc_auc_score(y_true, y_pred)
                score = 'auc'

                # # 计算top10的recall、precision、推荐物品覆盖率
                # # 计算top10的recall、precision、推荐物品覆盖率
                user_item = self.loader['valid_simple'].dataset.user_item
                items = torch.arange(self.n_items).long().to(self.device)
                hit, rec_count, test_count, all_rec_items = 0, 0, 0, set()
                train_ui = self.loader['train'].dataset.user_item
                for u in user_item:
                    target_items = user_item[u]

                    users = [int(u)] * self.n_items
                    users = torch.Tensor(users).long().to(self.device)
                    scores = self.predict(users, items)
                    if u in train_ui:
                        seen_items = np.array(list(train_ui[u].keys()))

                        scores[seen_items] = -1e9
                    else:
                        continue

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
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')
                print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
                print(hit, len(all_rec_items), len(user_item))

        return


parser = argparse.ArgumentParser()
parser.add_argument('--n_factors', type=int, default=20, help='隐空间维度')
parser.add_argument('--lr', type=float, default=0.1, help='学习率')
parser.add_argument('--topn', type=int, default=10, help='物品推荐数')
parser.add_argument('--batch_size', type=int, default=2048, help='batch大小')
parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
parser.add_argument('--weight_decay', type=float, default=0.005, help='adam学习器中的权值衰减')
parser.add_argument('--alpha', type=int, default=1, help='控制系数')
parser.add_argument('--gama', type=int, default=1, help='控制系数')
parser.add_argument('--beta', type=int, default=1, help='控制系数')
opt = parser.parse_args()

if __name__ == '__main__':
    model = CPLR("../data/ml-100k/u.data", n_factors=opt.n_factors, lr=opt.lr, topn=opt.topn,
                 weight_decay=opt.weight_decay, batch_size=opt.batch_size, alpha=opt.alpha, gama=opt.gama,
                 beta=opt.beta)
    model.fit(opt.epochs)
# precisioin=0.2898	recall=0.1363	coverage=0.3591
