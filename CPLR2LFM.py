import math
import random
from collections import defaultdict
from operator import itemgetter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions,CPLR_Interactions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

# 设置是否使用隐式反馈
IMPLICT=True
# 设置是否使用超小数据集测试
SMALL=False
EPS=1e-12

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
def cplr_loss(diff):
    sig = nn.Sigmoid()
    return -torch.log(sig(diff)).sum()

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
                C.setdefault(u,{})
                C[u].setdefault(v,0)
                C[u][v] += 1
    # 计算最终的用户相似度矩阵
    user_sim = dict()
    for u, related_users in C.items():
        user_sim[u]={}
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

def getDataLoader(data_path, batch_size=2048, K=10):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 5).astype(np.float32)
    le = preprocessing.LabelEncoder()
    le.fit(data_df['user_id'])
    data_df['user_id']=le.transform(data_df['user_id'])
    le.fit(data_df['item_id'])
    data_df['item_id']=le.transform(data_df['item_id'])
    #     data_df['user_id']-=1
    #     data_df['item_id'] -= 1
    #

    # get user number
    n_users = max(data_df['user_id'].values)+1
    # get item number
    n_items = max(data_df['item_id'].values)+1

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    df = {}
    df['train'] = data_df.sample(n=int(len(data_df) * 0.8), replace=False)
    df['valid'] = data_df.drop(df['train'].index, axis=0)
    loader = {}
    for phase in ['train', 'valid']:
        loader[phase] = data.DataLoader(
            Interactions(df[phase]), batch_size=batch_size, shuffle=(phase == 'train'))

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))

    # loader['valid_simple'] = data.DataLoader(
    #     Interactions(df['valid']), batch_size=batch_size, shuffle=False)

    return (n_users, n_items), loader
#
class CPLR(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, alpha=1, beta=1, gama=1, lr=0.1, weight_decay=0.005,device=torch.device("cpu"),
                 sparse=False, topn=10):

    # def __init__(self, n_users, n_items, n_factors=10, lr=0.01, weight_decay=0.01, sparse=False,topn=10, device=torch.device("cpu")):
        super(CPLR, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.n_factors = n_factors
        self.topn=topn # 推荐物品的topn
        self.device = device


        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)


        # weight_decay相当于L2正则化，因此Loss中不用考虑正则化项
        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        self=self.to(self.device)

    # 预测评分结果
    def predict(self, users, items):
        ues = self.user_embeddings(users) # B,F
        uis = self.item_embeddings(items) # B,F

        preds = self.user_biases(users) # B,1
        preds += self.item_biases(items) # B,1
        preds += (ues * uis).sum(dim=-1, keepdim=True) # B,1

        return preds.squeeze(-1) # B
    # 前向函数，用于计算loss
    def forward(self, users, items):
        return self.predict(users,items)

    def fit(self, loaders, epochs=5):
        # training cycle
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
                for batch_idx, ((users, items), ratings) in pbar:
                    self.optimizer.zero_grad()

                    users = users.long()
                    items = items.long()
                    ratings = ratings.float().to(self.device)
                    preds = self.forward(users, items)
                    loss = nn.MSELoss(reduction='sum')(preds, ratings)
                    # loss = -self.alpha * torch.log(torch.sigmoid(r_uit)+EPS) + \
                    #     -self.beta * torch.log(torch.sigmoid(r_utj)+EPS) + \
                    #     -self.gama * torch.log(torch.sigmoid(r_uij)+EPS)
                    # loss=loss.sum()
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / users.size()[0]
                    pbar.set_postfix(loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
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
                    preds = self.predict(row, col)
                    # if IMPLICT:
                    #     preds = sigmoid(preds.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += val.tolist()
                y_true,y_pred=np.array(y_true), np.array(y_pred)
                if IMPLICT:
                    epoch_score = roc_auc_score(y_true,y_pred)
                    score='auc'
                else:
                    epoch_score=sum([(y - x) ** 2 for x, y in zip(y_true, y_pred)]) / len(y_pred)
                    score='mse'


                # user_item=loaders['valid_simple'].dataset.user_item
                # items = torch.arange(self.n_items).long()
                # hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                # train_ui=loaders['train'].dataset.user_item
                # for u in user_item:
                #     target_items=user_item[u]
                #     if u not in train_ui:continue
                #     seen_items = np.array(list(train_ui[u].keys()))
                #
                #     users=[int(u)]*self.n_items
                #     users = torch.Tensor(users).long()
                #     scores=self.predict(users,items)
                #     scores[seen_items]=-1e9
                #     recs=np.argsort(scores)[-self.topn:].tolist()
                #
                #     for item in recs:  # 遍历给user推荐的物品
                #         if item in target_items:  # 测试集中有该物品
                #             hit += 1  # 推荐命中+1
                #         all_rec_items.add(item)
                #     rec_count += self.topn
                #     test_count += len(target_items)
                #     precision = hit / (1.0 * rec_count)
                #     recall = hit / (1.0 * test_count)
                #     coverage = len(all_rec_items) / (1.0 * self.n_items)

                # # 计算top10的recall、precision、推荐物品覆盖率
                # # 计算top10的recall、precision、推荐物品覆盖率
                user_item=loaders['valid'].dataset.user_item
                items = torch.arange(self.n_items).long().to(self.device)
                hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                train_ui=loaders['train'].dataset.user_item
                for u in user_item:
                    target_items=user_item[u]

                    users=[int(u)]*self.n_items
                    users = torch.Tensor(users).long().to(self.device)
                    scores=self.predict(users,items)
                    if u in train_ui:
                        seen_items = np.array(list(train_ui[u].keys()))

                        scores[seen_items]=-1e9
                    else:continue
                    # print('s',len(seen_items))
                    # seen_items = np.array(list(train_ui[u].keys()))
                    # scores[seen_items] = -1e9
                    # print('t',len(seen_items))
                    recs=np.argsort(scores)[-self.topn:].tolist()
                    # print('------------')
                    # print(seen_items)
                    # print(recs)
                    # print(scores[recs])

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

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = CPLR(input_size[0],input_size[1])
    model.fit(loader,10)
