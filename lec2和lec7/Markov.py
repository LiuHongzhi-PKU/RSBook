# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin


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
from utils import Interactions,PairwiseInteractions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from pandas.core.frame import DataFrame
import csv

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
    # user_basket：二维字典，user->basket->item的list
    # user_basket_test：list的字典，key是用户，value是一个长度为2的list，该用户的最后两个basket里的物品
    # self.Matrix：转移矩阵
    # topn：物品推荐数目
    # Matrix:共享转移矩阵

    def __init__(self, topn=10):
        super(FPMC, self).__init__()
        self.topn=topn

        self.loadData2(transPath="../data/trans.txt", usersPath="../data/users.txt", itemsPath="../data/items.txt")
        self.initModel()
        self.evaluate()



    def initModel(self):
        print("开始初始化")
        transCount={}
        self.Matrix=defaultdict(dict)
        for user , userdict in self.user_basket.items():
            print(user)
            for index,itemList in  userdict.items():
                if index==0:
                    continue;
                for j in itemList:
                    for i in userdict[index-1]:
                        self.Matrix[i].setdefault(j, 0)
                        transCount.setdefault(i,0)
                        self.Matrix[i][j]+=1 #i到j的次数
                        transCount[i]+=1 #i的总次数
        for i , jlist in self.Matrix.items():
            for j in jlist:
                self.Matrix[i][j]/=transCount[i]

    def predict(self, user):
        last_bucket=self.user_basket_test[user][0]
        length=len(last_bucket)
        score={}
        for l in last_bucket:
            for i in range(self.n_items):
                score.setdefault(i,0)
                if i not in self.Matrix[l].keys():
                    continue
                score[i]+=self.Matrix[l][i]/length
        rec_list = []
        rec_items = sorted(score.items(), key=itemgetter(1), reverse=True)[0:self.topn]
        for item, score in rec_items:
            rec_list.append(item)
        return rec_list

    def evaluate(self):
        print("开始评估")
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
            rec_list = self.predict(u)


            for item in rec_list:  # 遍历给user推荐的物品
                if item in target_items:  # 测试集中有该物品
                    hit += 1  # 推荐命中+1
                all_rec_items.add(item)
            rec_count += self.topn
            test_count += len(target_items)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.n_items)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
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



if __name__ == '__main__':
    model = FPMC()


# precisioin=0.2830	recall=0.6011	coverage=0.1178