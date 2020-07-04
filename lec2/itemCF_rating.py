# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn

from operator import itemgetter

np.random.seed(1024)


class itemCF():
    def __init__(self, data_file, K=20):
        self.K = K  # 近邻数
        self.readData(data_file)  # 读取数据
        self.initModel()  # 初始化模型

    def initModel(self):
        # 计算每个物品被用户评分的个数
        item_cnt = {}
        for user, items in self.train_data.items():
            for i in items:
                # count item popularity
                item_cnt.setdefault(i, 0)
                item_cnt[i] += 1

        # 计算每个项目的平均评分
        self.average_rating = {}
        for user, items in self.train_data.items():
            for i in items:
                self.average_rating.setdefault(i, 0)
                self.average_rating[i] += self.train_data[user][i] / item_cnt[i]

        # 相似度的分子部分
        C2 = dict()
        C3 = dict()
        C1 = dict()
        for user, items in self.train_data.items():
            for i in items:
                for j in items:
                    if i == j:
                        continue
                    C1.setdefault(i, {})
                    C1[i].setdefault(j, 0)
                    C2.setdefault(i, {})
                    C2[i].setdefault(j, 0)
                    C3.setdefault(i, {})
                    C3[i].setdefault(j, 0)

                    C1[i][j] += ((self.train_data[user][i] - self.average_rating[i]) * (
                                self.train_data[user][j] - self.average_rating[j]))
                    C2[i][j] += ((self.train_data[user][i] - self.average_rating[i]) * (
                            self.train_data[user][i] - self.average_rating[i]))
                    C3[i][j] += ((self.train_data[user][j] - self.average_rating[j]) * (
                            self.train_data[user][j] - self.average_rating[j]))

        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for i, related_items in C1.items():
            self.item_sim[i] = {}
            for j, cuv in related_items.items():
                if C1[i][j] == 0:
                    self.item_sim[i][j] = 0
                else:
                    self.item_sim[i][j] = C1[i][j] / math.sqrt(C2[i][j] * C3[i][j])

    def readData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)

        self.test_data = {}
        self.train_data = {}

        # 按照1:9划分数据集
        for (user, item, record, timestamp) in data.itertuples(index=False):
            if random.randint(0, 10) == 0:
                self.test_data.setdefault(user, {})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user, {})
                self.train_data[user][item] = record

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['item_id'].values))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

    def forward(self, user, item):
        rui = 0
        # 分子和分母
        C1 = 0
        C2 = 0
        for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                      key=itemgetter(1), reverse=True)[:self.K]:
            if similar_item not in self.train_data[user]:
                continue
            C1+=similarity_factor*self.train_data[user][similar_item]
            C2 += math.fabs(similarity_factor)
        if not C1==0:
            rui=(C1/C2)
        return rui


def evaluate(model):
    print('Evaluating start ...')
    count = 0
    sum_rui = 0

    for user in model.test_data:
        for movie in model.test_data[user]:
            if movie in model.item_sim:
                rui = model.forward(user, movie)  # 预测的评分
                if rui==0 :   #说明用户u评分过的物品中没有i的邻域
                    continue
                count += 1
                sum_rui += math.fabs(model.test_data[user][movie] - rui)

    print("count=",count)

    print("平均绝对值误差=", sum_rui / count)



if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data")
    evaluate(model)
    print('done!')
# 平均绝对值误差= 1.169491008089159