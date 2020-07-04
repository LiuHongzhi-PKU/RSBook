# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import random
import math
import pandas as pd
import numpy as np
import math
from operator import itemgetter
import argparse

np.random.seed(1024)


class userCF():
    def __init__(self, data_file, K=20):
        self.K = K  # 近邻数
        self.readData(data_file)  # 读取数据
        self.initModel()  # 初始化模型

    def initModel(self):
        # 计算每个用户的平均评分
        self.average_rating = {}
        for u, items in self.train_data.items():
            self.average_rating.setdefault(u, 0)
            for i in items:
                self.average_rating[u] += self.train_data[u][i] / len(items)

        # 建立item_user倒排表
        # item->set
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        # 相似度的分子部分
        C1 = dict()
        C2 = dict()
        C3 = dict()
        for i, users in item_users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    C1.setdefault(u, {})
                    C1[u].setdefault(v, 0)
                    C2.setdefault(u, {})
                    C2[u].setdefault(v, 0)
                    C3.setdefault(u, {})
                    C3[u].setdefault(v, 0)

                    C1[u][v] += ((self.train_data[u][i] - self.average_rating[u]) * (
                            self.train_data[v][i] - self.average_rating[v]))
                    C2[u][v] += ((self.train_data[u][i] - self.average_rating[u]) * (
                            self.train_data[u][i] - self.average_rating[u]))
                    C3[u][v] += ((self.train_data[v][i] - self.average_rating[v]) * (
                            self.train_data[v][i] - self.average_rating[v]))

        # 计算最终的用户相似度矩阵
        self.user_sim = dict()
        for u, related_users in C1.items():
            self.user_sim[u] = {}
            for v, cuv in related_users.items():
                # print(C1[u][v],"  ",C2[u][v],"  ",C3[u][v])
                if C1[u][v]==0:
                    self.user_sim[u][v]=0
                else:
                    self.user_sim[u][v] = C1[u][v] / math.sqrt(C2[u][v] * C3[u][v])

    def readData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)
        # 二维字典
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
        rui = self.average_rating[user]
        # 分子和分母
        C1 = 0
        C2 = 0
        for similar_user, similarity_factor in sorted(self.user_sim[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:self.K]:
            if item not in self.train_data[similar_user]:
                continue
            C1 += similarity_factor * (self.train_data[similar_user][item] - self.average_rating[similar_user])
            C2 += math.fabs(similarity_factor)
        if not C1==0:
            rui += (C1 / C2)
        else :
            rui=0
        return rui


# 产生推荐并通过准确率、召回率和覆盖率进行评估
def evaluate(model):
    print('Evaluating start ...')
    count = 0
    sum_rui = 0

    for user in model.test_data:
        for movie in model.test_data[user]:
            rui = model.forward(user, movie)  # 预测的评分
            if rui == 0:  # 说明用户u的邻域都没有对i评分过
                continue
            count += 1
            sum_rui += math.fabs(model.test_data[user][movie] - rui)

    print("平均绝对值误差=", sum_rui / count)
    print("count=", count)



if __name__ == '__main__':
    model = userCF("../data/ml-100k/u.data")
    evaluate(model)
    print('done!')
# 平均绝对值误差= 0.9615194990982957