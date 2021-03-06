# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import argparse
import random
import pandas as pd
import numpy as np
import math

from operator import itemgetter
from utils import evaluate
from utils import modelType
np.random.seed(1024)


class itemCF():
    # 参数:
    # K：近邻数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # average_rating：每个项目的平均评分、字典。项目->平均评分
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    # similarityMeasure：相似度度量，cosine或pearson
    def __init__(self, data_file, K=20,similarityMeasure="cosine"):
        self.K = K  # 近邻数
        self.similarityMeasure=similarityMeasure
        self.loadData(data_file)  # 读取数据
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

        # 计算用户的平均评分
        self.user_average_rating={}
        for user, items in self.train_data.items():
            self.user_average_rating.setdefault(user, 0)
            for i in items:
                self.user_average_rating[user] += self.train_data[user][i] / len(self.train_data[user])

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

                    if self.similarityMeasure=="cosine":
                        C1[i][j] += ((self.train_data[user][i] - self.user_average_rating[user]) * (
                                self.train_data[user][j] - self.user_average_rating[user]))
                        C2[i][j] += ((self.train_data[user][i] - self.user_average_rating[user]) * (
                                self.train_data[user][i] - self.user_average_rating[user]))
                        C3[i][j] += ((self.train_data[user][j] - self.user_average_rating[user]) * (
                                self.train_data[user][j] - self.user_average_rating[user]))
                    else:
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

        #每个物品的邻域
        self.item_nei={}
        for item in self.item_sim.keys():
            simi_items=sorted(self.item_sim[item].items(), key=itemgetter(1), reverse=True)[:self.K]
            self.item_nei[item]=simi_items

    def loadData(self, data_file):
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

    def predict(self, user, item):
        rui = 0
        # 分子和分母
        C1 = 0
        C2 = 0
        if not item in self.item_sim:
            return rui
        for interacted_item in self.train_data[user]:
            simi_items = self.item_nei[interacted_item]
            for similar_item, similarity_factor in simi_items:
                if item == similar_item:
                    C1 += similarity_factor * self.train_data[user][interacted_item]
                    C2 += math.fabs(similarity_factor)
        #
        # for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
        #                                               key=itemgetter(1), reverse=True)[:self.K]:
        #     if similar_item not in self.train_data[user]:
        #         continue
        #     C1 += similarity_factor * self.train_data[user][similar_item]
        #     C2 += math.fabs(similarity_factor)
        if not C1 == 0:
            rui = (C1 / C2)
        return rui



parser = argparse.ArgumentParser()
parser.add_argument('--K',type=int, default=20, help='近邻数')
parser.add_argument('--similarityMeasure', default="cosine", help='相似度度量，cosine或pearson')
opt = parser.parse_args()

if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data",K=opt.K,similarityMeasure=opt.similarityMeasure)
    ev=evaluate(modelType.rating)
    ev.evaluateModel(model)
    print('done!')
# 余弦相似度：平均绝对值误差= 1.1449563800161946 均方根误差= 2.22600813251409
# 余弦相似度：平均绝对值误差= 1.1399839529656934   均方根误差= 2.219480403341986
