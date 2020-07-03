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
    def __init__(self, data_file, K=20,N=10):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.readData(data_file)    # 读取数据
        self.initModel()            # 初始化模型

    def initModel(self):
        # 计算每个物品被用户评分的个数
        item_cnt = {}
        for user, items in self.train_data.items():
            for i in items:
                # count item popularity
                item_cnt.setdefault(i,0)
                item_cnt[i] += 1

        # 计算物品之间共同评分的物品数
        C = dict()
        for user, items in self.train_data.items():
            for u in items:
                for v in items:
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    C[u][v] += 1
        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for u, related_items in C.items():
            self.item_sim[u]={}
            for v, cuv in related_items.items():
                self.item_sim[u][v] = cuv / math.sqrt(item_cnt[u] * item_cnt[v])

    def readData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)

        self.test_data = {}
        self.train_data = {}

        # 按照1:9划分数据集
        for (user, item, record, timestamp) in data.itertuples(index=False):
            if random.randint(0,10) == 0:
                self.test_data.setdefault(user,{})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user,{})
                self.train_data[user][item] = record

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['item_id'].values))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

    def forward(self, user):
        rank = dict()
        interacted_items = self.train_data[user]

        # 对每个评分的物品寻找最近K个物品，构建评分列表
        for item, rating in interacted_items.items():
            for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                           key=itemgetter(1), reverse=True)[:self.K]:
                if similar_item in interacted_items:
                    continue
                rank.setdefault(similar_item, 0)
                rank[similar_item] += similarity_factor * rating

        # 返回最大N个物品
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]


# 产生推荐并通过准确率、召回率和覆盖率进行评估
def evaluate(model):
    print('Evaluating start ...')
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_movies = set()

    for i, user in enumerate(model.train_data):
        test_moives = model.test_data.get(user, {})  # 测试集中用户喜欢的电影
        rec_movies = model.forward(user)  # 得到推荐的电影及计算出的用户对它们的兴趣

        for movie, w in rec_movies:  # 遍历给user推荐的电影
            if movie in test_moives:  # 测试集中有该电影
                hit += 1  # 推荐命中+1
            all_rec_movies.add(movie)
        rec_count += model.N
        test_count += len(test_moives)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_movies) / (1.0 * model.n_items)

    print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data")
    evaluate(model)
    print('done!')