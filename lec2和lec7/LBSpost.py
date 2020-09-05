import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import argparse
from geopy.distance import geodesic
from operator import itemgetter
from tqdm import tqdm

np.random.seed(1024)

# 后过滤

class LBS():
    # 参数:
    # K：近邻数目
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    # similarityMeasure：相似度度量，cosine或conditional
    def __init__(self, data_file, K=20,N=10,similarityMeasure="cosine"):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.boundary=1000 # 预过滤的边界，单位米
        self.similarityMeasure = similarityMeasure
        self.loadData(data_file)    # 读取数据
        self.initModel()            # 初始化模型


    def initModel(self):
        # 计算每个物品被用户评分的个数
        item_cnt = {}
        for user, items in self.train_data.items():
            for i in items:
                # count item popularity
                item_cnt.setdefault(i,0)
                item_cnt[i] += 1

        # 计算物品之间共同评分的物品数,C为修正后的，count为修正前的。
        C = dict()
        count=dict()
        for user, items in self.train_data.items():
            for u in items:
                for v in items:
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    C[u][v] += math.log(self.n_items/len(items))

                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for u, related_items in C.items():
            self.item_sim[u]={}
            for v, cuv in related_items.items():
                if self.similarityMeasure=="cosine":
                    self.item_sim[u][v] = cuv / math.sqrt(item_cnt[u] * item_cnt[v])
                else:
                    self.item_sim[u][v] = count[u][v] / (item_cnt[u])

    def loadData(self, data_file):
        data = pd.read_csv("../data/poidata/Foursquare/mydata.txt", index_col=0)
        data = data.sort_values(by=["date", "hour", "minute"])

        le = preprocessing.LabelEncoder()
        le.fit(data['user_id'])
        data['user_id'] = le.transform(data['user_id'])
        le = preprocessing.LabelEncoder()
        le.fit(data['local_id'])
        data['local_id'] = le.transform(data['local_id'])

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['local_id'].values))


        self.test_data = {}
        self.train_data = {}
        self.locateUser = {}
        self.locateItem = {}

        # 按照1:9划分数据集
        length = len(data)
        train = data[:int(0.9 * length)]
        test = data[int(0.9 * length):]

        for (user, item, x, y, date, hour, minute) in train.itertuples(index=False):
            self.locateUser[user] = (x, y)
            self.locateItem[item] = (x, y)

            self.train_data.setdefault(user, {})
            self.train_data[user][item]=1

        for (user, item, x, y, date, hour, minute) in test.itertuples(index=False):
            self.test_data.setdefault(user, {})
            self.test_data[user][item] = 1

        print("开始计算距离")
        dis = {}
        pbar = tqdm(range(self.n_users))
        for i in pbar:
            if i not in self.locateUser:
                continue
            dis.setdefault(i, {})
            for j in range(self.n_items):
                if j not in self.locateItem:
                    continue
                dis[i][j] = geodesic((self.locateUser[i][1], self.locateUser[i][0]), (self.locateItem[j][1], self.locateItem[j][0])).m
        self.dis=dis

        # # 建立item_user倒排表
        # item_users = dict()
        # for u, items in self.train_data.items():
        #     for i in items:
        #         if i not in item_users:
        #             item_users[i] = set()
        #         item_users[i].add(u)
        #
        # self.item_users=item_users




        # for i in self.train_data:
        #     for j in range(self.n_items):
        #         if j not in self.train_data[i]:
        #             continue
        #         if self.dis[i][j]<self.boundary:
        #             self.train_data[i].pop(j)
        #
        # for i in self.test_data:
        #     for j in range(self.n_items):
        #         if j not in self.test_data[i]:
        #             continue
        #         if self.dis[i][j]<self.boundary:
        #             self.test_data[i].pop(j)

    def predict(self, user):
        rank = dict()
        interacted_items = self.train_data[user]
        print("inter",interacted_items)

        # 对每个评分的物品寻找最近K个物品，构建评分列表
        for item, rating in interacted_items.items():
            if item not in self.item_sim:
                continue
            for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                           key=itemgetter(1), reverse=True)[:self.K]:
                if similar_item in interacted_items:
                    continue
                rank.setdefault(similar_item, 0)
                # rank[similar_item] += similarity_factor * rating
                rank[similar_item] += similarity_factor

        # 对兴趣度进行惩罚
        for item in rank.keys():
            rank[item]=rank[item]/self.dis[user][item]

        rec_list = []
        rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)
        count = 0
        for item, score in rec_items:
            if item not in self.locateItem:
                continue
            rec_list.append(item)
            count += 1
            if count == self.N:
                break

        # 返回最大N个物品
        return rec_list


def evaluateTopN(model):
    print('Evaluating start ...')
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_movies = set()

    for i, user in enumerate(model.train_data):
        test_moives = model.test_data.get(user, {})  # 测试集中用户喜欢的电影
        if len(test_moives) == 0:
            continue
        rec_movies = model.predict(user)  # 得到推荐的电影及计算出的用户对它们的兴趣

        for movie in rec_movies:  # 遍历给user推荐的电影
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
    model = LBS("../data/ml-100k/u.data")
    evaluateTopN(model)
    print('done!')

# 前五百个用户 五百个物品
# precisioin=0.0074	recall=0.0388	coverage=0.8071

# # 前一千个用户 一千个物品
# precisioin=0.0059	recall=0.0261	coverage=0.8496