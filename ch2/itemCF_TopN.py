import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType
import argparse
from operator import itemgetter

np.random.seed(1024)


class itemCF():
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

    def predict(self, user):
        rank = dict()
        interacted_items = self.train_data[user]

        # 对每个评分的物品寻找最近K个物品，构建评分列表
        for item, rating in interacted_items.items():
            for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                           key=itemgetter(1), reverse=True)[:self.K]:
                if similar_item in interacted_items:
                    continue
                rank.setdefault(similar_item, 0)
                # rank[similar_item] += similarity_factor * rating
                rank[similar_item] += similarity_factor

        rec_list = []
        rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]
        for item, score in rec_items:
            rec_list.append(item)

        # 返回最大N个物品
        return rec_list

parser = argparse.ArgumentParser()
parser.add_argument('--K',type=int, default=20, help='近邻数')
parser.add_argument('--N',type=int, default=10, help='物品推荐数')
parser.add_argument('--similarityMeasure', default="cosine", help='相似度度量，cosine或conditional')
opt = parser.parse_args()


if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data",K=opt.K,N=opt.N,similarityMeasure=opt.similarityMeasure)
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

#余弦相似度 precisioin=0.1779	recall=0.1796	coverage=0.1272
# 条件概率 precisioin=0.1466	recall=0.1483	coverage=0.0642