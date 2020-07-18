import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType

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
    def __init__(self, data_file, K=20,N=10):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
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

        # 计算物品之间共同评分的物品数
        C = dict()
        for user, items in self.train_data.items():
            for u in items:
                for v in items:
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    C[u][v] += math.log(self.n_items/len(items))
        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for u, related_items in C.items():
            self.item_sim[u]={}
            for v, cuv in related_items.items():
                self.item_sim[u][v] = cuv / math.sqrt(item_cnt[u] * item_cnt[v])

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




if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data")
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

# precisioin=0.1799	recall=0.1847	coverage=0.1272