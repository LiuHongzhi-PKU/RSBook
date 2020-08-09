import random
import pandas as pd
import numpy as np
import math
from operator import itemgetter
from utils import evaluate
from utils import modelType

np.random.seed(1024)

class userCF():
    # 参数:
    # K：近邻数目
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # user_sim：用户之间的相似度。二维字典。u->v->相似度
    # similarityMeasure：相似度度量，cosine或jaccard
    def __init__(self, data_file, K=20,N=10,similarityMeasure="cosine"):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.similarityMeasure = similarityMeasure
        self.loadData(data_file)    # 读取数据
        self.initModel()            # 初始化模型

    def initModel(self):
        # 建立item_user倒排表
        # item->set
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        # 计算用户之间共同评分的物品数,C为修正后的，count为修正前的。
        C = dict()
        count = dict()
        for i, users in item_users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    # 对热门物品进行惩罚
                    C[u][v] += math.log(self.n_users/len(users))
                    # 计算最终的用户相似度矩阵

                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        self.user_sim = dict()
        for u, related_users in C.items():
            self.user_sim[u]={}
            for v, cuv in related_users.items():
                if self.similarityMeasure == "cosine":
                    self.user_sim[u][v] = cuv / math.sqrt(len(self.train_data[u]) * len(self.train_data[v]))
                else:
                    self.user_sim[u][v] = count[u][v] / (len(self.train_data[u])+len(self.train_data[v])-count[u][v])



    def loadData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)
        # 二维字典
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

        # 寻找最近的K个用户，利用它们的评分信息构造推荐列表
        for similar_user, similarity_factor in sorted(self.user_sim[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:self.K]:
            for movie in self.train_data[similar_user]:
                if movie in interacted_items:
                    continue
                rank.setdefault(movie, 0)
                # rank[movie] += similarity_factor*self.train_data[similar_user][movie]
                rank[movie] += similarity_factor

        rec_list=[]
        rec_items=sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]
        for item,score in rec_items:
            rec_list.append(item)
        return rec_list



if __name__ == '__main__':
    model = userCF("../data/ml-100k/u.data")
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

# 余弦相似度precisioin=0.1862	recall=0.1862	coverage=0.2598
# 杰卡德相似度：precisioin=0.1881	recall=0.1921	coverage=0.2331