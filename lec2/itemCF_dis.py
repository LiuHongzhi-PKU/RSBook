import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType
from tqdm import tqdm
from operator import itemgetter
from sklearn import preprocessing
import argparse
np.random.seed(1024)


# 基于距离的相似度度量
# 每个item的向量用购买过这个item的用户表示，n个用户，就为n维向量
# 如果该物品被m个用户购买过，则对应位置为1/m


class itemCF():
    # 参数:
    # K：近邻数目
    # N：物品推荐数目
    # r：闵可夫斯基距离中的r
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    def __init__(self, data_file, K=20, N=10, r=2):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.r = r  # 闵可夫斯基距离中的r
        self.loadData(data_file)  # 读取数据
        self.initModel()  # 初始化模型



    def initModel(self):
        # 建立item_user倒排表
        # item->set
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        # 计算距离和相似度
        simi = dict()
        pbar = tqdm(range(self.n_items))
        for i in pbar:
            for j in range(i + 1, self.n_items):

                simi.setdefault(i, {})
                simi[i].setdefault(j, 0)

                simi.setdefault(j, {})
                simi[j].setdefault(i, 0)

                if i not in item_users:
                    continue
                if j not in item_users:
                    continue

                seti = item_users[i]
                setj = item_users[j]

                leni = len(seti)
                lenj = len(setj)

                vec1 = np.zeros(self.n_users)
                vec2 = np.zeros(self.n_users)

                for user in seti:
                    vec1[user] = 1.0 / leni
                for user in setj:
                    vec2[user] = 1.0 / lenj

                dis = np.linalg.norm(vec1 - vec2, ord=self.r)

                # print(dis)
                sim = 1 / (1 + dis)
                # print(sim)
                simi[i][j] = sim
                simi[j][i] = sim
        self.item_sim = simi

    def loadData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)

        le = preprocessing.LabelEncoder()
        le.fit(data['user_id'])
        data['user_id'] = le.transform(data['user_id'])
        le.fit(data['item_id'])
        data['item_id'] = le.transform(data['item_id'])

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
parser.add_argument('--r',type=int, default=2, help='闵可夫斯基距离中的r')
opt = parser.parse_args()


if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data",K=opt.K,N=opt.N,r=opt.r)
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

# precisioin=0.1547	recall=0.1558	coverage=0.0660
