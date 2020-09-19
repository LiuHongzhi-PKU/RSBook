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
import argparse
np.random.seed(1024)

# 情境预过滤

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
    # dis：用户-地点距离
    # boundary：预过滤的边界
    # locateItem：地点的经纬度  i->(x,y)
    # locateUser：用户最后一次出现的经纬度
    def __init__(self, data_file, K=20,N=10,similarityMeasure="cosine",boundary=10000):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.boundary=boundary # 预过滤的边界，单位米
        self.similarityMeasure = similarityMeasure
        self.loadData(data_file)    # 读取数据
        self.initModel()            # 初始化模型


    def initModel(self):
        pass

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

        print("开始计算用户-地点距离")
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

        # print("开始计算地点-地点距离")
        # dis2 = {}
        # pbar = tqdm(range(self.n_items))
        # for i in pbar:
        #     if i not in self.locateItem:
        #         continue
        #     dis2.setdefault(i, {})
        #     for j in range(self.n_items):
        #         if j==i:
        #             continue
        #         if j not in self.locateItem:
        #             continue
        #         dis2[i][j] = geodesic((self.locateItem[i][1], self.locateItem[i][0]),
        #                              (self.locateItem[j][1], self.locateItem[j][0])).m
        # self.dis2 = dis2


    def predict(self, user):
        # 对该用户，范围内的地点
        location_in_scope=set()

        # 范围外的地点
        location_out_scope = set()

        for item in self.dis[user]:
            if self.dis[user][item]>self.boundary:
                location_out_scope.add(item)
            else:
                location_in_scope.add(item)

        # print(len(list(location_out_scope)))
        # print(len(list(location_in_scope)))

        # 物品-用户表
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                item_users.setdefault(i,{})
                item_users[i][u]=1

        # 删掉范围外的数据
        for item in range(self.n_items):
            if item not in item_users:
                continue
            if item in location_out_scope:
                item_users.pop(item)

        # 构建用户-物品表，即该用户的训练数据
        user_items=dict()
        for item,users in item_users.items():
            for u in users:
                user_items.setdefault(u,{})
                user_items[u][item]=1

        # print("user:",user_items)
        # print("item:", item_users)
        # 根据过滤后的数据，计算物品相似度

        # 计算每个物品被用户评分的个数
        item_cnt = {}
        for u, items in user_items.items():
            for i in items:
                # count item popularity
                item_cnt.setdefault(i, 0)
                item_cnt[i] += 1

        # print("item_cnt",item_cnt)

        # 计算物品之间共同评分的物品数,C为修正后的，count为修正前的。
        C = dict()
        count = dict()
        for u, items in user_items.items():
            for u in items:
                for v in items:
                    if u == v:
                        continue
                    C.setdefault(u, {})
                    C[u].setdefault(v, 0)
                    C[u][v] += math.log(self.n_items / len(items))

                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for u, related_items in C.items():
            self.item_sim[u] = {}
            for v, cuv in related_items.items():
                if self.similarityMeasure == "cosine":
                    self.item_sim[u][v] = cuv / math.sqrt(item_cnt[u] * item_cnt[v])
                else:
                    self.item_sim[u][v] = count[u][v] / (item_cnt[u])

        # print("sim",self.item_sim)

        #  接下来进行推荐
        rank = dict()
        interacted_items = user_items[user]

        # print("user",user_items)
        # print(user)
        # print("inter",interacted_items)

        # 对每个评分的物品寻找最近K个物品，构建评分列表
        for item, rating in interacted_items.items():
            if item not in self.item_sim:
                # print("pass")
                continue
            for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                           key=itemgetter(1), reverse=True)[:self.K]:
                if similar_item in interacted_items:
                    continue
                rank.setdefault(similar_item, 0)
                # rank[similar_item] += similarity_factor * rating
                rank[similar_item] += similarity_factor


        rec_list = []
        rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)
        # print(rec_items)
        count=0
        for item, score in rec_items:
            if item not in self.locateItem:
                # print(item)
                continue
            rec_list.append(item)
            count+=1
            if count==self.N:
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

    pbar = tqdm(range(model.n_users))
    for user in pbar:
        if user not in model.train_data:
            # print(user)
            continue
        test_moives = model.test_data.get(user, {})  # 测试集中用户喜欢的电影
        if len(test_moives) == 0:
            continue
        rec_movies = model.predict(user)  # 得到推荐的电影及计算出的用户对它们的兴趣
        # print(rec_movies)

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


parser = argparse.ArgumentParser()
parser.add_argument('--K',type=int, default=20, help='近邻数')
parser.add_argument('--N',type=int, default=10, help='物品推荐数')
parser.add_argument('--boundary',type=int, default=10000, help='预过滤边界，单位米')
parser.add_argument('--similarityMeasure', default="cosine", help='相似度度量，cosine或conditional')
opt = parser.parse_args()


if __name__ == '__main__':
    model = LBS("../data/ml-100k/u.data",K=opt.K,N=opt.N,similarityMeasure=opt.similarityMeasure,boundary=opt.boundary)
    evaluateTopN(model)
    print('done!')

# 前五百个用户 五百个物品
# 距离 50000
# precisioin=0.0082	recall=0.0431	coverage=0.8095

# 10000
# precisioin=0.0090	recall=0.0474	coverage=0.8310

# 5000
# precisioin=0.0066	recall=0.0345	coverage=0.6929

