import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType

from operator import itemgetter

np.random.seed(1024)


class userCF():
    # 参数:
    # K：近邻数目
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    # similarityMeasure：相似度度量，cosine或conditional
    def __init__(self, data_file, K=20,N=10,alpha=0,beta=0):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.alpha=alpha
        self.beta=beta
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




        # 计算相似度的分子
        C = dict()
        for i, users in item_users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    t1 = self.train_data[u][i]
                    t2 = self.train_data[v][i]
                    C.setdefault(u, {})
                    C[u].setdefault(v, 0)
                    # 对热门物品进行惩罚
                    C[u][v] += 1.0 / (1.0+self.alpha * math.fabs(t1 - t2))
 # 计算最终的用户相似度矩阵

        self.user_sim = dict()
        for u, related_users in C.items():
            self.user_sim[u] = {}
            for v, cuv in related_users.items():
                self.user_sim[u][v] = cuv / math.sqrt(len(self.train_data[u]) * len(self.train_data[v]))

    def loadData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)

        self.test_data = {}
        self.train_data = {}

        data = data.sort_values(by=['timestamp'])

        length = len(data)

        train = data[:int(0.9 * length)]
        test = data[int(0.9 * length):]
        # print(test)
        # print("asdassdasdasdas")
        print(data.iloc[0, 3]-test.iloc[0, 3])

        self.currentTime = test.iloc[0, 3]/60/60/24/30
        print("currentTime=", self.currentTime)

        for (user, item, record, timestamp) in train.itertuples(index=False):
            self.train_data.setdefault(user, {})
            self.train_data[user][item] = timestamp/60/60/24/30

        for (user, item, record, timestamp) in test.itertuples(index=False):
            self.test_data.setdefault(user, {})
            self.test_data[user][item] = timestamp

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['item_id'].values))

        print(len(self.train_data))
        print(len(self.test_data))

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
                time=self.train_data[similar_user][movie]
                rank[movie] += (similarity_factor/(1+self.beta*math.fabs(self.currentTime-time)))

        rec_list = []
        rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]
        for item, score in rec_items:
            rec_list.append(item)
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
        if len(test_moives)==0:
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
    model = userCF("../data/ml-100k/u.data")
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

# precisioin=0.1856	recall=0.0579	coverage=0.1546