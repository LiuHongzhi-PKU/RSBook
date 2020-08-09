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
    # test_data：测试数据，二维字典。user->item->时间
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    # similarityMeasure：相似度度量，cosine或conditional
    def __init__(self, data_file, K=20,N=10,alpha=1.0):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.alpha = alpha
        self.loadData(data_file)    # 读取数据
        self.initModel()            # 初始化模型


    def initModel(self):
        item_score = {}
        for user in self.train_data:
            for item in self.train_data[user]:
                if item not in item_score:
                    item_score[item] = 0
                t = self.train_data[user][item]
                item_score[item] += 1.0 / (1.0+self.alpha * (self.currentTime - t))
        self.item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))
        print(self.item_score)

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
        print(data.iloc[0, 3] - test.iloc[0, 3])

        self.currentTime = test.iloc[0, 3] / 60 / 60 / 24 / 30
        print("currentTime=", self.currentTime)

        for (user, item, record, timestamp) in train.itertuples(index=False):
            self.train_data.setdefault(user, {})
            self.train_data[user][item] = timestamp / 60 / 60 / 24 / 30

        for (user, item, record, timestamp) in test.itertuples(index=False):
            self.test_data.setdefault(user, {})
            self.test_data[user][item] = timestamp

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['item_id'].values))

        print(len(self.train_data))
        print(len(self.test_data))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

    def predict(self, user):
        user_items = set(self.train_data[user])
        rec_items = [x[0] for x in self.item_score if x[0] not in user_items]

        print("rec_items:",rec_items)
        return rec_items[:self.N]




if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data")
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

#precisioin=0.1567	recall=0.0489	coverage=0.0434