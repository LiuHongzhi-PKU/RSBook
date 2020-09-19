# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import random
import pandas as pd


from utils import evaluate
from utils import modelType
import argparse

class SLOPE():
    # 参数:
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_users：item_user倒排表。字典，item->user的set
    def __init__(self, data_file):
        self.loadData(data_file)  # 读取数据
        self.initModel()

    def initModel(self):
        # 建立item_user倒排表
        self.item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in self.item_users:
                    self.item_users[i] = set()
                self.item_users[i].add(u)

    def loadData(self, data_file):
        # load train data
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        data_df = pd.read_table(data_file, names=data_fields)

        self.test_data = {}
        self.train_data = {}

        # 按照1:9划分数据集
        for (user, item, record, timestamp) in data_df.itertuples(index=False):
            if random.randint(0, 10) == 0:
                self.test_data.setdefault(user, {})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user, {})
                self.train_data[user][item] = record

        # get user number
        self.n_users = len(set(data_df['user_id'].values))
        # get item number
        self.n_items = len(set(data_df['item_id'].values))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

    def predict(self, user, j):
        rating = 0
        cnt = 0
        if j not in self.item_users:
            return rating
        for i in self.train_data[user]:
            if i == j:
                continue
            if i not in self.item_users:
                continue
            common_users = self.item_users[i] & self.item_users[j]
            if len(common_users) == 0:
                continue
            dev_ij = 0
            cnt += 1
            for common_user in common_users:
                dev_ij += (self.train_data[common_user][j] - self.train_data[common_user][i])
            dev_ij /= len(common_users)
            rating += (dev_ij + self.train_data[user][i])
        rating /= cnt
        return rating

if __name__ == '__main__':
    model = SLOPE("../data/ml-100k/u.data")
    # for name, value in vars(model).items():
    #     print('%s' % (name))
    ev = evaluate(modelType.rating)
    ev.evaluateModel(model)
    print('done!')

# 平均绝对值误差= 0.739291913465804