# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import random
from collections import defaultdict
import argparse
import pandas as pd
from tqdm import tqdm

from utils import evaluate
from utils import modelType


class  spreadingActivation():
    # 参数
    # step: 扩散的步数
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # rec_item：在初始化模型时就已经给每个用户对所有物品排好序
    def __init__(self, data_file,step=5,N=10):
        self.step = step
        self.N = N  # 物品推荐数
        self.loadData(data_file)  # 读取数据
        self.initModel()


    def loadData(self, data_file):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table(data_file, names=data_fields)
        # 二维字典
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

    def initModel(self):
        # 建立item_user倒排表
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)


        user_item = self.train_data
        self.rec_item = defaultdict(list)
        for u in tqdm(user_item):
            # 每个物品第几步被扩散到
            visit_step = defaultdict(int)
            # 物品扩散到的次数
            visit_cnt = defaultdict(int)

            user_set = set()
            user_set.add(u)
            item_set = set()
            for k in range(self.step):
                if k % 2 == 0:
                    # 用户扩散项目
                    for user in user_set:
                        for item in user_item[user]:
                            if item not in visit_step:
                                visit_step[item] = k + 1
                            visit_cnt[item] += 1
                        item_set.update(set(user_item[user]))
                    user_set = set()
                else:
                    # 项目扩散用户
                    for item in item_set:
                        user_set.update(set(item_users[item]))
                    item_set = set()

            res = ((pd.DataFrame(visit_cnt, index=[0]).append(pd.DataFrame(visit_step, index=[1]))).T).sort_values(
                [1, 0], ascending=[1, 0]).index.tolist()

            self.rec_item[u] = [i for i in res if i not in user_item[u]]

    def predict(self, user):
        # print(self.rec_item[user][0:self.N])
        # 返回最大N个物品
        return self.rec_item[user][0:self.N]

parser = argparse.ArgumentParser()
parser.add_argument('--N',type=int, default=10, help='物品推荐数')
parser.add_argument('--step',type=int, default=10, help='扩散步长，item->user->item算两步')
opt = parser.parse_args()


if __name__ == '__main__':
    model = spreadingActivation("../data/ml-100k/u.data",N=opt.N,step=opt.step)
    # for name, value in vars(model).items():
    #     print('%s' % (name))
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')


# precisioin=0.1027	recall=0.1071	coverage=0.0434