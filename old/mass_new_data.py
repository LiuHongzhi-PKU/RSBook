import random
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn

from operator import itemgetter
import os
# for reproducibility
from tqdm import tqdm


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def getDataLoader(data_path, batch_size=32):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    data_df.rating = (data_df.rating >= 4).astype(np.float32)

    test_data = {}
    train_data = {}

    # 按照1:9划分数据集
    for (user, item, record, timestamp) in data_df.itertuples(index=False):
        if random.randint(0, 10) == 0:
            test_data.setdefault(user, {})
            test_data[user][item] = record
        else:
            train_data.setdefault(user, {})
            train_data[user][item] = record

    # get user number
    n_users = len(set(data_df['user_id'].values))
    # get item number
    n_items = len(set(data_df['item_id'].values))

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))

    loaders = {'train': train_data,
               'valid': test_data,
               }

    return (n_users, n_items), loaders

class userCF():
    def __init__(self, n_users, n_items, K=20,N=10):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.n_users = n_users
        self.n_items = n_items
        self.user_sim = dict()
        self.train_data = dict()
        self.rec_item = dict()

    def forward(self, user):
        # rank = dict()
        # interacted_items = self.train_data[user]

        # 寻找最近的K个用户，利用它们的评分信息构造推荐列表

        # 返回最大N个物品
        return self.rec_item[user][0:self.N]

    def fit(self, train_data, Step=4):
        self.train_data = train_data
        # 建立item_user倒排表
        item_users = dict()
        for u, items in train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        # 计算用户之间共同评分的物品数
        C = dict()
        for i, users in item_users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    C[u][v] += 1
        # 计算最终的用户相似度矩阵
        self.user_sim = dict()
        for u, related_users in C.items():
            self.user_sim[u]={}
            for v, cuv in related_users.items():
                self.user_sim[u][v] = cuv / math.sqrt(len(train_data[u]) * len(train_data[v]))

        user_item=train_data
        self.rec_item=defaultdict(list)
        for u in tqdm(user_item):
            item_cnt=defaultdict(int)
            user_cnt=defaultdict(int)
            for i in user_item[u]:
                item_cnt[i]=1
            for k in range(Step):
                if k %2 ==1:
                    # 用户扩散项目
                    for user in user_cnt:
                        for item in user_item[user]:
                            item_cnt[item]+=user_cnt[user]/len(user_item[user])
                    user_cnt=defaultdict(int)
                else:
                    # 项目扩散用户
                    for item in item_cnt:
                        for user in item_users[item]:
                            user_cnt[user]+=item_cnt[item]/(len(item_users[item]))
                    item_cnt = defaultdict(int)
            # print(len(visit_cnt))
            # item_cnt
            res=((pd.DataFrame(item_cnt,index=[0])).T).sort_values([0],ascending=[0]).index.tolist()
            # self.rec_item[u]=res
            self.rec_item[u] = [i for i in res if i not in user_item[u]]



# 产生推荐并通过准确率、召回率和覆盖率进行评估
def evaluate(model, test_data):
    print('Evaluating start ...')
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_items = set()

    for i, user in enumerate(model.train_data):
        test_items = test_data.get(user, {})  # 测试集中用户喜欢的物品
        rec_items = model.forward(user)  # 得到推荐的物品及计算出的用户对它们的兴趣

        for item in rec_items:  # 遍历给user推荐的物品
            if item in test_items:  # 测试集中有该物品
                hit += 1  # 推荐命中+1
            all_rec_items.add(item)
        rec_count += model.N
        test_count += len(test_items)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_items) / (1.0 * model.n_items)

    print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
    # precisioin=0.1004	recall=0.1014	coverage=0.0422
    # precisioin = 0.1177
    # recall = 0.1222
    # coverage = 0.0493

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = userCF(input_size[0],input_size[1])
    model.fit(loader['train'])
    # model = userCF("../data/ml-100k/u.data")
    evaluate(model, loader['valid'])
    print('done!')