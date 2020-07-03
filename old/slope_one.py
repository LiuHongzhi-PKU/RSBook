import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn

from operator import itemgetter
import os
# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def getDataLoader(data_path):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    # data_df.rating = (data_df.rating >= 4).astype(np.float32)

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

class SLOPE():
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.item_users = dict()
        self.train_data = dict()

    def forward(self, user, j):
        rating = 0
        cnt = 0
        for i in self.train_data[user]:
            if i == j:
                continue
            common_users = self.item_users[i] & self.item_users[j]
            if len(common_users) == 0:
                continue
            dev_ij = 0
            cnt += 1
            for common_user in common_users:
                dev_ij += self.train_data[common_user][j] - self.train_data[common_user][i]
            dev_ij /= len(common_users)
            rating += dev_ij
        rating /= cnt
        return rating

    def fit(self, train_data):
        self.train_data = train_data
        # 建立item_user倒排表
        item_users = dict()
        for u, items in train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)
        self.item_users = item_users


def evaluate(model, test_data):
    print('Evaluating start ...')

    cnt = 0
    loss = 0
    for i, user in enumerate(test_data):
        if user not in model.train_data:
            continue
        for item in test_data[user]:
            if item not in model.item_users:
                continue
            pred = model.forward(user, item)
            val = test_data[user][item]
            loss += (pred - val) ** 2
            cnt += 1

    loss /= cnt
    print('loss=%.4f' % (loss))

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = SLOPE(input_size[0],input_size[1])
    model.fit(loader['train'])
    # model = userCF("../data/ml-100k/u.data")
    evaluate(model, loader['valid'])
    print('done!')