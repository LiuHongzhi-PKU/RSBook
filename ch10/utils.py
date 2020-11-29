import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
np.random.seed(1024)
from operator import itemgetter
from enum import Enum

class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, df):
        df.index=range(len(df)) # 重设index
        self.df=df
        self.user_item = {}
        for (user, item, record) in df.itertuples(index=False):
            self.user_item.setdefault(user-1,{})
            self.user_item[user-1][item-1] = record

    def __getitem__(self, index):
        user = int(self.df.loc[index]['user_id'])
        item = int(self.df.loc[index]['item_id'])
        rating = float(self.df.loc[index]['rating'])
        return (user,item), rating

    def __len__(self):
        return len(self.df)

# 枚举类  模型评估时选择使用哪种方式
class modelType(Enum):
    topN=1
    rating=2

# 用于模型评估
class evaluate():
    def __init__(self,type):
        self.type=type

    def evaluateModel(self,model):
        if self.type == modelType.topN:
            self.evaluateTopN(model)
        else :
            self.evaluateRating(model)

    def evaluateRating(self,model):
        print('Evaluating start ...')
        count = 0
        mae = 0
        mse = 0

        for user in model.test_data:
            for item in model.test_data[user]:
                rui = model.predict(user, item)  # 预测的评分
                if rui == 0:  # 说明用户u评分过的物品中没有i的邻域
                    continue
                count += 1
                mae += math.fabs(model.test_data[user][item] - rui)
                mse += math.pow(model.test_data[user][item] - rui,2)
        mae /= count
        mse /= count
        print("平均绝对值误差=", mae)
        print("均方误差=", mse)
        print("均方根误差=", math.sqrt(mse))

    def evaluateTopN(self,model):
        print('Evaluating start ...')
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_items = set()

        for i, user in enumerate(model.train_data):
            test_items = model.test_data.get(user, {})  # 测试集中用户喜欢的物品
            if len(test_items) == 0:
                continue
            rec_items = model.predict(user)  # 得到推荐的物品及计算出的用户对它们的兴趣

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

