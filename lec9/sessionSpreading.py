# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType

from operator import itemgetter
from sklearn.utils import shuffle
import random
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from utils import evaluate
from utils import modelType
import csv
from sklearn import preprocessing
import argparse
class  sessionSpreading():
    # 参数
    # step: 扩散的步数
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # rec_item：在初始化模型时就已经给每个用户对所有物品排好序
    # user_item：用户到物品的映射
    # user_basket :用户到session的映射
    # basket_item:session到item的映射
    # weightS weightU :用户和session的扩散权重
    def __init__(self, data_file,step=5,N=10,weightU=1,weightS=1):
        self.step = step
        self.N = N  # 物品推荐数
        self.weightU = weightU  # 物品扩散给用户的权重
        self.weightS = weightS  # 物品扩散给session的权重

        self.loadData(data_file)  # 读取数据
        self.initModel()
        self.evaluate()

    def loadData(self, data_path="../data/ml-100k/u.data"):
        print("开始读数据")
        # load train data
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        data_df = pd.read_table(data_path, names=data_fields)
        # data_df=data_df[:100]
        data_df.rating = (data_df.rating >= 5).astype(np.float32)
        le = preprocessing.LabelEncoder()
        le.fit(data_df['user_id'])
        data_df['user_id'] = le.transform(data_df['user_id'])
        le.fit(data_df['item_id'])
        data_df['item_id'] = le.transform(data_df['item_id'])

        self.n_users = max(data_df['user_id'].values) + 1
        # get item number
        self.n_items = max(data_df['item_id'].values) + 1


        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

        df = data_df
        df.index = range(len(df))  # 重设index
        df = df.sort_values(['user_id', 'timestamp'])
        print(df)

        # 获取每个user对应的basket序列
        user_basket = {}
        basket = {0: [int(df.iloc[0].item_id)]}
        last_user = int(df.iloc[0].user_id)
        last_t = df.iloc[0].timestamp
        print(df.shape[0])
        for i in range(df.shape[0]):
            print(i)
            if i == 0 : continue

            user = int(df.iloc[i].user_id)
            if user != last_user:
                last_user = user
                basket = {}
            t = df.iloc[i].timestamp
            if t == last_t:
                basket[len(basket) - 1] += [int(df.iloc[i].item_id)]
            else:
                basket[len(basket)] = [int(df.iloc[i].item_id)]
                last_t = t
            user_basket[user] = basket


        user_basket_test=defaultdict(list)

        # print("去掉长度<3的session,去掉session数目<4的用户，然后每个用户拿4个session作为测试集")
        print("去掉session数目<2的用户,每个用户拿10%的session作为测试集")
        for i in range(self.n_users):
            # print(i)
            # for key in list(user_basket[i].keys()):
            #     if len(user_basket[i][key])<3:
            #         user_basket[i].pop(key)
            count=0
            for key in list(user_basket[i].keys()):
                user_basket[i][count]=user_basket[i].pop(key)
                count+=1
        for i in range(self.n_users):
            length = len(list(user_basket[i].keys()))
            if length < 2:
                user_basket.pop(i)

        count = 0
        for user in user_basket:
            count += len(user_basket[user])
        print("count=", count/len(user_basket))

        for i in list(user_basket.keys()):
            length = len(list(user_basket[i].keys()))

            len_test=int(length*0.1)
            if len_test ==0:
                len_test=1
            print(i,"  ",length, "   ", len_test)
            user_basket_test[i].append(user_basket[i].pop(length - len_test))

            for j in range(len_test):
                if j ==0 :
                    continue
                user_basket_test[i][0] += (user_basket[i].pop(length - j))

            # user_basket_test[i].append(user_basket[i].pop(length-4))
            # user_basket_test[i][0] += (user_basket[i].pop(length - 3))
            # user_basket_test[i][0] += (user_basket[i].pop(length - 2))
            # user_basket_test[i][0] += (user_basket[i].pop(length - 1))



        # for i in list(user_basket_test.keys()):
        #     print(user_basket_test[i])
        #
        # for i in list(user_basket.keys()):
        #     print(user_basket[i])

        self.user_basket_test=user_basket_test
        self.user_basket=user_basket

        user_item_test = {}
        for user in self.user_basket_test:
            user_item_test.setdefault(user, {})
            for item in self.user_basket_test[user][0]:
                user_item_test[user][item] = 1
        self.user_item_test = user_item_test




        count = 0
        for user in user_item_test:
            count += len(user_item_test[user])
        print("count=", count)
        # print("user_item_test:", self.user_item_test)

        user_item = {}
        for user in self.user_basket:
            user_item.setdefault(user, {})
            for basket in self.user_basket[user]:
                for item in user_basket[user][basket]:
                    user_item[user][item] = 1
        self.user_item = user_item
        # print("user_item:", self.user_item)

        basket_item = []
        for user in self.user_basket:
            for basket in self.user_basket[user]:
                basket_item.append(user_basket[user][basket])
        self.basket_item = basket_item
        # print("basket_item:", self.basket_item)


    def initModel(self):
        # 建立item_user倒排表
        item_users = dict()
        for u, items in self.user_item.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)


        # item_basket倒排表
        item_basket = dict()
        for basket in range(len(self.basket_item)):
            for i in self.basket_item[basket]:
                if i not in item_basket:
                    item_basket[i]=set()
                item_basket[i].add(basket)


        user_item = self.user_item
        basket_item = self.basket_item

        # print("here")
        # print(item_users)
        # print(user_item)
        # print(basket_item)
        # print(item_basket)


        self.rec_item = defaultdict(list)


        for u in tqdm(user_item):
            item_cnt = defaultdict(int)
            for i in user_item[u]:
                item_cnt[i] = 1

            for k in range(self.step):

                # 项目扩散给购物篮和用户
                basket_cnt = defaultdict(int)
                user_cnt = defaultdict(int)
                for item in item_cnt:
                    users=len(item_users[item])
                    basks=len(item_basket[item])
                    touser= self.weightU/ (users*self.weightU+basks*self.weightS)
                    tobask = self.weightS/ (users*self.weightU+basks*self.weightS)
                    for basket in item_basket[item]:
                        basket_cnt[basket] += item_cnt[item] * tobask
                    for user in item_users[item]:
                        user_cnt[user] += item_cnt[item] * touser

                item_cnt = defaultdict(int)




                # 购物篮扩散项目
                for basket in basket_cnt:
                    for item in basket_item[basket]:
                        item_cnt[item] += basket_cnt[basket] / (len(basket_item[basket]))



                # 用户扩散项目
                for user in user_cnt:
                    for item in user_item[user]:
                        item_cnt[item] += user_cnt[user] / len(user_item[user])


            # print(item_cnt)

            res = ((pd.DataFrame(item_cnt, index=[0])).T).sort_values([0], ascending=[0]).index.tolist()
            # print("res:",res)

            self.rec_item[u] = [i for i in res if i not in user_item[u]]
            # print("rec_item:",self.rec_item[u][0:self.N])

    def predict(self, user):
        # 返回最大N个物品
        # print(self.rec_item[user][0:self.N])
        return self.rec_item[user][0:self.N]


    def evaluate(self):
        print("开始评估")
        # # 计算top10的recall、precision、推荐物品覆盖率

        user_item = self.user_basket_test

        hit, rec_count, test_count, all_rec_items = 0, 0, 0, set()

        for u in user_item:

            target_items=list(self.user_item_test[u].keys())
            # print(target_items)

            rec_list = self.predict(u)
            # print("开始给",u)
            # print(target_items)
            # print(rec_list)


            for item in rec_list:  # 遍历给user推荐的物品
                if item in target_items:  # 测试集中有该物品
                    hit += 1  # 推荐命中+1
                all_rec_items.add(item)
            rec_count += self.N
            test_count += len(target_items)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.n_items)
        print('precisioin=%.8f\trecall=%.8f\tcoverage=%.8f' % (precision, recall, coverage))
        print(rec_count)
        print(test_count)
        print(hit)


parser = argparse.ArgumentParser()
parser.add_argument('--N',type=int, default=10, help='物品推荐数')
parser.add_argument('--step',type=int, default=10, help='扩散步长，item->user(session)->item算一步')
parser.add_argument('--weightU',type=int, default=1, help='item向user扩散的权重')
parser.add_argument('--weightS',type=int, default=1, help='item向session扩散的权重')
opt = parser.parse_args()



if __name__ == '__main__':
    model = sessionSpreading("../data/ml-100k/u.data",N=opt.N,step=opt.step,weightS=opt.weightS,weightU=opt.weightU)



# ml-100k
# 两个session
# 都为1
# precisioin=0.05595568	recall=0.06909526	coverage=0.04280618
# 不考虑session
# precisioin=0.04681440	recall=0.05780742	coverage=0.03032105
# 不考虑用户
# precisioin=0.06218837	recall=0.07679152	coverage=0.07372176

# 20%
# precisioin=0.11155885	recall=0.05886961	coverage=0.05112961

# 10%
# precisioin=0.06511135	recall=0.07097445	coverage=0.05410226