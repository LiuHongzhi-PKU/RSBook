import random
import pandas as pd
import numpy as np
import math
from utils import evaluate
from utils import modelType

from operator import itemgetter

import random
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from utils import evaluate
from utils import modelType
import csv
from sklearn import preprocessing



np.random.seed(1024)


class itemCF():
    # 参数:
    # K：近邻数目
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # item_sim：项目之间的相似度。二维字典。i->j->相似度
    # item_sim_basket：根据session得到的item相似度
    # similarityMeasure：相似度度量，cosine或conditional
    # user_basket :用户到session的映射
    # basket_item:session到item的映射
    # w1 w2 ：根据用户和session计算得到的物品相似度的权重
    def __init__(self, data_file, K=20,N=10,similarityMeasure="cosine",w1=1,w2=1):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.w1 = w1
        self.w2 = w2#     用户和session相似度的权重
        self.similarityMeasure = similarityMeasure
        self.loadData(data_file)
        self.initModel()            # 初始化模型
        self.evaluate()



    def initModel(self):
        # 计算每个物品被用户评分的个数
        item_cnt = {}
        for user, items in self.user_item.items():
            for i in items:
                # count item popularity
                item_cnt.setdefault(i,0)
                item_cnt[i] += 1

        # 计算物品之间共同评分的物品数,C为修正后的，count为修正前的。
        count=dict()
        for user, items in self.user_item.items():
            for u in items:
                for v in items:
                    if u == v:
                        continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        # 计算最终的物品相似度矩阵
        self.item_sim = dict()
        for u, related_items in count.items():
            self.item_sim[u]={}
            for v, cuv in related_items.items():
                self.item_sim[u][v] = cuv / math.sqrt(item_cnt[u] * item_cnt[v])


        # 计算每个物品在购物篮里的次数
        item_cnt_basket = {}
        for basket in range(len(self.basket_item)):
            for i in self.basket_item[basket]:
                # count item popularity
                item_cnt_basket.setdefault(i,0)
                item_cnt_basket[i] += 1

        count_basket= dict()
        for basket in range(len(self.basket_item)):
            for i in self.basket_item[basket]:
                for j in self.basket_item[basket]:
                    if i==j:
                        continue
                    count_basket.setdefault(i,{})
                    count_basket[i].setdefault(j, 0)
                    count_basket[i][j]+=1

        # session相似度

        self.item_sim_basket = dict()
        for u,related_items in count_basket.items():
            self.item_sim_basket[u]={}
            for v, cuv in related_items.items():
                self.item_sim_basket[u][v] = cuv / math.sqrt((item_cnt_basket[u]*item_cnt_basket[v]))

        for i in range(self.n_items):
            for j in range(self.n_items):
                self.item_sim.setdefault(i,{})
                self.item_sim[i].setdefault(j,0)

                self.item_sim_basket.setdefault(i, {})
                self.item_sim_basket[i].setdefault(j, 0)

        for i in self.item_sim:
            for j in self.item_sim[i]:
                self.item_sim[i][j]= self.item_sim[i][j] * self.w1 + self.item_sim_basket[i][j] *self.w2

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
            if i == 0: continue

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

        user_basket_test = defaultdict(list)

        # print("去掉长度<3的session,去掉session数目<4的用户，然后每个用户拿4个session作为测试集")
        print("去掉session数目<2的用户,每个用户拿10%的session作为测试集")
        for i in range(self.n_users):
            # print(i)
            # for key in list(user_basket[i].keys()):
            #     if len(user_basket[i][key])<3:
            #         user_basket[i].pop(key)
            count = 0
            for key in list(user_basket[i].keys()):
                user_basket[i][count] = user_basket[i].pop(key)
                count += 1
        for i in range(self.n_users):
            length = len(list(user_basket[i].keys()))
            if length < 2:
                user_basket.pop(i)

        count = 0
        for user in user_basket:
            count += len(user_basket[user])
        print("count=", count / len(user_basket))

        for i in list(user_basket.keys()):
            length = len(list(user_basket[i].keys()))

            len_test = int(length * 0.1)
            if len_test == 0:
                len_test = 1
            print(i, "  ", length, "   ", len_test)
            user_basket_test[i].append(user_basket[i].pop(length - len_test))

            for j in range(len_test):
                if j == 0:
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

        self.user_basket_test = user_basket_test
        self.user_basket = user_basket

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


    def predict(self, user):
        rank = dict()
        interacted_items = self.user_item[user]

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




if __name__ == '__main__':
    model = itemCF("../data/ml-100k/u.data")



# 2个会话作为测试集
# precisioin=0.06994460	recall=0.08636908	coverage=0.21165279
# 不考虑session
# precisioin=0.07146814	recall=0.08825038	coverage=0.19738407
# 不考虑用户
#precisioin=0.04916898	recall=0.06071490	coverage=0.53507729


# 10%
# precisioin=0.08419936	recall=0.09178130	coverage=0.16111772