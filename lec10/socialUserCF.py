import random
import pandas as pd
import numpy as np
import math
from enum import Enum
from operator import itemgetter
from sklearn import preprocessing
from collections import defaultdict
from utils import evaluate
from utils import modelType

np.random.seed(1111)

class socialUserCF():
    # 参数:
    # K：近邻数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # potential_users：可以进行评分预测的用户集合
    # average_rating：每个用户的平均评分、字典。用户->平均评分
    # familiarity：用户之间的熟悉度。二维字典。u->v->相似度
    # interest_sim：用户之间的兴趣相似度。二维字典。u->v->相似度
    # user_sim：由熟悉度和兴趣相似度构建的用户相似度。二维字典。u->v->相似度
    # alpha：计算用户相似度时熟悉度的权重，兴趣相似度的权重为1-alpha
    def __init__(self, data_file, K=20,alpha=0.8):
        self.K = K  # 近邻数
        self.alpha = alpha
        self.loadData(data_file)  # 读取数据
        self.initModel()  # 初始化模型

    def initModel(self):
        # 计算每个用户的平均评分
        self.average_rating = {}
        for u, items in self.train_data.items():
            self.average_rating.setdefault(u, 0)
            for i in items:
                self.average_rating[u] += self.train_data[u][i] / len(items)

        # 建立item_user倒排表
        # item->set
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)


        # 计算最终的用户相似度矩阵

        self.familiarity = dict()
        for u, trust_u in self.social_user.items():
            self.familiarity[u] = {}
            for v, trust_v in self.social_user.items():
                if u == v:
                    continue
                if len(trust_u.union(trust_v))!=0:
                    self.familiarity[u][v] = len(trust_u.intersection(trust_v)) / len(trust_u.union(trust_v))
                else:
                    self.familiarity[u][v] = 0

        self.interest_sim = dict()
        for u, items_u in self.train_data.items():
            self.interest_sim[u] = {}
            for v, items_v in self.train_data.items():
                if u == v:
                    continue
                set_u=set(items_u)
                set_v=set(items_v)
                self.interest_sim[u][v] = len(set_u.intersection(set_v)) / len(set_u.union(set_v))
                
        alpha = 0.3
        self.user_sim = dict()
        self.potential_users = set(self.train_data).union(set(self.social_user))
        for u in self.potential_users:
            self.user_sim[u] = {}
            for v in self.potential_users:
                if u == v:
                    continue
                self.user_sim[u][v] = 0
                if u in self.familiarity and v in self.familiarity[u]:
                  self.user_sim[u][v] += alpha * self.familiarity[u][v]
                if u in self.interest_sim and v in self.interest_sim[u]:
                  self.user_sim[u][v] += (1 - alpha) * self.interest_sim[u][v]
    def loadData(self, data_path):

        # 二维字典
        self.test_data = {}
        self.train_data = {}
        trainset_rate = 0.9

        data_df = pd.read_table(data_path+'data_df.txt',sep=' ')
        data_df['rating'] /= max(data_df['rating'])
        social_df = pd.read_table(data_path+'social_df.txt',sep=' ')

        le = preprocessing.LabelEncoder()
        le.fit(data_df['user_id'])
        data_df['user_id']=le.transform(data_df['user_id'])
        social_df['user_id']=le.transform(social_df['user_id'])
        social_df['user_id2']=le.transform(social_df['user_id2'])
        le.fit(data_df['item_id'])
        data_df['item_id']=le.transform(data_df['item_id'])

        df_train = data_df.sample(n=int(len(data_df) * trainset_rate), replace=False)
        df_test = data_df.drop(df_train.index, axis=0)

        for (user, item, record) in df_test.itertuples(index=False):
              self.test_data.setdefault(user, {})
              self.test_data[user][item] = record
        for (user, item, record) in df_train.itertuples(index=False):
              self.train_data.setdefault(user, {})
              self.train_data[user][item] = record

        self.social_user = {}
        for (user, user2, record) in social_df.itertuples(index=False):
            self.social_user.setdefault(user, set())
            self.social_user[user].add(user2)

        self.n_users = len(set(data_df['user_id'].values))
        self.n_items = len(set(data_df['item_id'].values))

        print("Initialize end. The user number is: %d, item number is: %d" % (self.n_users, self.n_items))
    def predict(self, user, item):
        if user not in model.potential_users:
          return 0
        # 分子和分母
        C1 = 0
        C2 = 0
        if user in self.train_data:
          r_ui = self.average_rating[user]
          for similar_user, similarity_factor in sorted(self.user_sim[user].items(),key=itemgetter(1), reverse=True)[0:self.K]:
              if similar_user not in self.train_data or item not in self.train_data[similar_user]:
                  continue
              C1 += similarity_factor * (self.train_data[similar_user][item] - self.average_rating[similar_user])
              C2 += math.fabs(similarity_factor)
        else:
          r_ui = 0
          for similar_user, similarity_factor in sorted(self.user_sim[user].items(),key=itemgetter(1), reverse=True)[0:self.K]:
              if similar_user not in self.train_data or item not in self.train_data[similar_user]:
                  continue
              C1 += similarity_factor * self.train_data[similar_user][item]
              C2 += math.fabs(similarity_factor)
        if not C1==0:
          r_ui += (C1 / C2)
        else:
          r_ui=0
        return r_ui


if __name__ == '__main__':
    model = socialUserCF("../data/epinions/")
    ev = evaluate(modelType.rating)
    ev.evaluateModel(model)
    print('done!')
        
    # Initialize end. The user number is: 3139, item number is: 8145
    # Evaluating start ...
    # 平均绝对值误差= 0.19047773008069102
    # done!
