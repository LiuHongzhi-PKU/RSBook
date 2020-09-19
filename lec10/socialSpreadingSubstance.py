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
from tqdm import tqdm

np.random.seed(1111)

class  socialSpreadingSubstance():
    # 参数
    # step: 扩散的轮数
    # N：物品推荐数目
    # test_data：测试数据，二维字典。user->item->评分
    # train_data：训练数据
    # n_users：用户数目
    # n_items：项目数目
    # rec_item：在初始化模型时就已经给每个用户对所有物品排好序
    def __init__(self, data_file,step=2,N=10):
        self.step = step
        self.N = N  # 物品推荐数
        self.loadData(data_file)  # 读取数据
        self.initModel()


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

    def initModel(self):
        # 建立item_user倒排表
        item_users = dict()
        for u, items in self.train_data.items():
            for i in items:
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)


        user_items = self.train_data
        user_users = self.social_user
        self.rec_item = defaultdict(list)
        for u in tqdm(user_items):
            item_cnt = defaultdict(int)
            user_cnt = defaultdict(int)
            user_cnt_new = defaultdict(int)
            for i in user_items[u]:
                item_cnt[i] = 1

            for k in range(3*self.step):
                if k % 2 == 0:
                    # 项目扩散用户
                    user_cnt = defaultdict(int)
                    for item in item_cnt:
                        for user in item_users[item]:
                            user_cnt[user] += item_cnt[item] / len(item_users[item])
                elif k % 2 ==1:
                    # 用户扩散项目和信任用户
                    item_cnt = defaultdict(int)
                    user_cnt_new = defaultdict(int)
                    for user in user_cnt:
                        for item in user_items[user]:
                            if user in user_users:
                                item_cnt[item] += user_cnt[user] / (len(user_items[user])+len(user_users[user]))
                            else:
                                item_cnt[item] += user_cnt[user] / len(user_items[user])
                        if user in user_users:
                            for trusted_user in user_users[user]:
                                user_cnt_new[trusted_user] += user_cnt[user] / (len(user_items[user])+len(user_users[user]))
                else:
                    # 用户扩散项目
                    for user in user_cnt_new:
                        for item in user_items[user]:
                            item_cnt[item] += user_cnt_new[user] / len(user_items[user])
            res = ((pd.DataFrame(item_cnt, index=[0])).T).sort_values([0], ascending=[0]).index.tolist()
            # print(res)

            self.rec_item[u] = [i for i in res if i not in user_items[u]]
            # print(self.rec_item[u][0:self.N])

    def predict(self, user):
        # 返回最大N个物品
        # print(self.rec_item[user][0:self.N])
        return self.rec_item[user][0:self.N]


if __name__ == '__main__':
    model = socialSpreadingSubstance("../data/epinions/")
    ev = evaluate(modelType.topN)
    ev.evaluateModel(model)
    print('done!')

    # 100%|██████████| 3123/3123 [30:44<00:00,  1.69it/s]
    # Evaluating start ...
    # precisioin=0.0156	recall=0.0415	coverage=0.0413
    # done!
