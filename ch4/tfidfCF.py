import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from operator import itemgetter
from tqdm import tqdm
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

class itemCF():
    def __init__(self, data_file, K=20, N=10):
        self.K = K  # 近邻数
        self.N = N  # 物品推荐数
        self.readData(data_file)  # 读取数据
        self.initModel(data_file)  # 初始化模型

    def initModel(self,data_file):
        news_df = pd.read_csv(data_file + 'news_df_small_new.csv')
        corpus = news_df.text.tolist()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  # max_features=1000,
                                           stop_words='english')
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            tfidf_vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵

        # word = tfidf_vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        SimMatrix = (tfidf * tfidf.T).A
        weight_norm = np.sqrt((weight * weight).sum(1))
        cos_sim = SimMatrix
        for i in tqdm(range(cos_sim.shape[0])):
            for j in range(cos_sim.shape[0]):
                cos_sim[i][j] = cos_sim[i][j] / (weight_norm[i] * weight_norm[j])
        item_sim = dict()
        for i in tqdm(range(weight.shape[0])):
            item_sim[news_df.iloc[i].news_id] = {}
            ids = np.argsort(cos_sim[i])[-100:]  # 相似的前100个物品
            for j in ids:
                if i == j: continue
                item_sim[news_df.iloc[i].news_id][news_df.iloc[j].news_id] = cos_sim[i][j]  # get_fast_cos(i,j)
        self.item_sim=item_sim

    def readData(self, data_file):
        df = pd.read_csv(data_file+'rating_small_20_new.csv')
        df = df.reset_index()
        test_df = df.groupby('user_id').apply(lambda x: x.iloc[int(x.shape[0] * 9. / 10):]['index'].tolist())
        test_id = [j for i in test_df.values.tolist() for j in i]
        self.test_data = {}
        self.train_data = {}

        for i in range(df.shape[0]):
            x = df.iloc[i]
            user = x['user_id']
            item = x['item_id']
            record = x['rating']
            if i not in test_id:
                self.test_data.setdefault(user, {})
                self.test_data[user][item] = record
            else:
                self.train_data.setdefault(user, {})
                self.train_data[user][item] = record

        data = df

        self.n_users = len(set(data['user_id'].values))
        self.n_items = len(set(data['item_id'].values))

        print("Initialize end.The user number is:%d,item number is:%d" % (self.n_users, self.n_items))

    def forward(self, user):
        rank = dict()
        interacted_items = self.train_data[user]

        # 对每个评分的物品寻找最近K个物品，构建评分列表
        for item, rating in interacted_items.items():
            for similar_item, similarity_factor in sorted(self.item_sim[item].items(),
                                                          key=itemgetter(1), reverse=True)[:self.K]:
                if similar_item in interacted_items:
                    continue
                rank.setdefault(similar_item, 0)
                # rank[similar_item] += similarity_factor * rating
                rank[similar_item] += similarity_factor

        # 返回最大N个物品
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]


# 产生推荐并通过准确率、召回率和覆盖率进行评估
def evaluate(model):
    print('Evaluating start ...')
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_movies = set()

    for i, user in enumerate(model.train_data):
        test_moives = model.test_data.get(user, {})  # 测试集中用户喜欢的电影
        rec_movies = model.forward(user)  # 得到推荐的电影及计算出的用户对它们的兴趣

        for movie, w in rec_movies:  # 遍历给user推荐的电影
            if movie in test_moives:  # 测试集中有该电影
                hit += 1  # 推荐命中+1
            all_rec_movies.add(movie)
        rec_count += model.N
        test_count += len(test_moives)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_movies) / (1.0 * model.n_items)

    print('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

if __name__ == '__main__':
    model = itemCF("../data/news/")
    evaluate(model)

# Initialize end.The user number is:1364,item number is:4253
# 100%|██████████| 4253/4253 [00:16<00:00, 261.33it/s]
# 100%|██████████| 4253/4253 [01:24<00:00, 50.54it/s]
# Evaluating start ...
# precision=0.0226	recall=0.0089	coverage=0.5537