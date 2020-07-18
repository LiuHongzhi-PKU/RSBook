# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

from enum import Enum
import math

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
        sum_rui = 0

        for user in model.test_data:
            for movie in model.test_data[user]:
                rui = model.predict(user, movie)  # 预测的评分
                if rui == 0:  # 说明用户u评分过的物品中没有i的邻域
                    continue
                count += 1
                sum_rui += math.fabs(model.test_data[user][movie] - rui)
        print("平均绝对值误差=", sum_rui / count)

    def evaluateTopN(self,model):
        print('Evaluating start ...')
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(model.train_data):
            test_moives = model.test_data.get(user, {})  # 测试集中用户喜欢的电影
            rec_movies = model.predict(user)  # 得到推荐的电影及计算出的用户对它们的兴趣

            for movie in rec_movies:  # 遍历给user推荐的电影
                if movie in test_moives:  # 测试集中有该电影
                    hit += 1  # 推荐命中+1
                all_rec_movies.add(movie)
            rec_count += model.N
            test_count += len(test_moives)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * model.n_items)

        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))











