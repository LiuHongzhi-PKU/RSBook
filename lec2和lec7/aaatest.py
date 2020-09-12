# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin
import math
import random
from collections import defaultdict
from operator import itemgetter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions,PPushCRInteractions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import argparse
from geopy.distance import geodesic




data=pd.read_csv("../data/poidata/Foursquare/mydata.txt",index_col=0)
data=data.sort_values(by=["date","hour","minute"])

le = preprocessing.LabelEncoder()
le.fit(data['user_id'])
data['user_id'] = le.transform(data['user_id'])
le = preprocessing.LabelEncoder()
le.fit(data['local_id'])
data['local_id'] = le.transform(data['local_id'])


n_users= len(set(data['user_id'].values))
n_items=len(set(data['local_id'].values))


length = len(data)
train = data[:int(0.9 * length)]
test = data[int(0.9 * length):]


test_data = {}
train_data = {}

locateUser={}
locateItem={}
print(train)
for (user, item, x,y,date,hour,minute) in train.itertuples(index=False):
    locateUser[user] = (x,y)
    locateItem[item] = (x, y)

# print(locateUser[78][0],"   ",locateUser[78][1])
# print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m)
dis={}
print("计算距离")
for i in range(n_users):
    if i not in  locateUser:
        continue
    dis.setdefault(i,{})
    for j in range(n_items):
        if j not in locateItem:
            continue
        dis[i][j]=geodesic((locateUser[i][1],locateUser[i][0]),(locateItem[j][1],locateItem[j][0])).m
        # print(i,"  ",j,"   ",dis[i][j])




