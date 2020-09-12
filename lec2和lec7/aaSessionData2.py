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
from utils import Interactions,PairwiseInteractions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn import preprocessing
import csv


data_path = '../data/IJCAI16_data/ijcai2016_taobao1.csv'
data_df = pd.read_table(data_path, sep=',')
data=data_df.dropna()
# print(data)

data = shuffle(data)

data = data.drop(data[data.act_ID>0].index)

le = preprocessing.LabelEncoder()
le.fit(data['use_ID'])
data['use_ID'] = le.transform(data['use_ID'])


le = preprocessing.LabelEncoder()
le.fit(data['ite_ID'])
data['ite_ID'] = le.transform(data['ite_ID'])

# le = preprocessing.LabelEncoder()
# le.fit(data['time'])
# data['time'] = le.transform(data['time'])



data = data.drop(data[data.use_ID>1999].index)

# data = data.drop(data[data.ite_ID>1000].index)

n_users = len(set(data['use_ID'].values))
n_sessions = len(set(data['time'].values))
n_items = len(set(data['ite_ID'].values))

# print(data)
print(n_users)
print(n_items)
print(n_sessions)

data = data[['use_ID', 'ite_ID', 'time']]
data=data.reset_index()
data = data[['use_ID', 'ite_ID', 'time']]

data=data.sort_values(by=['use_ID','time'])
data=data.reset_index()
data = data[['use_ID', 'ite_ID', 'time']]

le = preprocessing.LabelEncoder()
le.fit(data['use_ID'])
data['use_ID'] = le.transform(data['use_ID'])


le = preprocessing.LabelEncoder()
le.fit(data['ite_ID'])
data['ite_ID'] = le.transform(data['ite_ID'])

le = preprocessing.LabelEncoder()
le.fit(data['time'])
data['time'] = le.transform(data['time'])

print(data)

n_users = len(set(data['use_ID'].values))
n_sessions = len(set(data['time'].values))
n_items = len(set(data['ite_ID'].values))

# print(data)
print(n_users)
print(n_items)
print(n_sessions)

df=data

df.rename(columns={'use_ID':'user_id', 'time':'timestamp', 'ite_ID':'item_id'}, inplace = True)

print(df)

# 获取每个user对应的basket序列
user_basket = {}
basket = {0: [int(df.iloc[0].item_id)]}
last_user = int(df.iloc[0].user_id)
last_t = df.iloc[0].timestamp
print(df.shape[0])
for i in range(df.shape[0]):
    print("i=",i)
    if i == 0 : continue
    user = int(df.iloc[i].user_id)
    flag=False
    if user != last_user:
        last_user = user
        flag=True
        basket = {}

    t = df.iloc[i].timestamp
    if t == last_t and flag==False:
        basket[len(basket) - 1] += [int(df.iloc[i].item_id)]
    else:
        basket[len(basket)] = [int(df.iloc[i].item_id)]
        last_t = t
    user_basket[user] = basket

print(user_basket)

