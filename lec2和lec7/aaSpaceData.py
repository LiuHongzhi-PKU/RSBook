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



#显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('precision', 16)

# 对空间的数据集进行采样
data_path="../data/poidata/Foursquare/train.txt"
data_fields = ['user_id', 'local_id', 'coordinate', 'time','date']
data_df = pd.read_table(data_path, names=data_fields,sep='	')

data_df['y'], data_df['x'] = data_df['coordinate'].str.split(',', 1).str
data_df['hour'], data_df['minute'] = data_df['time'].str.split(':', 1).str
data = data_df[['user_id', 'local_id', 'x', 'y','date','hour','minute']]
data = shuffle(data)
print(data)

le = preprocessing.LabelEncoder()
le.fit(data['user_id'])
data['user_id'] = le.transform(data['user_id'])
le = preprocessing.LabelEncoder()
le.fit(data['local_id'])
data['local_id'] = le.transform(data['local_id'])
df=data
print("aaaaa")
print(df)
df["user_id"] = df["user_id"].astype("int")
df["local_id"] = df["local_id"].astype("int")
df["x"] = df["x"].astype("float")
df["y"] = df["y"].astype("float")
df["date"] = df["date"].astype("int")
df["hour"] = df["hour"].astype("int")
df["minute"] = df["minute"].astype("int")
print(df)
print(df.dtypes)

df = df.drop(df[df.user_id > 500].index)
df = df.drop(df[df.local_id > 500].index)
df=df.reset_index(drop=True)
le = preprocessing.LabelEncoder()
le.fit(data['user_id'])
data['user_id'] = le.transform(data['user_id'])
le = preprocessing.LabelEncoder()
le.fit(data['local_id'])
data['local_id'] = le.transform(data['local_id'])

print(df)
df.to_csv("../data/poidata/Foursquare/mydata.txt")