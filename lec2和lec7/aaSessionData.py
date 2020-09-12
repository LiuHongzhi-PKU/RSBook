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

data_path="../data/sample_train-item-views.csv"
# data_fields = ['user_id', 'local_id', 'coordinate', 'time','date']
data_df = pd.read_table(data_path, sep=';')

# print(data_df.dropna())
data=data_df.dropna()

data = shuffle(data)

le = preprocessing.LabelEncoder()
le.fit(data['user_id'])
data['user_id'] = le.transform(data['user_id'])


le = preprocessing.LabelEncoder()
le.fit(data['item_id'])
data['item_id'] = le.transform(data['item_id'])

le = preprocessing.LabelEncoder()
le.fit(data['session_id'])
data['session_id'] = le.transform(data['session_id'])

# data = data.drop(data[data.session_id > 500].index)

n_users = len(set(data['user_id'].values))
n_sessions = len(set(data['session_id'].values))
n_items = len(set(data['item_id'].values))
data=data.sort_values(by=['session_id','user_id'])
print(data)
print(n_users)
print(n_items)
print(n_sessions)

# data.to_csv("data.txt")