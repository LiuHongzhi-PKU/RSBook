import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
np.random.seed(1024)
from operator import itemgetter

class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, df, index_from_one=True):
        df.index=range(len(df)) # 重设index
        self.df=df
        self.user_item = {}
        self.index_from_one=index_from_one
        # for (user, item, record, timestamp) in df.itertuples(index=False):
        for i in range(len(df)):
            x=df.iloc[i]
            user, item, record=x['user_id'],x['item_id'],x['rating']
            if index_from_one:
                self.user_item.setdefault(user-1,{})
                self.user_item[user-1][item-1] = record
            else:
                self.user_item.setdefault(user,{})
                self.user_item[user][item] = record

    def __getitem__(self, index):
        if self.index_from_one:
            user = int(self.df.loc[index]['user_id'])-1
            item = int(self.df.loc[index]['item_id'])-1
            rating = float(self.df.loc[index]['rating'])
        else:
            user = int(self.df.loc[index]['user_id'])
            item = int(self.df.loc[index]['item_id'])
            rating = float(self.df.loc[index]['rating'])
        return (user,item), rating

    def __len__(self):
        return len(self.df)

class HinInteractions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, df, index_from_one=True):
        df.index=range(len(df)) # 重设index
        self.df=df
        self.user_item = {}
        self.index_from_one=index_from_one
        for (user, item, record, timestamp,hin_y) in df.itertuples(index=False):
            if index_from_one:
                self.user_item.setdefault(user-1,{})
                self.user_item[user-1][item-1] = record
            else:
                self.user_item.setdefault(user,{})
                self.user_item[user][item] = record

    def __getitem__(self, index):
        if self.index_from_one:
            user = int(self.df.loc[index]['user_id'])-1
            item = int(self.df.loc[index]['item_id'])-1
            rating = float(self.df.loc[index]['rating'])
        else:
            user = int(self.df.loc[index]['user_id'])
            item = int(self.df.loc[index]['item_id'])
            rating = float(self.df.loc[index]['rating'])
        return (user,item), (rating,self.df.loc[index]['y_hin'])

    def __len__(self):
        return len(self.df)

class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, df, n_items):
        df.index = range(len(df)) # 重设index
        self.df = df
        self.n_items = n_items

        self.user_item = {}
        self.neg_item={}
        for (user, item, record, timestamp) in df.itertuples(index=False):
            if record == 1:
                self.user_item.setdefault(user - 1, {})
                self.user_item[user-1][item-1] = record
            else:
                self.neg_item.setdefault(user - 1, {})
                self.neg_item[user - 1][item - 1] = record

        # self.user_item = {}
        # for (user, item, record, timestamp) in df.itertuples(index=False):
        #     self.user_item.setdefault(user-1,{})
        #     self.user_item[user-1][item-1] = record


    def __getitem__(self, index):
        user = int(self.df.loc[index]['user_id']) - 1
        found = False

        while not found:
            neg_col = np.random.randint(self.n_items)
            # print(self.user_item.shape,user)
            if neg_col in self.neg_item[user]:
                found = True
            # if neg_col not in self.user_item[user]:
            #     found = True

        pos_col = int(self.df.loc[index]['item_id']) - 1
        rating = float(self.df.loc[index]['rating'])
        return (user, (pos_col, neg_col)), rating

    def __len__(self):
        return len(self.df)

class FPMCInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, df):
        df.index = range(len(df)) # 重设index
        df = df.sort_values(['user_id', 'timestamp'])

        user_basket = {}
        basket = {0: [df.iloc[0].item_id]}
        last_user = df.iloc[0].user_id
        last_t = df.iloc[0].timestamp
        for i in range(df.shape[0]):
            if i == 0 or df.iloc[i].rating <= 0: continue

            user = df.iloc[i].user_id
            if user != last_user:
                last_user = user
                basket = {}
            t = df.iloc[i].timestamp
            if t == last_t:
                basket[len(basket) - 1] += [df.iloc[i].item_id]
            else:
                basket[len(basket)] = [df.iloc[i].item_id]
                last_t = t
            user_basket[user] = basket
        self.user_basket = user_basket

        res = []
        for i in user_basket:
            res += [(i, j) for j in range(len(user_basket[i]))]
        res = pd.DataFrame(res, columns=['user_id', 't'])
        self.df=res



    def __getitem__(self, index):
        user = int(self.df.loc[index]['user_id'])
        t= int(self.df.loc[index]['t'])
        # print(user,t)

        pos_set=self.user_basket[user][t]
        last_set=[]
        if t>0: last_set=self.user_basket[user][t-1]

        rating = float(1)
        return (user, (pos_set, last_set)), rating

    def __len__(self):
        return len(self.df)


class CPLR_Interactions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, df, n_items, user_sim, K):
        df.index = range(len(df)) # 重设index
        self.df = df
        self.n_items = n_items
        self.user_sim = user_sim
        self.K = K

        self.user_item = {}
        for (user, item, record, timestamp) in df.itertuples(index=False):
            self.user_item.setdefault(user-1,{})
            self.user_item[user-1][item-1] = record
        self.col_usr_item = {}
        for user in self.user_item:
            self.col_usr_item.setdefault(user, set())
            for similar_user, similarity_factor in sorted(self.user_sim[user].items(),
                                                          key=itemgetter(1), reverse=True)[0:self.K]:
                self.col_usr_item[user].update(self.user_item[similar_user])
            self.col_usr_item[user] = self.col_usr_item[user] - set(self.user_item[user])



    def __getitem__(self, index):
        user = int(self.df.loc[index]['user_id']) - 1
        found = False

        col_col = random.sample(self.col_usr_item[user], 1)[0]
        while not found:
            lef_col = np.random.randint(self.n_items)
            # print(self.user_item.shape,user)
            if lef_col not in self.user_item[user] and lef_col not in self.col_usr_item[user]:
                found = True

        pos_col = int(self.df.loc[index]['item_id']) - 1
        rating = float(self.df.loc[index]['rating'])
        return (user, (pos_col, col_col, lef_col)), rating

    def __len__(self):
        return len(self.df)

class FieldLoader(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, df, feature_split):
        df.index=range(len(df)) # 重设index
        self.feature_split=feature_split
        self.df=df

    def __getitem__(self, index):
        x=[]
        for j in range(len(self.feature_split) - 1):
            # print(self.df.values[index][self.feature_split[j]:self.feature_split[j + 1]])
            # print(self.feature_split[j])
            val = np.argmax(self.df.values[index][self.feature_split[j]:self.feature_split[j + 1]]) + self.feature_split[j]
            x.append(val)
        # user = int(self.df.loc[index]['user_id']) - 1
        # item = int(self.df.loc[index]['item_id']) - 1
        rating = float(self.df.loc[index]['rating'])
        return np.array(x), rating

    def __len__(self):
        return len(self.df)


def compute_mrr(data,model,U,V,test_users=None):
    """compute average Mean Reciprocal Rank of data according to factors
    params:
      data      : scipy csr sparse matrix containing user->(item,count)
      U         : user factors
      V         : item factors
      test_users: optional subset of users over which to compute MRR
    returns:
      the mean MRR over all users in data
    """
    mrr = []
    if test_users is None:
        test_users = range(len(U))
    for user in enumerate(test_users):
        items = test_users[user]
        users=torch.tensor([user for i in items])
        predictions=model.predict(users,items)
        predictions = np.sum(np.tile(U[i],(len(V),1))*V,axis=1)
        for rank,item in enumerate(np.argsort(predictions)[::-1]):
            if item in items:
                mrr.append(1.0/(rank+1))
                break
    assert(len(mrr) == len(test_users))
    return np.mean(mrr)