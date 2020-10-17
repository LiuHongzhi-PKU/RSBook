import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd

# 设置是否使用隐式反馈
IMPLICT=False
# 设置是否使用超小数据集测试
SMALL=False

# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# To compute probalities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 获得dataloader，将数据读入后使用Interactions加载，Interactions是pytorch的取样器，可以返回df对应的u,i,r，详情见utils.py
def getDataLoader(data_path, batch_size=2048):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 5).astype(np.float32)

    # 数据离散化编码
    le = preprocessing.LabelEncoder()
    le.fit(data_df['user_id'])
    data_df['user_id']=le.transform(data_df['user_id'])
    le.fit(data_df['item_id'])
    data_df['item_id']=le.transform(data_df['item_id'])

    df_train = data_df.sample(n=int(len(data_df) * 0.8), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)

    # get user number
    n_users = max(data_df['user_id'].values)+1
    # get item number
    n_items = max(data_df['item_id'].values)+1

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items ), loaders

#
class SVD(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=80,topn=10, sparse=False, device=torch.device("cpu")):
        super(SVD, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.topn = topn
        # get factor number
        self.n_factors = n_factors
        self.user_vec=None
        self.item_vec=None
        self=self.to(self.device)

    def _convert_df(self, user_num, item_num, df):
        ratings = list(df['rating'])
        rows = list(df['user_id'])
        cols = list(df['item_id'])
        mat = sp.csr_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

    def fit(self,loaders):
        # load train data
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        data_path="u.data"
        data_df = pd.read_table(data_path, names=data_fields)
        print(" SVD START")
        train_set = self._convert_df(self.n_users, self.n_items, data_df)
        U, sigma, Vt = randomized_svd(train_set,
                                      n_components=self.n_factors,
                                      random_state=2020)
        s_Vt = sp.diags(sigma) * Vt
        print('SVD END')
        self.user_vec = U
        self.item_vec = s_Vt.T

    def predict(self, u, i):
        return self.user_vec[u, :].dot(self.item_vec[i, :])


if __name__ == '__main__':
    input_size, loader=getDataLoader("u.data")
    # 从getDataLoader中得到模型需要的初始化参数，如用户数与物品数
    model = SVD(input_size[0],input_size[1])
    model.fit()
