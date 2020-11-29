import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from IPython import embed

IMPLICT=False
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

def drop_df(df):
    pos_cnt = df.groupby('user_id', as_index=False)['rating'].agg({"pos_cnt": 'sum'})
    tot_cnt = df.groupby('user_id', as_index=False)['rating'].agg({"tot_cnt": 'count'})
    df = pd.merge(df, pos_cnt, on=['user_id'], how='left')
    df = pd.merge(df, tot_cnt, on=['user_id'], how='left')
    df = df[(df.pos_cnt > 0) & (df.tot_cnt > df.pos_cnt)]
    df = df.drop(['pos_cnt', 'tot_cnt'], axis=1)
    return df

# 元路径寻找前10近邻
K=10
def calSim(path, M, hinSim, hinSimI):
    hinSim[path]=np.empty((M.shape[0],K))
    hinSimI[path]=np.empty((M.shape[0],K),dtype=np.int)
    M=M.tocsc()
    col_sum={}
    for i in range(M.shape[1]):
        col_sum[i]=M.getcol(i).toarray().sum()
    M=M.tocsr()
    row_sum={}
    for i in range(M.shape[0]):
        row_sum[i]=M.getrow(i).toarray().sum()
    for i in range(M.shape[0]):
        M_i=M.getrow(i).toarray()
        sim=[]
        for j in range(M_i.shape[1]):
            M_ij=M_i[0][j]
            M_i_=row_sum[i]
            M_j_=col_sum[j]

            S_ij=2*M_ij/(M_i_+M_j_)
            sim.append(S_ij)
        sim=np.array(sim)
        ids=np.argsort(-sim)
        hinSimI[path][i]=ids[:K]
        hinSim[path][i]=sim[ids[:K]]
    hinSim[path]=torch.from_numpy(hinSim[path]).float()
    hinSimI[path] = torch.from_numpy(hinSimI[path]).long()
    # from IPython import embed
    # embed()
    # print(hinSim[path].dtype, hinSimI[path].dtype)
    # xxx


# 使用每次getrow(i) 需要10min 速度太慢了
# 可以优化 预处理每个的row sum与col sum

def getDataLoader(data_path, batch_size=2048):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path+'', names=data_fields)
    # data_df=pd.read_csv(data_path+'user_ratedmovies.dat',sep='\t')
    # # embed()
    # data_df=data_df[['userID','movieID','rating']]
    # data_df.columns=['user_id', 'item_id', 'rating']
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 5).astype(np.float32)

    for i in ['user_id', 'item_id']:
        data_df[i] = data_df[i].map(dict(zip(data_df[i].unique(), range(1,data_df[i].nunique()+1))))

    # print(data_df.describe())

    df_train = data_df.sample(n=int(len(data_df) * 0.9), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    if IMPLICT:
        df_train=drop_df(df_train)
        df_test = drop_df(df_test)
    # get user number
    n_users = max(data_df['user_id'].values)
    # get item number
    n_items = max(data_df['item_id'].values)

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = torch.utils.data.DataLoader(
        Interactions(df_train,index_from_one=True), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        Interactions(df_test,index_from_one=True), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    from scipy.sparse import coo_matrix
    data = np.ones((data_df.shape[0]))
    row = data_df.user_id - 1
    col = data_df.item_id - 1
    UI = coo_matrix((data, (row, col)), shape=(n_users, n_items))
    UIU = UI.dot(UI.transpose())
    IUI = UI.transpose().dot(UI)
    hinSim = {}
    hinSimI = {}
    calSim('UIU', UIU, hinSim, hinSimI)
    calSim('IUI', IUI, hinSim, hinSimI)
    calSim('UI', UI, hinSim, hinSimI)
    metaPath = {'II': ['IUI'],'UU':['UIU'],'UI':'UI'}  # 也有可能是 IUI
    return (n_users,n_items, hinSim, hinSimI,metaPath), loaders

class heteMF(torch.nn.Module):
    def __init__(self, n_users, n_items, hinSim,
                 hinSimI, metaPath, n_factors=80, lr=0.01,
                 lambda_0=0.5,lambda_I=1, device=torch.device("cpu")):
        super(heteMF, self).__init__()

        self.n_users = n_users+1
        self.n_items = n_items+1
        self.n_factors = n_factors # 隐特征维度
        self.hinSim = hinSim # (元路径->前k相似度值)
        self.hinSimI = hinSimI  # (元路径->前k相似度下标)
        self.metaPath = metaPath # (UU or II or UI, 元路径的起点与终点划分)
        self.lambda_I = lambda_I
        self.device = device


        # 模型参数
        self.U = nn.Embedding(self.n_users, self.n_factors)
        self.V = nn.Embedding(self.n_items, self.n_factors)

        self.path2id = {} # 路径编码
        for p in self.metaPath:
            for pp in self.metaPath[p]:
                self.path2id[pp]=torch.LongTensor([len(self.path2id)])
        self.W_I=nn.Embedding(len(self.path2id), 1) # 路径参数

        self.optimizer = torch.optim.SGD(self.parameters(),
                                   lr=lr, weight_decay=lambda_0)
        self=self.to(self.device)


    # 预测评分
    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        ues = self.U(users)
        uis = self.V(items)
        preds = (ues * uis).sum(dim=-1)
        return preds

    def fit(self, loaders, epochs=5):
        # training cycle
        best_score = 0.
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}

            for phase in ['train', 'valid']:

                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                pbar = tqdm(enumerate(loaders[phase]),
                            total=len(loaders[phase]),
                            desc='({0}:{1:^3})'.format(phase, epoch+1))
                for batch_idx, ((batch_U, batch_I), batch_y) in pbar:
                    self.optimizer.zero_grad()

                    batch_U = batch_U.long()
                    batch_I = batch_I.long()
                    batch_y = batch_y.float().to(self.device)
                    y_pred = self.forward(batch_U, batch_I)
                    loss = nn.MSELoss(reduction='sum')(y_pred, batch_y)

                    WI_sum=0
                    loss_PI=0
                    try:
                        for p in self.metaPath['II']:
                            hin_value = self.hinSim[p][batch_I] # I*K->B*K  前K个的取值
                            hin_index = self.hinSimI[p][batch_I] # I*K -> B*K 前K个的下标
                            hin_embeddings = self.V(hin_index.reshape(-1,1)).reshape(
                                hin_index.shape[0],hin_index.shape[1],-1) # B*K*H

                            hin_reg = (self.V(batch_I).reshape(batch_I.shape[0], 1, -1) - hin_embeddings)\
                                .pow(2).sum(-1)  # B*K*1
                            loss_PI = loss_PI + (self.lambda_I *
                                        torch.exp(self.W_I(self.path2id[p])) * hin_value * hin_reg).sum()
                            WI_sum = WI_sum + torch.exp(self.W_I(self.path2id[p])).squeeze()
                    except Exception as ex:
                        print (ex)
                        from IPython import embed
                        embed()

                    loss+=loss_PI/WI_sum
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / batch_y.shape[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            with torch.no_grad():
                model.eval()
                y_pred,y_true = [],[]

                for ((row, col), val) in loaders['valid']:
                    row = row.long()
                    col = col.long()
                    val = val.float()
                    preds = self.forward(row, col)
                    if IMPLICT:
                        preds = sigmoid(preds.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += val.tolist()
                y_true,y_pred=np.array(y_true), np.array(y_pred)
                if IMPLICT:
                    epoch_score = roc_auc_score(y_true,y_pred)
                    score='auc'
                else:
                    epoch_score=sum([(y - x) ** 2 for x, y in zip(y_true, y_pred)]) / len(y_pred)
                    score='mse'


                user_item=loaders['valid'].dataset.user_item
                items = torch.arange(self.n_items).long()
                hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                train_ui=loaders['train'].dataset.user_item
                for u in user_item:
                    target_items=user_item[u]

                    users=[int(u)]*self.n_items
                    users = torch.Tensor(users).long()
                    scores=self.forward(users,items)
                    if u in train_ui:
                        seen_items = np.array(list(train_ui[u].keys()))
                        scores[seen_items]=-1e9
                    recs=np.argsort(scores)[-10:].tolist()

                    for item in recs:  # 遍历给user推荐的物品
                        if item in target_items:  # 测试集中有该物品
                            hit += 1  # 推荐命中+1
                        all_rec_items.add(item)
                    rec_count += 10
                    test_count += len(target_items)
                    precision = hit / (1.0 * rec_count)
                    recall = hit / (1.0 * test_count)
                    coverage = len(all_rec_items) / (1.0 * self.n_items)


            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')
                print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    # input_size, loader = getDataLoader("../data/hetrec2011-movielens-2k-v2/")

    model = heteMF(input_size[0],input_size[1],
                   input_size[2], input_size[3],input_size[4])
    model.fit(loader,15)


# Initialize end.The user number is:943,item number is:1682
# (train: 1 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=7.54]
# (valid: 1 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=7.86]
# (train: 2 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 1 train loss: 40.252 valid loss 7.412 mse 6.783
# precisioin=0.0055	recall=0.0051	coverage=0.7879
# (train: 2 ): 100%|██████████| 44/44 [00:34<00:00,  1.29it/s, train_loss=2.27]
# (valid: 2 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=2.6]
# (train: 3 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 2 train loss: 3.753 valid loss 2.543 mse 2.308
# precisioin=0.0132	recall=0.0122	coverage=0.7600
# (train: 3 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.971]
# (valid: 3 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.72]
# (train: 4 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 3 train loss: 1.046 valid loss 1.721 mse 1.576
# precisioin=0.0219	recall=0.0203	coverage=0.7421
# (train: 4 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.837]
# (valid: 4 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.4]
# (train: 5 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 4 train loss: 0.790 valid loss 1.427 mse 1.327
# precisioin=0.0408	recall=0.0378	coverage=0.6738
# (train: 5 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.755]
# (valid: 5 ): 100%|██████████| 5/5 [00:03<00:00,  1.40it/s, train_loss=1.25]
# (train: 6 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 5 train loss: 0.725 valid loss 1.271 mse 1.196
# precisioin=0.0656	recall=0.0607	coverage=0.4617
# (train: 6 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.751]
# (valid: 6 ): 100%|██████████| 5/5 [00:03<00:00,  1.38it/s, train_loss=1.18]
# (train: 7 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 6 train loss: 0.701 valid loss 1.181 mse 1.121
# precisioin=0.0749	recall=0.0694	coverage=0.2757
# (train: 7 ): 100%|██████████| 44/44 [00:34<00:00,  1.29it/s, train_loss=0.713]
# (valid: 7 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.1]
# (train: 8 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 7 train loss: 0.687 valid loss 1.120 mse 1.069
# precisioin=0.0819	recall=0.0758	coverage=0.1901
# (train: 8 ): 100%|██████████| 44/44 [00:34<00:00,  1.29it/s, train_loss=0.724]
# (valid: 8 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.07]
# epoch 8 train loss: 0.683 valid loss 1.077 mse 1.032
# precisioin=0.0849	recall=0.0786	coverage=0.1622
# (train: 9 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.716]
# (valid: 9 ): 100%|██████████| 5/5 [00:03<00:00,  1.40it/s, train_loss=1.06]
# (train:10 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 9 train loss: 0.679 valid loss 1.054 mse 1.012
# precisioin=0.0809	recall=0.0749	coverage=0.1503
# (train:10 ): 100%|██████████| 44/44 [00:34<00:00,  1.27it/s, train_loss=0.713]
# (valid:10 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.01]
# epoch 10 train loss: 0.679 valid loss 1.036 mse 0.997
# precisioin=0.0870	recall=0.0806	coverage=0.1480
# (train:11 ): 100%|██████████| 44/44 [00:34<00:00,  1.29it/s, train_loss=0.739]
# (valid:11 ): 100%|██████████| 5/5 [00:03<00:00,  1.39it/s, train_loss=1.03]
# (train:12 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 11 train loss: 0.678 valid loss 1.026 mse 0.988
# precisioin=0.0847	recall=0.0784	coverage=0.1349
# (train:12 ): 100%|██████████| 44/44 [00:34<00:00,  1.26it/s, train_loss=0.722]
# (valid:12 ): 100%|██████████| 5/5 [00:03<00:00,  1.40it/s, train_loss=0.988]
# (train:13 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 12 train loss: 0.678 valid loss 0.999 mse 0.962
# precisioin=0.0914	recall=0.0846	coverage=0.1450
# (train:13 ): 100%|██████████| 44/44 [00:34<00:00,  1.27it/s, train_loss=0.766]
# (valid:13 ): 100%|██████████| 5/5 [00:03<00:00,  1.39it/s, train_loss=1]
# (train:14 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 13 train loss: 0.678 valid loss 0.997 mse 0.961
# precisioin=0.0878	recall=0.0813	coverage=0.1396
# (train:14 ): 100%|██████████| 44/44 [00:34<00:00,  1.27it/s, train_loss=0.696]
# (valid:14 ): 100%|██████████| 5/5 [00:03<00:00,  1.41it/s, train_loss=1.01]
# (train:15 ):   0%|          | 0/44 [00:00<?, ?it/s]epoch 14 train loss: 0.676 valid loss 1.000 mse 0.965
# precisioin=0.0800	recall=0.0741	coverage=0.1325
# (train:15 ): 100%|██████████| 44/44 [00:34<00:00,  1.28it/s, train_loss=0.696]
# (valid:15 ): 100%|██████████| 5/5 [00:03<00:00,  1.40it/s, train_loss=0.993]
# epoch 15 train loss: 0.677 valid loss 0.989 mse 0.954
# precisioin=0.0889	recall=0.0823	coverage=0.1331
