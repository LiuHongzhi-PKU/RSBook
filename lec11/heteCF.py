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

def getDataLoader(data_path, batch_size=512):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
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
    # S = dok_matrix((5, 5), dtype=np.float32)
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
    metaPath = {'II': ['IUI'],'UU':['UIU'],'UI':['UI']}  # 也有可能是 IUI
    return (n_users,n_items, hinSim, hinSimI,metaPath), loaders

class heteMF(torch.nn.Module):
    def __init__(self, n_users, n_items, hinSim,
                 hinSimI, metaPath, n_factors=80, lr=0.01,
                 lambda_0=0.5,lambda_I=0.01,lambda_U=0.01,lambda_UI=0.00001, device=torch.device("cpu")):
        # __init__(self, n_users, n_items, hinSim,
        #          hinSimI, metaPath, n_factors=10, lr=0.01,
        #          lambda_0=0.001, lambda_I=1, device=torch.device("cpu")):

        super(heteMF, self).__init__()

        self.n_users = n_users+1
        self.n_items = n_items+1
        self.n_factors = n_factors # 隐特征维度
        self.hinSim = hinSim # (元路径->前k相似度值)
        self.hinSimI = hinSimI  # (元路径->前k相似度下标)
        self.metaPath = metaPath # (UU or II or UI, 元路径的起点与终点划分)
        self.lambda_I = lambda_I
        self.lambda_U = lambda_U
        self.lambda_UI = lambda_UI

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
        try:
            users = users.to(self.device)
            items = items.to(self.device)
            ues = self.U(users)
            uis = self.V(items)
            preds = (ues * uis).sum(dim=-1)
        except Exception as ex:
            print(ex)
            # embed()
            raise ex
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
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    try:
                        batch_U = batch_U.long()
                        batch_I = batch_I.long()
                        batch_y = batch_y.float().to(self.device)
                        # embed()
                        y_pred = self.forward(batch_U, batch_I)
                        loss = nn.MSELoss(reduction='sum')(y_pred, batch_y)
                    except Exception as ex:
                        print(ex)
                        embed()

                    WI_sum=0
                    loss_PI=0
                    for p in self.metaPath['II']:
                        hin_value = self.hinSim[p][batch_I] # I*K->B*K  前K个的取值
                        hin_index = self.hinSimI[p][batch_I] # I*K -> B*K 前K个的下标
                        hin_embeddings = self.V(hin_index.reshape(-1,1)).reshape(hin_index.shape[0],hin_index.shape[1],-1) # B*K*H
                        hin_pow = (self.V(batch_I).reshape(batch_I.shape[0],1,-1)-hin_embeddings).pow(2).sum(-1) # B*K*1
                        # print(hin_pow.shape)
                        loss_PI = loss_PI + (torch.exp(self.W_I(self.path2id[p])) * hin_value * hin_pow).sum()
                        WI_sum= WI_sum+torch.exp(self.W_I(self.path2id[p])).squeeze()

                    loss+=self.lambda_I * loss_PI/WI_sum

                    WI_sum = 0
                    loss_PI = 0
                    for p in self.metaPath['UU']:
                        hin_value = self.hinSim[p][batch_U]  # I*K->B*K  前K个的取值
                        hin_index = self.hinSimI[p][batch_U]  # I*K -> B*K 前K个的下标
                        hin_embeddings = self.U(hin_index.reshape(-1, 1)).reshape(hin_index.shape[0], hin_index.shape[1],
                                                                                  -1)  # B*K*H
                        hin_pow = (self.U(batch_U).reshape(batch_U.shape[0], 1, -1) - hin_embeddings).pow(2).sum(
                            -1)  # B*K*1
                        # print(hin_pow.shape)
                        loss_PI = loss_PI + (
                                    torch.exp(self.W_I(self.path2id[p])) * hin_value * hin_pow).sum()
                        WI_sum = WI_sum + torch.exp(self.W_I(self.path2id[p])).squeeze()

                    loss += self.lambda_U * loss_PI / WI_sum

                    WI_sum = 0
                    loss_PI = 0
                    for p in self.metaPath['UI']:
                        hin_value = self.hinSim[p][batch_U]  # U*K->B*K  前K个的取值
                        # hin_index = self.hinSimI[p][batch_U]  # U*K -> B*K 前K个的下标

                        y_pred = self.forward(batch_U, batch_I).reshape(y_pred.shape[0],1,-1)
                        loss_PI=loss_PI+(torch.exp(self.W_I(self.path2id[p]))*nn.MSELoss(reduction='sum')(y_pred, hin_value)).sum()

                        WI_sum = WI_sum + torch.exp(self.W_I(self.path2id[p])).squeeze()

                    loss += self.lambda_UI * loss_PI / WI_sum
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / batch_y.shape[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)
            # print('epoch done')
            # after each epoch check if we improved roc auc and if yes - save model
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

            # if ((epoch + 1) % 1) == 0:
            #     print(
            #         f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')


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

            # if ((epoch + 1) % 1) == 0:
            #     print(
            #         f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = heteMF(input_size[0],input_size[1],
                   input_size[2], input_size[3],input_size[4])
    model.fit(loader,15)


# (train: 1 ): 100%|██████████| 176/176 [00:40<00:00,  4.36it/s, train_loss=3.16]
# (valid: 1 ):  95%|█████████▌| 19/20 [00:03<00:00,  5.22it/s, train_loss=3.15]D:\Anaconda3\lib\site-packages\torch\nn\modules\loss.py:443: UserWarning: Using a target size (torch.Size([272, 10])) that is different to the input size (torch.Size([272, 1, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   return F.mse_loss(input, target, reduction=self.reduction)
# (valid: 1 ): 100%|██████████| 20/20 [00:03<00:00,  5.41it/s, train_loss=2.22]
# epoch 1 train loss: 22.947 valid loss 2.865 mse 2.405
# precisioin=0.0319	recall=0.0295	coverage=0.7302
# (train: 2 ): 100%|██████████| 176/176 [00:40<00:00,  4.31it/s, train_loss=1.74]
# (valid: 2 ): 100%|██████████| 20/20 [00:03<00:00,  5.32it/s, train_loss=1.74]
# epoch 2 train loss: 1.969 valid loss 2.202 mse 1.767
# precisioin=0.0995	recall=0.0921	coverage=0.1064
# (train: 3 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=1.94]
# (valid: 3 ): 100%|██████████| 20/20 [00:03<00:00,  5.41it/s, train_loss=1.65]
# epoch 3 train loss: 1.969 valid loss 2.096 mse 1.659
# precisioin=0.0839	recall=0.0777	coverage=0.0463
# (train: 4 ): 100%|██████████| 176/176 [00:40<00:00,  4.37it/s, train_loss=1.89]
# (valid: 4 ): 100%|██████████| 20/20 [00:03<00:00,  5.29it/s, train_loss=1.6]
# (train: 5 ):   0%|          | 0/176 [00:00<?, ?it/s]epoch 4 train loss: 2.005 valid loss 2.069 mse 1.625
# precisioin=0.0794	recall=0.0735	coverage=0.0428
# (train: 5 ): 100%|██████████| 176/176 [00:40<00:00,  4.36it/s, train_loss=1.94]
# (valid: 5 ): 100%|██████████| 20/20 [00:03<00:00,  5.33it/s, train_loss=1.64]
# epoch 5 train loss: 2.020 valid loss 2.062 mse 1.617
# precisioin=0.0902	recall=0.0835	coverage=0.0416
# (train: 6 ): 100%|██████████| 176/176 [00:40<00:00,  4.37it/s, train_loss=1.71]
# (valid: 6 ): 100%|██████████| 20/20 [00:03<00:00,  5.43it/s, train_loss=1.68]
# (train: 7 ):   0%|          | 0/176 [00:00<?, ?it/s]epoch 6 train loss: 2.030 valid loss 2.061 mse 1.619
# precisioin=0.0839	recall=0.0777	coverage=0.0416
# (train: 7 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=2.01]
# (valid: 7 ): 100%|██████████| 20/20 [00:03<00:00,  5.04it/s, train_loss=1.66]
# epoch 7 train loss: 2.032 valid loss 2.063 mse 1.619
# precisioin=0.0798	recall=0.0739	coverage=0.0386
# (train: 8 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=2]
# (valid: 8 ): 100%|██████████| 20/20 [00:03<00:00,  5.25it/s, train_loss=1.63]
# (train: 9 ):   0%|          | 0/176 [00:00<?, ?it/s]epoch 8 train loss: 2.037 valid loss 2.052 mse 1.602
# precisioin=0.0923	recall=0.0855	coverage=0.0392
# (train: 9 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=1.87]
# (valid: 9 ): 100%|██████████| 20/20 [00:03<00:00,  5.34it/s, train_loss=1.69]
# epoch 9 train loss: 2.037 valid loss 2.059 mse 1.615
# precisioin=0.0758	recall=0.0702	coverage=0.0392
# (train:10 ): 100%|██████████| 176/176 [00:40<00:00,  4.38it/s, train_loss=1.98]
# (valid:10 ): 100%|██████████| 20/20 [00:03<00:00,  5.35it/s, train_loss=1.62]
# (train:11 ):   0%|          | 0/176 [00:00<?, ?it/s]epoch 10 train loss: 2.039 valid loss 2.060 mse 1.612
# precisioin=0.0737	recall=0.0682	coverage=0.0398
# (train:11 ): 100%|██████████| 176/176 [00:40<00:00,  4.36it/s, train_loss=1.95]
# (valid:11 ): 100%|██████████| 20/20 [00:03<00:00,  5.31it/s, train_loss=1.62]
# epoch 11 train loss: 2.038 valid loss 2.075 mse 1.634
# precisioin=0.0840	recall=0.0778	coverage=0.0386
# (train:12 ): 100%|██████████| 176/176 [00:40<00:00,  4.38it/s, train_loss=2.06]
# (valid:12 ): 100%|██████████| 20/20 [00:03<00:00,  5.28it/s, train_loss=1.63]
# epoch 12 train loss: 2.040 valid loss 2.045 mse 1.597
# precisioin=0.0837	recall=0.0775	coverage=0.0410
# (train:13 ): 100%|██████████| 176/176 [00:40<00:00,  4.36it/s, train_loss=2.05]
# (valid:13 ): 100%|██████████| 20/20 [00:03<00:00,  5.36it/s, train_loss=1.66]
# (train:14 ):   0%|          | 0/176 [00:00<?, ?it/s]epoch 13 train loss: 2.041 valid loss 2.065 mse 1.619
# precisioin=0.0802	recall=0.0743	coverage=0.0386
# (train:14 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=1.76]
# (valid:14 ): 100%|██████████| 20/20 [00:03<00:00,  5.42it/s, train_loss=1.65]
# epoch 14 train loss: 2.040 valid loss 2.075 mse 1.633
# precisioin=0.0720	recall=0.0667	coverage=0.0374
# (train:15 ): 100%|██████████| 176/176 [00:40<00:00,  4.35it/s, train_loss=1.92]
# (valid:15 ): 100%|██████████| 20/20 [00:03<00:00,  5.41it/s, train_loss=1.65]
# epoch 15 train loss: 2.041 valid loss 2.063 mse 1.617
# precisioin=0.0785	recall=0.0727	coverage=0.0357