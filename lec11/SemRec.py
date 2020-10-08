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

def getDataLoader(data_path, batch_size=2048):
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

    df_train = data_df.sample(n=int(len(data_df) * 0.8), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    # df_train=drop_df(df_train)
    # df_test = drop_df(df_test)
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
    metaPath = {'II': ['IUI'],'UU':['UIU'],'UI':['UI']}  # 也有可能是 IUI
    return (n_users,n_items, hinSim, hinSimI,metaPath), loaders

class SemRec(torch.nn.Module):
    def __init__(self, n_users, n_items, hinSim, hinSimI, metaPath, n_factors=20, lr=0.1, lambda_0=0.5,lambda_I=0, device=torch.device("cpu")):
        super(SemRec, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors # 隐特征维度
        self.hinSim = hinSim # (元路径，前k相似度值)
        self.hinSimI = hinSimI  # (元路径，前k相似度下标)
        self.metaPath = metaPath # (UU or II, 元路径)
        self.lambda_I = lambda_I
        self.device = device


        # 模型参数
        self.U = {}#nn.Embedding(self.n_users, self.n_factors)
        self.V = {}#nn.Embedding(self.n_items, self.n_factors)

        self.path2id = {}
        for pp in self.metaPath['UI']:
            self.path2id[pp]=torch.LongTensor([len(self.path2id)])
            self.U[pp] = nn.Embedding(self.n_users, self.n_factors)
            self.V[pp] = nn.Embedding(self.n_items, self.n_factors)
            self.add_module("U_emb_"+pp, self.U[pp])
            self.add_module("V_emb_" + pp, self.V[pp])

        for pp in self.metaPath['UU']:
            self.path2id[pp]=torch.LongTensor([len(self.path2id)])
        self.W_U=nn.Embedding(self.n_users, len(self.path2id))

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=lambda_0)
        embed()
        self=self.to(self.device)


    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        # print(items)
        preds=torch.zeros(users.shape,dtype=torch.float32).to(self.device)
        WI_sum = torch.zeros(users.shape, dtype=torch.float32).to(self.device) # B
        try:
            for p in self.metaPath['UI']:
                ues = self.U[p](users)
                uis = self.V[p](items)
                w_ui=self.W_U(users)[:,self.path2id[p]].reshape(-1)
                w_ui=torch.exp(w_ui)
                preds = preds + w_ui*(ues * uis).sum(dim=-1)
                WI_sum+=w_ui

        except Exception as ex:
            print(ex)
            from IPython import embed
            embed()
        # return preds
        return preds/WI_sum

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

                    try:
                        loss_PI=0

                        for p in self.metaPath['UU']:
                            s_p_k = self.hinSim[p][batch_U] # U*K->B*K  前K个的取值
                            hin_index = self.hinSimI[p][batch_U] # U*K -> B*K 前K个的下标
                            w_p_k = self.W_U(hin_index)[:,:,self.path2id[p]].reshape(s_p_k.shape) # B,K,1
                            w_p_i = self.W_U(batch_U)[:,self.path2id[p]] # B,1
                            hin_reg=(w_p_i.reshape(-1)-(w_p_k*s_p_k).sum(-1)).pow(2).sum()
                            loss_PI = loss_PI + (self.lambda_I  * hin_reg).sum()
                    except Exception as ex:
                        print (ex)
                        from IPython import embed
                        embed()
                    loss+=loss_PI
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
    # input_size, loader=getDataLoader("../data/ml-100k/u.data")
    input_size, loader = getDataLoader("../data/hin/UI.txt")

    model = SemRec(input_size[0],input_size[1], input_size[2], input_size[3],input_size[4])
    model.fit(loader,10)
    from IPython import embed
    print('done')
    embed()


# (train: 1 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=4.32]
# (valid: 1 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=3.38]
# (train: 2 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 1 train loss: 16.551 valid loss 3.399 mse 3.399
# precisioin=0.0821	recall=0.0386	coverage=0.4477
# (train: 2 ): 100%|██████████| 40/40 [00:36<00:00,  1.09it/s, train_loss=1.03]
# (valid: 2 ): 100%|██████████| 10/10 [00:08<00:00,  1.12it/s, train_loss=1.29]
# (train: 3 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 2 train loss: 1.530 valid loss 1.289 mse 1.289
# precisioin=0.1055	recall=0.0496	coverage=0.2687
# (train: 3 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=1.04]
# (valid: 3 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.12]
# (train: 4 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 3 train loss: 0.928 valid loss 1.095 mse 1.095
# precisioin=0.1071	recall=0.0504	coverage=0.1778
# (train: 4 ): 100%|██████████| 40/40 [00:36<00:00,  1.09it/s, train_loss=0.941]
# (valid: 4 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.09]
# (train: 5 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 4 train loss: 0.885 valid loss 1.072 mse 1.072
# precisioin=0.1162	recall=0.0546	coverage=0.1350
# (train: 5 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=0.888]
# (valid: 5 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.09]
# (train: 6 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 5 train loss: 0.883 valid loss 1.072 mse 1.072
# precisioin=0.1194	recall=0.0562	coverage=0.1213
# (train: 6 ): 100%|██████████| 40/40 [00:36<00:00,  1.09it/s, train_loss=1.29]
# (valid: 6 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.09]
# (train: 7 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 6 train loss: 0.876 valid loss 1.051 mse 1.051
# precisioin=0.1243	recall=0.0585	coverage=0.1231
# (train: 7 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=1.07]
# (valid: 7 ): 100%|██████████| 10/10 [00:08<00:00,  1.12it/s, train_loss=1.09]
# (train: 8 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 7 train loss: 0.865 valid loss 1.060 mse 1.060
# precisioin=0.1290	recall=0.0607	coverage=0.1361
# (train: 8 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=0.916]
# (valid: 8 ): 100%|██████████| 10/10 [00:08<00:00,  1.12it/s, train_loss=1.15]
# epoch 8 train loss: 0.862 valid loss 1.063 mse 1.063
# precisioin=0.1198	recall=0.0563	coverage=0.1260
# (train: 9 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=0.835]
# (valid: 9 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.09]
# (train:10 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 9 train loss: 0.866 valid loss 1.081 mse 1.081
# precisioin=0.1269	recall=0.0597	coverage=0.1361
# (train:10 ): 100%|██████████| 40/40 [00:36<00:00,  1.10it/s, train_loss=0.916]
# (valid:10 ): 100%|██████████| 10/10 [00:08<00:00,  1.13it/s, train_loss=1.13]
# epoch 10 train loss: 0.870 valid loss 1.080 mse 1.080
# precisioin=0.1066	recall=0.0502	coverage=0.1302

