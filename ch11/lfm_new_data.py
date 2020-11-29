import random
import math
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

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

def getDataLoader(data_path, batch_size=2048):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 4).astype(np.float32)
    # ua_base = allData.sample(n=90570, replace=False)
    df_train = data_df.sample(n=int(len(data_df) * 0.8), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    if IMPLICT:
        df_train=drop_df(df_train)
        df_test = drop_df(df_test)
    # get user number
    n_users = max(set(data_df['user_id'].values))+1
    # get item number
    n_items = max(set(data_df['item_id'].values))+1

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items ), loaders

class LFM(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, lr=0.1, weight_decay=0.001, sparse=False,topn=10, device=torch.device("cpu")):
        super(LFM, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.topn=topn

        # get factor number
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        self.sparse = sparse

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=0.5)
        self=self.to(self.device)


    def forward(self, users, items):
        users=users.to(self.device)
        items = items.to(self.device)
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users) # b 1
        preds += self.item_biases(items)# b 1
        # preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)
        preds += ((ues) * (uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()

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
                for batch_idx, ((row, col), val) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    row = row.long()
                    col = col.long()
                    val = val.float().to(self.device)
                    preds = self.forward(row, col)
                    loss = nn.MSELoss(reduction='sum')(preds, val)

                    losses[phase] += loss.item()
                    batch_loss = loss.item() / row.size()[0]
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

                # # 计算top10的recall、precision、推荐物品覆盖率
                # user_item=loaders['valid'].dataset.user_item
                # items = torch.arange(self.n_items).long().to(self.device)
                # hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                # train_ui=loaders['train'].dataset.user_item
                # for u in user_item:
                #     target_items=user_item[u]
                #
                #     users=[int(u)]*self.n_items
                #     users = torch.Tensor(users).long().to(self.device)
                #     scores=self.forward(users,items)
                #     if u in train_ui:
                #         seen_items = np.array(list(train_ui[u].keys()))
                #         scores[seen_items]=-1e9
                #     recs=np.argsort(scores)[-self.topn:].tolist()
                #
                #     for item in recs:  # 遍历给user推荐的物品
                #         if item in target_items:  # 测试集中有该物品
                #             hit += 1  # 推荐命中+1
                #         all_rec_items.add(item)
                #     rec_count += self.topn
                #     test_count += len(target_items)
                #     precision = hit / (1.0 * rec_count)
                #     recall = hit / (1.0 * test_count)
                #     coverage = len(all_rec_items) / (1.0 * self.n_items)
                user_item=loaders['valid'].dataset.user_item
                items = torch.arange(self.n_items).long()
                hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                train_ui=loaders['train'].dataset.user_item
                for u in user_item:
                    target_items=user_item[u]
                    # seen_items = np.array(list(train_ui[u].keys()))

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
                    rec_count += self.topn
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
    model = LFM(input_size[0],input_size[1])
    model.fit(loader,10)

# batch_size很大影响 2048比32的auc大很多


# Initialize end.The user number is:943,item number is:1682
# (train: 1 ): 100%|#############################################################################################################################################| 2813/2813 [00:35<00:00, 80.00it/s, train_loss=2.27]
# (valid: 1 ): 100%|##############################################################################################################################################| 313/313 [00:03<00:00, 84.73it/s, train_loss=0.809]
# epoch 1 train loss: 7.579 valid loss 1.813 mse 1.813
# (train: 2 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 77.69it/s, train_loss=0.769]
# (valid: 2 ): 100%|################################################################################################################################################| 313/313 [00:03<00:00, 85.52it/s, train_loss=0.8]
# epoch 2 train loss: 1.171 valid loss 1.160 mse 1.160
# (train: 3 ): 100%|#############################################################################################################################################| 2813/2813 [00:35<00:00, 78.45it/s, train_loss=0.82]
# (valid: 3 ): 100%|###############################################################################################################################################| 313/313 [00:03<00:00, 87.12it/s, train_loss=1.16]
# epoch 3 train loss: 0.942 valid loss 1.087 mse 1.087
# (train: 4 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 78.06it/s, train_loss=0.906]
# (valid: 4 ): 100%|###############################################################################################################################################| 313/313 [00:03<00:00, 85.85it/s, train_loss=1.14]
# epoch 4 train loss: 0.883 valid loss 1.043 mse 1.043
# (train: 5 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 77.42it/s, train_loss=0.716]
# (valid: 5 ): 100%|###############################################################################################################################################| 313/313 [00:03<00:00, 83.22it/s, train_loss=1.42]
# epoch 5 train loss: 0.837 valid loss 1.044 mse 1.044


# 加了weight_decay
# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python lfm_new_data.py
# Initialize end.The user number is:943,item number is:1682
# (train: 1 ): 100%|#############################################################################################################################################| 2813/2813 [00:36<00:00, 76.82it/s, train_loss=1.12]
# (valid: 1 ): 100%|###############################################################################################################################################| 313/313 [00:03<00:00, 88.55it/s, train_loss=1.07]
# epoch 1 train loss: 5.408 valid loss 1.325 mse 1.325
# (train: 2 ): 100%|############################################################################################################################################| 2813/2813 [00:37<00:00, 75.33it/s, train_loss=0.728]
# (valid: 2 ): 100%|###############################################################################################################################################| 313/313 [00:03<00:00, 83.50it/s, train_loss=0.89]
# epoch 2 train loss: 1.001 valid loss 1.009 mse 1.009
# (train: 3 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 77.64it/s, train_loss=0.665]
# (valid: 3 ): 100%|##############################################################################################################################################| 313/313 [00:03<00:00, 84.64it/s, train_loss=0.961]
# epoch 3 train loss: 0.877 valid loss 0.966 mse 0.966
# (train: 4 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 77.06it/s, train_loss=0.985]
# (valid: 4 ): 100%|##############################################################################################################################################| 313/313 [00:03<00:00, 82.73it/s, train_loss=0.943]
# epoch 4 train loss: 0.831 valid loss 0.940 mse 0.940
# (train: 5 ): 100%|############################################################################################################################################| 2813/2813 [00:36<00:00, 77.66it/s, train_loss=0.663]
# (valid: 5 ): 100%|##############################################################################################################################################| 313/313 [00:03<00:00, 83.98it/s, train_loss=0.872]
# epoch 5 train loss: 0.806 valid loss 0.953 mse 0.953



# (base) C:\Users\Ares\Desktop\实验室相关\code\mycode>python lfm_new_data.py
# Initialize end.The user number is:943,item number is:1682
# (train: 1 ): 100%|############################################################################################################################################| 2813/2813 [00:43<00:00, 64.54it/s, train_loss=0.207]
# (valid: 1 ): 100%|##############################################################################################################################################| 313/313 [00:04<00:00, 71.55it/s, train_loss=0.291]
# epoch 1 train loss: 1.703 valid loss 0.321 auc 0.675
# (train: 2 ): 100%|############################################################################################################################################| 2813/2813 [00:43<00:00, 64.92it/s, train_loss=0.222]
# (valid: 2 ): 100%|##############################################################################################################################################| 313/313 [00:04<00:00, 71.74it/s, train_loss=0.233]
# epoch 2 train loss: 0.236 valid loss 0.228 auc 0.725
# (train: 3 ): 100%|############################################################################################################################################| 2813/2813 [00:42<00:00, 65.50it/s, train_loss=0.194]
# (valid: 3 ): 100%|##############################################################################################################################################| 313/313 [00:04<00:00, 72.48it/s, train_loss=0.266]
# epoch 3 train loss: 0.214 valid loss 0.223 auc 0.731
# (train: 4 ): 100%|############################################################################################################################################| 2813/2813 [00:42<00:00, 65.85it/s, train_loss=0.198]
# (valid: 4 ): 100%|##############################################################################################################################################| 313/313 [00:04<00:00, 66.00it/s, train_loss=0.221]
# epoch 4 train loss: 0.216 valid loss 0.221 auc 0.737
# (train: 5 ): 100%|############################################################################################################################################| 2813/2813 [00:44<00:00, 63.70it/s, train_loss=0.263]
# (valid: 5 ): 100%|##############################################################################################################################################| 313/313 [00:04<00:00, 65.91it/s, train_loss=0.221]
# epoch 5 train loss: 0.216 valid loss 0.222 auc 0.734

# 0919
# D:\Anaconda3\python.exe C:/Users/Ares/Desktop/实验室相关/code/mycode/lfm_new_data.py
# Initialize end.The user number is:944,item number is:1683
# (train: 1 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=1.82]
# (valid: 1 ): 100%|██████████| 10/10 [00:07<00:00,  1.37it/s, train_loss=1.87]
# epoch 1 train loss: 11.046 valid loss 1.871 mse 1.871
# precisioin=0.0343	recall=0.0162	coverage=0.3809
# (train: 2 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=1.31]
# (valid: 2 ): 100%|██████████| 10/10 [00:07<00:00,  1.36it/s, train_loss=1.15]
# (train: 3 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 2 train loss: 1.074 valid loss 1.136 mse 1.136
# precisioin=0.0836	recall=0.0394	coverage=0.2038
# (train: 3 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=0.958]
# (valid: 3 ): 100%|██████████| 10/10 [00:07<00:00,  1.38it/s, train_loss=1.01]
# epoch 3 train loss: 0.857 valid loss 1.002 mse 1.002
# precisioin=0.0938	recall=0.0442	coverage=0.1260
# (train: 4 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=0.886]
# (valid: 4 ): 100%|██████████| 10/10 [00:07<00:00,  1.36it/s, train_loss=0.962]
# (train: 5 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 4 train loss: 0.818 valid loss 0.961 mse 0.961
# precisioin=0.1200	recall=0.0565	coverage=0.1099
# (train: 5 ): 100%|██████████| 40/40 [00:30<00:00,  1.33it/s, train_loss=0.841]
# (valid: 5 ): 100%|██████████| 10/10 [00:07<00:00,  1.35it/s, train_loss=0.958]
# (train: 6 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 5 train loss: 0.797 valid loss 0.939 mse 0.939
# precisioin=0.1296	recall=0.0610	coverage=0.1111
# (train: 6 ): 100%|██████████| 40/40 [00:30<00:00,  1.33it/s, train_loss=0.821]
# (valid: 6 ): 100%|██████████| 10/10 [00:07<00:00,  1.35it/s, train_loss=0.932]
# (train: 7 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 6 train loss: 0.784 valid loss 0.928 mse 0.928
# precisioin=0.1284	recall=0.0604	coverage=0.1224
# (train: 7 ): 100%|██████████| 40/40 [00:30<00:00,  1.33it/s, train_loss=0.953]
# (valid: 7 ): 100%|██████████| 10/10 [00:07<00:00,  1.36it/s, train_loss=0.95]
# (train: 8 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 7 train loss: 0.768 valid loss 0.939 mse 0.939
# precisioin=0.1359	recall=0.0640	coverage=0.1266
# (train: 8 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=0.677]
# (valid: 8 ): 100%|██████████| 10/10 [00:07<00:00,  1.37it/s, train_loss=0.945]
# (train: 9 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 8 train loss: 0.759 valid loss 0.936 mse 0.936
# precisioin=0.1388	recall=0.0653	coverage=0.1272
# (train: 9 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=0.826]
# (valid: 9 ): 100%|██████████| 10/10 [00:07<00:00,  1.36it/s, train_loss=0.957]
# (train:10 ):   0%|          | 0/40 [00:00<?, ?it/s]epoch 9 train loss: 0.754 valid loss 0.936 mse 0.936
# precisioin=0.1464	recall=0.0689	coverage=0.1390
# (train:10 ): 100%|██████████| 40/40 [00:29<00:00,  1.35it/s, train_loss=0.699]
# (valid:10 ): 100%|██████████| 10/10 [00:07<00:00,  1.36it/s, train_loss=0.958]
# epoch 10 train loss: 0.751 valid loss 0.940 mse 0.940
# precisioin=0.1363	recall=0.0641	coverage=0.1343