import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from collections import defaultdict
from utils import Interactions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

# 设置是否使用隐式反馈
IMPLICT=False
# 设置是否使用超小数据集测试
SMALL=True

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

def epinionsPreprocessing(data_path):
    data_df = pd.read_table(data_path+'ratings_data.txt', names=['user_id', 'item_id', 'rating'],sep=' ')
    # print(data_df.head())

    user_id_series = data_df['user_id']
    user_id_series = user_id_series.sample(n=int(len(set(user_id_series)) * 0.1), replace=False).reset_index(drop=True)
    item_id_series = data_df['item_id']
    item_id_series = item_id_series.sample(n=int(len(set(item_id_series)) * 0.1), replace=False).reset_index(drop=True)

    data_df = data_df[(data_df['user_id'].isin(user_id_series.values)) & (data_df['item_id'].isin(item_id_series.values))].reset_index(drop=True)
    social_df = pd.read_table(data_path+'trust_data.txt', names=['user_id','user_id2','trust'],sep=' ')
    social_df = social_df[(social_df['user_id'].isin(data_df['user_id'].values)) & (social_df['user_id2'].isin(data_df['user_id'].values))]

    data_df.to_csv(data_path+'data_df.txt',sep=' ',index=False)
    social_df.to_csv(data_path+'social_df.txt',sep=' ',index=False)

    print("Done.")
    return data_df, social_df
    
def getDataLoader(data_path, trainset_rate=0.8, batch_size=4096):
    # load train data
    # all data file
    if os.path.exists(data_path+'data_df.txt') and os.path.exists(data_path+'social_df.txt'):
      print("Epinions dataset has been preprocessed.")
    else:
      print("Epinions dataset has not been preprocessed.")
      epinionsPreprocessing(data_path)
    data_df = pd.read_table(data_path+'data_df.txt',sep=' ')
    data_df['rating'] /= max(data_df['rating'])
    social_df = pd.read_table(data_path+'social_df.txt',sep=' ')

    le = preprocessing.LabelEncoder()
    le.fit(data_df['user_id'])
    data_df['user_id']=le.transform(data_df['user_id'])
    social_df['user_id']=le.transform(social_df['user_id'])
    social_df['user_id2']=le.transform(social_df['user_id2'])
    le.fit(data_df['item_id'])
    data_df['item_id']=le.transform(data_df['item_id'])

    social_user = defaultdict(set)
    for (user, user2, record) in social_df.itertuples(index=False):
        social_user.setdefault(user, set())
        social_user[user].add(user2)

    df_train = data_df.sample(n=int(len(data_df) * trainset_rate), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    # get user number
    # get item number
    n_users = max(set(data_df['user_id'].values))+1
    n_items = max(set(data_df['item_id'].values))+1


    print("Initialize end. The user number is: %d, item number is: %d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items, social_user), loaders


class SoReg(torch.nn.Module):
    def __init__(self, n_users, n_items, social_user, n_factors=10, lr=0.01, lambda_1=0.01, lambda_2=0.001,sparse=False, device=torch.device("cuda")):
        super(SoReg, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.social_user = social_user
        self.device = device

        # get factor number
        self.n_factors = n_factors
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self=self.to(self.device)

    # 前向计算，主要是计算评分
    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)
        preds = (ues*uis).sum(dim=1,keepdim=True)
        return preds.squeeze()

    def fit(self, loaders, epochs=10):
        # training cycle
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
                    self.optimizer.zero_grad()
                    row = row.long()
                    col = col.long()
                    val = val.float().to(self.device)
                    preds = self.forward(row, col)
                    loss = nn.MSELoss(reduction='sum')(preds, val)
                    for i in range(row.shape[0]):
                        # print(row[i], self.social_user[int(row[i])])
                        if len(self.social_user[int(row[i])])>0:
                            user = row[i].to(self.device)
                            user_emb = self.user_embeddings(user)
                            for user_2 in self.social_user[int(row[i])]:
                                user_2 = torch.tensor(user_2).long().to(self.device)
                                sim_emb = self.user_embeddings(user_2)
                                loss+= self.lambda_1 * ((user_emb - sim_emb) * (user_emb - sim_emb)).sum()/ len(self.social_user[int(row[i])])
                    loss += self.lambda_2 * (self.item_embeddings.weight.norm() + self.user_embeddings.weight.norm())
                    losses[phase] += loss.item()
                    batch_loss = loss.item() / row.size()[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()

                y_pred,y_true = [],[]

                for ((row, col), val) in loaders['valid']:
                    row = row.long()
                    col = col.long()
                    val = val.float()
                    preds = self.forward(row, col)
                    y_pred += preds.tolist()
                    y_true += val.tolist()
                y_true,y_pred=np.array(y_true), np.array(y_pred)
                # 根据不同任务设定不同评分
                if IMPLICT:
                    epoch_score = roc_auc_score(y_true,y_pred)
                    score='auc'
                    print(
                      f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss: {losses["valid"]:.3f} {score}: {epoch_score:.3f}')
                else:
                    epoch_score=np.sqrt(sum([(y - x) ** 2 for x, y in zip(y_true, y_pred)]) / len(y_pred))
                    score='rmse'
                    epoch_score1=sum([np.abs(y - x) for x, y in zip(y_true, y_pred)]) / len(y_pred)
                    score1='mae'
                    print(                    
                      f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss: {losses["valid"]:.3f} {score}: {epoch_score:.3f} {score1}: {epoch_score1:.3f}')
        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/epinions/")
    # 从getDataLoader中得到模型需要的初始化参数，如用户数与物品数
    model = SoReg(input_size[0],input_size[1],input_size[2],device="cuda")
    model.fit(loader,5)
    torch.save(model.state_dict(), 'SoReg.pkl')