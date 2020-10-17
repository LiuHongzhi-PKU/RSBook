import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils_torch import Under_sampling_Interactions
import os
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

# 设置是否使用隐式反馈
IMPLICT=True
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
def getDataLoader(data_path, batch_size=2048,sigma=0.6):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        dropidx=data_df[data_df.rating<3].index
        for i in dropidx:
            if random.random()>sigma:
                data_df.drop(i,axis=0,inplace=True)
        data_df.rating = (data_df.rating).astype(np.float32)

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
        Under_sampling_Interactions(df_train,sigma), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Under_sampling_Interactions(df_test,sigma), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items ), loaders

#
class USMF(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=80, lr=0.001,topn=10, reg_2=0.5,sparse=False, device=torch.device("cpu")):
        super(USMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.topn = topn
        # get factor number
        self.n_factors = n_factors
        self.reg_2 = reg_2
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self=self.to(self.device)

    # 前向计算，主要是计算评分
    def forward(self, users, items):
        users=users.to(self.device)
        items = items.to(self.device)
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)
        preds = (ues*uis).sum(dim=1,keepdim=True)
        return preds.squeeze(1)

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
                    loss += self.reg_2 * (self.item_embeddings.weight.norm() + self.user_embeddings.weight.norm())
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

                # # 计算top10的recall、precision、推荐物品覆盖率
                user_item=loaders['valid'].dataset.user_item
                items = torch.arange(self.n_items).long().to(self.device)
                hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                train_ui=loaders['train'].dataset.user_item
                for u in user_item:
                    target_items=user_item[u]

                    users=[int(u)]*self.n_items
                    users = torch.Tensor(users).long().to(self.device)
                    scores=self.forward(users,items)
                    if u in train_ui:
                        seen_items = np.array(list(train_ui[u].keys()))
                        scores[seen_items]=-1e9 # 不推荐已评分过的物品
                    else:continue # 跳过不在训练集中的用户

                    recs=np.argsort(scores)[-self.topn:].tolist()

                    for item in recs:  # 遍历给user推荐的物品
                        if item in target_items:  # 测试集中有该物品
                            hit += 1  # 推荐命中+1
                        all_rec_items.add(item)
                    rec_count += self.topn
                    test_count += len(target_items)
                precision = hit / (1.0 * rec_count)
                recall = hit / (1.0 * test_count)
                coverage = len(all_rec_items) / (1.0 * self.n_items)

            if ((epoch + 1) % 1) == 0: # 这个永远为true，可省略
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} ')
                print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
                print(hit, len(all_rec_items), len(user_item))

        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("u.data",sigma=0.6)
    # 从getDataLoader中得到模型需要的初始化参数，如用户数与物品数
    model = USMF(input_size[0],input_size[1])
    model.fit(loader,20)

    # precisioin = 0.2100
    # recall = 0.1064
    # coverage = 0.0465