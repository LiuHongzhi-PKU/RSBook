import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import Interactions
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

def getDataLoader(data_path, batch_size=2048):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)
    if SMALL:
        data_df = data_df.sample(n=int(len(data_df) * 0.1), replace=False)
    if IMPLICT:
        data_df.rating = (data_df.rating >= 5).astype(np.float32)
    le = preprocessing.LabelEncoder()
    le.fit(data_df['user_id'])
    data_df['user_id']=le.transform(data_df['user_id'])
    le.fit(data_df['item_id'])
    data_df['item_id']=le.transform(data_df['item_id'])

    # ua_base = allData.sample(n=90570, replace=False)
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

class CPMF(torch.nn.Module):
    # 核心思想：使用历史物品的嵌入修正用户表示，进行矩阵分解
    def __init__(self, n_users, n_items, n_factors=20, lr=0.1, weight_decay=0.001, sparse=False,topn=10, device=torch.device("cpu")):
        super(CPMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.topn=topn

        self.n_factors = n_factors
        self.user_biases = nn.Embedding(self.n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(self.n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)
        self.W = nn.Embedding(self.n_items, self.n_factors, sparse=sparse)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        self=self.to(self.device)


    def forward(self, users, items, UI):
        users=users.to(self.device)
        items = items.to(self.device)
        UI=UI.to(self.device)
        ues = self.user_embeddings(users) #(B,F)
        uis = self.item_embeddings(items) # (B,F)
        # 得生成一个(B,M,F)嵌入矩阵，然后UI(B,M)相乘进行mask求和取平均->使用(B,M)*(M,F)即可

        # W = self.W(torch.arange(self.n_items).to(self.device).long()).unsqueeze(0)  # (1,M,F)
        # W=W.expand(users.shape[0],-1,-1) #(B,M,F)
        # UI=UI.unsqueeze(2)#(B,M,1)
        # W=(UI*W).sum(1)/UI.sum(1) # (B,F)

        W = self.W(torch.arange(self.n_items).to(self.device).long())
        W = UI.matmul(W)/UI.sum(1,keepdim=True)

        ues+=W
        preds = self.user_biases(users) # b 1
        preds += self.item_biases(items)# b 1
        preds += ((ues) * (uis)).sum(dim=1, keepdim=True)

        return preds.squeeze(1)

    def fit(self, loaders, epochs=5):
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
                user_item = loaders[phase].dataset.user_item
                for batch_idx, ((row, col), val) in pbar:
                    self.optimizer.zero_grad()

                    row = row.long()
                    col = col.long()
                    val = val.float().to(self.device)
                    # 根据u,i对生成(B,I)的每个用户的物品01向量
                    # 然后利用01向量改进用户表示，进行forward
                    UI=torch.zeros([row.shape[0], self.n_items]).to(self.device)
                    for i in range(row.shape[0]):
                        u=row[i]
                        for item in user_item.get(u.item(),[]):
                            UI[i][item]=1

                    row = row.long()
                    col = col.long()
                    val = val.float().to(self.device)

                    preds = self.forward(row, col,UI)
                    loss = nn.MSELoss(reduction='sum')(preds, val)

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
                    UI = torch.zeros([row.shape[0], self.n_items]).to(self.device)
                    for i in range(row.shape[0]):
                        u = row[i]
                        for item in user_item.get(u.item(), []):
                            UI[i][item] = 1

                    row = row.long()
                    col = col.long()
                    val = val.float().to(self.device)

                    preds = self.forward(row, col, UI)
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

                # 计算top10的recall、precision、推荐物品覆盖率
                user_item=loaders['valid'].dataset.user_item
                items = torch.arange(self.n_items).long().to(self.device)
                hit, rec_count, test_count,all_rec_items = 0,0,0,set()
                train_ui=loaders['train'].dataset.user_item
                for u in user_item:
                    target_items=user_item[u]

                    users=[int(u)]*self.n_items
                    users = torch.Tensor(users).long().to(self.device)
                    UI = torch.zeros([1, self.n_items]).to(self.device)
                    for item in user_item.get(u, []):
                        UI[0][item] = 1
                    UI=UI.expand((self.n_items, UI.shape[1]))

                    scores=self.forward(users,items, UI)
                    if u in train_ui:
                        seen_items = np.array(list(train_ui[u].keys()))
                        scores[seen_items]=-1e9
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

            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} {score} {epoch_score:.3f}')
                print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = CPMF(input_size[0],input_size[1])
    model.fit(loader,10)
