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

def getDataLoader(data_path, batch_size=32):
    # load train data
    data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
    # all data file
    data_df = pd.read_table(data_path, names=data_fields)

    # ua_base = allData.sample(n=90570, replace=False)
    df_train = data_df.sample(n=int(len(data_df) * 0.9), replace=False)
    df_test = data_df.drop(df_train.index, axis=0)
    # get user number
    n_users = len(set(data_df['user_id'].values))
    # get item number
    n_items = len(set(data_df['item_id'].values))

    print("Initialize end.The user number is:%d,item number is:%d" % (n_users, n_items))
    train_loader = data.DataLoader(
        Interactions(df_train), batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        Interactions(df_test), batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader,
               'valid': test_loader}

    return (n_users,n_items ), loaders

class NCF(torch.nn.Module):
    def __init__(self, n_users, n_items, emb_size=64, dropout_p=0.02, lr=0.01, weight_decay=0., sparse=False):
        super(NCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items

        # get factor number
        self.emb_size = emb_size
        self.user_embeddings_GMF = nn.Embedding(self.n_users, self.emb_size, sparse=sparse)
        self.item_embeddings_GMF = nn.Embedding(self.n_items, self.emb_size, sparse=sparse)
        self.user_embeddings_MLP = nn.Embedding(self.n_users, self.emb_size, sparse=sparse)
        self.item_embeddings_MLP = nn.Embedding(self.n_items, self.emb_size, sparse=sparse)

        self.MLP_1 = nn.Linear(emb_size * 2, emb_size * 2)
        self.MLP_2 = nn.Linear(emb_size * 2, emb_size)
        self.MLP_3 = nn.Linear(emb_size, emb_size // 2 )
        self.merge_linear = nn.Linear(emb_size + emb_size // 2, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.activate = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)


    def forward(self, users, items):
        # print('test!!!!!!!!')
        # print(users.shape, users)
        # print(items.shape, items)

        u_emb_GMF = self.user_embeddings_GMF(users)
        i_emb_GMF = self.item_embeddings_GMF(items)
        out_GMF = (u_emb_GMF * i_emb_GMF)

        u_emb_MLP = self.user_embeddings_MLP(users)
        i_emb_MLP = self.item_embeddings_MLP(items)

        ipt_MLP = torch.cat([u_emb_MLP,i_emb_MLP],dim=1)
        # print(ipt_MLP.shape)
        out_MLP = self.dropout(self.activate(self.MLP_1(ipt_MLP)))
        out_MLP = self.dropout(self.activate(self.MLP_2(out_MLP)))
        out_MLP = self.dropout(self.activate(self.MLP_3(out_MLP)))
        # print(out_MLP.shape,out_GMF.shape)
        # print('!!!!!!!')

        merge = torch.cat([out_GMF, out_MLP], dim=1)
        out = self.merge_linear(merge)
        # print(out.shape)

        return out.squeeze()

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
                            desc='({0:^3})'.format(epoch))
                for batch_idx, ((row, col), val) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()

                    row = row.long()
                    col = col.long()
                    val = val.float()
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

            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f}')
        return

if __name__ == '__main__':
    input_size, loader=getDataLoader("../data/ml-100k/u.data")
    model = NCF(input_size[0],input_size[1])
    model.fit(loader,5)
