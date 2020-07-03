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
np.random.seed(1024)
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from utils import FieldLoader
import torch.nn.functional as F
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

def getDataLoader(data_path, device=torch.device("cpu"), batch_size=2048):
    # load train data
    # Step1: 获取基本信息
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('../data/Movielens100K/u.user', sep='|', names=header)
    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv('../data/Movielens100K/u.item', sep='|', names=header, encoding="ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])

    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                                    '90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])

    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    cols = user_features + movie_features
    cols.remove('user_id')
    cols.remove('item_id')

    # Step2: 把特征进行归类处理(分成4个field)
    # 这里, 如果我们把Field分成4类, Gender, Occupation, Age, Other
    field_index, feature2field = {}, {}
    other_idxs = []
    for idx, col in enumerate(cols):
        infos = col.split('_')
        if len(infos) == 2:
            field = infos[0]
            field_index[field] = field_index.get(field, len(field_index))
            feature2field[idx] = field_index[field]
        if len(infos) == 1:
            other_idxs.append(idx)
    for idx in other_idxs:
        feature2field[idx] = len(field_index)

    # Step3: 根据user_id, item_id进行Merge, 得到对应的数据
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('../data/Movielens100K/ua.base', sep='\t', names=header)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')

    df_test = pd.read_csv('../data/Movielens100K/ua.test', sep='\t', names=header)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')

    # Step4: Label的变换, 以避免Cuda中报错
    # 需要对Label进行一定的转换, 因为原始的Label是[1, 2, 3, 4, 5]
    # 而 cuda中, 如果直接以这种Label的话, 会报错(Label 需要在[0, n_class - 1]范围
    # 因此, 需要转成[0, 1, 2, 3, 4]
    map_dict = dict()
    label_set = sorted(set(df_train['rating']) | set(df_test['rating']))
    for x in label_set:
        map_dict[x] = map_dict.get(x, len(map_dict))

    df_train['rating'] = df_train.rating.apply(lambda x: map_dict[x])
    df_test['rating'] = df_test.rating.apply(lambda x: map_dict[x])

    # # 如果想要使用"二分类"的话, 可以使用下面的方式来处理
    df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) >= 3 else 0)
    df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) >= 3 else 0)

    df_train = df_train[df_train.columns[4:].tolist() + ['rating']]
    df_test = df_test[df_test.columns[4:].tolist() + ['rating']]
    feature_split=[0,2,23,33,51]

    loaders = {'train': data.DataLoader(
        FieldLoader(df_train, feature_split), batch_size=batch_size, shuffle=True),
               'valid': data.DataLoader(
        FieldLoader(df_test, feature_split), batch_size=batch_size, shuffle=False)}

    feat_size = len(df_train.columns) - 1
    field_size = len(feature_split)-1
    # input_size = X_tensor.shape[1]
    return (feat_size, field_size,feature2field), loaders

class DeepFM(torch.nn.Module):
    def __init__(self, feat_size, field_size, feature2field, emb_size=10,
                 lr=0.01, weight_decay=0., layer_sizes=[64,64,64], dropout_deep=0.1, device=torch.device('cuda')):
        super(DeepFM, self).__init__()

        self.feat_size = feat_size
        self.field_size = field_size
        self.feature2field = feature2field
        self.emb_size = emb_size
        self.dropout_deep= dropout_deep
        self.layer_sizes = layer_sizes
        # first order term parameters embedding
        self.first_weights = nn.Embedding(feat_size, 1)  # None * M * 1
        nn.init.xavier_uniform_(self.first_weights.weight)
        # 需要定义一个 Embedding
        self.feat_embeddings = nn.Embedding(feat_size, emb_size)  # None * M * K
        nn.init.xavier_uniform_(self.feat_embeddings.weight)

        # 神经网络方面的参数
        all_dims = [self.field_size * self.emb_size] + layer_sizes
        for i in range(1, len(layer_sizes) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_deep))

        # 最后一层全连接层
        self.fc = nn.Linear(field_size + emb_size + all_dims[-1], 1)
        self.device = device


        self.optimizer = torch.optim.Adam(self.parameters(),
                                   lr=lr, weight_decay=weight_decay)
        print(self.first_weights)
        self = self.to(device)
        print(self.first_weights)

    def forward(self, feat_index):   # x: (batch_size, feat_num)
        feat_index = torch.tensor(feat_index).to(self.device)
        feat_value = torch.ones(feat_index.shape).to(self.device)

        feat_value = torch.unsqueeze(feat_value, dim=2)                       # None * F * 1

        # Step1: 先计算一阶线性的部分 sum_square part
        first_weights = self.first_weights(feat_index)                        # None * F * 1
        first_weight_value = torch.mul(first_weights, feat_value)
        y_first_order = torch.sum(first_weight_value, dim=2)                  # None * F
        # y_first_order = nn.Dropout(self.dropout_fm[0])(y_first_order)         # None * F

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)                      # None * F * K
        feat_emd_value = secd_feat_emb * feat_value                           # None * F * K(广播)

        # sum_square part
        summed_feat_emb = torch.sum(feat_emd_value, 1)                        # None * K
        interaction_part1 = torch.pow(summed_feat_emb, 2)                     # None * K

        # squared_sum part
        squared_feat_emd_value = torch.pow(feat_emd_value, 2)                 # None * K
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)          # None * K

        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        # y_secd_order = nn.Dropout(self.dropout_fm[1])(y_secd_order)

        # Step3: Deep部分
        y_deep = feat_emd_value.reshape(-1, self.field_size * self.emb_size)  # None * (F * K)
        y_deep = nn.Dropout(self.dropout_deep)(y_deep)

        for i in range(1, len(self.layer_sizes) + 1):
            y_deep = getattr(self, 'linear_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = F.relu(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        concat_input = torch.cat((y_first_order, y_secd_order, y_deep), dim=1)
        output = self.fc(concat_input)
        return output.squeeze(-1)

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
                for batch_id,(batch_x, batch_y) in pbar:
                # for batch_x, batch_y in loaders[phase]:
                    self.optimizer.zero_grad()
                    out = self.forward(batch_x)
                    batch_y = batch_y.type(torch.float32).to(self.device)
                    loss = nn.BCEWithLogitsLoss()(out, batch_y)
                    losses[phase] += loss.item() * batch_x.size(0)
                    batch_loss = loss.item() / batch_x.size()[0]
                    pbar.set_postfix(train_loss=batch_loss)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
                            #                             scheduler.step()
                            self.optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()

                y_pred,y_true = [],[]
                for batch_x, batch_y in loaders['valid']:
                    out = self.forward(batch_x)
                    preds = sigmoid(out.cpu().numpy())
                    y_pred += preds.tolist()
                    y_true += batch_y.tolist()
                epoch_score = roc_auc_score(np.array(y_true), np.array(y_pred))


            if ((epoch + 1) % 1) == 0:
                print(
                    f'epoch {epoch + 1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} valid roc auc {epoch_score:.3f}')
        return

if __name__ == '__main__':
    (feat_size, field_size, feature2field), loader=getDataLoader('../data/fm_kaggle/')
    model = DeepFM(feat_size, field_size, feature2field)
    model.fit(loader,5)
