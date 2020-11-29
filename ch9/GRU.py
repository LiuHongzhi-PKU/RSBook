# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import argparse
import pickle
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import datetime
import math
import numpy as np
import torch
from torch import nn
import argparse
class GRU(nn.Module):
    """GRU
    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int):
    """
    def __init__(self, opt, n_node):
        super(GRU, self).__init__()
        self.n_items = n_node + 1 # 物品id从1开始
        self.hidden_size = opt.hiddenSize
        self.batch_size = opt.batchSize
        self.embedding_dim = opt.hiddenSize
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.n_items, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def forward(self, seq, mask):
        # 输入为 B,L    mask为0,1
        embs = self.emb_dropout(self.emb(seq))  # B,L,H
        gru_out, hidden = self.gru(embs)  # gru_out: B,L,H
        hidden = gru_out

        # 获得有效位的最后的隐层状态 B,H （不考虑有效位就是 ht=hidden[:,-1]）
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        # 全连接输出评分
        scores = self.dense(ht)
        return nn.Softmax()(scores)

    def get_loss(self, scores, targets):
        return nn.CrossEntropyLoss()(scores, targets)
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        # i是这个batch的id list
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        return inputs, mask, targets

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='历史遗留参数，不需要改动')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
opt = parser.parse_args()
print(opt)

class RNN():
    def __init__(self):
        self.loadData()
        self.model = trans_to_cuda(GRU(opt, self.n_node))
        self.fit()

    def loadData(self):
        train_data = pickle.load(open('../data/' + opt.dataset + '/train.txt', 'rb'))

        test_data = pickle.load(open('../data/' + opt.dataset + '/test.txt', 'rb'))

        self.train_data = Data(train_data, shuffle=True)
        self.test_data = Data(test_data, shuffle=False)
        self.n_node = 310

    def forward(self,i):
        # i是slice的编号，对应一个batch的数据
        data=self.train_data
        model=self.model
        inputs, mask, targets = data.get_slice(i)
        inputs = trans_to_cuda(torch.Tensor(inputs).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        y_pred = model(inputs, mask)
        return targets, y_pred

    def fit(self,epochs=10):
        model=self.model
        train_data=self.train_data
        test_data=self.test_data

        for epoch in range(epochs):
            model.scheduler.step()
            print('start training: ', datetime.datetime.now())
            model.train()
            total_loss = 0.0
            slices = train_data.generate_batch(model.batch_size)
            for i, j in zip(slices, np.arange(len(slices))):
                model.optimizer.zero_grad()
                targets, scores = self.forward(i)
                targets = trans_to_cuda(torch.Tensor(targets).long())
                # loss = model.loss_function(scores, targets)
                loss = model.get_loss(scores, targets)
                loss.backward()

                model.optimizer.step()
                total_loss += loss
            #     if j % int(len(slices) / 5 + 1) == 0:
            #         print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
            # print('\tLoss:\t%.3f' % total_loss)

            print('start predicting: ', datetime.datetime.now())
            model.eval()
            hit, mrr = [], []
            slices = test_data.generate_batch(model.batch_size)
            for i in slices:
                targets, scores = self.forward(i)
                sub_scores = scores.topk(20)[1]
                sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                for score, target, mask in zip(sub_scores, targets, test_data.mask):
                    hit.append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target)[0][0] + 1))
            Recall = np.mean(hit) * 100
            MMR = np.mean(mrr) * 100

            print("Recall=",Recall)
            print("MMR=", MMR)

        # return hit, mrr



model=RNN()


# GRU
# Recall= 65.65656565656566
# MMR= 41.3641641582818

# LSTM
# Recall= 69.6969696969697
# MMR= 50.16093412378242

# RNN
# Recall= 61.61616161616161
# MMR= 34.969667389453484