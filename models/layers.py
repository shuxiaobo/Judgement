#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class StackRnn(nn.Module):
    '''
    Apply for paragraph representation. sentence -> paragraph to hidden size
    '''

    def __init__(self, input_size, hidden_size, output_size, words_size, feature_size, bidirection=True,
                 number_layers=3,
                 dropout_rate=0.5,
                 rnn_type=nn.LSTM):
        super(StackRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(words_size, hidden_size)
        self.rnns = nn.ModuleList()

        for i in range(self.number_layers):
            in_size = input_size + feature_size if i == 0 else 2 * hidden_size
            self.rnns.append(
                self.rnn_type(in_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirection))
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def init_weight(self):
        # nn.init.xavier_uniform(self.embedding.state_dict()['weight'])
        for rn in self.rnns:
            for name, param in rn.state_dict().items():
                if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)

    def forward(self, x, x_mask, x_feature, clause, cls):
        '''
        这里第一版2017年12月21日00:57:58
        建议将篇章分成句子，句子里在分词，那么这里就应该表示为：
            B * para_len(num_sentence) * seq_len
        然后对一个Batch进行单独的:
            1. encoding
            2. StackRnn(dropout): 句子表示->篇章表示
            3. FC (预想这里在第二版换成RL，做成多决策的RL)
        x: list , contain multi paragraph. batch first
        x_mask: data mask : batch * seq_len_max
        :param x:
        :param x_mask:
        :return:
        '''

        paragraph_representation = []  # will be (batch * para_hidden)

        for i, para in enumerate(x):
            para_emd = self.embedding(para)  # para_len(num_sentence) * seq_len * hidden_size

            # 改为Dictionary，不然embedding(0)会报错

            sen_represen = []
            # pack one sentence
            for j, sen in enumerate(para_emd):
                sen_len = x_mask[i][j].eq(0).long().sum(0).data[0]  # 0 for origin, 1 for pad
                if sen_len == 0:
                    continue
                sen = torch.unsqueeze(sen[:sen_len], 0)  # (1 * seq_len * hidden_size)
                # add feature to sen
                sen_feature = torch.unsqueeze(x_feature[i, j, :sen_len], 0)
                sen = torch.cat([sen, sen_feature], 2)

                # 不根据句子长度排序，送进去，因为在forward之前已经pad了
                # Note ：要根据句子长度取最后的有效隐藏状态
                sen_out = self.rnns[0](sen)[0].squeeze()[sen_len - 1]  # 得到最后的隐藏状态
                sen_represen.append(sen_out)

            para_repre = self.rnns[1](torch.cat(sen_represen, 0).view(-1, 1, self.hidden_size * 2))[1][0]
            paragraph_representation.append(para_repre)  # 拿到篇章的表示

        out = F.log_softmax(self.fc(torch.cat(paragraph_representation, 0).view(-1, x.size(0),
                                                                                self.hidden_size * 2)).squeeze(),
                            dim=0)  # batch * output_size

        return out
