#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.autograd import Variable
import numpy as np

sys.path.append('..')
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class CNN_F(nn.Module):

    def __init__(self, num_words, embedding_size, channel_size, kermel_size, pool_size, dropout_prob, is_training=True):
        super(CNN_F, self).__init__()
        self.embedding_size = embedding_size
        self.filter_size = kermel_size
        self.pool_size = pool_size
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_words, embedding_size)

        self.atten = Hierarchy_attention(embedding_size)

        self.cnn_list = nn.Sequential()

        for i in range(len(kermel_size)):
            self.cnn_list.add_module(name='conv-{}'.format(i),
                                     module=nn.Conv2d(channel_size[i], channel_size[i + 1], kermel_size[i], stride=1,
                                                      # padding=2,
                                                      padding=(
                                                          int((kermel_size[i][0] - 1) / 2),
                                                          int((kermel_size[i][1] - 1) / 2)),
                                                      dilation=1, groups=1, bias=True))
            self.cnn_list.add_module(name='bn-{}'.format(i), module=nn.BatchNorm2d(channel_size[i + 1]))
            # self.cnn_list.add_module(name='pool-{}'.format(i),
            #                          module=nn.MaxPool2d(kernel_size=tuple(pool_size[i]), stride=tuple(pool_size[i]),
            #                                              padding=0, dilation=1,
            #                                              return_indices=False, ceil_mode=False))
            self.cnn_list.add_module(name='relu-{}'.format(i), module=nn.ReLU())

    def forward(self, x, x_mask, sentences_count):
        embed = []
        for i in range(x.shape[0]):
            embed.append(self.embedding(x[i]))
        embed = torch.stack(embed, 0)

        cnned = self.cnn_list(embed.transpose(1, 3)).transpose(1, 3)

        # paraph = []
        # for i in range(cnned.shape[0]):
        #     score = self.word_attn(cnned[i], x_mask[i])
        #     paraph.append(self.weighted_avg(x[i], score))
        # embed_atten = torch.stack(paraph, 0)
        # embed_atten = self.atten(cnned.transpose(1, 3).contiguous(), x_mask)

        droped = F.dropout(cnned, self.dropout_prob, training=self.training)

        # return the sentence mask
        seq_lens = [np.ceil(i / (np.sum(self.pool_size, axis=0))) for i in
                    sentences_count]

        return droped, seq_lens


class Cnn_rnn(nn.Module):

    def __init__(self, num_words, embedding_size, channel_size, kermel_size, pool_size, dropout_prob, hidden_size,
                 output_size, bidirection=True,
                 number_layers=1,
                 rnn_type=nn.LSTM):
        super(Cnn_rnn, self).__init__()
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.number_layers = number_layers
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_prob

        self.cnn = CNN_F(num_words, embedding_size, channel_size, kermel_size, pool_size, dropout_prob,
                         is_training=True)

        self.word_attn = LinearSelfAttn(embedding_size)

        self.rnn = self.rnn_type(channel_size[-1], hidden_size, num_layers=number_layers,
                                 bidirectional=bidirection, batch_first=True)

        dire = 2 if bidirection else 1

        self.sen_attn = LinearSelfAttn(hidden_size * dire)

        bidre = 2 if bidirection else 1

        self.fc = nn.Linear(hidden_size * self.number_layers * bidre, 128)

        self.fc2 = nn.Linear(128, output_size)

    def init_weight(self, args):
        if args.finetune:
            nn.init.xavier_uniform(self.cnn.embedding.state_dict()['weight'])
        for name, param in self.rnn.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param, 10)

        for k, state in self.cnn.cnn_list.state_dict().items():
            if k[:4] == 'conv' and not k[:-4] == 'bias':
                nn.init.xavier_normal(state)

        for k, state in self.cnn.atten.state_dict().items():
            if not k[-4:] == 'bias':
                nn.init.xavier_normal(state)

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc2.bias.data.fill_(0)

    def forward(self, x, x_mask, x_feature, sentences_count, clause, cls):
        # 使用层级attention, 词级的atten，扩展cnn维度，到句子级attention，到RNN，fc

        cnned, _ = self.cnn(x, x_mask, sentences_count)

        paraph = []
        for i in range(cnned.shape[0]):
            score = self.word_attn(cnned[i], x_mask[i])
            paraph.append(weighted_avg(cnned[i], score))
        embed_atten = torch.stack(paraph, 0)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embed_atten, sentences_count, batch_first=True)

        paragraph_output, paragraph_hidden = self.rnn(packed)

        paragraph_output, _ = torch.nn.utils.rnn.pad_packed_sequence(paragraph_output)
        paragraph_output = paragraph_output.transpose(0, 1).contiguous()
        seq_mask = Variable(torch.sum(1 - x_mask.data, 2) == 0)
        # paraph = torch.stack(paragraph_output, 0)
        sen_score = self.sen_attn(paragraph_output, seq_mask)
        paraph = weighted_avg(paragraph_output, sen_score)

        hidden_fc = self.fc(paraph)

        hidden_fc = F.relu(hidden_fc)

        hidden_fc2 = self.fc2(hidden_fc)
        hidden_fc2 = F.relu(hidden_fc2)

        # paragraph_output = F.relu(self.fc2(paragraph_output))

        out = F.log_softmax(hidden_fc2, dim=0)  # batch * output_size

        return out


class Hierarchy_attention(nn.Module):

    def __init__(self, input_size):
        super(Hierarchy_attention, self).__init__()
        self.input_size = input_size
        self.sen_attn = LinearSelfAttn(input_size)
        self.word_attn = LinearSelfAttn(input_size)

    def forward(self, x, x_mask):
        paraph = []
        for i in range(x.shape[0]):
            score = self.word_attn(x[i], x_mask[i])
            paraph.append(self.weighted_avg(x[i], score))
        seq_mask = Variable(torch.sum(1 - x_mask.data, 2) == 0)
        paraph = torch.stack(paraph, 0)
        sen_score = self.sen_attn(paraph, seq_mask)
        paraph = self.weighted_avg(paraph, sen_score)

        return paraph


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.input_size = input_size
        self.linear_attn = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        :param x: batch * len * hdim
        :param x_mask: batch * len (0 for true, 1 for pad)
        :return:
         alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        x_score = self.linear_attn(x_flat).view(x.size(0), x.size(1))
        # x_score = self.linear_attn(x_flat).view(x.size(0), x.size(1))
        x_score.data.masked_fill_(x_mask.data, -float('inf'))
        # why fill inf? for softmax, to make alpha to 0 because exp(-inf) almost 0
        alpha = F.softmax(x_score)
        return alpha


class StackRnn(nn.Module):
    '''
    Apply for paragraph representation. sentence -> paragraph to hidden size
    '''

    def __init__(self, input_size, hidden_size, output_size, words_size, feature_size, tags_len, bidirection=True,
                 number_layers=3,
                 dropout_rate=0.5,
                 rnn_type=nn.LSTM):
        super(StackRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.tags_len = tags_len
        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(words_size, hidden_size)
        self.rnns = nn.ModuleList()

        self.sel_attn = LinearSelfAttn(hidden_size * 2)

        for i in range(self.number_layers):
            in_size = input_size + feature_size if i == 0 else 2 * hidden_size
            self.rnns.append(
                self.rnn_type(in_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirection))
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.fc2 = nn.Linear(hidden_size * 2, tags_len)

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


class AvgRnn(nn.Module):
    '''
    Apply for paragraph representation. sentence -> paragraph to hidden size
    '''

    def __init__(self, input_size, hidden_size, output_size, words_size, feature_size, tags_len, bidirection=True,
                 number_layers=1,
                 dropout_rate=0.5,
                 rnn_type=nn.LSTM):
        super(AvgRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.tags_len = tags_len
        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(words_size, hidden_size)

        # self.sel_attn_sen = nn.Linear(hidden_size, hidden_size)  # mend
        self.sel_attn_sen = LinearSelfAttn(hidden_size)  # mend
        self.sel_attn_sen_v = nn.Parameter(FloatTensor(hidden_size, 1))
        self.rnn = self.rnn_type(input_size + feature_size, hidden_size, num_layers=number_layers,
                                 bidirectional=bidirection)

        bidre = 2 if bidirection else 1

        self.sel_attn_para = LinearSelfAttn(hidden_size * self.number_layers * bidre)

        self.fc = nn.Linear(hidden_size * self.number_layers * bidre, output_size)
        # self.fc = nn.Linear(hidden_size * self.number_layers * bidre, 128)
        self.bn = nn.BatchNorm1d(output_size)

        self.fc2 = nn.Linear(128, output_size)

    def init_weight(self, args):
        if args.finetune:
            nn.init.xavier_uniform(self.embedding.state_dict()['weight'])
        for name, param in self.rnn.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param, 10)

        nn.init.xavier_normal(self.sel_attn_sen_v)
        nn.init.xavier_normal(self.sel_attn_sen.state_dict()['linear_attn.weight'])

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc2.bias.data.fill_(0)

    def forward(self, x, x_mask, x_feature, sentences_len, clause, cls):
        '''
        这里第一版2018年02月01日12:56:55
        建议将篇章分成句子，句子里在分词，那么这里就应该表示为：
            B * para_len(num_sentence) * seq_len
        然后对一个Batch进行单独的:
            1. encoding
            2. Avg emb(dropout): 句子表示->篇章表示
            3. rnn
            3. FC (预想这里在第二版换成RL，做成多决策的RL)
        x: list , contain multi paragraph. batch first
        x_mask: data mask : batch * seq_len_max
        :param x:
        :param x_mask:
        :return:
        '''

        batch_size = x.size(0)

        paragraph_representation = []  # will be (batch * para_hidden)

        max_sen_len = max(sentences_len)

        for i, para in enumerate(x):
            para_emd = self.embedding(para)  # para_len(num_sentence) * seq_len * hidden_size

            sen_represen = []
            # pack one sentence

            for j, sen in enumerate(para_emd):
                sen_len = x_mask[i][j].eq(0).long().sum(0).data[0]  # 0 for origin, 1 for pad
                if sen_len == 0:
                    continue

                # att_words_score = torch.mm(F.relu(self.sel_attn_sen(sen)), self.sel_attn_sen_v)
                att_words_score = F.relu(self.sel_attn_sen(sen, x_mask[i][j]))
                # att_words_score = F.softmax(att_words_score, 0)
                sen = sen * att_words_score
                sen = torch.unsqueeze(sen[:sen_len], 0)  # (1 * seq_len * hidden_size)
                # # add feature to sen
                sen_feature = torch.unsqueeze(x_feature[i, j, :sen_len], 0)

                sen = torch.cat([sen, sen_feature], 2)
                sen_represen.append(sen.sum(1))

            if max_sen_len - len(sen_represen) > 0:
                for i in range(max_sen_len - len(sen_represen)):
                    sen_represen.append(Variable(FloatTensor(1, self.feature_size + self.hidden_size).float().fill_(0)))

            paragraph_representation.append(torch.cat(sen_represen, 0).unsqueeze(1))  # 拿到篇章的表示

        packed = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(paragraph_representation, 1), sentences_len)

        paragraph_output, paragraph_hidden = self.rnn(packed)
        # F.log_softmax(self.fc(torch.nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0])[0]))
        # paragraph_output, paragraph_len = torch.nn.utils.rnn.pad_packed_sequence(paragraph_output)

        paragraph_hidden = paragraph_hidden[0].view(batch_size, -1)

        # paragraph_hidden = self.bn_rnn(paragraph_hidden)
        paragraph_output = F.relu(self.bn(self.fc(paragraph_hidden)))

        # paragraph_output = F.relu(self.fc2(paragraph_output))

        out = F.log_softmax(paragraph_output, dim=0)  # batch * output_size
        return out


class SentenceRnn(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, words_size, feature_size, tags_len, bidirection=True,
                 number_layers=1,
                 dropout_rate=0.5,
                 rnn_type=nn.LSTM):
        super(SentenceRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.tags_len = tags_len
        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(words_size, hidden_size)

        self.rnn = self.rnn_type(input_size + feature_size, hidden_size, num_layers=number_layers,
                                 bidirectional=bidirection)

        bidre = 2 if bidirection else 1

        self.fc = nn.Linear(hidden_size * self.number_layers * bidre, output_size)

    def init_weight(self, args):
        if args.finetune:
            nn.init.xavier_uniform(self.embedding.state_dict()['weight'])
        for name, param in self.rnn.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)

    def forward(self, x, x_mask, x_feature, sentences_len, clause, cls):
        '''
        这里第一版2018年02月01日12:56:55
        建议将篇章分成句子，句子里在分词，那么这里就应该表示为：
            B * para_len(num_sentence) * seq_len
        然后对一个Batch进行单独的:
            1. encoding
            2. Avg emb(dropout): 句子表示->篇章表示
            3. rnn
            3. FC (预想这里在第二版换成RL，做成多决策的RL)
        x: list , contain multi paragraph. batch first
        x_mask: data mask : batch * seq_len_max
        :param x:
        :param x_mask:
        :return:
        '''

        batch_size = x.size(0)

        paragraph_representation = []  # will be (batch * para_hidden)

        for i, para in enumerate(x):
            para_emd = self.embedding(para)  # para_len(num_sentence) * seq_len * hidden_size

            para_emd = torch.cat([para_emd, x_feature[i]], 2)

            paragraph_representation.append(para_emd.unsqueeze(0))  # 拿到篇章的表示

        paragraph_representation = torch.cat(paragraph_representation, 0).sum(2)

        # packed = torch.nn.utils.rnn.pack_padded_sequence(paragraph_representation, sentences_len)

        paragraph_output, paragraph_hidden = self.rnn(paragraph_representation.transpose(0, 1))
        # F.log_softmax(self.fc(torch.nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0])[0]))
        # paragraph_output, paragraph_len = torch.nn.utils.rnn.pad_packed_sequence(paragraph_output)

        paragraph_hidden = paragraph_hidden[0].view(batch_size, -1)

        # paragraph_hidden = self.bn_rnn(paragraph_hidden)
        paragraph_output = F.relu(self.fc(paragraph_hidden))

        # paragraph_output = F.relu(self.fc2(paragraph_output))

        out = F.log_softmax(paragraph_output, dim=0)  # batch * output_size
        return out


class AvgRnnNew(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, words_size, feature_size, tags_len, bidirection=True,
                 number_layers=1,
                 dropout_rate=0.5,
                 rnn_type=nn.LSTM):
        super(AvgRnnNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirection = bidirection
        self.tags_len = tags_len
        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(words_size, hidden_size)

        # self.sel_attn_sen = nn.Linear(hidden_size, hidden_size)  # mend
        self.sel_attn_sen = LinearSelfAttn(hidden_size)  # mend
        self.sel_attn_sen_v = nn.Parameter(FloatTensor(hidden_size, 1))
        self.rnn = self.rnn_type(input_size + feature_size, hidden_size, num_layers=number_layers,
                                 bidirectional=bidirection)

        bidre = 2 if bidirection else 1

        self.sel_attn_para = LinearSelfAttn(hidden_size * bidre)

        # self.fc = nn.Linear(hidden_size * self.number_layers * bidre, output_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.hidden = None

        self.bn = nn.BatchNorm1d(output_size)

        self.fc2 = nn.Linear(128, output_size)

    def init_weight(self, args):
        if args.finetune:
            nn.init.xavier_uniform(self.embedding.state_dict()['weight'])
        for name, param in self.rnn.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param, 10)

        nn.init.xavier_normal(self.sel_attn_sen_v)
        nn.init.xavier_normal(self.sel_attn_sen.state_dict()['linear_attn.weight'])

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)
        nn.init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc2.bias.data.fill_(0)

    def init_hidden(self, batch_size, sen_len):
        h0 = Variable(FloatTensor(2, batch_size, self.hidden_size))
        c0 = Variable(FloatTensor(2, batch_size, self.hidden_size))
        self.hidden = (h0, c0)
        return (h0, c0)

    def forward(self, x, x_mask, x_feature, sentences_len, clause, cls, hidden=None):
        '''
        这里第一版2018年02月01日12:56:55
        建议将篇章分成句子，句子里在分词，那么这里就应该表示为：
            B * para_len(num_sentence) * seq_len
        然后对一个Batch进行单独的:
            1. encoding
            2. Avg emb(dropout): 句子表示->篇章表示
            3. rnn
            3. FC (预想这里在第二版换成RL，做成多决策的RL)
        x: list , contain multi paragraph. batch first
        x_mask: data mask : batch * seq_len_max
        :param x:
        :param x_mask:
        :return:
        '''

        batch_size = x.size(0)

        paragraph_representation = []  # will be (batch * para_hidden)

        max_sen_len = max(sentences_len)

        for i, para in enumerate(x):
            para_emd = self.embedding(para)  # para_len(num_sentence) * seq_len * hidden_size

            sen_represen = []
            # pack one sentence

            for j, sen in enumerate(para_emd):
                sen_len = x_mask[i][j].eq(0).long().sum(0).data[0]  # 0 for origin, 1 for pad
                if sen_len == 0:
                    continue

                # att_words_score = torch.mm(F.relu(self.sel_attn_sen(sen)), self.sel_attn_sen_v)
                att_words_score = F.relu(self.sel_attn_sen(sen, x_mask[i][j]))
                # att_words_score = F.softmax(att_words_score, 0)
                sen = sen * att_words_score
                sen = torch.unsqueeze(sen[:sen_len], 0)  # (1 * seq_len * hidden_size)
                # # add feature to sen
                sen_feature = torch.unsqueeze(x_feature[i, j, :sen_len], 0)

                sen = torch.cat([sen, sen_feature], 2)
                sen_represen.append(sen.sum(1))

            if max_sen_len - len(sen_represen) > 0:
                for i in range(max_sen_len - len(sen_represen)):
                    sen_represen.append(Variable(FloatTensor(1, self.feature_size + self.hidden_size).float().fill_(0)))

            paragraph_representation.append(torch.cat(sen_represen, 0).unsqueeze(1))  # 拿到篇章的表示

        packed = torch.nn.utils.rnn.pack_padded_sequence(torch.cat(paragraph_representation, 1), sentences_len)

        paragraph_output, hidden = self.rnn(packed, hidden)
        # F.log_softmax(self.fc(torch.nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0])[0]))
        paragraph_output, paragraph_len = torch.nn.utils.rnn.pad_packed_sequence(paragraph_output)

        # paragraph_output = torch.nn.utils.rnn.pad_packed_sequence(paragraph_output)[0].transpose(0, 1).contiguous().view(batch_size, -1)

        # paragraph_hidden = self.bn_rnn(paragraph_hidden)
        paragraph_output = F.relu(
            self.bn(self.fc(paragraph_output[-1].transpose(0, 1).contiguous().view(batch_size, -1))))

        # paragraph_output = F.relu(self.fc2(paragraph_output))

        out = F.log_softmax(paragraph_output, dim=0)  # batch * output_size
        return out, hidden
