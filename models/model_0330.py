#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 29/03/2018


import logging
import sys

sys.path.append('..')
from models.layers import *
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
import copy

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(filename)s %(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class CnnRnnModel():
    def __init__(self, input_size, hidden_size, output_size, feature_size, words_size, words_dict, tags_len, class_len,
                 feature_dict,
                 bidirection=True, number_layers=2, dropout_rate=0.5, rnn_type=nn.LSTM, args=None):
        """
        for classify first

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param words_size:
        :param words_dict:
        :param tags_len:
        :param class_len:
        :param feature_dict:
        :param bidirection:
        :param number_layers:
        :param dropout_rate:
        :param rnn_type:
        """
        self.loss_fun = CrossEntropyLoss()
        self.words_dict = words_dict
        self.tags_len = tags_len
        self.class_len = class_len
        self.feature_dict = feature_dict
        self.feature_size = feature_size
        self.optimizer = None
        self.args = args
        self.updates = 0
        self.words_num = words_size

        kermel_size = [[5, 5], [5, 5], [5, 5]]
        pool_size = [[2, 2], [2, 2], [2, 2]]
        channel_sizes = [input_size, 256, input_size, 128]

        self.network = Cnn_rnn(words_size, input_size, channel_sizes, kermel_size, pool_size, dropout_rate, hidden_size,
                               output_size,
                               bidirection=bidirection,
                               number_layers=number_layers,
                               rnn_type=rnn_type)
        if USE_CUDA:
            self.network.cuda()

    def init_optim(self, lr1, lr2=None, weight_decay=0):
        self.network.init_weight(self.args)
        if not self.optimizer:
            ignore_param = list(map(id, self.network.cnn.embedding.parameters()))
            base_param = filter(lambda x: id(x) not in ignore_param, self.network.parameters())
            if lr2 is None: lr2 = lr1 * 0.5
            optimizer = Adam([dict(params=base_param, lr=lr1, weight_decay=weight_decay),
                              {'params': self.network.cnn.embedding.parameters(), 'lr': lr2}])
            self.optimizer = optimizer
        logger.info('Initiate Optim Over...')

    def quick_load_embed(self, embed_ndarray):
        logger.info(
            'Load the embedding %d words %d dimension per word' % (embed_ndarray.shape[0], embed_ndarray.shape[1]))
        embedding = self.network.embedding.weight.data
        embedding = torch.from_numpy(embed_ndarray)
        logger.info('Load the embedding over..')

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.words_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file, encoding='utf-8') as f:
            words_num, size = f.readline().strip().split(' ')
            # assert (size == embedding.size(1) + 1)
            # assert ( self.words_num < words_num)
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                # w = self.words_dict.normalize(parsed[0])  # No need for chinese
                w = parsed[0]
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.words_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.words_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.words_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Run forward
        score = self.network(*ex)

        # Compute loss and accuracies of the label
        loss = self.loss_fun(score, ex[5])
        # Compute loss of the classify

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if self.args.grad_clipping:
            torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                          self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        # self.reset_parameters()

        logger.info('Get Loss %.2f' % loss.data[0])

        return loss.data[0], ex[0].size(0)

    def predict(self, ex):

        self.network.eval()  # 测试模式

        cls, _ = self.network(*ex)

        return np.argmax(cls.data.cpu().numpy(), 1)

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.words_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'words_dict': self.words_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.words_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    # @staticmethod
    # def load(filename, new_args=None, normalize=True):
    #     logger.info('Loading model %s' % filename)
    #     saved_params = torch.load(
    #         filename, map_location=lambda storage, loc: storage
    #     )
    #     word_dict = saved_params['word_dict']
    #     feature_dict = saved_params['feature_dict']
    #     state_dict = saved_params['state_dict']
    #     args = saved_params['args']
    #     if new_args:
    #         args = override_model_args(args, new_args)
    #     return DocReader(args, word_dict, feature_dict, state_dict, normalize)
    #
    # @staticmethod
    # def load_checkpoint(filename, normalize=True):
    #     logger.info('Loading model %s' % filename)
    #     saved_params = torch.load(
    #         filename, map_location=lambda storage, loc: storage
    #     )
    #     word_dict = saved_params['word_dict']
    #     feature_dict = saved_params['feature_dict']
    #     state_dict = saved_params['state_dict']
    #     epoch = saved_params['epoch']
    #     optimizer = saved_params['optimizer']
    #     args = saved_params['args']
    #     model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
    #     model.init_optimizer(optimizer)
    #     return model, epoch
