import logging

import torch
import torch.nn as nn
from layers import StackRnn
from torch.nn import MultiLabelMarginLoss
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.autograd import Variable

logger = logging.getLogger(__file__)


class ReaderModel():
    def __init__(self, input_size, hidden_size, output_size, feature_size, words_size, words_dict, tags_len, class_len, feature_dict,
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
        self.loss_fun = NLLLoss()
        self.words_dict = words_dict
        self.tags_len = tags_len
        self.class_len = class_len
        self.feature_dict = feature_dict
        self.feature_size = feature_size
        self.optimizer = None
        self.args = args
        self.updates = 0
        self.words_num = words_size

        self.network = StackRnn(input_size, hidden_size, output_size, words_size, feature_size, bidirection=bidirection,
                                number_layers=number_layers,
                                dropout_rate=dropout_rate,
                                rnn_type=rnn_type)

    def init_optim(self, lr1, lr2=None, weight_decay=0):
        if not self.optimizer:
            ignore_param = list(map(id, self.network.embedding.parameters()))
            base_param = filter(lambda x: id(x) not in ignore_param, self.network.parameters())
            if lr2 is None: lr2 = lr1 * 0.5
            optimizer = Adam([dict(params=base_param, lr=lr1, weight_decay=weight_decay),
                              {'params': self.network.embedding.parameters(), 'lr': lr2}])
            self.optimizer = optimizer

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
            words_num , size = f.readline().strip().split(' ')
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
        loss = self.loss_fun(score, ex[4])
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

        return loss.data[0], ex[0].size(0)
