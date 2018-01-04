import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .layers import StackRnn

logger = logging.getLogger(__file__)


class ReaderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, words_size, words_dict, tags_len, class_len, feature_dict,
        bidirection=True, number_layers=2, dropout_rate=0.5, rnn_type=nn.LSTM):
        self.words_dict = words_dict
        self.tags_len = tags_len
        self.class_len = class_len
        self.feature_dict = feature_dict

        self.network = StackRnn(input_size, hidden_size, output_size, words_size, bidirection=bidirection,
                                number_layers=number_layers,
                                dropout_rate=dropout_rate,
                                rnn_type=rnn_type)

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
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])  # No need for chinese
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

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

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:5]]
            target_s = Variable(ex[5].cuda(async=True))
            target_e = Variable(ex[6].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[5])
            target_e = Variable(ex[6])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex[0].size(0)
