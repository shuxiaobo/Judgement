import argparse

import torch.nn as nn
from torch.nn import NLLLoss
from torch.optim import SGD

from .layers import StackRnn

input_size = 128
hidden_size = 128
output_size = 8
class_size = 452
bidirection = True
number_layers = 3
dropout_rate = 0.5
lr = 0.005


def init_model(words_size):
    model = StackRnn(input_size, hidden_size, output_size, words_size, bidirection=True, number_layers=3,
                     dropout_rate=dropout_rate,
                     rnn_type=nn.LSTM)
    optimizer = SGD(params=model.parameters(), lr=lr)
    nll_loss = NLLLoss()
    return model

def train(data):

    # get batch



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # args.add_argument('--input-size', type=int, default=)
