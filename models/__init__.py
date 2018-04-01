#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys

sys.path.append('..')
import torch
import sys

sys.path.append('..')
from .model import ReaderModel
from .layers import StackRnn
from .model_0330 import CnnRnnModel

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
