#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import logging

import sys

sys.path.append('..')
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import time

from models import ReaderModel
from utils import util
from utils.dataset import RnnDataSet
from utils.evaluation import Evaluate

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(filename)s %(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

input_size = 128
hidden_size = 128
feature_size = 49
output_size = 8
class_size = 452
bidirection = True
number_layers = 1
dropout_rate = 0.2
lr = 0.005
batch_size = 32
train_file = '../data/precessed/pos_ner_content.txt'
word2vec_file = '../data/precessed/word2vec.bin'
TAGS_LEN = 452
CLASS_LEN = 8

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def init_model(words_dict, feature_dict, args):
    logger.info('Initiate Model...')
    model = ReaderModel(input_size, hidden_size, output_size, len(feature_dict), len(words_dict), words_dict, TAGS_LEN,
                        CLASS_LEN, feature_dict=feature_dict,
                        bidirection=True, number_layers=number_layers,
                        dropout_rate=dropout_rate,
                        rnn_type=nn.LSTM, args=args)
    model.init_optim(lr1=lr)
    return model


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = util.AverageMeter()
    epoch_time = util.Timer()
    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))  # run on one batch

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def make_dataset(data, model):
    rnn_dataset = RnnDataSet(data, model)
    data_loader = DataLoader(rnn_dataset, batch_size=batch_size, shuffle=True, collate_fn=util.collate_batch,
                             pin_memory=False,
                             drop_last=True)
    return data_loader


def main(args):
    # data, word2ids = util.load_train_data(train_file, word2vec_file)
    data, word2ids, embed_arr = util.load_data()
    feature_dict = util.build_feature_dict(args, data)
    model = init_model(words_dict=word2ids, feature_dict=feature_dict, args=args)
    # model.quick_load_embed(embed_arr)
    data_loader = make_dataset(data, model)

    start_epoch = 0

    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Train now! Output loss every %d batch...' % args.display_iter)
    stats = {'timer': util.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        train(args, data_loader, model, stats)

        result = evaluate(model, data_loader, global_stats=stats)

        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


def evaluate(model, data_loader, global_stats, mode='train'):
    # Use precision for classify
    eval_time = util.Timer()
    start_acc = util.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s = model.predict(ex)
        answer = ex[5]
        # We get metrics for independent start/end and joint start/end
        start_acc.update(Evaluate.accuracies(pred_s, answer.cpu().data.numpy()), 1)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial use Accuracy: Epoch = %d | acc = %.2f | ' %
                (mode, global_stats['epoch'], start_acc.avg) +
                ' = %d | ' %
                (examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'acc': start_acc.avg}


if __name__ == '__main__':
    logger.info("System initiate...")
    parser = argparse.ArgumentParser(
        'AI Challenger',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model specific details
    detail = parser.add_argument_group('AI Challenger Model Details')
    detail.add_argument('--concat-rnn-layers', type=bool, default=True,
                        help='Combine hidden states from each encoding layer')
    detail.add_argument('--use-pos', type=bool, default=True,
                        help='Whether to use pos features')
    detail.add_argument('--use-ner', type=bool, default=True,
                        help='Whether to use ner features')
    detail.add_argument('--use-tf', type=bool, default=False,
                        help='Whether to use term frequency features')

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type=bool, default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type=bool, default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=10,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--grad-clipping', type=bool, default=True,
                         help='Grad clipping')
    runtime.add_argument('--checkpoint', type=bool, default=True,
                         help='checkpoint')
    runtime.add_argument('--model-file', type=str, default='check' + str(time.time().is_integer()) + '.model',
                         help='Model file save path')
    runtime.add_argument('--finetune', type=bool, default=False,
                         help='Model file save path')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid-metric', type=str, default='acc',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')

    args = parser.parse_args()

    main(args)

    # args.add_argument('--input-size', type=int, default=)
