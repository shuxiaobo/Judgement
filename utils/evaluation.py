#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 30/01/2018
import sys

sys.path.append('..')
import torch


class Evaluate():

    def __init__(self):
        pass

    @classmethod
    def accuracies(self, predicts, answers):
        """
        It should be list 、tensor with one dim or numpy vector.
        :param predicts:
        :param answers:
        :param com: com should be a function used for compare
        :return:
        """
        if torch.is_tensor(answers):
            answers = answers.numpy()
        if torch.is_tensor(predicts):
            predicts = predicts.numpy()
        correct = sum([1 if predicts[i] == con else 0 for i, con in enumerate(answers)])

        return (correct * 1.0) / len(predicts)

    @classmethod
    def recall(self, predicts, answers):
        """
        It should be list 、tensor with one dim or numpy vector.
        :param predicts:
        :param answers:
        :param com: com should be a function used for compare
        :return:
        """
        if torch.is_tensor(answers):
            answers = answers.numpy()
        correct = sum([1 if predicts[i] == con else 0 for i, con in enumerate(answers)])

        return (correct * 1.0) / len(predicts)