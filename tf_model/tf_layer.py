#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 23/03/2018

import tensorflow as tf
import numpy as np


class AvgRNN():

    def __init__(self, embedding_mat, emb_size, hidden_unit, filter_sizes, num_filters,
                 max_pool_size, num_classes, sequence_length, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length, None], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, emb_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_regu = tf.constant([0.0])

        with tf.name_scope('embedding'), tf.device('/cpu:0'):
            w = tf.Variable(name='w', initial_value=embedding_mat)
            self.embedding = tf.nn.embedding_lookup(w, self.input_x)
            self.embedding = tf.reduce_mean(self.embedding, axis=2)
            emb = tf.expand_dims(self.embedding, -1)

        pool_cat = []
        pool_reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool % s' % filter_size) as scope:
                filter_shape = [filter_size, emb_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name='b')
                conv1 = tf.nn.conv2d(emb, filter=W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu')

                h_pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                                          padding='SAME',
                                          data_format="NHWC", name=None)
                h_pooled = tf.reshape(h_pooled, shape=[-1, pool_reduced, num_filters])

                pool_cat.append(h_pooled)

        pool_cat = tf.concat(pool_cat, axis=2)

        pool_cat = tf.nn.dropout(pool_cat, keep_prob=self.dropout_keep_prob)

        gru_cell = tf.contrib.rnn.rnn_cell.GRUCell(num_units=hidden_unit)

        gru_cell = tf.contrib.rnn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_keep_prob)

        gru_inputs = [input_ for input_ in tf.split(pool_cat, num_or_size_splits=pool_reduced, axis=1)]

        self._init_state = gru_cell.zero_state(self.batch_size, tf.float32)

        outputs, hidden = tf.contrib.rnn.static_rnn(gru_cell, gru_inputs, initial_state=self._initial_state,
                                                    sequence_length=self.real_len)
        with tf.name_scope('output') as scope:
            tf.get_variable_scope().reuse_variables()
            out = outputs[0]
            ones = tf.ones(shape=[1, hidden_unit])
            for i in range(1, len(outputs)):
                ind = self.real_len < (i - 1)
                ind = tf.float32(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, ones)

                out = tf.add(tf.multiply(out, mat) + tf.multiply(out, (1.0 - mat)))

        with tf.name_scope('fc-predict') as scope:
            self.w = tf.Variable(name='w', initial_value=tf.constant(
                tf.truncated_normal(shape=[hidden_unit, num_classes], stddev=0.1)))
            b = tf.Variable(initial_value=tf.constant([0.1], shape=[num_classes]), name='b')

            self.scores = tf.nn.xw_plus_b(out, w, b, name='scores')

            self.predict = tf.argmax(self.scores, axis=1, name='predict')
        with tf.name_scope('loss') as scope:

            l2_regu += tf.nn.l2_loss(w)
            l2_regu += tf.nn.l2_loss(b)

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores, name='loss')
            self.loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_regu
        with tf.name_scope('accuracy') as scope:
            correct_count = tf.equal(tf.argmax(self.input_y, 1), self.predict)
            self.accracy = tf.reduce_mean(tf.cast(correct_count, dtype=tf.float32), name='accuracy')


