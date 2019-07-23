# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import Myattention


class LSTM_Attention(object):

    def CNN(self,embeded_chars, embedding_size,dropout_keep_prob,max_sent_len,name):

        with tf.variable_scope("CNN_"+name,reuse=tf.AUTO_REUSE):

            pool_out = []
            filter_sizes = [1,3,5]
            num_filters = 150

            embeded_chars = tf.expand_dims(embeded_chars,-1)

            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size , reuse=tf.AUTO_REUSE):

                    filter_shape = [filter_size, embedding_size ,1 ,num_filters ]
                    name_f = 'W%s' % i
                    W = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=filter_shape, name=name_f)
                    b = tf.Variable(tf.constant(0.0,shape=[num_filters]), name=name_f)

                    conv = tf.nn.conv2d(
                        embeded_chars,
                        W,
                        strides=[1,1,1,1],
                        padding='VALID',
                        name= 'conv')

                    h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')

                    pooled = tf.nn.max_pool(
                    h,
                    ksize=[1 ,max_sent_len-filter_size+1 ,1 ,1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name = 'pool'
                    )
                    pool_out.append(pooled)

            self.h_pool = tf.concat(pool_out, 3)
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

            with tf.variable_scope("dropout"+name,reuse=tf.AUTO_REUSE):
                self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

            return self.h_pool_flat,num_filters_total

    def __init__(self,
                 max_sent_len,
                 embedding_size,
                 hidden_units,
                 l2_reg_lambda,
                 batch_size,
                 W_embedding,
                 margin):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.hidden_units = hidden_units

        self.max_sent_len = max_sent_len

        self.input_x1 = tf.placeholder(tf.int32, [None,None], name="input_x1")

        self.input_y = tf.placeholder(tf.float32,[None,5], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        initializer = tf.contrib.layers.xavier_initializer()  # 变量初始化
        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            self.W = tf.get_variable(initializer=W_embedding, dtype=tf.float32, trainable=True, name="vocabulary")
            self.word_embedding = tf.concat([tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32)),
                                             tf.get_variable("unk", [1, self.embedding_size], dtype=tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer()),
                                             self.W], 0)
            self.embeded_chars1 = tf.nn.embedding_lookup(self.word_embedding, self.input_x1)  # 查词表

        with tf.name_scope("word_out"):

            self.word_level_output1,num_filters_total1 = self.CNN(self.embeded_chars1, self.embedding_size, self.dropout_keep_prob,self.max_sent_len ,name="word")

        with tf.name_scope("full_connect"):
            self.w_all = tf.get_variable(initializer=initializer,shape=[num_filters_total1,5] ,dtype=tf.float32,name="w_all")
            self.b_all = tf.get_variable(initializer=initializer,shape=[5],dtype=tf.float32,name="b_all")

            l2_loss +=tf.nn.l2_loss(self.w_all)
            l2_loss +=tf.nn.l2_loss(self.b_all)

            with tf.name_scope("score1"):
                self.score1 = tf.nn.xw_plus_b(self.word_level_output1, self.w_all,self.b_all,name='score1')

            self.predictions = tf.argmax(self.score1, axis=1, name='predictions', output_type=tf.int32)
            self.scores = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.score1)

            correct_acc = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1, output_type=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_acc, 'float'), name='accuracy')

        with tf.name_scope("loss"):

            self.sum_loss = tf.reduce_sum(self.scores)
            self.loss = tf.reduce_mean(self.scores) + l2_reg_lambda* l2_loss

