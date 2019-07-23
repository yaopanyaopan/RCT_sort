# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import Myattention


class LSTM_Attention(object):

    def length_word(self,input_x):

        used = tf.sign(tf.abs(input_x))
        length = tf.reduce_sum(used, 2)
        length = tf.reshape(length,[-1])
        length = tf.cast(length, tf.int32)
        return length

    def length_sentence(self,input_x):
        used = tf.sign(tf.abs(input_x))
        length = tf.reduce_sum(used, 2)
        length = tf.sign(length)
        length = tf.reduce_sum(length, 1)
        length = tf.reshape(length,[-1])
        length = tf.cast(length,tf.int32)
        return length

    def Bi_LSTM(self,x,drop_out, step, hidden_units,name,input_x):
        if name=='word':
            length = self.length_word(input_x)
        else:
            length = self.length_sentence(input_x)

        with tf.variable_scope("biLSTM_word"+name,reuse=tf.AUTO_REUSE):

            fw_cell = tf.contrib.rnn.LSTMCell(hidden_units)
            # fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=drop_out)
            bw_cell = tf.contrib.rnn.LSTMCell(hidden_units)

            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                x,
                dtype=tf.float32,
                sequence_length=length,
                )

            outputs = tf.concat((fw_outputs, bw_outputs),2)
            return outputs


    def CNN(self,embeded_chars, embedding_size,dropout_keep_prob,max_sent_len, max_seq_len ,hidden_units,name="word"):

        pool_out = []
        filter_sizes = [1,3,5]
        num_filters = 150
        embeded_chars = tf.expand_dims(embeded_chars, -1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size , reuse=tf.AUTO_REUSE):

                filter_shape = [filter_size, embedding_size ,1 ,num_filters ]  # 高、宽、输入、输出
                name = 'W%s' % i
                W = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=filter_shape, name=name)
                b = tf.Variable(tf.constant(0.0,shape=[num_filters]), name='b')

                conv = tf.nn.conv2d(
                    embeded_chars,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name= 'conv')

            h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')

            pooled = tf.nn.max_pool(
                h,
                ksize = [1 ,max_sent_len-filter_size+1 ,1 ,1],
                strides = [1,1,1,1],
                padding = 'VALID',
                name = 'pool'
            )
            pool_out.append(pooled)

        self.h_pool = tf.concat(pool_out, 3)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,max_seq_len,num_filters_total])

        return self.h_pool_flat

    def __init__(self,
                 max_seq_len,
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

        self.max_seq_len = max_seq_len
        self.max_sent_len = max_sent_len


        self.input_x1 = tf.placeholder(tf.int32, [None,None,None], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None,None,None], name="input_x2")

        self.input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        initializer = tf.contrib.layers.xavier_initializer()  # 变量初始化

        with tf.name_scope("embedding"):
            self.W = tf.get_variable(initializer=W_embedding, dtype=tf.float32, trainable=True, name="vocabulary")
            self.word_embedding = tf.concat([tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32)),
                                             tf.get_variable("unk", [1, self.embedding_size], dtype=tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer()),
                                             self.W], 0)
            self.embeded_chars1 = tf.nn.embedding_lookup(self.word_embedding, self.input_x1)  # 查词表
            self.embeded_chars2 = tf.nn.embedding_lookup(self.word_embedding, self.input_x2)

        with tf.name_scope("word_out"):
            self.embeded_chars1 = tf.reshape(self.embeded_chars1,[-1, self.max_sent_len,self.embedding_size])
            self.embeded_chars2 = tf.reshape(self.embeded_chars2,[-1, self.max_sent_len,self.embedding_size])

            self.word_level_output1 = self.CNN(self.embeded_chars1, self.embedding_size, self.dropout_keep_prob,self.max_sent_len,self.max_seq_len ,self.hidden_units,name="word")
            self.word_level_output2 = self.CNN(self.embeded_chars2, self.embedding_size, self.dropout_keep_prob,self.max_sent_len,self.max_seq_len , self.hidden_units, name="word")


        with tf.name_scope("sentence_out"):

            self.sentence_output1 = self.Bi_LSTM(self.word_level_output1, self.dropout_keep_prob,self.max_seq_len,hidden_units,name="sentence",input_x=self.input_x1)
            self.sentence_output2 = self.Bi_LSTM(self.word_level_output2, self.dropout_keep_prob,self.max_seq_len,hidden_units,name="sentence",input_x=self.input_x2)

            self.sentence_level_output1 ,self.alphas11 = Myattention.attention(self.sentence_output1, self.max_seq_len, self.hidden_units*2, return_alphas=True,name="sentence_level" ,input_x=self.input_x1)
            self.sentence_level_output2 ,self.alphas22 = Myattention.attention(self.sentence_output2, self.max_seq_len , self.hidden_units*2, return_alphas=True,name="sentence_level",input_x=self.input_x2)

        with tf.name_scope("full_connect"):
            self.w_all = tf.get_variable(initializer=initializer,shape=[hidden_units*2,1] ,dtype=tf.float32,name="w_all")
            self.b_all = tf.get_variable(initializer=initializer,shape=[1],dtype=tf.float32,name="b_all")

            with tf.name_scope("score1"):

                self.score1 = tf.nn.xw_plus_b(self.sentence_level_output1, self.w_all,self.b_all,name='score1')
            with tf.name_scope("score2"):
                self.score2 = tf.nn.xw_plus_b(self.sentence_level_output2, self.w_all, self.b_all, name='score2')

            self.scores = tf.maximum(0.0, -self.input_y * (self.score1 - self.score2) + margin)

        with tf.name_scope("loss"):

            self.sum_loss = tf.reduce_sum(self.scores)

            self.loss = tf.reduce_mean(self.scores)






