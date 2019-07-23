# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import Myattention

class LSTM_Attention(object):
    def length_word(self,input_x):

        used = tf.sign(tf.abs(input_x))
        length = tf.reduce_sum(used, 1)
        length = tf.reshape(length,[-1])
        length = tf.cast(length, tf.int32)
        return length

    def Bi_LSTM(self,x,drop_out, step, hidden_units,name,input_x):
        if name=='word':
            length = self.length_word(input_x)

        with tf.variable_scope("biLSTM_word"+name,reuse=tf.AUTO_REUSE):

            fw_cell = tf.contrib.rnn.LSTMCell(hidden_units)
            # fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=drop_out)
            bw_cell = tf.contrib.rnn.LSTMCell(hidden_units)

            ((fw_outputs, bw_outputs), (fw_outputs_final, bw_outputs_final)) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                x,
                dtype=tf.float32,
                sequence_length=length,
                )

            outputs = tf.concat((fw_outputs, bw_outputs),2)
            final = tf.concat((fw_outputs_final.h, bw_outputs_final.h), 1)

            return outputs,final

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
        self.input_x2 = tf.placeholder(tf.int32, [None,None], name="input_x2")

        self.input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        initializer = tf.contrib.layers.xavier_initializer()      # 变量初始化

        with tf.name_scope("embedding"):

            self.W = tf.get_variable(initializer=W_embedding,dtype=tf.float32,trainable=True,name="vocabulary")
            self.word_embedding = tf.concat([tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32)),
                                             tf.get_variable("unk", [1, self.embedding_size], dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer()),
                                             self.W], 0)
            self.embeded_chars1 = tf.nn.embedding_lookup(self.word_embedding, self.input_x1)   # 查词表
            self.embeded_chars2 = tf.nn.embedding_lookup(self.word_embedding, self.input_x2)
            # print("词表",self.W)

        with tf.name_scope("word_out"):
        # with tf.variable_scope("word_out"):

            self.word_out1,self.word_out1_final = self.Bi_LSTM(self.embeded_chars1, self.dropout_keep_prob,self.max_sent_len, self.hidden_units,name="word",input_x=self.input_x1)
            self.word_out2,self.word_out2_final = self.Bi_LSTM(self.embeded_chars2, self.dropout_keep_prob,self.max_sent_len, self.hidden_units,name="word",input_x=self.input_x2)

            # with tf.variable_scope('dropout'):
            #     self.word_level_output1 = tf.nn.dropout(
            #         self.word_level_output1,keep_prob=self.dropout_keep_prob
            #     )
            #     self.word_level_output2 = tf.nn.dropout(
            #         self.word_level_output2, keep_prob=self.dropout_keep_prob
            #     )

        with tf.name_scope("full_connect"):

            self.w_all = tf.get_variable(initializer=initializer,shape=[hidden_units*2,1] ,dtype=tf.float32,name="w_all")
            self.b_all = tf.get_variable(initializer=initializer,shape=[1],dtype=tf.float32,name="b_all")

            with tf.name_scope("score1"):
                self.score1 = tf.nn.xw_plus_b(self.word_out1_final, self.w_all,self.b_all,name='score1')
            with tf.name_scope("score2"):
                self.score2 = tf.nn.xw_plus_b(self.word_out2_final, self.w_all, self.b_all, name='score2')


            self.scores = tf.maximum(0.0, -self.input_y * (self.score1 - self.score2)+margin)


        with tf.name_scope("loss"):

            self.sum_loss = tf.reduce_sum(self.scores)
            self.loss = tf.reduce_mean(self.scores)



