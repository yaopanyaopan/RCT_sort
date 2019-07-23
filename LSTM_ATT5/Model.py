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

        self.input_y = tf.placeholder(tf.float32,[None,5], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        initializer = tf.contrib.layers.xavier_initializer()  # 变量初始化
        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):

            self.W = tf.get_variable(initializer=W_embedding,dtype=tf.float32,trainable=True,name="vocabulary")
            self.word_embedding = tf.concat([tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32)),
                                             tf.get_variable("unk", [1, self.embedding_size], dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer()),
                                             self.W], 0)
            self.embeded_chars1 = tf.nn.embedding_lookup(self.word_embedding, self.input_x1)   # 查词表

            # print("词表",self.W)

        with tf.name_scope("word_out"):
        # with tf.variable_scope("word_out"):
            self.embeded_chars1 = tf.reshape(self.embeded_chars1,[-1, self.max_sent_len,self.embedding_size])

            self.word_out1 = self.Bi_LSTM(self.embeded_chars1, self.dropout_keep_prob,self.max_sent_len, self.hidden_units,name="word",input_x=self.input_x1)

            self.word_level_output1, self.alphas1 = Myattention.attention(self.word_out1, self.max_sent_len, self.hidden_units*2,return_alphas=True,name="word_level",input_x=self.input_x1)

            #
            with tf.variable_scope('dropout'):
                self.word_level_output1 = tf.nn.dropout(
                    self.word_level_output1,keep_prob=self.dropout_keep_prob
                )


        with tf.name_scope("sentence_out"):
        # with tf.variable_scope("sentence_out"):
            self.sentence_input1 = tf.reshape(self.word_level_output1,[-1, self.max_seq_len, self.hidden_units*2])

            self.sentence_level_output1 ,self.alphas11 = Myattention.attention(self.sentence_input1, self.max_seq_len, self.hidden_units*2, return_alphas=True,name="sentence_level",input_x=self.input_x1 )

            tf.add_to_collection('attention1', self.alphas11)

            with tf.variable_scope('dropout'):
                self.sentence_level_output1 = tf.nn.dropout(
                self.sentence_level_output1, keep_prob=self.dropout_keep_prob
                )


        with tf.name_scope("full_connect"):

            self.w_all = tf.get_variable(initializer=initializer,shape=[hidden_units*2,5] ,dtype=tf.float32,name="w_all")
            self.b_all = tf.get_variable(initializer=initializer,shape=[5],dtype=tf.float32,name="b_all")

            l2_loss += tf.nn.l2_loss(self.w_all)
            l2_loss += tf.nn.l2_loss(self.b_all)

            with tf.name_scope("score1"):
                self.score1 = tf.nn.xw_plus_b(self.sentence_level_output1, self.w_all,self.b_all,name='score1')

            self.predictions = tf.argmax(self.score1, axis=1, name='predictions', output_type=tf.int32)
            self.scores = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score1)

            correct_acc = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1, output_type=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_acc, 'float'), name='accuracy')


        with tf.name_scope("loss"):

            self.sum_loss = tf.reduce_sum(self.scores)
            self.loss = tf.reduce_mean(self.scores) + l2_reg_lambda* l2_loss


