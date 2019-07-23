# -*-coding:utf-8 -*-
__author__ = '$'

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import input_helpers
import Model
from tensorflow.contrib import learn
import csv
import pickle
import sklearn
import math
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
inpH = input_helpers.InputHelper
FLAGS = tf.flags.FLAGS

print('\n Parameters:')
for attr, value in sorted(FLAGS._flags().items()):
    print('{}={}'.format(attr.upper(),value))
print('')

pkl_x_raw1 = open('../data/test_W5.pickle', 'rb')
x_test = pickle.load(pkl_x_raw1)
pkl_y_test = open('../data/label_test_W5.pickle', 'rb')
y_test = pickle.load(pkl_y_test)

# print(np.shape(x_test))
print('测试集数量： ',len(y_test))

#----------------------评 估----------------------------------------------------------
print('\nEvaluating...\n')
with tf.Session() as sess:
    checkpoint_file = tf.train.get_checkpoint_state('runs/1560431582.8990076')

    for path in checkpoint_file.all_model_checkpoint_paths:

        print('模型路径 ：', path)

        saver = tf.train.import_meta_graph('{}.meta'.format(path))

        saver.restore(sess, path)
        graph = tf.get_default_graph()

        input_x = graph.get_tensor_by_name('input_x1:0')
        dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        predictions = graph.get_tensor_by_name('full_connect/predictions:0')
        scores = graph.get_tensor_by_name('full_connect/score1/score1:0')

        batches = inpH.batch_iter1(list(zip(x_test, y_test)), 256, False)

        all_predictions = []
        Label = []
        Pre = []
        P = []
        for db in batches:
            # print(len(sco))
            x_test_batch, y_dev_b = zip(*db)

            label = np.argmax(y_dev_b, 1)
            # print(label)
            for la in label:
                Label.append(la)

            batch_predictions,batch_scores = sess.run(
                [predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0}
            )
            # print(batch_predictions)

            def softmax(x):
                x_exp = np.exp(x)
                # 如果是列向量，则axis=0
                x_sum = np.sum(x_exp, axis=1, keepdims=True)
                s = x_exp / x_sum
                return s


            pro_score = softmax(batch_scores)

            for ps in pro_score:
                max_value = 0.0
                for ss in ps:
                    if ss > max_value:
                        max_value = ss
                P.append(max_value)

            for p in batch_predictions:
                Pre.append(p)

        # cl = sklearn.metrics.classification_report(Label, Pre)
        # print(cl)
        threads = 2000

        c = sorted(enumerate(Label), key=lambda x: x[1], reverse=True)
        # print(c)
        sum_I = 0.0
        for index, val in enumerate(c[:threads]):
            res_val = val[1] + 1
            sum_I = sum_I + float(math.pow(2, res_val) - 1) / math.log((index + 1) + 1, 2)

        t5 = []
        t4 = []
        t3 = []
        t2 = []
        t1 = []
        cal1 = []
        cal2 = []
        cal3 = []
        cal4 = []
        cal5 = []
        New_d = []
        for ids, ps_val in enumerate(Pre):
            if ps_val == 0:  # 预测的值
                t1.append(P[ids])  # 置信度
                cal1.append(Label[ids])
            elif ps_val == 1:  # 预测的值
                t2.append(P[ids])  # 置信
                cal2.append(Label[ids])
            elif ps_val == 2:  # 预测的
                t3.append(P[ids])  # 置信度
                cal3.append(Label[ids])
            elif ps_val == 3:  # 预测的值
                t4.append(P[ids])  # 置信度
                cal4.append(Label[ids])
            else:
                t5.append(P[ids])  # 置信度
                cal5.append(Label[ids])
        # print(t1)
        # print(cal1)
        c5 = sorted(enumerate(t5), key=lambda x: x[1], reverse=True)
        for res in c5:
            New_d.append(cal5[res[0]])
        c4 = sorted(enumerate(t4), key=lambda x: x[1], reverse=True)
        for res in c4:
            New_d.append(cal4[res[0]])
        c3 = sorted(enumerate(t3), key=lambda x: x[1], reverse=True)
        for res in c3:
            New_d.append(cal3[res[0]])
        c2 = sorted(enumerate(t2), key=lambda x: x[1], reverse=True)
        for res in c2:
            New_d.append(cal2[res[0]])
        c1 = sorted(enumerate(t1), key=lambda x: x[1], reverse=True)
        for res in c1:
            New_d.append(cal1[res[0]])

        # print(New_d)
        sum_D = 0.0
        for index, val in enumerate(New_d[:threads]):
            res_val = val + 1
            sum_D = sum_D + float(math.pow(2, res_val) - 1) / math.log((index + 1) + 1, 2)

        print('NDCG: ', float(sum_D / sum_I))



