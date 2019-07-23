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
from tensorflow.python import pywrap_tensorflow


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
inpH = input_helpers.InputHelper
FLAGS = tf.flags.FLAGS

print('\n Parameters:')
for attr, value in sorted(FLAGS._flags().items()):
    print('{}={}'.format(attr.upper(),value))

pkl_x_raw1 = open('../data/test.pickle', 'rb')
x_test = pickle.load(pkl_x_raw1)
pkl_y_test = open('../data/label_test.pickle', 'rb')
y_test = pickle.load(pkl_y_test)

# print(np.shape(x_test))
print('测试集数量： ',len(y_test))

#----------------------评 估----------------------------------------------------------
print('\nEvaluating...\n')

#checkpoint_file = tf.train.latest_checkpoint('runs/1556849414.1516447')


# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file.model_checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# print(len(var_to_shape_map))
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
    # print(reader.get_tensor(key))
with tf.Session() as sess:
     checkpoint_file = tf.train.get_checkpoint_state('runs/1558798897.901901')
     for path in checkpoint_file.all_model_checkpoint_paths:

            print('模型路径 ：',path)

            saver = tf.train.import_meta_graph('{}.meta'.format(path))

            saver.restore(sess,path)
            graph = tf.get_default_graph()

            input_x = graph.get_tensor_by_name('input_x1:0')
            dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
            predictions = graph.get_tensor_by_name('full_connect/score1/score1:0')


            batches = inpH.batch_iter1(list(zip(x_test, y_test)), 256, False)

            all_predictions = []
            rank = []
            sco = []
            for db in batches:
                # print(len(sco))
                x_test_batch, y_dev_b = zip(*db)

                for i in y_dev_b:

                    rank.append(i)

                # print(rank)

                batch_predictions = sess.run(
                    predictions,{input_x:x_test_batch,dropout_keep_prob:1.0}
                )
                for i in batch_predictions:
                    sco.append(i[0])


            ndcg, P = input_helpers.NDCG(rank, sco)

            print('NDCG :',ndcg)
            print('P :',P)



