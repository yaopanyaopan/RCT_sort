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


os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
     checkpoint_file = tf.train.get_checkpoint_state('runs/1560406529.8086884')
     max_value = 0.0
     top_value= []
     top_test = []
     top_attention = []
     for path in checkpoint_file.all_model_checkpoint_paths:

            print('模型路径 ：',path)

            saver = tf.train.import_meta_graph('{}.meta'.format(path))

            saver.restore(sess,path)
            graph = tf.get_default_graph()

            input_x = graph.get_tensor_by_name('input_x1:0')
            dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
            predictions = graph.get_tensor_by_name('full_connect/score1/score1:0')

            attention = tf.get_collection('attention1')

            batches = inpH.batch_iter1(list(zip(x_test, y_test)), 256, False)

            all_predictions = []
            rank = []
            sco = []
            for db in batches:
                # print(len(sco))
                x_test_batch, y_dev_b = zip(*db)

                for i in y_dev_b:
                    # print(i)
                    rank.append(i)

                batch_predictions ,batch_attention= sess.run(
                    [predictions,attention],{input_x:x_test_batch,dropout_keep_prob:1.0}
                )
                for index,i in enumerate(batch_predictions):

                    sco.append(i[0])

                    batch_attention = np.array(batch_attention)
                    top_value.append(i[0])
                    top_test.append(x_test_batch[index])
                    top_attention.append(batch_attention[0][index])

            print(len(top_value))
            c = sorted(enumerate(top_value), key=lambda x: x[1], reverse=True)
            # print(c)
            res_test = []
            res_attention = []
            for i in c:
                res_test.append(top_test[i[0]])
                res_attention.append(top_attention[i[0]])

            # last_test = res_test[2565]
            # last_attention = res_attention[2565]
            last_test = res_test[-30]
            last_attention = res_attention[-30]
            # print(last_test)
            print(last_attention)

            ndcg, P = input_helpers.NDCG_1(rank, sco)
            print('NDCG :',ndcg)
            print('P :',P)

            stop_list = []
            with open('../data/stop_words.txt', 'r', encoding='utf8') as fr:
                for line in fr:
                    if line.strip() != ' ':
                        stop_list.append(line.strip())

            with open("../data/word_freq.csv", "r", encoding='utf8') as f:
                vocab = []
                vocab.append("#PADDING#")
                vocab.append("#UNK#")
                i = 2
                word_freq = csv.reader(f)
                for word, freq in word_freq:
                    if int(freq) >= 3 and word not in stop_list:  # 词表去停用词 , 停用词去除 “ 。！？ ”
                        vocab.append(word)

            string_a = []
            for hang in range(len(last_test)):
                sent = []
                for lie in range(len(last_test[hang])):
                    if last_test[hang][lie]>1:
                        sent.append(vocab[int(last_test[hang][lie])])
                string_a.append(sent)

            # print(string_a)
            visualize = True
            if visualize == True:
                f = open('visualize.html','w')
                f.write(
                    '<html style="margin:0;padding:0;"><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><body style="margin:0;padding:0;">\n')

                if visualize == True:
                    f.write('<div style="margin:15px;">\n')
                    # f.write('<p style="margin:14px;>')
                    for k in range(len(last_attention)):

                        f.write('%s<span style="margin-left:3px;background-color:rgba(255,0,0,%s)">%s 。</span>' % (k+1,last_attention[k], ''.join(string_a[k])))
                    # f.write('</p>\n')
                    f.write('</div>\n')
                f.write('</body></html>')
                f.close()

