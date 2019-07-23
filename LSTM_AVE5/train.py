# -*- coding: utf-8 -*
__author__ = '$'

import sys
import tensorflow as tf
import numpy as np
import math
import input_helpers
# from My_model import input_helpers
import os
import re
import time
import datetime
import Model
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.flags.DEFINE_integer("embedding_dim",300,"...")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"...")
tf.flags.DEFINE_float("l2_reg_lambda",0.0,"...")
tf.flags.DEFINE_integer("hidden_units",150,"...")
tf.flags.DEFINE_integer("batch_size",64,"...")
tf.flags.DEFINE_integer("num_epochs",10,"...")
tf.flags.DEFINE_float("margin",5,"...")

tf.flags.DEFINE_integer("evaluate_every",1,"...")    # 验证集上评估
tf.flags.DEFINE_integer("checkpoint_every",1,"...")          # 保存节点步数

Flags = tf.flags.FLAGS

print("参数:\n")
for attr,value in Flags.flag_values_dict().items():
    print("{} = {}".format(attr.upper(), value))

inpH = input_helpers.InputHelper

chinese_copurs = True
if chinese_copurs:
    train,dev,label_train,label_dev, max_sentence_size,max_word_size , W_embedding = input_helpers.getDataSets()
    # train_x1,train_x2,label_train ,rank_train = input_helpers.combine_input(train,label_train)  # 合并为pairwise
    # dev_x, label_dev = input_helpers.only1_input(dev,label_dev)   #验证得分排名就行
else:
    train, dev, label_train, label_dev, max_sentence_size, max_word_size, W_embedding = input_helpers.getDataSets_english()
    train_x1, train_x2, label_train, rank_train = input_helpers.combine_input_english(train, label_train)  # 合并为pairwise
    dev_x, label_dev = input_helpers.only1_input_english(dev, label_dev)  # 验证得分排名就行

print("初始化模型")

with tf.Session() as sess:

        Model = Model.LSTM_Attention(
            max_seq_len=max_sentence_size,
            max_sent_len=max_word_size,
            embedding_size=Flags.embedding_dim,
            hidden_units=Flags.hidden_units,
            l2_reg_lambda=Flags.l2_reg_lambda,
            batch_size=Flags.batch_size,
            W_embedding=W_embedding,
            margin = Flags.margin
        )

        global_step = tf.Variable(0,name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        grads_and_vars = optimizer.compute_gradients(Model.loss)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 记录梯度值
        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)
        timestap = str(time.time())
        out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestap))

        loss_summary = tf.summary.scalar("loss", Model.loss)   # 记录损失值
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir,"summaries","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("初始化所有变量")

        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open(os.path.join(checkpoint_dir, "graphpb.txt"), "w") as f:
            f.write(graphpb_txt)

        def train_step(x1_batch, y_batch,epoch):

            feed_dict = {
                Model.input_x1: x1_batch,
                Model.input_y: y_batch,
                Model.dropout_keep_prob: Flags.dropout_keep_prob,
            }

            _,step,loss, scores,score1  = sess.run(
                [tr_op_set, global_step, Model.loss,Model.scores,Model.score1],feed_dict)

            print("epoch: %i ,  step: %i  , loss:  %f" %(epoch,step, loss ))


            summary_op_out = sess.run(train_summary_op, feed_dict = feed_dict)
            train_summary_writer.add_summary(summary_op_out, step)
            return loss


        def dev_step1(x1_batch,y_batch):
            feed_dict = {
                Model.input_x1: x1_batch,
                Model.input_y: y_batch,
                Model.dropout_keep_prob: 1.0,
            }
            step,loss,accuracy =sess.run(
                [global_step, Model.loss,Model.accuracy], feed_dict)
            return  loss,accuracy

        ptr = 0
        max_validation_acc = 0.0
        print("开始训练")
        epoch = []
        train_loss = []
        dev_loss = []
        dev_acc = []

        for ep in range(0,int(Flags.num_epochs)):
            epoch.append(ep)
            if ep % Flags.evaluate_every == 0:
                print("评估：")
                dev_batches = inpH.batch_iter1(list(zip(dev, label_dev)), Flags.batch_size, False)
                res_loss = []
                res_acc = []
                for db in dev_batches:
                    if len(db) < 1:
                        continue
                    x1_dev_b,  y_dev_b= zip(*db)

                    loss, accuracy  = dev_step1(x1_dev_b,y_dev_b)
                    res_loss.append(loss)
                    res_acc.append(accuracy)

                dev_loss.append(sum(res_loss)/len(res_loss))
                dev_acc.append(sum(res_acc)/len(res_acc))

                sum_acc = sum(res_acc)/len(res_acc)
                print('acc:',sum_acc)
            if ep % Flags.checkpoint_every == 0 and ep>1:
                if sum_acc >= max_validation_acc:  # 保存在验证集准确率最大的模型
                    max_validation_acc = sum_acc
                    current_step = tf.train.global_step(sess, global_step)
                    saver.save(sess, checkpoint_dir, global_step=ep)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix,
                                         "graph" + str(time.time()) + ".pb", as_text=False)

            batches = inpH.batch_iter(
                list(zip(train,label_train)),
                Flags.batch_size,
                 True)
            sum_loss = []
            for batch in batches:
                if len(batch) < 1:
                    continue
                x1_batch ,  y_batch  = zip(*batch)

                loss = train_step(x1_batch,y_batch,ep)    # 训练
                sum_loss.append(loss)

            train_loss.append(sum(sum_loss)/len(sum_loss))
            print('\n')
            print('epoch : %i , average-loss : %f'%(ep,sum(sum_loss)/len(sum_loss)))
            print('\n')
        # 画图
        def to_picture(title, x_content, y_content, xlabel, ylabel, xlim, ylim,xticks,yticks, path):
            print("    - [Info] Plotting metrics into picture " + path)

            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = True

            plt.figure(figsize=(10, 5))
            plt.grid(linestyle="--")
            plt.xlim(xlim)
            plt.ylim(ylim)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.title(title, fontsize=14, fontweight='bold')
            plt.plot(x_content, y_content)
            plt.xticks(xticks)
            plt.yticks(yticks)
            plt.xlabel(xlabel, fontsize=13, fontweight='bold')
            plt.ylabel(ylabel, fontsize=13, fontweight='bold')
            plt.savefig(path, format='png')
            plt.clf()

        out_dir = os.path.join(out_dir)
        to_picture(title='dev-loss', x_content=epoch, y_content=dev_loss, xlabel='Epoch', ylabel='loss', xlim=(0,int(Flags.num_epochs)) ,ylim=(0,1),xticks=np.linspace(0,5,6,endpoint=True),yticks=np.linspace(0,1,21,endpoint=True), path=out_dir+'/'+'dev-loss.png')
        to_picture(title='dev-acc', x_content=epoch, y_content=dev_acc, xlabel='Epoch', ylabel='acc', xlim=(0, int(Flags.num_epochs)),ylim=(0,1),xticks=np.linspace(0,5,6,endpoint=True),yticks=np.linspace(0,1,21,endpoint=True),
                   path=out_dir+'/' + 'dev-P_rct.png')
        to_picture(title='loss', x_content=epoch, y_content=train_loss, xlabel='Epoch', ylabel='loss', xlim=(0, int(Flags.num_epochs)),ylim=(0,1),xticks=np.linspace(0,5,6,endpoint=True),yticks=np.linspace(0,1,21,endpoint=True),
                   path=out_dir +'/'+ 'train_loss.png')


