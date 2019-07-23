# -*- coding: utf-8 -*
import numpy as np
import re
import itertools
from collections import Counter
import time
import gc
# from tensorflow.contrib import learn
import gensim
from gensim.models.word2vec import Word2Vec
import gzip
import random
import sys, os
import jieba
import csv
import tensorflow as tf
import pickle
from collections import defaultdict
import math
import nltk
import matplotlib.pyplot as plt

class InputHelper(object):

    def batch_iter(data,batch_size, only_part):
        """
        Generates a batch iterator for a dataset.
        """
        shuffle=True
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/ batch_size) + 1
        print('随机采样 :',only_part,' -----1个epoch 的 step数-----:',num_batches_per_epoch)

        shuffled_data = []
        if shuffle==True:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                if only_part==True:
                    for shuffle_indice in shuffle_indices[:int(data_size)]:  # 每个epoch只随机取一部分数据
                        shuffled_data.append(data[shuffle_indice])
                else:
                    for shuffle_indice in shuffle_indices:
                        shuffled_data.append(data[shuffle_indice])

        for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                yield shuffled_data[start_index:end_index]

    def batch_iter1(data,batch_size, only_part):
        """
        Generates a batch iterator for a dataset.
        """
        shuffle=False
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        # print('随机采样 :',only_part,' -----1个epoch 的 step数-----:',num_batches_per_epoch)

        shuffled_data = []
        if shuffle==True:
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)
                if only_part==True:
                    for shuffle_indice in shuffle_indices[:int(data_size)]:  # 每个epoch只随机取一部分数据
                        shuffled_data.append(data[shuffle_indice])
                else:
                    for shuffle_indice in shuffle_indices:
                        shuffled_data.append(data[shuffle_indice])
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                yield shuffled_data[start_index:end_index]


def to_train_dev(title, x_content, y_content, xlabel, ylabel, xlim, ylim,xticks,yticks, path):
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

def NDCG(n_rank,n_sco):


    IDCG = sorted(n_rank,reverse=True)
    # print(IDCG)
    # print(n_rank)
    sum_I = 0.0
    for index,val in enumerate(IDCG):
        sum_I += float(math.pow(2,val)-1)/math.log((index+1)+1,2)
        # sum_I += float( val) / math.log((index + 1) + 1, 2)

    reli = []
    c = sorted(enumerate(n_sco),key=lambda x:x[1],reverse=True)
    # print(n_rank)
    # print(c)
    for i in c:
        reli.append(n_rank[i[0]])
    # print(reli)
    sum_D = 0.0
    for index,val in enumerate(reli):
        sum_D += float(math.pow(2,val)-1)/math.log((index+1)+1,2)
        # sum_D += float(val) / math.log((index + 1) + 1, 2)
    # print(sum_D)
    # print(sum_I)

    count = reli.count(5.0)
    count_rct = 0

    num =0
    # print(count)
    for index,rct in enumerate(reli):

        if num < count:

             num +=1
             if rct==5.0:
                 count_rct +=1
    # print(reli.count(1.0))
    # print(reli.count(2.0))
    # print(reli.count(3.0))
    # print(reli.count(4.0))
    # print(reli.count(5.0))

    print('top数量RCT：',count_rct)
    print('总共RCT篇数：',count)

    P = count_rct / count

    return  float(sum_D/sum_I) ,P

def NDCG_english(n_rank,n_sco):

    IDCG = sorted(n_rank,reverse=True)
    # print(IDCG)
    # print(n_rank)
    sum_I = 0.0
    for index,val in enumerate(IDCG):
        sum_I += float(math.pow(2,val)-1)/math.log((index+1)+1,2)
        # sum_I += float( val) / math.log((index + 1) + 1, 2)

    reli = []
    c = sorted(enumerate(n_sco),key=lambda x:x[1],reverse=True)
    # print(n_rank)
    # print(c)
    for i in c:
        reli.append(n_rank[i[0]])
    # print(reli)
    sum_D = 0.0
    for index,val in enumerate(reli):
        sum_D += float(math.pow(2,val)-1)/math.log((index+1)+1,2)
        # sum_D += float(val) / math.log((index + 1) + 1, 2)
    # print(sum_D)
    # print(sum_I)

    count = reli.count(1.0)
    count_rct = 0

    num =0
    # print(count)
    for index,rct in enumerate(reli):

        if num < count:

             num +=1
             if rct==1.0:
                 count_rct +=1
    # print(reli.count(1.0))

    print('top数量RCT：',count_rct)
    print('总共RCT篇数：',count)

    P = count_rct / count

    return  float(sum_D/sum_I) ,P


def pre_process():

    jieba.enable_parallel(16)
    with open('../data/medical.csv','r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            # print(i[0])
            jieba.add_word(i[0])

    word_freq = defaultdict(int)
    data = []
    with open("../data/total_data.csv","r",encoding="utf8") as fr:

        reader = csv.reader(fr)
        for i in reader:
            data.append(i)
            ab = jieba.lcut(i[1])
            for word in ab:
                word_freq[word] +=1


        with open("../data/word_freq.csv","w",encoding='utf8') as fw:
            res = []
            for k,v in word_freq.items():
                res.append([k,v])

            writer_res = csv.writer(fw)
            writer_res.writerows(res)

    DD = [i for i in range(len(data))]

    random.shuffle(DD)

    with open("../data/total_data_new.csv","w") as fw:
        writer = csv.writer(fw)
        for i in DD:
            writer.writerow(data[i])

def pre_process_english():

    word_freq = defaultdict(int)
    data = []
    with open("../data/english_corpus.csv","r",encoding="utf8") as fr:

        reader = csv.reader(fr)
        count = 0
        for i in reader:
            print(count)
            count += 1
            data.append(i)

            ab = nltk.tokenize.word_tokenize(i[1].strip().lower())
            for word in ab:
                word_freq[word] += 1
            ab = nltk.tokenize.word_tokenize(i[2].strip().lower())
            for word in ab:
                word_freq[word] += 1
            ab = nltk.tokenize.word_tokenize(i[3].strip().lower())
            for word in ab:
                word_freq[word] += 1

        with open("../data/english_word_freq.csv","w",encoding='utf8') as fw:
            res = []
            for k,v in word_freq.items():
                res.append([k,v])

            writer_res = csv.writer(fw)
            writer_res.writerows(res)

    DD = [i for i in range(len(data))]

    random.shuffle(DD)

    with open("../data/english_total_data_new.csv","w") as fw:
        writer = csv.writer(fw)
        for i in DD:
            writer.writerow(data[i])


def getDataSets():

    stop_list = []
    with open('../data/stop_words.txt', 'r', encoding='utf8') as fr:
        for line in fr:
            if line.strip() != ' ':
                stop_list.append(line.strip())

    jieba.enable_parallel(16)
    with open('../data/medical.csv', 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        for i in reader:
            jieba.add_word(i[0])

    data_set = []
    labels = []
    doc = []
    seg = ['。', '？', '！', '?', '!']
    vocab = {}
    vocab["#PADDING#"] = 0
    vocab["#UNK#"] = 1
    i = 2

    with open("../data/word_freq.csv", "r",encoding='utf8') as f:
        word_freq = csv.reader(f)
        for word, freq in word_freq:

            if int(freq) >= 3 and word not in stop_list:   # 词表去停用词 , 停用词去除 “ 。！？ ”
                vocab[word] = i
                i += 1

        # 混合预训练词向量
        word_vec = np.zeros(shape=(len(vocab)-2,300),dtype=np.float32)

        trained_embedding = load_embedding()   #{词：[]}
        for word in vocab.keys():
            if word != '#PADDING#' and word != '#UNK#':
               if word not in trained_embedding.keys() :
                    word_vec[vocab[word]-2] = np.random.uniform(-0.25,0.25,300)
               else:
                     word_vec[vocab[word]-2] = trained_embedding[word]

    max_word_size = 0
    avg_word_size = []
    with open("../data/total_data_new.csv", 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        for i in reader:
            words =[]
            result = jieba.lcut(i[1])
            labels.append(i[2])
            for j in result:
                if j in vocab.keys() and j!='':
                    words.append(j)
            avg_word_size.append(len(words))
            if len(words) > max_word_size:
                max_word_size = len(words)

            data_set.append(words)

        # print(len(data_set))

    print("最大词数", max_word_size)
    print('平均词数', sum(avg_word_size)/len(avg_word_size))

    train = []
    dev = []
    test = []
    label_train = []
    label_dev = []
    label_test = []

    data_new = []

    max_word_size = int(2*sum(avg_word_size)/len(avg_word_size))

    for doc in data_set:
        document = np.zeros(max_word_size)
        for i,wo in enumerate(doc):
            if i < max_word_size:
                document[i] = vocab.get(wo,"#UNK#")
        data_new.append(document)

    for i in range(len(data_new)):
        if i < 0.6* len(data_new):
            train.append(data_new[i])
            label_train.append(float(labels[i]))
        elif 0.6 * len(data_new) <= i <= 0.7 * len(data_new):
            dev.append(data_new[i])
            label_dev.append(float(labels[i]))
        else:
            test.append(data_new[i])
            label_test.append(float(labels[i]))

    res_labels = []
    for i in label_train:
        if i == 1.0:
            res_labels.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif i == 4.0:
            res_labels.append([0.0, 1.0, 0.0, 0.0, 0.0])
        elif i == 3.0:
            res_labels.append([0.0, 0.0, 0.0, 1.0, 0.0])
        elif i == 2.0:
            res_labels.append([0.0, 0.0, 1.0, 0.0, 0.0])
        else:
            res_labels.append([0.0, 0.0, 0.0, 0.0, 1.0])
    label_train=res_labels

    res_labels = []
    for i in label_dev:
        if i == 1.0:
            res_labels.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif i == 4.0:
            res_labels.append([0.0, 1.0, 0.0, 0.0, 0.0])
        elif i == 3.0:
            res_labels.append([0.0, 0.0, 0.0, 1.0, 0.0])
        elif i == 2.0:
            res_labels.append([0.0, 0.0, 1.0, 0.0, 0.0])
        else:
            res_labels.append([0.0, 0.0, 0.0, 0.0, 1.0])
    label_dev = res_labels

    with open("../data/test_W5.pickle","wb") as f3:
        pickle.dump(test,f3)

    with open("../data/label_test_W5.pickle","wb") as f4:
        res_labels = []
        for i in label_test:
                if i == 1.0:
                    res_labels.append([1.0,0.0,0.0,0.0,0.0])
                elif i == 4.0:
                    res_labels.append([0.0,1.0,0.0,0.0,0.0])
                elif i == 3.0:
                    res_labels.append([0.0,0.0,0.0,1.0,0.0])
                elif i == 2.0:
                    res_labels.append([0.0,0.0,1.0,0.0,0.0])
                else:
                    res_labels.append([0.0,0.0,0.0,0.0,1.0])

        print(res_labels[:15])
        pickle.dump(res_labels,f4)

    return train,dev,label_train,label_dev,max_word_size, word_vec

def getsinomedDataSets():

    stop_list = []
    with open('../data/stop_words.txt', 'r', encoding='utf8') as fr:
        for line in fr:
            if line.strip() != ' ':
                stop_list.append(line.strip())

    jieba.enable_parallel(16)
    with open('../data/medical.csv', 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        for i in reader:
            jieba.add_word(i[0])

    data_set = []
    labels = []
    doc = []
    seg = ['。', '？', '！', '?', '!']
    vocab = {}
    vocab["#PADDING#"] = 0
    vocab["#UNK#"] = 1
    i = 2

    with open("../sinomed/sinomedword_freq.csv", "r",encoding='utf8') as f:
        word_freq = csv.reader(f)
        for word, freq in word_freq:

            if int(freq) >= 3 and word not in stop_list:   # 词表去停用词 , 停用词去除 “ 。！？ ”
                vocab[word] = i
                i += 1

        # 混合预训练词向量
        word_vec = np.zeros(shape=(len(vocab)-2,300),dtype=np.float32)

        trained_embedding = load_embedding()   #{词：[]}
        for word in vocab.keys():
            if word != '#PADDING#' and word != '#UNK#':
               if word not in trained_embedding.keys() :
                    word_vec[vocab[word]-2] = np.random.uniform(-0.25,0.25,300)
               else:
                     word_vec[vocab[word]-2] = trained_embedding[word]

    max_word_size = 0
    avg_word_size = []
    with open("../sinomed/sinomed_RCT_new.csv", 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        count = 0
        for i in reader:
            if(count==0):
                count +=1
                continue
            words =[]
            result = jieba.lcut(i[3])
            labels.append(1.0)
            for j in result:
                if j in vocab.keys() and j!='':
                    words.append(j)
            avg_word_size.append(len(words))
            if len(words) > max_word_size:
                max_word_size = len(words)

            data_set.append(words)
    with open("../sinomed/sinomed_noneRCT_new.csv", 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        count = 0
        for i in reader:
            if (count == 0):
                count += 1
                continue
            words = []
            result = jieba.lcut(i[3])
            labels.append(0.0)
            for j in result:
                if j in vocab.keys() and j != '':
                    words.append(j)
            avg_word_size.append(len(words))
            if len(words) > max_word_size:
                max_word_size = len(words)

            data_set.append(words)

        # print(len(data_set))

    print("最大词数", max_word_size)
    print('平均词数', sum(avg_word_size)/len(avg_word_size))

    train = []
    dev = []
    test = []
    label_train = []
    label_dev = []
    label_test = []

    data_new = []

    max_word_size = int(2*sum(avg_word_size)/len(avg_word_size))

    for doc in data_set:
        document = np.zeros(max_word_size)
        for i,wo in enumerate(doc):
            if i < max_word_size:
                document[i] = vocab.get(wo,"#UNK#")
        data_new.append(document)

##$############### 临时 shuffle  测试 #####
    DD=[i for i in range(len(data_new))]
    random.shuffle(DD)
    r_d = []
    r_l = []
    for i in DD:
        r_d.append(data_new[i])
        r_l.append(labels[i])
    data_new = r_d
    labels = r_l
##$############### 临时 shuffle  测试 #####

    for i in range(len(data_new)):
        if i < 0.6* len(data_new):
            train.append(data_new[i])
            label_train.append(float(labels[i]))
        elif 0.6 * len(data_new) <= i <= 0.7 * len(data_new):
            dev.append(data_new[i])
            label_dev.append(float(labels[i]))
        else:
            test.append(data_new[i])
            label_test.append(float(labels[i]))

    res_labels = []
    for i in label_train:
        if i == 1.0:
            res_labels.append([0.0, 1.0])   # RCT
        else:
            res_labels.append([1.0, 0.0])
    label_train = res_labels

    res_labels = []
    for i in label_dev:
        if i == 1.0:
            res_labels.append([0.0, 1.0])
        else:
            res_labels.append([1.0, 0.0])
    label_dev = res_labels

    with open("test_W2_sinomed.pickle","wb") as f3:
        pickle.dump(test,f3)

    with open("label_test_W2_sinomed.pickle","wb") as f4:
        res_labels = []
        for i in label_test:
                if i == 1.0:
                    res_labels.append([0.0, 1.0])
                else:
                    res_labels.append([1.0, 0.0])

        print(res_labels[:15])
        pickle.dump(res_labels,f4)

    return train,dev,label_train,label_dev,max_word_size, word_vec

def getDataSets_english():

    # nltk.download('stopwords')
    # nltk.download('punkt')
    stop_list = nltk.corpus.stopwords.words('english')

    jieba.enable_parallel(16)
    data_set = []
    labels = []
    doc = []

    vocab = {}
    vocab["#PADDING#"] = 0
    vocab["#UNK#"] = 1
    i = 2

    with open("../data/english_word_freq.csv", "r",encoding='utf8') as f:
        word_freq = csv.reader(f)
        for word, freq in word_freq:
            # print(word,freq)
            if int(freq) >= 3 and word not in stop_list:   # 词表去停用词 , 停用词去除 “ 。！？ ”
                vocab[word] = i
                i += 1
        # 混合预训练词向量
        word_vec = np.zeros(shape=(len(vocab)-2,300),dtype=np.float32)

        trained_embedding = load_embedding_english()   #{词：[]}
        for word in vocab.keys():
            if word != '#PADDING#' and word != '#UNK#':
               if word not in trained_embedding.keys() :
                    word_vec[vocab[word]-2] = np.random.uniform(-0.25,0.25,300)
               else:
                    word_vec[vocab[word]-2] = trained_embedding[word]

    max_sentence_size = 0
    max_word_size = 0
    avg_sentence_size = []
    avg_word_size = []
    with open("../data/english_total_data_new.csv", 'r', encoding='utf8') as fr:
        reader = csv.reader(fr)
        for i in reader:
            result = nltk.tokenize.sent_tokenize(i[2].strip().lower())
            # print(result)
            if 'Randomized Controlled Trial' in i[3].strip():
                labels.append(1.0)
            else:
                labels.append(0.0)

            for sen in result:
                sen_res = nltk.tokenize.word_tokenize(sen)
                # print(sen_res)
                sentence = []
                for j in sen_res:
                    if j.strip() in vocab.keys():
                        sentence.append(j.strip())

                avg_word_size.append(len(sentence))
                if len(sentence) > max_word_size:
                    max_word_size = len(sentence)
                doc.append(sentence)

            avg_sentence_size.append(len(result))
            if len(result) > max_sentence_size:
                max_sentence_size = len(result)
            data_set.append(doc)
            doc = []
        # print(len(data_set))

    print("最大句子数", max_sentence_size)
    print("最大词数", max_word_size)
    print('平均句子数', sum(avg_sentence_size)/len(avg_sentence_size))
    print('平均词数', sum(avg_word_size)/len(avg_word_size))

    train = []
    dev = []
    test = []
    label_train = []
    label_dev = []
    label_test = []

    data_new = []

    max_sentence_size = int(2*sum(avg_sentence_size)/len(avg_sentence_size))
    max_word_size = int(2*sum(avg_word_size)/len(avg_word_size))

    for doc in data_set:
        document = np.zeros((max_sentence_size, max_word_size))
        for i,sent in enumerate(doc):
            if i < max_sentence_size:
                for j,word in enumerate(sent):
                    if j < max_word_size:
                        document[i][j] = vocab.get(word,"#UNK#")
        data_new.append(document)

    DD = [i for i in range(len(data_new))]
    random.shuffle(DD)

    for i in range(len(data_new)):
    # for i in DD:
        if i < 0.6* len(data_new):
            train.append(data_new[i])
            label_train.append(float(labels[i]))
        elif 0.6 * len(data_new) <= i <= 0.7 * len(data_new):
            dev.append(data_new[i])
            label_dev.append(float(labels[i]))
        else:
            test.append(data_new[i])
            label_test.append(float(labels[i]))

    with open("../data/english_test.pickle","wb") as f3:
        pickle.dump(test,f3)

    with open("../data/english_label_test.pickle","wb") as f4:
        pickle.dump(label_test,f4)

    return train,dev,label_train,label_dev,max_sentence_size,max_word_size, word_vec


def combine_input(train_set,label):

        labels = []

        for i in label:  # 重新排序
            if i == 1.0:
                labels.append(1.0)
            elif i== 4.0:
                labels.append(2.0)
            elif i == 3.0:
                labels.append(4.0)
            elif i == 2.0:
                labels.append(3.0)
            else:
                labels.append(5.0)
        rank = []
        new_label = []
        new_trainset1 = []
        new_trainset2 = []
        for index_low in range(len(labels)):
            for index_high in range(index_low + 1, len(labels),1):

                if labels[index_low] == labels[index_high]:
                    continue
                    # new_trainset1.append(train_set[index_low])
                    # new_trainset2.append(train_set[index_high])
                    # new_label.append([0.5])

                elif labels[index_low] > labels[index_high]:
                    new_trainset1.append(train_set[index_low])
                    new_trainset2.append(train_set[index_high])
                    new_label.append([1.0])
                else:
                    new_trainset1.append(train_set[index_low])
                    new_trainset2.append( train_set[index_high])
                    new_label.append([-1.0])

                rank.append((labels[index_low],labels[index_high]))

        print("合并后数据",len(new_trainset1))

        return new_trainset1,new_trainset2 , new_label , rank


def combine_input_english(train_set, label):
    labels = label

    rank = []
    new_label = []
    new_trainset1 = []
    new_trainset2 = []
    for index_low in range(len(labels)):
        for index_high in range(index_low + 1, len(labels), 1):

            if labels[index_low] == labels[index_high]:
                continue
                # new_trainset1.append(train_set[index_low])
                # new_trainset2.append(train_set[index_high])
                # new_label.append([0.5])

            elif labels[index_low] > labels[index_high]:
                new_trainset1.append(train_set[index_low])
                new_trainset2.append(train_set[index_high])
                new_label.append([1.0])
            else:
                new_trainset1.append(train_set[index_low])
                new_trainset2.append(train_set[index_high])
                new_label.append([-1.0])

            rank.append((labels[index_low], labels[index_high]))

    print("合并后数据", len(new_trainset1))

    return new_trainset1, new_trainset2, new_label, rank


def only1_input(dev,label):
    labels = []

    for i in label:  # 重新排序
        if i == 1.0:
            labels.append(1.0)
        elif i == 4.0:
            labels.append(2.0)
        elif i == 3.0:
            labels.append(4.0)
        elif i == 2.0:
            labels.append(3.0)
        else:
            labels.append(5.0)

    new_trainset1 = []

    for index_low in range(len(labels)):

                new_trainset1.append(dev[index_low])

    print("验证集数据", len(new_trainset1))

    return new_trainset1,  labels

def only1_input_english(dev,label):
    labels = label
    new_trainset1 = []

    for index_low in range(len(labels)):

        new_trainset1.append(dev[index_low])

    print("验证集数据", len(new_trainset1))

    return new_trainset1,  labels

def train_embedding():

    stop_list =[]
    with open('./data/stop_words.txt','r',encoding='utf8') as fr:
        for line in fr:
            if line.strip()!=' ':
                stop_list.append(line.strip())

    print(len(stop_list))

    jieba.enable_parallel(16)
    with open('./data/medical.csv', 'r') as fr:
        reader = csv.reader(fr)
        for i in reader:
            # print(i[0])
            jieba.add_word(i[0])
    sentences = []
    with open('./data/corpus.txt','r',encoding='utf8') as fr:
        lines = fr.readlines()
        for line in lines:
            sentence = jieba.lcut(line.strip())
            res = []
            for i in sentence:
                if i not in stop_list:
                    res.append(i)
            if len(res)>0:
                sentences.append(res)
    model = gensim.models.Word2Vec(sentences,size=300,window=5,min_count=0,workers=16)
    # print(model.wv.word_vec('口腔'))

    model.wv.save_word2vec_format('wv300.bin')


def load_embedding():

    data = {}
    with open('../wv300.bin','r') as fr:
        for line in fr:
            aa = line.split(' ')
            # print(len(aa))
            val = aa[1:]
            data[aa[0]] = val
    return data

def load_embedding_english():
    data = {}
    with open('../wv300.bin', 'r') as fr:
        for line in fr:
            aa = line.split(' ')
            # print(len(aa))
            val = aa[1:]
            data[aa[0]] = val
    return data

def fun():

    name_list = ['RCT', '随机不明对照试验','信息不足无法判断', '非随机对照试验', '其他']
    res = []
    for i in name_list:
        res.append(i)
    name_list = res

    # name_list = [1,2,3,4,5]
    num_list = [2454, 402, 887, 3836, 4657]
    plt.bar(range(len(num_list)), num_list, color='black', tick_label=name_list)
    plt.show()

if __name__=="__main__":
    fun()

    # train_embedding()
    # load_embedding()
    # pre_process_english()
    # pre_process()
    # getDataSets()
    # h=0


