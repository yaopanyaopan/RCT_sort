# -*- coding: utf-8 -*
__author__ = '$'

import sys

import numpy as np
import math
import input_helpers
# from My_model import input_helpers
import os
import re
import time
import datetime
import sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

inpH = input_helpers.InputHelper

chinese_copurs = True
if chinese_copurs:
    # train, label_train ,test , label_test = input_helpers.getDataSets_svm()

    train,label_train , test , label_test = input_helpers.getsinomedDataSets_svm()
else:
    train, label_train ,test , label_test= input_helpers.getDataSets_english_svm()

print("训练模型")

tuned_parameters = [{ 'loss':['hinge'],
                     'C': [0.405,0.41,0.415],
                      'verbose':[True]
                      }
                    ]

# clf = sklearn.svm.LinearSVC(penalty='l2',loss='hinge',C=2.145,max_iter=1000,class_weight='balanced') # chinese_copurs,5
clf = sklearn.svm.LinearSVC(penalty='l2',loss='hinge',class_weight='balanced',C=0.41) # sinomed


# dual=True,  multi_class='ovr',fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0,
# scores = cross_val_score(clf,train,label_train,cv=5)
# print(scores)

# clf = GridSearchCV(clf,tuned_parameters,cv=5)
clf.fit(train,label_train)

# print(clf.best_params_)

print('训练完毕')
score_dev = clf.score(test,label_test)
print(score_dev)

label_pred = clf.predict(test)
cl = sklearn.metrics.classification_report(label_test,label_pred)
print(cl)

cm = sklearn.metrics.confusion_matrix(label_test,label_pred)
print('混淆矩阵：')
print(cm)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_test,label_pred, pos_label=1)
print(fpr)
print(tpr)
print(thresholds)
auc = sklearn.metrics.auc(fpr,tpr)
print(auc)


