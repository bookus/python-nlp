#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/12 11:20
# @Author  : XIAOCHUANG
# @Site    : www.fengxiaochuang.com/ 
# @File    : bayes.py
# @Software: PyCharm

import os
import jieba
import sklearn.feature_extraction
import sklearn.naive_bayes as nb
from sklearn.externals import joblib


def clipper(txt):
    """
    结巴分词
    :param txt:
    :return:
    """
    return jieba.cut(txt)


def train(corpus_path, save_path):
    corpus_path = "D:\语料\分类\文本分类语料库\搜狗文本分类语料库精简版"
    # path = "D:\语料\分类\文本分类语料库\搜狗文本分类语料库微型版"
    if not os.path.isdir(corpus_path):
        print("Error Path")
    else:
        file_path = os.path.split(corpus_path)
        print(file_path)
        cateList = [x for x in os.listdir(corpus_path)]
        print(cateList)
        pageList = []
        targetList = []
        catIndex = 1
        for p in cateList:
            cPath = os.path.join(corpus_path, p)
            for f in os.listdir(cPath):
                f = os.path.join(cPath, f)
                with open(f, 'r') as page:
                    tmpword = []
                    for line in page:
                        line = line.replace(" ", '').replace("\n", '').replace('\u3000', '')
                        tmpword += [i for i in clipper(line)]
                    pageList += [tmpword]
                    targetList += [catIndex]
            catIndex += 1

            # that MODEL.
        gnb = nb.MultinomialNB(alpha=0.01)
        # Hashing Trick, transfrom dict -> vector
        fh = sklearn.feature_extraction.FeatureHasher(n_features=15000,
                                                      non_negative=True,
                                                      input_type='string')

        X = fh.fit_transform(pageList)
        gnb.fit(X, targetList)
        # clean context
        # result = gnb.predict(X)
        # rate = (result == targetList).sum() * 100.0 / len(result) * 1.0
        # print("rate %.2f %% " % rate)
        targetList = []
        pageList = []

        joblib.dump(gnb, save_path)


def load(model_path):
    model = joblib.load(model_path)
    return model