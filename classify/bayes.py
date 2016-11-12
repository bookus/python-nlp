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
                with open(f, 'r', encoding='utf-8') as page:
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


def predict(model, txt):
    words = []
    content = txt.replace(" ", '').replace("\n", '').replace('\u3000', '')
    words += [i for i in clipper(content)]
    fh = sklearn.feature_extraction.FeatureHasher(n_features=15000,
                                                  non_negative=True,
                                                  input_type='string')
    X = fh.fit_transform([words])
    return model.predict(X)


if __name__ == '__main__':
    # corpus_path = "D:\语料\分类\文本分类语料库\搜狗文本分类语料库微型版"
    save_path = "../model/classify/classify.bin"
    # train(corpus_path, save_path)
    model = load(save_path)
    res = predict(model,
                  "10月末，本外币贷款余额110.18万亿元，同比增长12.3%。月末人民币贷款余额104.77万亿元，同比增长13.1%，增速比上月末高0.1个百分点，比去年同期低2.3个百分点。当月人民币贷款增加6513亿元，同比多增1377亿元（注：前值12200亿）")
    print(res)
