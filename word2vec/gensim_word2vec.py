#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/11 21:01
# @Author  : XIAOCHUANG
# @Site    : www.fengxiaochuang.com/ 
# @File    : gensim_word2vec.py
# @Software: PyCharm
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os


class word2vec(object):
    def __init__(self, model_path=None):
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path):
        if (os.path.isfile(model_path) == False):
            raise Exception(model_path + "模型文件不存在")
        else:
            self.model = Word2Vec.load(model_path)

    def train_model(self, corpus):
        if (os.path.isfile(corpus) == False):
            raise Exception(corpus + "文件不存在")
        else:
            sentences = LineSentence(corpus)
            self.model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
            return self.model

    def save(self, save_path):
        self.model.save(save_path)

    def set_model(self, model):
        self.model = model

    def similar_by_word(self, word):
        return self.model.similar_by_word(word)

    def similarity(self, word1, word2):
        return self.model.similar_by_word(word1, word2)

    def most_similar(self, word1, word2):
        return self.model.most_similar(positive=[word1, word2])


if __name__ == '__main__':
    model = word2vec()

    # train_file = "../corpus/word2vec/text8"
    # save_path = "../model/word2vec/text8.bin"

    # model.train_model(train_file)
    # model.save(save_path)

    model_path = "../model/word2vec/text8.bin"

    # 其实可以使用原生API去调用的
    real_model = Word2Vec.load(model_path)
    real_model.similar_by_word('king')
    real_model.most_similar(positive=['king', 'queen'])

