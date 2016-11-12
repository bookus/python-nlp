#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/11/11 21:54
# @Author  : XIAOCHUANG
# @Site    : www.fengxiaochuang.com/ 
# @File    : k-means.py
# @Software: PyCharm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans

vectorizer = CountVectorizer()
corpus = [
    '奥运 拳击 入场券 基本 分罄 邹市明 夺冠 对手 浮出 水面',
    '股民 要 清楚 自己 的 目的',
    '印花税 之 股民 四季',
    '杭州 股民 放 鞭炮 庆祝 印花税 下调',
    '残疾 女 青年 入围 奥运 游泳 比赛 创 奥运 历史 两 项 第一',
    '介绍 一 个 ASP.net MVC 系列 教程',
    '在 asp.net 中 实现 观察者 模式 ，或 有 更 好 的 方法 续',
    '输 大钱 的 股民 给 我们 启迪',
    'Asp.Net 页面 执行 流程 分析',
    '运动员 行李 将 “后 上 先 下” 奥运 相关 人员 行李 实名制',
    'asp.net 控件 开发 显示 控件 内容',
    '奥运 票务 网上 成功 订票 后 应 及时 到 银行 代售 网点 付款',
    '某 心理 健康 站 开张 后 首 个 咨询 者 是 位 新 股民',
    'ASP.NET 自定义 控件 复杂 属性 声明 持久性 浅析',
]
# corpus = [
#    'This is the first document.',
#    'This is the second second document.',
#    'And the third one.',
#    'Is this the first document?',
# ]
X = vectorizer.fit_transform(corpus)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)

km = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=1,
            verbose=False)
km.fit(tfidf)
print(km.labels_)
