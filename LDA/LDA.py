# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:21:01 2018

@author: dzf
@description：LDA二分类简单实现
"""
import numpy as np
from sklearn.datasets.samples_generator import make_classification
from matplotlib import pyplot as plt

#计算类内散度
def withinMat(Xset1, Xset2):
    """
    计算类内散度矩阵
    param: 
        Xset1: 第一类样本集
        Xset2: 第二类样本集
    return:
        Sw: 类内散度矩阵
    """
    #计算平均点
    mu1 = np.mean(Xset1, axis=0)
    mu2 = np.mean(Xset2, axis=0)
    #计算协方差矩阵
    cov1 = np.dot((Xset1 - mu1).T, (Xset1 - mu1))
    cov2 = np.dot((Xset2 - mu2).T, (Xset2 - mu2))
    #计算散度矩阵
    Sw = cov1 + cov2
    return np.mat(Sw)

#定义LDA算法
def LDA(sample, label):
    """
    LDA
    param:
        sample: 样本向量
        label: 标签向量
    return:
        w: 系数
    """
    #0类样本
    Xset0 = np.array([sample[i] for i in range(len(sample)) if label[i] == 0])
    #1类样本
    Xset1 = np.array([sample[i] for i in range(len(sample)) if label[i] == 1])
    #计算均值
    mu0 = np.mean(Xset0, axis=0)
    mu1 = np.mean(Xset1, axis=0)
    #计算类内散度
    Sw = withinMat(Xset0, Xset1)
    #计算权重
    w = Sw.I.dot((mu0 - mu1))
    return np.array(w)

#计算变化后的样本点
def transform(w, x):
    return np.dot(x, w.T)


#生成数据集 
x, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2, 
                           n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

#计算权重系数
w = LDA(x, y)
#计算变换后的样本
x_ = transform(w, x)
#画出原始数据散点图
plt.scatter(x.T[0], x.T[1], c = y)
plt.show()
#画出变化后的散点图
plt.scatter(x_.T, y, c = y)
plt.show()






