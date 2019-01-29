# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:22:18 2019

@author: zmddzf
"""
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class LVQ:
    """
    学习向量化算法实现
    attributes:
        train:LVQ
        predict: 预测一个样本所属的簇
    """
    def __init__(self, D, T, lr, maxEpoch):
        """
        初始化LVQ, 构造器
        :param D: 训练集, 格式为[[array, label],...]
        :param T: 原型向量类别标记
        :param lr: 学习率，0-1之间
        :param maxEpoch: 最大迭代次数
        """
        self.D = D
        self.T = T
        self.lr = lr
        self.maxEpoch = maxEpoch
        self.P = []
        #初始化原型向量，随机选取
        for t in T:
            while True:
                p = random.choice(self.D)
                if p[1] != t:
                    pass
                else:
                    self.P.append(p)
                    break

    def __dist(self, p1, p2):
        """
        私有属性，计算距离
        :param p1: 向量1
        :param p2: 向量2
        :return dist: 距离
        """        
        dist = np.linalg.norm(p1 - p2)
        return dist
    
    def train(self):
        """
        训练LVQ
        :return self.P: 训练后的原型向量
        """
        for epoch in tqdm(range(self.maxEpoch)):
            x = random.choice(self.D) #从训练集随机选取样本
            dist = []
            for p in self.P:
                dist.append(self.__dist(p[0], x[0])) #计算距离列表
            
            t = self.P[dist.index(min(dist))][1] #确定对应最小距离原型向量的类别
            if t == x[1]:
                #若类别一致, 则靠拢
                self.P[dist.index(min(dist))][0] = self.P[dist.index(min(dist))][0] + self.lr*(x[0] - self.P[dist.index(min(dist))][0])
            else:
                #若类别不同, 则远离
                self.P[dist.index(min(dist))][0] = self.P[dist.index(min(dist))][0] - self.lr*(x[0] - self.P[dist.index(min(dist))][0])
        return self.P
    
    def predict(self, x):
        """
        预测样本所属的簇
        :param x: 样本向量
        :return label: 样本的分类结果
        """
        dist = []
        for p in self.P:
            dist.append(self.__dist(p[0], x))
        label = self.P[dist.index(min(dist))][1]
        return label
            


#生成实验数据集，数据集是两个正态分布二维点集
mu1 = 2; sigma1 = 1
mu2 = 4; sigma2 = 1
#生成第一个正态分布
samples1 = np.array([np.random.normal(mu1, sigma1, 50), np.random.normal(mu1, sigma1, 50)])
samples1 = samples1.T.tolist()
label1 = [1 for i in range(50)]
#生成第二个正态分布
samples2 = np.array([np.random.normal(mu2, sigma2, 50), np.random.normal(mu2, sigma2, 50)])
samples2 = samples2.T.tolist()
label2 = [0 for i in range(50)]
#合并生成数据集
samples = samples1 + samples2
labels = label1 + label2

#修改数据格式
data = []
for s, l in zip(samples, labels):
    data.append([np.array(s), l])

#开始训练
lvq = LVQ(data, [0, 1], 0.1, 5000)
vector = lvq.train()

#使用lvq分类
prediction = []
for i in data:
    prediction.append(lvq.predict(i[0]))

#计算accuracy
accuracy = 0
for pred, label in zip(prediction, labels):
    if pred == label:
        accuracy += 1
accuracy = accuracy / len(data)
print("accuracy of LVQ:", accuracy)

#画图展示原型向量和散点
plt.figure(figsize=(15,10))
plt.scatter(np.array(samples).T[0], np.array(samples).T[1], c = labels)
plt.scatter(vector[0][0][0], vector[0][0][1], marker = '*', s = 300)
plt.scatter(vector[1][0][0], vector[1][0][1], marker = '*', s = 300)

plt.show()







    
















