# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:07:14 2019

@author: zmddzf
"""
import random
import numpy as np

class KMeans:
    """
    k-Means算法类
    attributes:
        train: 训练k-Means聚类器
        predict: 预测一个样本所属的簇
    """
    def __init__(self, k, D):
        """
        构造器，传入簇个数(k), 训练集(D), 初始化中心
        :param k: 簇个数
        :param D: 训练集
        """
        self.k = k
        self.D = D
        self.Mu = random.sample(self.D, k = self.k)
        self.cluster = [i for i in range(self.k)]
    
    def __dist(self, P1, P2):
        """
        私有属性, 计算两点距离
        """
        dist = 0
        for p1, p2 in zip(P1, P2):
            dist += (p1 - p2)**2
        dist = dist**0.5
        return dist
    
    def __computeMu(self, c):
        """
        私有属性, 计算簇的中心点
        """
        c = np.array(c)
        mu = c.mean(axis = 0).tolist()
        return mu
        
    def train(self):
        """
        训练聚类器
        :return histMu: 每一轮迭代的簇中心点
        :return Mu: 训练完成后的簇中心点
        """
        histMu = [] #历史中心点
        while True:            
            Mu_ = self.Mu.copy() #将中心点拷贝为训练前的中心点
            histMu.append(Mu_) #历史中心点数组加入
            C = [[] for item in range(self.k)] #初始化簇点集合
            
            for d in self.D:
                dist = [] #初始化点与各个中心点的距离数组
                for mu in self.Mu:
                    dist.append(self.__dist(d, mu)) #计算距离
                C[dist.index(min(dist))].append(d) #找出点所属的簇
            
            for i in range(self.k):                
                self.Mu[i] = self.__computeMu(C[i]) #更新簇中心点
            
            if Mu_ == self.Mu:
                #如果中心点数组不再更新 则停止迭代
                break
            
        return histMu, self.Mu
    
    def predict(self, p):
        """
        对样本进行簇的预测
        :param p: 样本点数组
        :return cluster: 样本所属的簇
        """
        dist = []
        for mu in self.Mu:
            dist.append(self.__dist(p, mu))
        cluster = self.cluster[dist.index(min(dist))]
        return cluster
            
    
if __name__ == '__main__':
    from sklearn.datasets import load_iris #导入iris数据集函数
    import matplotlib.pyplot as plt #导入可视化库
    
    D = load_iris()['data'].tolist() #加载数据集
    km = KMeans(3, D) #实例化对象
    histMu, Mu = km.train() #训练
    
    #保存聚类结果
    clusters = []
    for d in D:
        clusters.append(km.predict(d))
    
    #可视化聚类结果
    D = np.array(D).T
    plt.scatter(D[2], D[3], c = clusters)
    plt.show()
    
    #可视化聚类中心的变动
    for mu in histMu:
        plt.scatter(np.array(mu).T[2], np.array(mu).T[3], c = [0,1,2])
    plt.show()
        

