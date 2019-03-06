# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:03:37 2019

@author: zmddzf
"""
from sklearn.preprocessing import scale 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义异常类
class ParamError(Exception): # 继承异常类
    def __init__(self, name, reason):
        self.name = name
        self.reason = reason


class PCA:
    def __init__(self, k=None, ratio=None):
        """
        构造函数
        :param k: 降维之后保留的维数，默认值为None
        :param ratio: 累积贡献率，默认值为None
        """
        self.k = k
        self.ratio = ratio
    
    def __mean(self, X):
        """
        私有函数，计算每一个特征的均值
        :param X: 训练数据集
        :return X_: 各个维度的均值向量, shape(X.shape[0], 1)
        """
        X_ = []
        for x in X:
            X_.append(x.mean())
        X_ = np.array(X_)
        self.mean_ = X_.reshape(X_.shape[0], 1)
        return X_
    
    def __cov(self, X):
        """
        私有函数，计算协方差矩阵
        :param X: 训练数据集
        :return cov: 协方差矩阵
        """
        self.__mean(X)
        X_adjust = X - self.mean_ # 对数据集去均值
        cov = X_adjust.dot(X_adjust.T) / X_adjust.shape[1] # 计算协方差矩阵
        self.cov = cov
        return cov
    
    def __eig(self, cov):
        """
        私有函数，计算特征值与特征向量，并进行排序
        :param cov: 协方差矩阵
        :return eig_pair: 特征值-特征向量排序后的元组列表
        """
        eig_val, eig_vec = np.linalg.eig(cov) # 调用np函数计算特征值与特征向量
        eig_pair = [(val, vec) for val, vec in zip(eig_val, eig_vec)] # 将特征值与特征向量组成数对列表
        eig_pair.sort(reverse = True) # 按照特征值从大到小进行排序
        self.eig_pair = eig_pair
        
        eig_val = []
        eig_vec = []
        for item in eig_pair:
            eig_val.append(item[0])
            eig_vec.append(item[1])
        self.eig_val = eig_val
        self.eig_vec = eig_vec
        return eig_pair
    
    def __ratio(self, eig_val):
        """
        私有函数，计算特征值的累计贡献率
        :param eig_val: 特征值有序列表
        :return cumulative: 累计贡献率列表
        """
        cumulative = []
        eig_val_sum = sum(eig_val)
        s = 0
        for val in eig_val:
            s += val
            cumulative.append(s/eig_val_sum)
        return cumulative
    
    def fit(self, X):
        """
        训练PCA
        """
        if self.ratio == None and self.k == None: # 如果出现参数都为None, 则抛出参数异常
            raise ParamError("ParamError", "PCA params missing!")
        cov = self.__cov(X) # 计算协方差
        eig_pair = self.__eig(cov) # 计算特征值与特征向量
        cumulative = self.__ratio(self.eig_val) # 计算累计贡献率
        self.cumulative = cumulative
        
        if self.k != None and self.k > 0 and type(self.k) == int: # 对k的值进行检查
            self.n_components = np.array(self.eig_vec[:self.k+1])
        elif self.ratio != None and self.ratio > 0 and self.ratio <= 1: # 对ratio的值进行检查
            n = None
            for i in range(len(cumulative)):
                if self.ratio >= cumulative[i-1] and self.ratio <= cumulative[i]:
                    n = i + 1
            self.n_components = np.array(self.eig_vec[:n])
        else: # 都不合要求, 则抛出异常
            raise ParamError("ParamError", "PCA params error!")
            
    def transform(self, X):
        """
        对原始数据进行变换
        """
        X_adjusted = X - self.mean_
        X_scaled = self.n_components.dot(X_adjusted)
        return X_scaled
            
            
        
if __name__ == "__main__":
    data = pd.read_csv('data.txt', sep='\t', index_col=0)
    data = scale(data) #Z-Score去除量纲影响
    X = data.T
    pca = PCA(ratio=0.85)       
    pca.fit(X)

    X_new = pca.transform(X)
    
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    ax.scatter(X_new[0], X_new[1], X_new[2], marker='o', s=60)
    plt.show()
    
    
            
        
        
        
        
        
        