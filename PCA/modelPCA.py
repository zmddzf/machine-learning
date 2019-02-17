# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:25:00 2019

@author: Administrator
"""

import numpy as np

class PCA:
    def __init__(self, X, n_dim):
        """
        构造器
        :param X: 训练数据，是一个每一列为一个样本的行向量，若数据集样本容量认为p，特征数为N，则X为N*p
        :param n_dim: 所需降至的维度
        """
        self.X = X
        self.n_dim = n_dim
        self.mean_array = np.array([[np.mean(i)] for i in self.X])
        self.scaled_X = self.X - self.mean_array


    def __compute_covariance(self):
        """
        计算协方差矩阵，特征向量及特征值
        """
        self.cov = np.cov(self.scaled_X)
        self.eig_val, self.eig_vec = np.linalg.eig(self.cov)
    
    def __sort_eig(self):
        """
        对(特征根，特征值)数对进行排序
        """
        self.eig_pairs = [(np.abs(self.eig_val[i]), self.eig_vec[:,i]) for i in range(len(self.eig_val))]
        self.eig_pairs.sort(reverse=True)
    
    def __choose_n_dim(self):
        """
        选择所需要维度数目的分量
        """
        new_eig_pairs = self.eig_pairs[:self.n_dim]
        new_eig_pairs = np.array(new_eig_pairs)
        self.components = np.array([pair[1] for pair in new_eig_pairs])
    
    def fit(self):
        """
        训练pca
        """
        self.__compute_covariance()
        self.__sort_eig()
        self.__choose_n_dim()
    
    def transform(self, X):
        """
        将转化数据
        """
        X_new = self.components.dot(X - self.mean_array)
        return X_new


from sklearn import datasets
import matplotlib.pyplot as plt
data = datasets.load_iris()
X = data.data
y = data.target
pca = PCA(X.T, 2)
pca.fit()
X_new = pca.transform(X.T)
plt.scatter(X_new[0], X_new[1], c = y)
plt.show()

        
        
        