# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:31:52 2019

@author: zmddzf
"""

import numpy as np
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
np.random.seed(0)

class NeuralNetwork:
    
    def __init__(self, inputNode, hiddenNode, outputNode, lr):
        """
        构造器，定义网络各层的节点数目与学习率，并初始化权重矩阵
        :param inputNode: 输入层的节点数
        :param hiddenNode: 隐层的节点数
        :param outputNode: 输出层的节点数
        :param lr: 学习率
        """
        self.inputNode = inputNode
        self.hiddenNode = hiddenNode
        self.outputNode = outputNode
        self.lr = lr
        #初始化权重矩阵，矩阵形状与各层节点个数有关
        self.weightsInToHidden = np.random.normal(0.0, self.hiddenNode**-0.5, 
                                       ( self.hiddenNode, self.inputNode))
        
        self.weightsHiddentoOut = np.random.normal(0.0, self.outputNode**-0.5, 
                                       (self.outputNode, self.hiddenNode))
        
        self.b1 = 0.5*np.random.rand(hiddenNode,1)-0.1
        self.b2 = 0.5*np.random.rand(outputNode,1)-0.1
        
        
    def __activeFunction(self, arr):
        """
        激活函数，sigmoid函数
        :param arr: 输入向量
        :return: 返回计算结果
        """
        fun = 1/(1 + np.exp(-arr))
        return fun
    
    
    def __feedForward(self, inputList):
        """
        前向传播算法
        :param inputList: 训练集列表，每个样本是一行
        :returns: 
            inputs: 输入训练集矩阵
            hiddenOutputs: 隐层输出
            outOutput: 输出层的输出
        """
        inputs = np.array(inputList, ndmin=2).T #将输入训练集变成每一列为一个样本
        
        hiddenInputs = np.dot(self.weightsInToHidden, inputs).transpose() + self.b1.transpose() #隐层输入计算
        hiddenOutputs = self.__activeFunction(hiddenInputs).transpose() #隐层输出计算
        outInputs = (np.dot(self.weightsHiddentoOut, hiddenOutputs).transpose()\
                     + self.b2.transpose()).transpose() #计算输出层的输入
        outOutput = outInputs #计算输出层的输出

        return inputs, hiddenOutputs, outOutput
    
    def __backPropagation(self, inputs, hiddenOutputs, outOutput, targetList):
        """
        BP误差反向传播算法
        :param targetList: 训练集的真实结果
        :param hiddenOutputs: 隐层输出结果
        :param outOutput: 输出层输出结果
        :return: 
            outputError: 输出误差大小
        """
        n = len(targetList)
        targets = np.array(targetList, ndmin=2)
        
        
        #计算误差反向传播，由于是回归问题，因此把输出层激活函数设为x
        outputError = targets - outOutput
        hiddenError = np.dot(self.weightsHiddentoOut.transpose(), outputError) * hiddenOutputs*(1-hiddenOutputs)
        
        dw2 = np.dot(outputError, hiddenOutputs.transpose())
        db2 = np.dot(outputError, np.ones((n, 1)))
        
        dw1 = np.dot(hiddenError, inputs.transpose())
        db1 = np.dot(hiddenError, np.ones((n, 1)))
        
        
        #权值梯度下降
        self.weightsHiddentoOut += dw2 * self.lr
        self.weightsInToHidden += dw1 * self.lr
        
        self.b1 += db1*self.lr
        self.b2 += db2*self.lr
        
        return outputError
    
    def train(self, inputList, targetList, maxEpcho, showTime = -1):
        """
        按照最大迭代次数训练网络
        :param inputList: 训练集列表，每个样本是一行
        :param targetList: 训练集的真实结果
        :param maxEpcho: 最大迭代次数
        :param showTime: 间隔几轮显示一次均方误差
        :return:
            historySSE: 每轮训练误差平方和列表
        """    
        historySSE = [] #记录每一伦的误差平方和
        for epcho in tqdm(range(maxEpcho)):
            
            inputs, hiddenOutputs, outOutput = self.__feedForward(inputList) #正向传播
            outputError = self.__backPropagation(inputs, hiddenOutputs, outOutput, targetList) #误差反向传播
            SSE = sum(sum(outputError**2)) #计算误差平方和
            historySSE.append(SSE)
            
            if showTime < 0:
                pass
            elif epcho % showTime == 0:
                print("epcho %d   SSE="%epcho, SSE)
        
        return historySSE
    
    def predict(self, inputList):
        """
        预测输出结果
        :param inputList: 训练集列表
        :return:
            outOutput: 预测结果
        """
        inputs, hiddenOutputs, outOutput = self.__feedForward(inputList)
        return outOutput
        
                
            
if __name__ == '__main__':
    data = load_boston()['data']
    target = load_boston()['target']
    
    #划分数据集
    trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size=0.33, random_state=42)

    #归一化数据集
    scaler = MinMaxScaler( )
    scaler.fit(trainData)
    trainData = scaler.transform(trainData)
    trainTarget = (trainTarget - trainTarget.mean()) / (trainTarget.max() - trainTarget.min())
    
    scaler = MinMaxScaler( )
    scaler.fit(testData)
    testData = scaler.transform(testData)
    testTarget = (testTarget - testTarget.mean()) / (testTarget.max() - testTarget.min())
    
    #开始训练
    nn = NeuralNetwork(13, 20, 1, 0.0032)
    historySSE = nn.train(trainData, trainTarget, 60000, showTime = 10000)
    
    #绘制出SSE变化图
    plt.figure(figsize=(50, 10))
    plt.plot(range(1, 60001), historySSE)
    plt.show()
    
    #进行预测
    predict = nn.predict(testData)
    
    #绘制预测与真实价格折线图
    plt.figure(figsize=(50, 10))
    plt.plot(range(167), predict.T)
    plt.plot(range(167), testTarget)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    