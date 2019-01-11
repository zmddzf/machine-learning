# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:40:31 2019

@author: zmddzf
"""
import pandas as pd

def readCSV(path):
    """
    用于读取数据
    :param path: 数据文件的路径
    :returns:
        names: 商品名称列表
        labels: 标签列表
    """
    data = pd.read_csv(path)
    names = data["商品名称"].tolist()
    names = [name.split(' ') for name in names]
    labels = data['商品编码'].tolist()
    return names, labels

def computePriori(labels):
    """
    计算先验概率P(c)
    :param labels: 标签列表
    :return:
        priorDict: 字典形式的每一类文档出现的频率
    """
    labelSet = set(labels)
    priorDict = dict()
    for label in labelSet:
        priorDict[label] = labels.count(label) / len(labels)
    return priorDict

def computeConditional(names, labels):
    """
    计算条件概率，这里需要注意，P(w|c)，此处分母应该是c类文档中的词的总数，而不是c类文档数
    :param names: 商品名称列表
    :param labels: 商品标签列表
    :return:
        conditionalDict: 条件概率字典
    """
    conditionalDict = dict()
    cwDict = dict((lab, 0) for lab in set(labels)) #初始化各类文档词的总数字典
    
    #统计每个词再每一类中出现的频数
    for name, label in zip(names, labels):
        for word in name:
            if word not in conditionalDict.keys():
                conditionalDict[word] = dict([(i, j) for i, j in zip(set(labels), [1 for i in range(len(set(labels)))])])
            else:
                conditionalDict[word][label] += 1
    
    #统计每一类中的总词数
    for lab in cwDict:
        for i in conditionalDict.values():
            cwDict[lab] += i[lab]
    
    #计算条件概率
    for item in conditionalDict:
        for priori in set(labels):
            conditionalDict[item][priori] = conditionalDict[item][priori] / cwDict[priori]
    return conditionalDict

def predict(name, priorDict, conditionalDict):
    """
    对商品名称进行分类
    :param name: 一个商品名称的分词列表
    :param priorDict: 先验概率字典
    :param conditionalDict: 条件概率字典
    :return:
        label: 分类结果
    """
    probDict = dict()
    for label in priorDict:
        probDict[label] = priorDict[label] #初始化各类概率字典
        #计算后验概率
        for word in name:
            if word in conditionalDict:
                probDict[label] *= conditionalDict[word][label] #利用贝叶斯公式计算后验概率
            else:
                pass #如果词汇超出了词典范围，则不做处理
    label = max(probDict,key=probDict.get) #求最大概率值的标签
    return label


if __name__ == "__main__":
    names, labels = readCSV('data.csv')
    priorDict = computePriori(labels)
    conditionalDict = computeConditional(names, labels)
    
    #对训练集进行分类，观察分类正确率
    count = 0
    for name, label in zip(names, labels):
        predLabel = predict(name, priorDict, conditionalDict)
        if predLabel == label:
            count += 1
    
    print("Accuracy:", count / len(labels))
        
    
    
    
    
    
    
    
    
    
    
    
    
    