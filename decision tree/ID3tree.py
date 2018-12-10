# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:38:02 2018

@author: dzf
@description: ID3决策树简单实现
"""
from collections import Counter
import numpy as np

class Node:
    def __init__(self, parent):
        self._feature = None
        self._category = None
        self._parent = parent
        self._children = dict()
    
    def addChild(self, name, node):
        self._children[name] = node
    
    def setCategory(self, category):
        self._category = category
    
    def setFeature(self, feature):
        self._feature = feature

def sameCategory(D):
    #D = {'samples': [[],[],...], 'labels' = [,..,...], 'mapping': {name:index}}
    d = set(D['labels'])
    if len(d) == 1:
        return True
    else:
        return False


def treeGenerate(D, A, chooseFeature):
    root = Node(None)
    if sameCategory(D) == 1:
        root.setCategory(D['labels'][0])
        return root
    
    if A == [] or D['samples'].count(D['samples'][0]) == len(D['samples']):
        root.setCategory(Counter(D['labels']).most_common(1)[0][0])
        return root
    
    feature = chooseFeature(D, A)
    
    del A[A.index(feature)]
    for i in set(np.array(D['samples']).T[D['mapping'][feature]].tolist()):
        node = Node(root)
        root.addChild(node._feature, node)
        node.setFeature(feature)
        Dv = [D['samples'][j] for j in range(len(D['samples'])) if D['samples'][j][D['mapping'][feature]] == i]
        label = [D['labels'][j] for j in range(len(D['samples'])) if D['samples'][j][D['mapping'][feature]] == i]
        
        if len(Dv) == 0:
            node.setCategory(Counter(label).most_common(1)[0][0])
            return root
        else:
            node.addChild(feature, treeGenerate(D, A, chooseFeature))
            return root
        
def entropy(samples, labels):
    ent = 0
    for i in set(labels):
        p = len([samples[j] for j in range(len(samples)) if labels[j] == i]) / len(samples)
        ent -= p * np.log2(p)
    return ent
        

def chooseFeature(D, A):
    gain = {}
    for i in A:
        e = 0
        for j in set(np.array(D['samples']).T[D['mapping'][i]].tolist()):
            Dv = [D['samples'][k] for k in range(len(D['samples'])) if D['samples'][k][D['mapping'][i]] == j]
            label = [D['labels'][k] for k in range(len(D['samples'])) if D['samples'][k][D['mapping'][i]] == j]
            e += (len(Dv)/len(D)) * entropy(Dv, label)
            
        gain[i] = entropy(D['samples'], D['labels']) -  e
        feature = max(gain, key = gain.get)
        return feature
    


D = {'samples': [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]], 'labels': [1,1,0,0,0], 'mapping': {'a': 0, 'b': 1}}
A = ['a', 'b']


tree = treeGenerate(D, A, chooseFeature)













