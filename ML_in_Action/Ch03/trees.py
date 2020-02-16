#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : lalala
'''
Created on 2017-10-08
Decision Tree Source Code for Machine Learning in Action Ch. 3
'''
from math import log
import operator
import treePlotter as tP


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# myDat, labels = createDataSet()


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 所有行数
    labelCounts = {}
    for featVec in dataSet:  # 每次一行
        currentLabel = featVec[-1]  # 每行最后一个元素，也就是标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 求信息量
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''在dataSet中，找出第axis项为value的那些行，然后返回一个去除axis项之后的一个retDataSet'''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 获取该值前面所有项
            reducedFeatVec.extend(featVec[axis + 1:])  # 获取该值后面所有项
            retDataSet.append(reducedFeatVec)  # 将该值前后所有项合并在一起
    return retDataSet

# 详细的算法可以具体参考《统计学习方法》


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 去掉最后一个标签项
    baseEntropy = calcShannonEnt(dataSet)  # 数据集D的经验熵H(D)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 得到每一列的特征值
        uniqueVals = set(featList)  # 得到每一列特征值的集合
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 对应于特征A将D划分的各个区间
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 求出条件熵H(D|A)
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if(infoGain > bestInfoGain):  # 求出信息增益最大的那个
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同，则停止划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 无法简单地返回唯一的类标签，需要选择出现次数最多的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 标签是越用越少的
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')  # 因为存储的是对象，必须使用二进制形式写进文件,这是py3和py2在pickle上不兼容的原因
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')  # 用二进制写入，就得用二进制读出
    return pickle.load(fr)


def classifyLenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree


print('This is ', __name__, '.py')
if __name__ == "trees":
    print('trees.py')
    # myDat, labels = createDataSet()
    tP.createPlot(classifyLenses())
else:
    print("nothing happend")
