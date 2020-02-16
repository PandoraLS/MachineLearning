# -*- coding: utf-8 -*-
# @Time    : 2019/8/22 19:57
# @Author  : Li Sen

from math import log


def createDataSet():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的熵
    :param dataSet: 二维矩阵数据(M,N),前N-1列为特征，最后一列为Label 
    :return: 返回该dataSet的熵
    """
    numEntries = len(dataSet)
    labelCounts = {}  # 统计Label及其数量，并保存在字典中
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 记录每行元素的最后一项，Label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt




if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(calcShannonEnt(dataSet))
