# -*- coding: utf-8 -*-
# @Author  : lalala


'''
Created on 2017-10-07
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label
'''

from numpy import *
import operator
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX: 测试数据
    :param dataSet:数据集
    :param labels:数据标签
    :param k:k临近
    :return:发生频率最高的数据对应的标签
    """

    dataSetSize = dataSet.shape[0]  # 计算行数
    # 把inX重复 dataSetSize 行，1列，最终形成的矩阵再减去dataSet
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 得到的结果为两点只差，未必是2维的，可以是多维的
    sqDiffMat = diffMat ** 2  # 把矩阵中的每一个元素平方
    sqDistances = sqDiffMat.sum(axis=1)
    # 现在对于数据的处理更多的还是numpy。没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 把距离从小到大排，返回的是索引
    classCount = {}  # 创建一个空字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # i 是从小到大排列的，索引对应着的也是从小到大排列的，利用for把数据从小到大按顺序和labels里面的标签一一对应
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # print(classCount) #{'B': 2, 'A': 2}
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 按照字典的键值对来进行排序，其中以[值]部分作为排序的标准，降序
    return sortedClassCount[0][0] # 返回出现次数最多的分类的名称


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()  # 获取该文件总共有多少行内容
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()  # 用于移除字符串头尾指定的字符，默认为空格  获取该行内容
        listFromLine = line.split('\t')  # 把该行变成分开成列表
        returnMat[index, :] = listFromLine[0:3]  # 获取前三列的数据
        classLabelVector.append(int(listFromLine[-1]))  # 获取最后一列的数据
        index += 1  # 改变行数
    return returnMat, classLabelVector


# 画图
def paint():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    plt.figure(1)
    plt.subplot(111)  # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    plt.xlabel('玩视频游戏所耗时间百分比')  # 设置X轴标签
    plt.ylabel('每年获取的飞行常客里程数')  # 设置Y轴标签
    plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], s=15.0 *
                np.array(datingLabels), c=15.0 * np.array(datingLabels))
    '''
    切换到figure(2)，想对figure(1)和figure(2)进行控制，可以来回切换，详见paint5.py
    '''
    plt.figure(2)
    plt.subplot(111)  # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    # plt.title('Scatter Plot')  # 设置标题
    plt.xlabel('玩视频游戏所耗时间百分比')  # 设置X轴标签
    plt.ylabel('每周消费的冰淇淋公升数量')  # 设置Y轴标签
    # plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 *
    #             np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.scatter(datingDataMat[:, 0], datingDataMat[:, 2], s=15.0 *
                np.array(datingLabels), c=15.0 * np.array(datingLabels))

    plt.figure(3)
    plt.subplot(111)  # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    # plt.title('Scatter Plot')  # 设置标题
    plt.xlabel('每年获取的飞行常客里程数')  # 设置X轴标签
    plt.ylabel('每周消费的冰淇淋公升数量')  # 设置Y轴标签
    plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], s=15.0 *
                np.array(datingLabels), c=15.0 * np.array(datingLabels))
    plt.show()  # 可以用命令kNN.plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # numpy中的min(),其中axis = 0表示将矩阵按列进行比较
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # 读取dataSet的行和列，以(行，列)的形式返回
    m = dataSet.shape[0]  # 获取dataSet的行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# normMat, ranges, minVals = autoNorm(datingDataMat)


def datingClassTest():
    horatio = 0.10  # 测试数据比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 总数据行数
    # print(m)
    numTestVecs = int(m * horatio)  # 用于测试的数据行数
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        # normMat[i, :]为要测试的每一行，normMat[numTestVecs:m, :]为后面90%的数据
        # datingLabels[numTestVecs:m]后面90%数据对应的标签,k这里取4
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['no at all', 'in small does', 'in large does']
    percentTats = float(input("玩视频游戏所耗时间百分比?"))
    ffMiles = float(input("每年获取的飞行常客里程数?"))
    iceCream = float(input("每周消费的冰淇淋公升数量?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("you will like this person:", resultList[classfierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()  # 读取该行，包括后面的'\n'
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    # 实际上这个文件的读取方式是按照字典序读取的0_0.txt，0_1.txt，0_10.txt，0_100.txt，0_101.txt，0_102.txt
    # 读取结果放到列表中
    m = len(trainingFileList)  # 能确定有多少个txt文件
    trainingMat = zeros((m, 1024))  # 每个文件放一行，共m行
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 单个txt文件，第i个txt文件
        fileStr = fileNameStr.split('.')[0]  # 选取文件名，比如3_90.txt，取3_90
        classNumStr = int(fileStr.split('_')[0])  # 选取_的前半部分，比如3_90选取3
        hwLabels.append(classNumStr)  # 将选取的数字x放到list中
        # print('trainingDigits/%s'%fileNameStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    print(trainingMat)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) # vectorUnderTest.shape=(1,1024)，对应一幅图像
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 4)
        #vectorUnderTest待测试数据，trainingMat为训练数据，hwLabels为训练数据对应的标签，k=4
        print("the classifier came back with:%d,the real answer is: %d" %
              (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is:%d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


if __name__ == "__main__":
    print("this is kNN")
    # paint()
    handwritingClassTest()
