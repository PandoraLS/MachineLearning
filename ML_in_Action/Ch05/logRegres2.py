# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/2/26 12:05
# Logistic Regression Working Module

import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # 删除空白符后进行切分
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 100 * 3
        labelMat.append(int(lineArr[2]))  # 标签列表testSet.txt的最后一列100 * 1
    return dataMat, labelMat  # dataMat: 100行3列, labelMat :1行 100 列


def sigmoid(inZ):
    return 1.0 / (1 + np.exp(-inZ))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 100 行 3列的一个矩阵
    labelMat = np.mat(classLabels).transpose()  # 100 行 1列的矩阵
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 3行1列
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)  # 100行1列
        # dataMatrix_trans = dataMatrix.transpose()
        # d_e = dataMatrix.transpose() * error
        weights = weights + alpha * dataMatrix.transpose() * error
        # 梯度上升
    return weights


def plotBestFit(weights):
    # 画出数据集和Logistic回归最佳拟合直线的函数
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # print(x.shape)
    y = (-weights[0] - weights[1] * x) / weights[2]
    y = y.tolist()
    y_temp = np.array(y[0])
    # print(y_temp.shape)
    ax.plot(x, y_temp) # 划线x,y_temp
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 默认迭代次数numIter = 500
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))  # 这里和python2不兼容
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is: %f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % \
          (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print(weights)
    plotBestFit(weights)
