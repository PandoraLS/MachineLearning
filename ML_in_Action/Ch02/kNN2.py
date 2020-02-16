# -*- coding: utf-8 -*-
# @Time: 2019/7/14 19:00
# @Author: Li Sen

import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0., 0.1]])
    labels = ['A', 'A', 'B', 'B']
    # labels = ['A', 'B', 'C', 'D']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    给定单一数据inX, 根据dataSet和labels以及k来判断inX最可能是哪个类别（包含在labels中）
    :param inX:  给定的数据（待测数据），一条数据
    :param dataSet: 训练用的数据，若干数据
    :param labels: 训练用的数据对应的labels，若干数据
    :param k: kNN中的k，确定最邻近的k个
    :return: 返回inX最可能属于哪一个类别
    """
    dataSetSize = dataSet.shape[0] # 训练数据集中样本的数量
    # aaa = tile(inX, (dataSetSize,1)) 共内外两个维度，将最里层的维度重复dataSetSize次，将最外层的维度重复1次
    # 将一条测试数据inX重复dataSetSize次，与dataSet相减，其实就是算inX与dataSet中每一条数据的差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 # 将整个矩阵的差值求平方(element-wise)
    sqDistances = np.sum(sqDiffMat, axis=1)  # 在numpy中，轴axis=0表示最外层的维度，将差值的平方加起来
    distances = sqDistances ** 0.5 # 求平方根
    sortedDistIndicies = np.argsort(distances)  # 对distances从小到大排序并返回对应数据的索引
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]  # 第i个数据（总共为k个）对应的标签(distances从小到大对应的标签)
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        # 统计前k个数据中，各个标签的数量
        # .get返回指定键的值，如果值不在字典中返回默认值default,并将等号右边的数值作为对该标签数量的统计
    # print(classCount) # 统计各个标签的数量
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda classCount: classCount[1], reverse=True)
    # reverse = True降序，key指定取待排序元素的哪一项进行排序，# 按照字典的键值对来进行排序，其中以[值]部分作为排序的标准，降序
    # <class 'list'>: [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]  # 返回出现次数最多的分类的名称(就是其对应的Label)


def file2matrix(filename):
    """
    将文本记录转换为Numpy的解析程序
    :param filename: 文件名
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def ClassTest(filename):
    hoRatio = 0.05  # 测试数据所占的比例
    DataMat, Labels = file2matrix(filename)
    hoRatio = 0.1  # 测试数据所占的比例
    DataMat, Labels = file2matrix(filename)  # DataMat的格式为(1000,3), Labels格式为(1000,1)
    normMat, ranges, minVals = autoNorm(DataMat)
    # normMat:(1000,3) 每一列的数值归一化, ranges:(1000,3)，每一列最大值-最小值, minVals:(1000,3)，每一列的最小值
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):  # 前100个是测试集，后面900个是训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], Labels[numTestVecs:m], k=4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, Labels[i]))
        if classifierResult != Labels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input("玩视频游戏所耗时间百分比?"))
    ffMiles = float(input("每年获取的飞行常客里程数?"))
    iceCream = float(input("每周消费的冰淇淋公升数量?"))
    InArr = np.array([ffMiles, percentTats, iceCream])  # 这里要和datingTestSet2.txt对应数据格式匹配
    DataMat, Labels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(DataMat)
    classifierResult = classify0((InArr - minVals) / ranges, normMat, Labels, 3)
    print("you will like this person:", resultList[classifierResult - 1])


def autoNorm(dataSet):
    minVals = np.min(dataSet, axis=0)
    maxVals = np.max(dataSet, axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # np.tile(xxx,(m,1))这种格式适合将一行的数据复制m行，形成一个新的矩阵
    normDataSet = (dataSet - np.tile(minVals, (m, 1))) / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def plot(DataMat, DataLabels):
    fig1 = plt.figure()
    fig1Ax = fig1.add_subplot(111)
    # ax.scatter(DataMat[:, 0], DataMat[:, 1], 15.0 * np.array(DataLabels), 15 * np.array(DataLabels))
    fig1Ax.scatter(DataMat[:, 1], DataMat[:, 2], s=15.0 * np.array(DataLabels), c=np.array(DataLabels))
    # s决定绘制散点的大小，c决定绘制散点的
    plt.xlabel('玩视频游戏所耗时间百分比')  # 设置X轴标签
    plt.ylabel('每周消费的冰淇淋公升数量')  # 设置Y轴标签

    fig2 = plt.figure()
    fig2Ax = fig2.add_subplot(111)
    fig2Ax.scatter(DataMat[:, 0], DataMat[:, 1], s=15.0 * np.array(DataLabels), c=np.array(DataLabels))
    plt.xlabel('每年获取的飞行常客里程数')  # 设置X轴标签
    plt.ylabel('玩视频游戏所耗时间百分比')  # 设置Y轴标签

    plt.show()


###################################################################################
# 手写识别
def img2vector(filename):
    """
    将txt文件中的字符串转换成数字并且存到一个一维数组(1*32)和(32*32的数组中)
    :param filename: 读取的文件名称
    :return: 一维数组和二维数组
    """
    returnVect = np.zeros((1, 1024))
    imgVect2D = np.zeros((32, 32))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline().strip('\n')
        # lineStr = fr.readline() # 读取txt文件中的一行（包括'\n'符号）
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])

    # 顺便将数字写成矩阵的形式
    for i in range(32):
        imgVect2D[i,:] = returnVect[:, i * 32:i * 32 + 32]
    return returnVect, imgVect2D


def handwritingClass(TrainDir, TestDir):
    hwLabels = []
    trainFileList = os.listdir('trainingDigits')
    m = len(trainFileList)
    trainMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        fileRoot = TrainDir + '/' + fileNameStr
        trainMat[i, :], _ = img2vector(fileRoot)

    testLabels = []
    testFileList = os.listdir(TestDir)
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classTestNum = int(fileStr.split('_')[0])
        testLabels.append(classTestNum)
        testRoot = TestDir + '/' + fileNameStr
        testVector, _ = img2vector(testRoot)
        classifierResult = classify0(testVector, trainMat, hwLabels, 4)
        print("predicted:%d, ground truth: %d" % (classifierResult, classTestNum))
        if(classifierResult !=classTestNum):
            errorCount += 1.
    print("预测错误的数量：%d/%d"%(errorCount,mTest))
    print("预测错误的比率：%.4f"%(errorCount/float(mTest)))



if __name__ == "__main__":
    print("kNN")
    ########################################################################
    ####dating部分######
    group, labels = createDataSet()
    a = classify0([1, 0], group, labels, 3)
    # filename = 'datingTestSet2.txt'
    # DataMat, Labels = file2matrix(filename)
    # # plot(DataMat, Labels)
    # normMat, ranges, minVals = autoNorm(DataMat)
    # # print(normMat)
    # # ClassTest(filename)
    # classifyPerson()

    ##########################################################################
    #############手写数字识别部分#################
    imgFile = 'test_0_13.txt'
    imgVector, _ = img2vector(imgFile)
    # # print(imgVector)
    # # print("------------------")
    # # print(_)
    # trainDir = "trainingDigits"
    # testDir = 'testDigits'
    # handwritingClass(trainDir, testDir)
