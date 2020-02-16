# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/2/15 15:51
# SMO算法公式推导过程：https://zhuanlan.zhihu.com/p/29212107
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels


def clip(alpha, L, H):
    # 修建alpha的值到L和H之间.
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def select_j(i, m):
    # 在m中随机选择除了i之外剩余的数
    l = list(range(m))
    seq = l[: i] + l[i + 1:]
    return np.random.choice(seq)


def get_w(alphas, dataset, labels):
    # 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    # w的计算方式：见[机器学习_周志华]的123页的公式(6.9)
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels) 
    # alphas的shape(100,)，dataset的shape的(100,2)，labels的shape(100,)
    # np_labels_re = labels.reshape(1, -1) # shape(1,100)，即原有的labels变成1行100列的形式
    # np_labels_re_T = np_labels_re.T # 转置为100行1列的形式(100,1)
    # np_labels_re_T_x_1 = np_labels_re_T * np.array([1, 1]) # 100行2列的形式，就是把原来的向量再复制1列 (100,2)

    yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataset  # (100,2)对应元素相乘
    # yx_T = yx.T
    w = np.dot(yx.T, alphas)  # 得到2个w值 (2,100)x(100,1)=(2,1)
    # wList = w.tolist()
    return w.tolist()


def simple_smo(dataset, labels, C, max_iter):
    ''' 
    简化版SMO算法实现，未使用启发式方法对alpha对进行选择.
    :param dataset: 所有特征数据向量[二维向量，这里是坐标]
    :param labels: 所有的数据标签[+1或-1]
    :param C: 软间隔常数, 0 <= alpha_i <= C
    :param max_iter: 外层循环最大迭代次数
    '''
    dataset = np.array(dataset)  # 将数据变成numpy矩阵格式
    m, n = dataset.shape  # m为行，n为列，m行即m个样本
    labels = np.array(labels)  # 将labels变成numpy矩阵格式
    # 初始化参数
    alphas = np.zeros(m)  # alphas初始化为0向量，shape(100,)
    b = 0
    iter = 0  # 迭代变量iter，该变量存储的是没有任何alpha改变的情况下遍历数据集的次数，该变量达到maxIter时，函数结束运行，并推出

    def f(x):
        "SVM分类器函数 y = w^Tx + b"
        # Kernel function vector.
        x = np.mat(x).T  # 选取一个样本变量，x.shape(2,1)
        data = np.mat(dataset)  # 所有的变量，data.shape(100,2)
        kernel = data * x  # K(x_i,X) 其中X为所有的变量，kernel.shape(100,1)
        # Predictive value.
        # alphas_labels = alphas * labels
        # np_alphas_labels = np.mat(alphas_labels)
        wx = np.mat(alphas * labels) * kernel  # sigma_j(alpha_i*y_i*k(x_i,x_j))
        fx = wx + b  # fx是对变量x的预测值 结果为1行1列的标量
        # fx00 = fx[0, 0]
        return fx[0, 0]  # 返回预测值

    while iter < max_iter:
        alphaPairsChanged = 0  # 每次迭代前，alphaPairsChanged先置0，然后对整个数据集[顺序]遍历
        for i in range(m):  # i为数据集中的每一个数据的下标
            # 对alphaI计算出预测值f(x_1)以及其与真实值的误差E_1
            a_i, x_i, y_i = alphas[i], dataset[i], labels[i]  # 
            fx_i = f(x_i)
            E_i = fx_i - y_i
            j = select_j(i, m)  # 在m中随机选择除了i之外剩余的数
            a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j  # 对随机选取的alphaJ计算出预测值f(x_2)以及其与真实值的误差E_2

            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2 * K_ij  # eta为alphaJ的最优修改量，是alphaJ修改量的度量系数
            if eta <= 0:
                # 实际上eta的数学表达式可以写成完全平方式,eta = (a-b)^2,所以eta>=0
                # 是如果eta为0，后面求解alphaJ的时候会出现除零错误，则需要退出当前for循环
                print('WARNING  eta <= 0')
                continue
            # 获取更新的alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j * (E_i - E_j) / eta

            # 对alphaJ进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(C, C + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - C)
                H = min(C, a_j_old + a_i_old)
            a_j_new = clip(a_j_new, L, H)  # 裁剪alphaJ
            a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)  # 更新alphaI

            if abs(a_j_new - a_j_old) < 0.00001:
                # 如果alphaJ改变的比较轻微(0.00001)，我们认为alphaJ没有什么改变，则退出当前for循环
                # print('WARNING   alpha_j not moving enough')
                continue
            alphas[i], alphas[j] = a_i_new, a_j_new  # 将得到的更新的alpha填到对应的位置

            # 更新阈值b
            b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + b
            b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + b
            if 0 < a_i_new < C:  # 当alphaI不在边界时：
                b = b_i
            elif 0 < a_j_new < C:  # 当alphaJ不在边界时：
                b = b_j
            else:  # 当两个乘子alphaI,alphaJ都在边界上，且L!=H,b1和b2之间的值就是和KKT条件一致的阈值。SMO选择他们的中点作为新的阈值
                b = (b_i + b_j) / 2.0
            alphaPairsChanged += 1
            print('INFO   iteration:{}  第{}个样本,J为{}  alphaPairsChanged:{}'.format(iter, i, j, alphaPairsChanged))
        # 关于for循环，当执行完一轮时，即是遍历了所有的数据集，此时如果alpha已经更新
        # 将iter置0，重新再来一轮遍历(可能还会造成alpha的更新)，直到alpha不再更新
        # 这里以alpha是否更新作为迭代轮数iter更替的标志事件，并且发现alpha更新后就将iter置0
        # 如果一连max_iter(40轮)都没更新alpha，那么我们认为此时alpha已经更新的差不多了
        # 然后就可以退出循环，返回alpha和b
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: {}'.format(iter))
    return alphas, b


if '__main__' == __name__:
    # 加载训练数据
    dataset, labels = load_data('testSet.txt')
    # 使用简化版SMO算法优化SVM
    alphas, b = simple_smo(dataset, labels, 0.6, 40) # alphas.shape=(100,),b为常数标量
    # 分类数据点
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            classified_pts['+1'].append(point) # 所有"+1"的点(的坐标)
        else:
            classified_pts['-1'].append(point) # 所有"-1"的点(的坐标)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    # 绘制分割线
    w = get_w(alphas, dataset, labels)
    x1, _ = max(dataset, key=lambda x: x[0]) # 横坐标最大值
    x2, _ = min(dataset, key=lambda x: x[0]) # 横坐标最小值
    a1, a2 = w
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2])
    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    plt.show()
