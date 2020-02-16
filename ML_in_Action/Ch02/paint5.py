import matplotlib.pyplot as plt
plt.figure(1)                # 第一张图
plt.subplot(211)             # 第一张图中的第一张子图
plt.plot([1, 2, 3])
plt.subplot(212)             # 第一张图中的第二张子图
plt.plot([4, 5, 6])


plt.figure(2)                # 第二张图
plt.plot([4, 5, 6])            # 默认创建子图subplot(111)

plt.figure(1)                # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
plt.subplot(211)             # 第一张图中的第一张子图
plt.title('figure1_1')   # 添加subplot 211 的标题
plt.xlabel('X1_1,figure1')  # 设置X轴标签
plt.ylabel('Y1_1,figure1')  # 设置Y轴标签

plt.figure(2)
plt.title('figure2')

plt.figure(1)
plt.subplot(212)             # 第一张图中的第一张子图
plt.title('figure1_2')   # 添加subplot 211 的标题
plt.xlabel('X1_2,figure1')  # 设置X轴标签
plt.ylabel('Y1_2,figure1')  # 设置Y轴标签

plt.show()
