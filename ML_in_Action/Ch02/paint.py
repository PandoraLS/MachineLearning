# 导入必要的模块
import numpy as np
import matplotlib.pyplot as plt
# 产生测试数据
x = np.arange(1, 10)
y = x
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Scatter Plot')  # 设置标题
plt.xlabel('X')  # 设置X轴标签
plt.ylabel('Y')  # 设置Y轴标签
ax1.scatter(x, y, c='r', marker='o')  # 画散点图
plt.legend('x1')  # 设置图标
plt.show()  # 显示所画的图
