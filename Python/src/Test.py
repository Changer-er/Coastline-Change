import numpy as np

# 创建两个信号，其中 y 是 x 的滞后版本
x = np.array([1, 2, 3, 4, 5])
y = np.array([0, 1, 2, 3, 4])  # y 比 x 滞后了 1

# 计算互相关
cross_corr = np.correlate(x, y, mode='full')

# 计算滞后
lag = np.argmax(cross_corr) - (len(x) - 1)

print("互相关序列:", cross_corr)
print("最大互相关的索引:", np.argmax(cross_corr))
print("滞后量 lag:", lag)
print("滞后量 lag:", np.argmax(cross_corr))
