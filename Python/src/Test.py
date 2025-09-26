import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模拟数据
np.random.seed(42)
time = np.arange(2000, 2010, 0.5)            # 时间
trend = 0.2 * (time - 2000)                  # 线性趋势
seasonal = 0.5 * np.sin(2 * np.pi * (time-2000)) # 季节性波动
noise = np.random.normal(0, 0.1, len(time))  # 随机噪声

data = trend + seasonal + noise

# 去趋势化
data_detrended = data - trend

# 减去长期平均
data_anomaly = data - np.mean(data)

# 绘图
plt.figure(figsize=(12,6))
plt.plot(time, data, label='Original Data', color='blue')
plt.plot(time, data_detrended, label='Detrended Data', color='orange')
plt.plot(time, data_anomaly, label='Minus Long-Term Mean', color='green')
plt.xlabel('Year')
plt.ylabel('Shoreline Position (m)')
plt.title('Comparison: Original vs Detrended vs Minus Long-Term Mean')
plt.legend()
plt.grid(True)
plt.show()


