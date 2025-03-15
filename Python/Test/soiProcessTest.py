import pandas as pd
import matplotlib.pyplot as plt
from smooth_algorithms.smoothn import smoothn
import numpy as np

# 读取 TXT 文件（空格作为分隔符）
# df = pd.read_csv('SOI derived_data.txt', sep=r'\s+', header=None, engine='python')
# # 保存为 CSV（逗号分隔）
# df.to_csv('SOI output.csv', index=False, header=False)
data = pd.read_csv('../derived_data/SOI output.csv')

# 设置年份区间和月份名称
months = data.columns[1:]

filtered_data = data[(data["YEAR"] >= 2022) & (data["YEAR"] <= 2022)].reset_index(drop=True)
num_rows = len(filtered_data)

# 绘制折线图

if num_rows == 1:
    fig= plt.figure(figsize=(8,6))
    values = filtered_data.iloc[0, 1:].values
    print(values)
    plt.plot(months, values, label=filtered_data["YEAR"])
    plt.ylabel('Value')
    plt.title(filtered_data["YEAR"])
    plt.legend()
else:
    fig, axes = plt.subplots(num_rows,1 ,figsize=(12, 14))  # 设置图像大小
    # before smoothing algorithm
    for index, row in filtered_data.iterrows():
        axes[index].plot(months, row[1:], label=row["YEAR"])  # X轴是月份，Y轴是对应值
        axes[index].set_title(row["YEAR"])
        axes[index].legend(loc='best')
        axes[index].set_ylabel("Value")
        axes[index].grid(True, linestyle="--", alpha=0.6)  # 添加网格
    #数据已经平滑到极限

plt.tight_layout()
# 显示图像

plt.show()




