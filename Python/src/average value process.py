#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:17:55 2024
@author: chang
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import average
from scipy.stats import pearsonr
from smooth_algorithms.smoothn import smoothn
import pandas as pd
import matplotlib.dates as mdates
from SOIFunction import plot_soi_data
import os
from DataProcess import *

filePath = '../raw_data/nzd_188.csv'

#Read the NZD original data and calculate the mean for each row
nzd_filter, filename, coastValueName= filter_data(filePath)

# Get the hard copy of nzd filtered data for plotting
nzd_filter['dates'] = pd.to_datetime(nzd_filter['dates'])
nzd_orig = nzd_filter.copy()

# smooth parameter s
s = 3

# Smooth the filtered data firstly and calculate the average of nzd data on a monthly basis
coast_monthly_avg = calc_mean_monthly(nzd_filter, s)

# Preprocess SOI data, Select the specified column and change the column name
start_date = '1999-09-01'
end_date = '2024-10-30'
SOI_monthly_avg = preprocess_soi(start_date, end_date, s)

# merge the both data, Leave out months that don't exist
merged_data = merge_nzd_soi(coast_monthly_avg,SOI_monthly_avg)

numerical_data = merged_data.select_dtypes(include=['float64'])

# correlation = numerical_data.corr()# 计算皮尔逊相关系数
x = numerical_data[f'{filename}_Average_Value'].values
y = numerical_data['Value'].values
cross_corr = np.correlate(x - np.mean(x), y - np.mean(y), mode="full") / (np.std(x) * np.std(y) * len(x))

# 找到最大相关性
lag = np.argmax(cross_corr) - (len(x) - 1)
print("=======================================================")
print("Maximum correlation lag:", lag)
print("Maximum cross-correlation coefficient:",max(cross_corr))
# 获取对应的滞后值
top_5_indices = np.argsort(cross_corr)[-5:][::-1]
lags = np.arange(-len(x) + 1, len(x))
top_5_lags = lags[top_5_indices]
top_5_values = cross_corr[top_5_indices]
print("=======================================================")
print("Top 5 滞后值和对应的互相关系数：")
for lag, value in zip(top_5_lags, top_5_values):
    print(f"Lag: {lag}, Correlation: {value:.4f}")
print(np.argmax(cross_corr),len(x)-1)



# 获得属性值
coastName = nzd_filter.columns
num = len(coastName)

#设置变量储存前一个子图
ax_prev = None

# 绘制海岸线变化情况以及soi变化情况
fig, axes = plt.subplots( int(num/2) ,2, sharex=False, figsize=(20,8))
merged_data["Year-Month"] = merged_data["Year-Month"].dt.to_timestamp()
axes[1].plot(merged_data["Year-Month"], merged_data[coastValueName],'b-.', label=f"{filename}_Coastline_value")
axes[1].legend(loc='upper left')
#绘制soi平滑后数据
ax1 = axes[1].twinx()
ax1.plot(merged_data["Year-Month"], merged_data["Value"], 'g-', label="SOI value")
ax1.set_ylabel("SOI value")
#绘制子图标题，设置基本格式
ax1.tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
ax1.grid(True)
ax1.legend(loc='best')


axes[0].plot(nzd_orig["dates"],nzd_orig[coastValueName],'b-', label="188nzd value")
axes[0].set_ylabel("188nzd value")
#子图标题，设置基本格式
axes[0].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
axes[0].grid(True)
axes[0].legend(loc='best')
# 多次
# for ax, (i, col) in zip(axes.flat, enumerate(coastName)):
#     if i % 2 == 1:
#         df = df.dropna(subset=[col])
#         coastValue = df[col].values
#         coastDate = pd.to_datetime(df[df.columns[i-1]].values)
#         coastSmooth = smoothn(coastValue, isrobust=True)[0]
#         #绘制海岸线平滑后的变化数据
#         ax.plot(coastDate, coastSmooth, 'b-.', label=col)
#         ax.set_ylabel("Coastline change")
#         ax.set_ylim(345,370)
#
#         #绘制soi平滑后数据
#         ax1 = ax.twinx()
#         ax1.plot(df_melted["Date"], soi_smooth, 'g-', label="SOI value")
#         ax1.set_ylabel("SOI value")
#
#         #绘制子图标题，设置基本格式
#         ax.set_title(col + '_smooth and SOI_smooth')
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         ax.tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
#         ax.grid(True)
#         ax1.legend(loc='best')
#         ax.legend(loc='best')
#
#         if ax_prev is not None: #绘制海岸线变化数据
#             ax_prev.plot(coastDate, coastValue, label=col + '_original', color='r')
#             ax_prev.set_title(col + '_original')
#             ax_prev.legend(loc='best')
#             ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#             ax.tick_params(axis='x', rotation=45)
#     ax_prev = ax


plt.tight_layout()
plt.show()

# plot_soi_data('../derived_data/SOI_value.csv', '1999-09-01', '2022-10-31')