#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:17:55 2024
@author: chang
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from DataProcess import *
from Crosscorrelation import *


names, quadrant, nzd_file = extract_filename()

# start_date = '2000-01-01'
# end_date = '2025-12-31'
start_date = '2010-01-01'
end_date = '2014-12-31'
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
start_year = start_date.year
end_year = end_date.year

smooth_soi = input("Use smooth SOI? (y/n): ").strip().lower() in ("y", "yes", "1", "true")

# smooth parameter s
s = 0
print(f"=======================================SOI Value====================================")
SOI_before_smooth, SOI_smooth = preprocess_soi(start_date, end_date, s)

for index, CoastlineName in names.items():
    # Coastline_ts = f'../Clean_coastline_data/{CoastlineName}/transect_time_series_tidally_corrected.csv'
    filepath = f'../Trend_data/{CoastlineName}/{CoastlineName}.xlsx'

    #Read the NZD original data and calculate the mean for each row
    result = filter_data(filepath, CoastlineName, start_year, end_year)

    if result is None:
        print(f"=========================={CoastlineName}================================")
        print(f'No data found for {CoastlineName}')
        continue
    else:
        nzd_filter, coastValueName = result

    # Smooth the filtered data firstly and calculate the average of nzd data on a monthly basis
    nzd_monthly = calc_mean_monthly(nzd_filter, s)

    # merge the both data, Leave out months that don't exist
    if smooth_soi:
        merged_data = merge_nzd_soi(nzd_monthly, SOI_smooth)
        numerical_data = merged_data.select_dtypes(include=['float64'])
        y = numerical_data["Smooth_Value"].values
        SOI_value = "Smooth_Value"
    else:
        merged_data = merge_nzd_soi(nzd_monthly, SOI_before_smooth)
        numerical_data = merged_data.select_dtypes(include=['float64'])
        y = numerical_data["Value"].values
        SOI_value = "Value"

    merged_data["Year-Month"] = merged_data["Year-Month"].dt.to_timestamp()
    nzd_monthly["Year-Month"] = nzd_monthly["Year-Month"].dt.to_timestamp()

    # 创建的互相关性计算公式
    x = numerical_data['monthly_anomaly'].values
    (max_select_lag, max_select_corr, max_p, select_x, select_y, select_p) = calc_cross_corr(x, y)

    #2：Bootstrap估计置信区间
    print(f"======================================={CoastlineName}====================================")
    print("- max_lag:", max_select_lag)
    print("- max_cc:", max_select_corr)
    print(f"- max_p: {max_p:.4f}")

    # 绘制海岸线变化情况以及soi变化情况
    fig, axes = plt.subplots( 2 ,2, sharex=False, figsize=(20,8))

    axes[0, 0].plot(nzd_monthly["Year-Month"],nzd_monthly['monthly_anomaly'], 'g-', label=f"{CoastlineName}_value")
    axes[0, 0].set_ylabel("coastline value")
    axes[0, 0].set_title(f"{CoastlineName}_Coastline and SOI before smoothing")
    #子图标题，设置基本格式
    axes[0, 0].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='best')
    # ax = axes[0, 0].twinx()
    # ax.plot(merged_data["Year-Month"], merged_data["Value"], 'b-', label="SOI value")
    # ax.set_ylabel("SOI value")
    # ax.tick_params(axis='x', rotation=45)  # X轴标签旋转，防止重叠
    # ax.grid(True)
    # ax.legend(loc='best')

    axes[0, 1].plot(merged_data["Year-Month"], merged_data['monthly_anomaly'],'g-.', label=f"{CoastlineName}_Coastline_value")
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set_title(f"{CoastlineName}_Coastline_and_SOI_after_smoothing during {start_date} to {end_date}")
    axes[0, 1].set_xlabel("Year-Month")
    axes[0, 1].set_ylabel("coastline smooth value")
    #绘制soi平滑后数据
    ax1 = axes[0, 1].twinx()
    ax1.plot(merged_data["Year-Month"], merged_data[SOI_value], 'b-', label="SOI value")
    ax1.set_ylabel("SOI smooth value")
    #绘制子图标题，设置基本格式
    ax1.tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    ax1.grid(True)
    ax1.legend(loc='best')

    if max_select_lag > 0:
        x_plot = x[max_select_lag:]
        y_plot = y[:-max_select_lag]
    else:
        x_plot = x
        y_plot = y
    print(x_plot)
    print(y_plot)
    def zscore_nan(a):
        mu = np.nanmean(a)  # 忽略 NaN 的均值
        sigma = np.nanstd(a, ddof=1)  # 忽略 NaN 的样本标准差（ddof=1）
        return (a - mu) / sigma  # 原来是 NaN 的位置仍是 NaN

    x_plot_z = zscore_nan(x_plot)
    y_plot_z = zscore_nan(y_plot)

    axes[1,0].scatter(x_plot_z, y_plot_z, alpha=0.7)
    axes[1,0].set_xlabel(f"{CoastlineName}_Value (shifted for lag={max_select_lag})")
    axes[1,0].set_ylabel(f"SOI_Value")
    axes[1,0].grid(True)
    axes[1,0].set_title(f"Cross-Correlation at Lag = {max_select_lag}")
    axes[1, 0].set_aspect('equal', adjustable='box')

    axes[1, 1].plot(select_x, select_y, 'b-', label="Cross correlation")
    axes[1, 1].set_ylabel("The cross-correlation")
    axes[1, 1].set_title("The relation between lags and cross-correlation within one reason")
    #子图标题，设置基本格式
    axes[1, 1].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc='best')
    axes[1, 1].axvline(x=max_select_lag, color='r', linestyle='--', label=f"Max value at lag after 0 = {max_select_lag}")
    axes[1, 1].axhline(y=max_select_corr, color='r', linestyle='--', label=f"Max value after 0= {max_select_corr}")
    axes[1, 1].annotate(
        f'Max Corr = {max_select_corr:.3f}\nLag = {max_select_lag}',
        xy=(max_select_lag, max_select_corr),
        xytext=(max_select_lag + 0.05, max_select_corr + 0.01),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

    outputfile = f"../derived_data/Figure/{start_year}_{end_year}/"
    os.makedirs(outputfile, exist_ok=True)
    fig.savefig(os.path.join(outputfile,f"{CoastlineName}_Coastline.png"), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    nzd_file.loc[index, 'p-value'] = max_p
    nzd_file.loc[index, 'Max_lag'] = max_select_lag
    nzd_file.loc[index, 'Max_Correlation'] = max_select_corr

nzd_file.to_csv(f"../derived_data/Results/{start_year}_{end_year}/Coastline_summary.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(merged_data["Year-Month"], merged_data["Value"], color='red', marker = 'o')
plt.title("SOI change over time")
plt.xlabel("Year-Month")
plt.ylabel("SOI value")
plt.grid(True)
plt.tight_layout()
plt.show()