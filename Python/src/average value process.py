#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:17:55 2024
@author: chang
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np

from DataProcess import *

names, orientation, nzd_file = extract_filename()

start_date = '1999-09-01'
end_date = '2024-10-30'
# start_date = '2007-01-01'
# end_date = '2016-12-31'
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
start_year = start_date.year
end_year = end_date.year

# smooth parameter s
s = 4
print(f"=======================================SOI Value====================================")
print(f'The Smoothing Parameters we set: "s"= {s} ')
SOI_before_smooth, SOI_monthly_avg = preprocess_soi(start_date, end_date, s)

for index, value in names.items():
    filename = value
    filePath = f'../Coastline_data/{filename}/transect_time_series_tidally_corrected.csv'
    #Read the NZD original data and calculate the mean for each row
    nzd_filter, coastValueName= filter_data(filePath, filename, start_year, end_year)

    # Get the hard copy of nzd filtered data for plotting
    nzd_filter['dates'] = pd.to_datetime(nzd_filter['dates'])
    nzd_orig = nzd_filter.copy()

    # Smooth the filtered data firstly and calculate the average of nzd data on a monthly basis
    nzd_monthly, nzd_smooth_monthly = calc_mean_monthly(nzd_filter, start_date, end_date, s)

    # merge the both data, Leave out months that don't exist
    merged_data = merge_nzd_soi(nzd_smooth_monthly,SOI_monthly_avg)
    merged_data.to_csv()
    numerical_data = merged_data.select_dtypes(include=['float64'])
    merged_data["Year-Month"] = merged_data["Year-Month"].dt.to_timestamp()
    nzd_monthly["Year-Month"] = nzd_monthly["Year-Month"].dt.to_timestamp()

    # 1. Calculate correlation
    x = numerical_data[f'{filename}_Average_Value'].values
    y = numerical_data['Smooth_Value'].values
    (max_lag, max_corr, max_select_lag, max_select_corr,
     lags, cross_corr, select_lags_x, select_lags_y) = calc_cross_correlation(x, y)

    # 2. Define the test statistic (sum of squares)
    selected_lags = np.arange(0, 12)
    observed_stat = compute_test_statistic(cross_corr, selected_lags, len(x))
    print(f"Observed test statistic (Q): {observed_stat:.4f}")

    # 3. Bootstrap joint inspection
    np.random.seed(42)
    bootstrap_stats = bootstrap_test(x, y, selected_lags, n_bootstrap=5000)
    print(f"Bootstrap test statistic (Q): {bootstrap_stats}")

    # 4. 计算 p 值
    p_value = np.mean(bootstrap_stats >= observed_stat)

    #2：Bootstrap估计置信区间
    print(f"======================================={filename}====================================")
    print(f"Observed test statistic (Q): {observed_stat:.4f}")
    print(f"Bootstrap p-value: {p_value:.4f}")
    print("\n")
    print("Maximum correlation lag:", max_lag)
    print("Maximum cross-correlation coefficient:",max_corr)


    nzd_file.loc[index, 'Max_Lag'] = max_lag
    nzd_file.loc[index, 'Max_Correlation'] = max_corr
    nzd_file.loc[index, 'Max_Selected_lag'] = max_select_lag
    nzd_file.loc[index, 'Max_Selected_Correlation'] = max_select_corr


    # 绘制海岸线变化情况以及soi变化情况
    fig, axes = plt.subplots( 2 ,2, sharex=False, figsize=(20,8))

    axes[0, 0].plot(nzd_monthly["Year-Month"],nzd_monthly[coastValueName], 'g-', label=f"{filename}_value")
    axes[0, 0].set_ylabel("188nzd value")
    axes[0, 0].set_title(f"{filename}_Coastline_before_smoothing")
    #子图标题，设置基本格式
    axes[0, 0].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='best')

    axes[0, 1].plot(merged_data["Year-Month"], merged_data[coastValueName],'g-.', label=f"{filename}_Coastline_value")
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set_title(f"{filename}_Coastline_and_SOI_after_smoothing")
    #绘制soi平滑后数据
    ax1 = axes[0, 1].twinx()
    ax1.plot(merged_data["Year-Month"], merged_data["Smooth_Value"], 'b-', label="SOI value")
    ax1.set_ylabel("SOI value")
    #绘制子图标题，设置基本格式
    ax1.tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    ax1.grid(True)
    ax1.legend(loc='best')

    axes[1, 0].plot(lags, cross_corr, 'g-', label="Cross correlation")
    axes[1, 0].set_ylabel("The cross-correlation")
    axes[1, 0].set_title("The relation between lags and cross-correlation")
    #子图标题，设置基本格式
    axes[1, 0].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc='best')
    axes[1, 0].axvline(x = max_lag, color='r', linestyle='--', label=f"Max value at lag = {max_lag}")
    axes[1, 0].axhline(y = max_corr, color='r', linestyle='--', label=f"Max value = {max_corr}")
    axes[1, 0].annotate(
        f'Max Corr = {max_corr:.3f}\nLag = {max_lag}',
        xy=(max_lag, max_corr),
        xytext=(max_lag + 15, max_corr - 0.2),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )


    axes[1, 1].plot(select_lags_x, select_lags_y, 'b-', label="Cross correlation")
    axes[1, 1].set_ylabel("The cross-correlation")
    axes[1, 1].set_title("The relation between lags and cross-correlation within 2 year")
    #子图标题，设置基本格式
    axes[1, 1].tick_params(axis='x', rotation=45)# X轴标签旋转，防止重叠
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc='best')
    axes[1, 1].axvline(x=max_select_lag, color='r', linestyle='--', label=f"Max value at lag after 0 = {max_select_lag}")
    axes[1, 1].axhline(y=max_select_corr, color='r', linestyle='--', label=f"Max value after 0= {max_select_corr}")
    axes[1, 1].annotate(
        f'Max Corr = {max_select_corr:.3f}\nLag = {max_select_lag}',
        xy=(max_select_lag, max_select_corr),
        xytext=(max_select_lag + 5, max_select_corr - 0.2),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

    outputfile = f"../fig/nzd_{start_year}_{end_year}/"
    os.makedirs(outputfile, exist_ok=True)
    fig.savefig(os.path.join(outputfile,f"{filename}_Coastline.png"), dpi=300, bbox_inches='tight')

    # plt.hist(bootstrap_stats, bins=50, alpha=0.7, label="Bootstrap Distribution")
    # plt.axvline(observed_stat, color="red", label="Observed Statistic")
    # plt.title("Bootstrap Distribution of Test Statistic (Q)")
    # plt.xlabel("Sum of Squared Cross-Correlations")
    # plt.legend()
    # plt.show()

    plt.tight_layout()
    plt.show()

nzd_file.to_csv(f"../derived_data/nzd_results_{start_year}_{end_year}.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(SOI_before_smooth["Year-Month"].dt.to_timestamp(), SOI_before_smooth["Value"], color='red', marker = 'o')
plt.title("SOI change over time")
plt.xlabel("Year-Month")
plt.ylabel("SOI value")
plt.grid(True)
plt.tight_layout()
plt.show()