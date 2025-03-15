import numpy as np
import pandas as pd
import os
from smooth_algorithms.smoothn import smoothn
from SOIFunction import *

coastName = ""
filename = ""

def filter_data(filePath):
    #计算所有行数据平均值，计算该海岸平均距离， nzd_filtered表格
    global filename,  coastName
    df_nzd = pd.read_csv(filePath)
    filename = os.path.splitext(os.path.basename(filePath))[0]
    coastName = f'{filename}_Average_Value'
    df_nzd[coastName] = df_nzd.iloc[:,1:-1].mean(axis=1)
    nzd_filter = df_nzd.iloc[:, [0, -1]]  # 只保留第一列（日期）和最后一列（平均值）
    #df_new.to_csv(f'../derived_data/{filename}_filtered.csv', index=False)
    return nzd_filter, filename, coastName

def calc_mean_monthly(nzd_filtered):
    #df_orig = df_new.copy()
    coast_smooth_value = smoothn(nzd_filtered[coastName].values, s=3)[0]
    s = smoothn(nzd_filtered[coastName].values)[1]
    print(f'未指定情况下，海岸线数据平滑参数 = {s} ')
    nzd_filtered[coastName] = coast_smooth_value
    nzd_filtered["dates"] = nzd_filtered["dates"].dt.tz_localize(None)
    nzd_filtered["Year-Month"] = nzd_filtered["dates"].dt.to_period("M")  # 只保留年-月
    nzd_monthly_avg = nzd_filtered.groupby("Year-Month")[coastName].mean().reset_index()
    # coast_monthly_avg.to_csv(f'../derived_data/{filename}_average_monthly.csv', index=False)
    return nzd_monthly_avg

def  preprocess_soi(start_date, end_date):
    soi_filter, soi_smooth = plot_soi_data('../derived_data/SOI output.csv', start_date, end_date)
    soi_filter["Year-Month"] = soi_filter["Date"].dt.to_period("M")
    SOI_monthly_avg = soi_filter.iloc[:, [2, 4]]
    SOI_monthly_avg.loc[:, "Value"] = soi_smooth
    return SOI_monthly_avg

def merge_nzd_soi(coast_monthly_avg, SOI_monthly_avg):
    merged_data = pd.merge(coast_monthly_avg, SOI_monthly_avg, on='Year-Month', how='inner')
    merged_data.to_csv(f'../derived_data/{filename}_soi_Merged', index=False)
    return merged_data