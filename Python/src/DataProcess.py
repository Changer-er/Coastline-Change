import numpy as np
import pandas as pd
import os
from smooth_algorithms.smoothn import smoothn

coastName = ""
filename = ""

#计算所有行数据平均值，计算该海岸平均距离， nzd_filtered表格
def filter_data(filePath):
    global filename,  coastName
    df_nzd = pd.read_csv(filePath)
    filename = os.path.splitext(os.path.basename(filePath))[0]
    coastName = f'{filename}_Average_Value'
    df_nzd[coastName] = df_nzd.iloc[:,1:-1].mean(axis=1)
    nzd_filter = df_nzd.iloc[:, [0, -1]]  # 只保留第一列（日期）和最后一列（平均值）
    #处理异常值
    Q1 = nzd_filter[coastName].quantile(0.25)
    Q3 = nzd_filter[coastName].quantile(0.75)
    IQR = Q3 - Q1
    nzd_filter = nzd_filter[(nzd_filter[coastName] >= (Q1 - 1.5 * IQR)) & (nzd_filter[coastName] <= (Q3 + 1.5 * IQR))]
    nzd_filter.to_csv(f'../derived_data/{filename}_average.csv', index=False)
    return nzd_filter, filename, coastName

# 先进行平滑处理，计算月平均值
def calc_mean_monthly(nzd_filtered, s):
    #df_orig = df_new.copy()
    coast_smooth_value = smoothn(nzd_filtered[coastName].values, s=s)[0]
    coast_smooth = smoothn(nzd_filtered[coastName].values)[1]
    print(f'original coast smoothing parameters "s"= {coast_smooth} ')
    nzd_filtered["dates"] = nzd_filtered["dates"].dt.tz_localize(None)
    nzd_filtered["Year-Month"] = nzd_filtered["dates"].dt.to_period("M")  # 只保留年-月
    nzd_monthly = nzd_filtered.groupby("Year-Month")[coastName].mean().reset_index()
    nzd_monthly.to_csv(f'../derived_data/{filename}_monthly.csv', index=False)

    nzd_filtered[coastName] = coast_smooth_value
    nzd_smooth_monthly = nzd_filtered.groupby("Year-Month")[coastName].mean().reset_index()
    nzd_smooth_monthly.to_csv(f'../derived_data/{filename}_smooth_monthly.csv', index=False)
    return nzd_monthly, nzd_smooth_monthly

# 先计算月平均值, 进行平滑处理
# def calc_mean_monthly_copy(nzd_filtered, s):
#     #df_orig = df_new.copy()
#     nzd_monthly_smooth = pd.read_csv(f'../derived_data/{filename}_monthly.csv')
#     nzd_smooth_value = smoothn(nzd_monthly_smooth[coastName].values, s=s)[0]
#     nzd_monthly_smooth[coastName] = nzd_smooth_value
#     nzd_monthly_smooth.to_csv(f'../derived_data/{filename}_monthly_smooth.csv', index=False)
#     return nzd_monthly_smooth

#读取soi数据，筛选时间范围，导出备份，对soi数据进行平滑处理, 导出平滑后的数据集
def preprocess_soi(start_date, end_date, s):
    df_melted = plot_soi_data('../raw_data/SOI_value.csv')
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    df_melted = df_melted[(df_melted["Date"] >= start_date) & (df_melted["Date"] <= end_date)].reset_index(drop=True)
    soi_filter = df_melted.copy()
    soi_filter.to_csv(f'../derived_data/soi_date.csv', index=False)
    soi_smooth = smoothn(df_melted["Value"].values, s=s)[0]
    soi_s = smoothn(df_melted["Value"].values)[1]
    print(f'original soi smoothing parameters "s"= {soi_s} ')
    soi_filter["Year-Month"] = soi_filter["Date"].dt.to_period("M")
    SOI_monthly_avg = soi_filter.iloc[:, [4, 2]].copy()
    SOI_monthly_avg["Smooth_Value"] = soi_smooth
    SOI_monthly_avg.to_csv(f'../derived_data/soi_smooth_monthly.csv', index=False)
    return SOI_monthly_avg

def plot_soi_data(csv_path):
    # 重新读取 CSV
    df = pd.read_csv(csv_path)
    # 2. 将数据转换成长格式
    df_melted = df.melt(id_vars=["YEAR"], var_name="Month", value_name="Value")
    # 3. 确保月份按正确顺序排列, 修改月份为category类型，具有顺序特性
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    df_melted["Month"] = pd.Categorical(df_melted["Month"], categories=month_order, ordered=True)
    # 4. 创建时间序列列并进行排序，通过sort_values进行排序
    df_melted["Date"] = pd.to_datetime(df_melted["YEAR"].astype(str) + "-" + df_melted["Month"].astype(str),format="%Y-%b")
    df_melted = df_melted.sort_values(by="Date")
    return df_melted


def merge_nzd_soi(coast_monthly_avg, SOI_monthly_avg):
    merged_data = pd.merge(coast_monthly_avg, SOI_monthly_avg, on='Year-Month', how='inner')
    merged_data.to_csv(f'../derived_data/Combined_{filename}_soi.csv', index=False)
    return merged_data

def merge_raw(coast_monthly_avg, SOI_monthly_avg):
    merged_data = pd.merge(coast_monthly_avg, SOI_monthly_avg, on='Year-Month', how='inner')
    merged_data.to_csv(f'../derived_data/Combined_raw_{filename}_soi.csv', index=False)
    return merged_data
# def filter_SOI_data(filepath, start_date, end_date, s):
#     df_melted = pd.read_csv(filepath)
#     start_date = pd.Timestamp(start_date)
#     end_date = pd.Timestamp(end_date)
#     start_date = start_date.strftime('%Y-%m')
#     end_date = end_date.strftime('%Y-%m')
#     df_melted = df_melted[(df_melted["Date"] >= start_date) & (df_melted["Date"] <= end_date)].reset_index(drop=True)
#
#     # 6.数据进行平滑处理
#     soi_smooth = smoothn(df_melted["Value"].values, s=s)[0]
#     s = smoothn(df_melted["Value"].values)[1]
#     print(f'smooth soi data, soi_smooth = {soi_smooth}')
#     print(f'original soi smoothing parameters "s"= {s} ')
#     return df_melted, soi_smooth