import numpy as np
import pandas as pd
import os
from smooth_algorithms.smoothn import smoothn

def extract_filename():
    nzd_file = pd.read_csv("../raw_data/nzd group.CSV")
    nzd_column = nzd_file['nzd_name']
    nzd_orientation = nzd_file['orientation']
    return nzd_column, nzd_orientation, nzd_file

#计算所有行数据平均值，计算该海岸平均距离， nzd_filtered表格
def filter_data(filePath, filename, start_year, end_year):
    global coastName, coastline, output_dir
    df_nzd = pd.read_csv(filePath)
    coastline = filename

    coastName = f'{coastline}_Average_Value'
    output_dir = f'../derived_data/{start_year}_{end_year}/{coastline}/'
    os.makedirs(output_dir, exist_ok=True)
    df_nzd[coastName] = df_nzd.iloc[:,1:-1].mean(axis=1)
    nzd_filter = df_nzd.iloc[:, [0, -1]]  # 只保留第一列（日期）和最后一列（平均值）

    #处理异常值
    Q1 = nzd_filter[coastName].quantile(0.25)
    Q3 = nzd_filter[coastName].quantile(0.75)
    IQR = Q3 - Q1
    nzd_filter = nzd_filter[(nzd_filter[coastName] >= (Q1 - 1.5 * IQR)) &
                            (nzd_filter[coastName] <= (Q3 + 1.5 * IQR))]

    nzd_filter.to_csv(os.path.join(output_dir, f'{coastline}_transects_average.csv'), index=False)
    return nzd_filter, coastName

# 先筛选年份，然后进行平滑处理，计算月平均值
def calc_mean_monthly(nzd_filtered, start_date, end_date, s):
    #df_orig = df_new.copy()
    nzd_filtered["dates"] = nzd_filtered["dates"].dt.tz_localize(None)

    #首先筛选时间，然后平滑处理，最后互相关水平会略微降低，降低0.01左右
    # nzd_filtered = nzd_filtered[(nzd_filtered["dates"] >= start_date) & (nzd_filtered["dates"] <= end_date)].reset_index(drop=True)
    nzd_filtered_copy = nzd_filtered.copy()

    coast_smooth_value = smoothn(nzd_filtered_copy[coastName].values, s=s)[0]
    coast_smooth = smoothn(nzd_filtered_copy[coastName].values)[1]
    print(f'original coast smoothing parameters "s"= {coast_smooth} ')

    nzd_filtered_copy["Year-Month"] = nzd_filtered_copy["dates"].dt.to_period("M")  # 只保留年-月

    # 输出未平滑处理的月平均值
    nzd_monthly = nzd_filtered_copy.groupby("Year-Month")[coastName].mean().reset_index()
    # nzd_monthly.to_csv(f'../derived_data/{coastline}_monthly.csv', index=False)

    # 平滑处理后的月平均值
    nzd_filtered_copy[coastName] = coast_smooth_value
    nzd_smooth_monthly = nzd_filtered_copy.groupby("Year-Month")[coastName].mean().reset_index()
    nzd_smooth_monthly.to_csv(os.path.join(output_dir, f'{coastline}_smooth_monthly.csv'), index=False)
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
    df_melted = df_melted[(df_melted["Date"] >= start_date) & (df_melted["Date"] <= end_date)].reset_index(drop=True)
    soi_filter = df_melted.copy()

    # soi_filter.to_csv(f'../derived_data/soi_date.csv', index=False)
    soi_smooth = smoothn(soi_filter["Value"].values, s=s)[0]
    soi_s = smoothn(soi_filter["Value"].values)[1]
    print(f'original soi smoothing parameters "s"= {soi_s} ')

    soi_filter["Year-Month"] = soi_filter["Date"].dt.to_period("M")
    SOI_before_smooth = soi_filter.iloc[:, [4, 2]]

    SOI_monthly_avg = soi_filter.iloc[:, [4, 2]].copy()
    SOI_monthly_avg["Smooth_Value"] = soi_smooth

    soi_dir = f'../derived_data/{start_date.year}_{end_date.year}/'
    os.makedirs(soi_dir, exist_ok=True)
    SOI_monthly_avg.to_csv( f'../derived_data/{start_date.year}_{end_date.year}/SOI_smooth.csv', index=False)

    return SOI_before_smooth, SOI_monthly_avg

def plot_soi_data(csv_path):
    # 1. 读取 CSV
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
    merged_data.to_csv(os.path.join(output_dir, f'Combined_{coastline}_soi.csv'), index=False)
    return merged_data


def calc_cross_correlation(x, y):
    n = len(x)
    cross_corr, lags = compute_cross_correlation(x,y)
    # 找到最大相关性
    max_indices = np.argmax(np.abs(cross_corr))

    max_lag = lags[max_indices]
    max_corr = cross_corr[max_indices]

    # 绘制滞后值前后12个月的内容
    window = 12
    start = n
    end = n + window
    select_lags_x = lags[start - window: end]
    select_lags_y = cross_corr[start - window: end]

    select_lags = lags[start:end]
    select_corr = cross_corr[start:end]

    max_select_indices = np.argmax(np.abs(select_corr))
    max_select_lag = select_lags[max_select_indices]
    max_select_corr = select_corr[max_select_indices]

    return (max_lag, max_corr, max_select_lag, max_select_corr,
            lags, cross_corr, select_lags_x, select_lags_y)


def compute_cross_correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    cross_corr = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * len(x))
    lags = np.arange(-len(x) + 1, len(x))
    return cross_corr, lags

def compute_test_statistic(cross_corr, selected_lags, n):
    """计算选定滞后的平方和"""
    return np.sum(cross_corr[selected_lags + (n - 1)]**2)  # 调整滞后索引

def bootstrap_test(x, y, selected_lags, n_bootstrap=1000):
    """Bootstrap 生成 Q 的分布"""
    n = len(x)
    stat_samples = []
    block_size = 12
    n_blocks = n // block_size
    for _ in range(n_bootstrap):
        # 随机选择块并拼接
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        indices = np.concatenate([
            np.arange(i * block_size, (i + 1) * block_size) for i in block_indices
        ])
        x_boot = x[indices[:n]]  # 确保长度匹配
        y_boot = y[indices[:n]]

        # 计算互相关
        _, cross_corr_boot = compute_cross_correlation(x_boot, y_boot)
        stat = compute_test_statistic(cross_corr_boot, selected_lags, n)
        stat_samples.append(stat)

    return np.array(stat_samples)


# def merge_raw(coast_monthly_avg, SOI_monthly_avg):
#     merged_data = pd.merge(coast_monthly_avg, SOI_monthly_avg, on='Year-Month', how='inner')
#     merged_data.to_csv(f'../derived_data/Combined_raw_{coastline}_soi.csv', index=False)
#     return merged_data

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