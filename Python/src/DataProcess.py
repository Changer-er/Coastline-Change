import numpy as np
import pandas as pd
import os
from smooth_algorithms.smoothn import smoothn

# 读取选择的coastline 并分配象限
def extract_filename():
    nzd_file = pd.read_csv("../raw_data/nzd group.CSV")
    nzd_column = nzd_file['nzd_name']
    nzd_orientation = nzd_file['orientation']

    nzd_radians = nzd_orientation * np.pi / 180
    x = np.sin(nzd_radians)
    y = np.cos(nzd_radians)
    conditions = [
        (x > 0) & (y > 0),
        (x > 0) & (y < 0),
        (x < 0) & (y < 0),
        (x < 0) & (y > 0),
    ]
    choices = [1,2,3,4]

    nzd_file['quadrant'] = np.select(conditions, choices, default=0)
    quadrant = nzd_file['quadrant']
    return nzd_column, quadrant, nzd_file


#读取soi数据，筛选时间范围，导出备份，对soi数据进行平滑处理, 导出平滑后的数据集
def preprocess_soi(start_date, end_date, s):
    df_melted = sort_SOI('../raw_data/SOI_value.csv')
    df_melted = df_melted[(df_melted["Date"] >= start_date) & (df_melted["Date"] <= end_date)].reset_index(drop=True)
    soi_filter = df_melted.copy()

    # soi_filter.to_csv(f'../derived_data/soi_date.csv', index=False)
    if s == 0:
        print("we use default smoothing parameter")

    soi_smooth_value = smoothn(soi_filter["Value"].values, s=s)[0]
    soi_s = smoothn(soi_filter["Value"].values, s=s)[1]
    print(f"The defualt Smoothing Parameters: {soi_s}")

    soi_filter["Year-Month"] = soi_filter["Date"].dt.to_period("M")
    SOI_before_smooth = soi_filter.iloc[:, [4, 2]]

    SOI_smooth = SOI_before_smooth.copy()
    SOI_smooth["Smooth_Value"] = soi_smooth_value

    soi_dir = f'../derived_data/Results/{start_date.year}_{end_date.year}/'
    os.makedirs(soi_dir, exist_ok=True)
    SOI_smooth.to_csv(f"{soi_dir}/SOI_smooth.csv", index=False)

    return SOI_before_smooth, SOI_smooth

def sort_SOI(csv_path):
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



# 减去长期平均值，去趋势化，计算月均异常值
def filter_data(filePath, CoastlineName, start_year, end_year):
    global coast_average, coastline, output_dir
    coastline = CoastlineName

    df_nzd = pd.read_excel(filePath, sheet_name=None)
    Coastline_ts = df_nzd['Intersects']
    Trend_data = df_nzd['Transects']
    params = Trend_data[['id', 'intercept', 'trend']].copy()

    #----------previous process-----------
    # Previous_data = Coastline_ts.copy()
    # Previous_data["monthly_anomaly"] = Previous_data.iloc[:, 1:-1].mean(axis=1)
    # Previous_data = Previous_data.iloc[:, [0, -1]].dropna()

    # ----------R2_score > 0.05-----------
    # 选出可靠性高的id
    # Significant_Trend = Trend_data[Trend_data['r2_score'] > 0.05]
    # Significant_ids = Significant_Trend['id']

    # #判断是否有显著transects
    # if Significant_Trend.empty:
    #     return  # 退出函数
    #
    # #转为datatime64时期格式，计算基于开始时间的年份比例

    Coastline_ts['dates'] = pd.to_datetime(Coastline_ts['dates'], utc=True)
    Coastline_ts = Coastline_ts.drop(columns=["satname"])

    # # 获得显著性海岸线的dataframe
    # Significant_coastline = Coastline_ts[['dates'] + Significant_ids.tolist()].copy()
    # # 获得series的长期平均值
    # longterm_average = Significant_coastline[Significant_ids].mean(axis=0)

    # calculate the average over full periods
    longterm_average = Coastline_ts.iloc[:, 1:].mean(axis=0)
    # # 自动广播映射，获得异常值，去中心化
    # Significant_coastline[Significant_ids] = Significant_coastline[Significant_ids] - longterm_average
    Coastline_ts.iloc[:,1:] = Coastline_ts.iloc[:,1:] - longterm_average

    params = params.set_index(["id"])
    params["intercept"] = params["intercept"]- longterm_average

    # 对显著性的海岸线进行去趋势化
    coastline_detrend = detrend(Coastline_ts, params)

    coast_average = f'{coastline}_Average_Value'

    #计算transects mean
    coastline_detrend[coast_average] = coastline_detrend.iloc[:, 1:].mean(axis=1)

    nzd_filter = coastline_detrend.iloc[:, [0, -1]].dropna()  # 只保留第一列（日期）和最后一列（平均值）
    output_dir = f'../derived_data/Results/{start_year}_{end_year}/{coastline}/'
    os.makedirs(output_dir, exist_ok=True)

    nzd_filter.to_csv(os.path.join(output_dir, f'{coastline}_transects_average.csv'), index=False)
    return nzd_filter, coast_average #, Previous_data

# LinearRegression to detrend
def detrend(df, params):
    df = df.copy()
    df['dates'] = pd.to_datetime(df['dates'])

    #循环不同的transects
    for col_name, col_values in df.iloc[:, 1:].items():

        df_col = df[['dates', col_name]].dropna()
        X = (df_col['dates'] - df_col['dates'].min()).dt.days.values / 365.25

        intercept = params.loc[col_name, 'intercept']
        slope = params.loc[col_name, 'trend']
        trend_line = pd.Series((slope * X + intercept), index=df_col.index)

        detrended = df_col[col_name] - trend_line
        df.loc[df_col.index, col_name] = detrended

    return df


# 先筛选年份，然后进行平滑处理，计算月平均值
# coast_average列是未减去多年平均月份的数据，monthly_anomaly是减去多年平均月异常值的数据
def calc_mean_monthly(nzd_filter, remove_monthly_climatology = False):

    # 处理数据的日期数据
    nzd_filter['dates'] = pd.to_datetime(nzd_filter['dates'])
    nzd_filter["dates"] = nzd_filter["dates"].dt.tz_localize(None)
    nzd_filter["Year-Month"] = nzd_filter["dates"].dt.to_period("M")  # 只保留年-月

    if remove_monthly_climatology == True:
        # remove monthly climatology
        nzd_monthly = remove_clim_trend(nzd_filter)
    else:
        nzd_monthly = (nzd_filter
                       .groupby("Year-Month", as_index=False)[coast_average].mean()
                       .rename(columns={coast_average: 'monthly_anomaly'})
                       )

    nzd_monthly.to_csv(os.path.join(output_dir, f'{coastline}_monthly.csv'), index=False)
    return nzd_monthly

def remove_clim_trend(df):

    df['month'] = df['Year-Month'].dt.month

    monthly_mean_anomalies = (
        df
        .groupby("Year-Month", as_index=False)[coast_average].mean()
    )

    clim = (
        df
        .groupby("month", as_index=False)[coast_average].mean()
        .rename(columns={coast_average: 'climatology'})
    )
    monthly_mean_anomalies['month'] = monthly_mean_anomalies['Year-Month'].dt.month
    out = monthly_mean_anomalies.merge(clim, on="month", how="left")
    out['monthly_anomaly'] = out[coast_average] - out['climatology']
    return out

def merge_nzd_soi(coast_monthly, SOI_monthly):
    merged_data = pd.merge(coast_monthly, SOI_monthly, on='Year-Month', how='inner')
    merged_data.to_csv(os.path.join(output_dir, f'Combined_{coastline}_soi.csv'), index=False)

    return merged_data

#
# def calc_cross_correlation(x, y):
#     n = len(x)
#     cross_corr, lags = compute_cross_correlation(x,y)
#     # 找到最大相关性
#     max_indices = np.argmax(np.abs(cross_corr))
#     max_lag = lags[max_indices]
#     max_corr = cross_corr[max_indices]
#
#     # 绘制滞后值前后12个月的内容
#     window = 12
#     origin = n - 1
#
#     # 选择在6个月最大correlation处的lag值
#     select_lags = lags[origin: origin + 6]
#     select_corr = cross_corr[origin: origin + 6]
#
#     max_select_indices = np.argmax(np.abs(select_corr))
#     max_select_lag = select_lags[max_select_indices]
#     max_select_corr = select_corr[max_select_indices]
#
#     return (max_lag, max_corr, max_select_lag, max_select_corr,
#             lags, cross_corr, select_lags, select_corr)
#
#
# def compute_cross_correlation(x, y):
#     x = x - np.mean(x)
#     y = y - np.mean(y)
#     cross_corr = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * len(x))
#     lags = np.arange(-len(x) + 1, len(x))
#     return cross_corr, lags
#
# def compute_test_statistic(cross_corr, period, n):
#     """计算选定滞后的平方和"""
#     return np.sum(cross_corr[period + (n - 1)]**2)  # 调整滞后索引
#
# def bootstrap_test(x, y, period, n_bootstrap=1000):
#     """Bootstrap 生成 Q 的分布"""
#     n = len(x)
#     stat_samples = []
#     block_size = 6
#     n_blocks = n // block_size
#     for _ in range(n_bootstrap):
#         # 随机选择块并拼接
#         block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
#         indices = np.concatenate([
#             np.arange(i * block_size, (i + 1) * block_size) for i in block_indices
#         ])
#         x_boot = x[indices[:n]]  # 确保长度匹配
#         y_boot = y[indices[:n]]
#
#         # 计算互相关
#         cross_corr_boot, _ = compute_cross_correlation(x_boot, y_boot)
#         stat = compute_test_statistic(cross_corr_boot, period, n)
#         stat_samples.append(stat)
#
#     return np.array(stat_samples)

