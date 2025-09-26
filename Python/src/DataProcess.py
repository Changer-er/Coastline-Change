import numpy as np
import pandas as pd
import os
from smooth_algorithms.smoothn import smoothn
import openpyxl
from sklearn.linear_model import LinearRegression

#读取soi数据，筛选时间范围，导出备份，对soi数据进行平滑处理, 导出平滑后的数据集
def preprocess_soi(start_date, end_date, s):
    df_melted = sort_SOI('../raw_data/SOI_value.csv')
    df_melted = df_melted[(df_melted["Date"] >= start_date) & (df_melted["Date"] <= end_date)].reset_index(drop=True)
    soi_filter = df_melted.copy()

    # soi_filter.to_csv(f'../derived_data/soi_date.csv', index=False)
    soi_smooth_value = smoothn(soi_filter["Value"].values, s=s)[0]
    soi_s = smoothn(soi_filter["Value"].values)[1]
    print(f'original soi smoothing parameters "s"= {soi_s} ')

    soi_filter["Year-Month"] = soi_filter["Date"].dt.to_period("M")
    SOI_before_smooth = soi_filter.iloc[:, [4, 2]]

    SOI_smooth = SOI_before_smooth.copy()
    SOI_smooth["Smooth_Value"] = soi_smooth_value

    soi_dir = f'../derived_data/Results/{start_date.year}_{end_date.year}/'
    os.makedirs(soi_dir, exist_ok=True)
    SOI_smooth.to_csv( f'../derived_data/Results/{start_date.year}_{end_date.year}/SOI_smooth.csv', index=False)

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

# 去趋势化，减去长期平均值，计算月均异常值
def filter_data(filePath, CoastlineName, start_year, end_year):
    global coast_average, coastline, output_dir
    coastline = CoastlineName

    df_nzd = pd.read_excel(filePath, sheet_name=None)
    Coastline_ts = df_nzd['Intersects']
    Trend_data = df_nzd['Transects']

    # 选出可靠性高的id
    Significant_Trend = Trend_data[Trend_data['r2_score'] > 0.05]
    Significant_ids = Significant_Trend['id']

    #判断是否有显著transects
    if Significant_Trend.empty:
        return  # 退出函数

    #转为datatime64时期格式，计算基于开始时间的年份比例
    Coastline_ts['dates'] = pd.to_datetime(Coastline_ts['dates'], utc=True)


    # 排序显著性的trend的index
    trend = Significant_Trend['trend']
    # 扩展 trend 成与 Coastline_ts[Significant_ids] 同形状的 DataFrame
    trend_df = pd.DataFrame([trend.values] * len(Coastline_ts), columns=Significant_ids)

    # 获得显著性海岸线的dataframe
    Significant_coastline = Coastline_ts[['dates'] + Significant_ids.tolist()].copy()
    # 获得series的长期平均值
    longterm_average = Significant_coastline[Significant_ids].mean(axis=0)

    # 自动广播映射，获得异常值
    Significant_coastline[Significant_ids] = Significant_coastline[Significant_ids] - longterm_average

    # 对显著性的海岸线进行去趋势化
    coastline_detrend = detrend(Significant_coastline)

    coast_average = f'{coastline}_Average_Value'
    #计算的到按月平均异常值
    coastline_detrend[coast_average] = coastline_detrend.iloc[:,1:].mean(axis=1)

    nzd_filter = coastline_detrend.iloc[:, [0, -1]].dropna()  # 只保留第一列（日期）和最后一列（平均值）
    output_dir = f'../derived_data/Results/{start_year}_{end_year}/{coastline}/'
    os.makedirs(output_dir, exist_ok=True)
    #处理异常值
    # Q1 = nzd_filter[coastName].quantile(0.25)
    # Q3 = nzd_filter[coastName].quantile(0.75)
    # IQR = Q3 - Q1
    # nzd_filter = nzd_filter[(nzd_filter[coastName] >= (Q1 - 1.5 * IQR)) &
    #                         (nzd_filter[coastName] <= (Q3 + 1.5 * IQR))]

    nzd_filter.to_csv(os.path.join(output_dir, f'{coastline}_transects_average.csv'), index=False)
    return nzd_filter, coast_average

# LinearRegression to detrend
def detrend(df):
    df['dates'] = pd.to_datetime(df['dates'])

    #循环不同的transects
    for col_name, col_values in df.iloc[:, 1:].items():
        df_col = df[['dates', col_name]].dropna()
        X = (df_col['dates'] - df_col['dates'].min()).dt.days.values / 365.25
        X = X.reshape(-1, 1)
        y = df_col[col_name].values

        model = LinearRegression().fit(X, y)

        dates = (df['dates'] - df['dates'].min()).dt.days.values / 365.25
        dates = dates.reshape(-1, 1)
        trend = model.predict(dates)

        detrend = df[col_name] - trend
        df[col_name] = detrend
    df_detrend = df
    return df_detrend



# 先筛选年份，然后进行平滑处理，计算月平均值
def calc_mean_monthly(nzd_filter, s):

    # 处理数据的日期数据
    nzd_filter['dates'] = pd.to_datetime(nzd_filter['dates'])
    nzd_filter["dates"] = nzd_filter["dates"].dt.tz_localize(None)
    nzd_filter["Year-Month"] = nzd_filter["dates"].dt.to_period("M")  # 只保留年-月

    #计算平滑后的数据
    coast_smooth_value = smoothn(nzd_filter[coast_average].values, s=s)[0]
    coast_smooth = smoothn(nzd_filter[coast_average].values)[1]
    print(f'original coast smoothing parameters "s"= {coast_smooth} ')

    # 计算未平滑处理的月平均值
    nzd_monthly = nzd_filter.groupby("Year-Month")[coast_average].mean().reset_index()

    # 计划平滑处理后的月平均值
    nzd_filter[coast_average] = coast_smooth_value
    nzd_smooth_monthly = nzd_filter.groupby("Year-Month")[coast_average].mean().reset_index()
    nzd_smooth_monthly.to_csv(os.path.join(output_dir, f'{coastline}_smooth_monthly.csv'), index=False)

    return nzd_monthly, nzd_smooth_monthly



def merge_nzd_soi(coast_monthly, SOI_monthly):
    merged_data = pd.merge(coast_monthly, SOI_monthly, on='Year-Month', how='inner')
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
    origin = n - 1

    # 选择在6个月最大correlation处的lag值
    select_lags = lags[origin: origin + 6]
    select_corr = cross_corr[origin: origin + 6]

    max_select_indices = np.argmax(np.abs(select_corr))
    max_select_lag = select_lags[max_select_indices]
    max_select_corr = select_corr[max_select_indices]

    return (max_lag, max_corr, max_select_lag, max_select_corr,
            lags, cross_corr, select_lags, select_corr)


def compute_cross_correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    cross_corr = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * len(x))
    lags = np.arange(-len(x) + 1, len(x))
    return cross_corr, lags

def compute_test_statistic(cross_corr, period, n):
    """计算选定滞后的平方和"""
    return np.sum(cross_corr[period + (n - 1)]**2)  # 调整滞后索引

def bootstrap_test(x, y, period, n_bootstrap=1000):
    """Bootstrap 生成 Q 的分布"""
    n = len(x)
    stat_samples = []
    block_size = 6
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
        cross_corr_boot, _ = compute_cross_correlation(x_boot, y_boot)
        stat = compute_test_statistic(cross_corr_boot, period, n)
        stat_samples.append(stat)

    return np.array(stat_samples)

