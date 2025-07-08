import numpy as np
import pandas as pd
import os
from DataProcess import *
from scipy.stats import pearsonr

filePath = f'../Coastline_data/nzd0046/transect_time_series_tidally_corrected.csv'

filename = f'nzd0046'
start_date = '2007-01-01'
end_date = '2016-12-31'

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
start_year = start_date.year
end_year = end_date.year

s = 4
print(f"=======================================SOI Value====================================")
print(f'The Smoothing Parameters we set: "s"= {s} ')
SOI_before_smooth, SOI_smooth = preprocess_soi(start_date, end_date, s)

nzd_filter, coastValueName= filter_data(filePath, filename, start_year, end_year)

# Get the hard copy of nzd filtered data for plotting
nzd_filter['dates'] = pd.to_datetime(nzd_filter['dates'])
nzd_orig = nzd_filter.copy()

# Smooth the filtered data firstly and calculate the average of nzd data on a monthly basis
nzd_monthly, nzd_smooth = calc_mean_monthly(nzd_filter, start_date, end_date, s)

# merge the both data, Leave out months that don't exist
merged_data = merge_nzd_soi(nzd_smooth,SOI_smooth)
merged_data.to_csv()
numerical_data = merged_data.select_dtypes(include=['float64'])

merged_data["Year-Month"] = merged_data["Year-Month"].dt.to_timestamp()


# 1. Calculate correlation
x = numerical_data[f'{filename}_Average_Value'].values
y = numerical_data['Smooth_Value'].values
(max_lag, max_corr, max_select_lag, max_select_corr,
lags, cross_corr, select_lags_x, select_lags_y) = calc_cross_correlation(x, y)
#
# 2. Define the test statistic (sum of squares)
period = np.arange(0, 12)
observed_stat = compute_test_statistic(cross_corr, period, len(x))

# 3. Bootstrap joint inspection
np.random.seed(42)
bootstrap_stats = bootstrap_test(x, y, period, n_bootstrap=5000)

# 4. 计算 p 值
p_value = np.mean(bootstrap_stats >= observed_stat)

x2 = x[max_select_lag: len(x)]
y2 = y[0: len(x) - max_select_lag]
cc, p = pearsonr(y2, x2)

print(f"======================================={filename}====================================")
print(f"pearsonr cross-correlation: {cc:.4f}")
print(f"p-value: {p:.4f}")
print(f"Bootstrap p-value: {p_value:.4f}")
print("- Maximum correlation lag:", max_select_lag)
print("- Maximum cross-correlation coefficient:",max_select_corr)


# print(f"Observed test statistic (Q): {observed_stat:.4f}")
# print(f"Bootstrap test statistic (Q): {bootstrap_stats}")
# print(f"Bootstrap p-value: {p_value:.4f}")

