import numpy as np
import pandas as pd
import os

filePath = f'../Coastline_data/nzd0100/transect_time_series_tidally_corrected.csv'
df_nzd = pd.read_csv(filePath)

start_date = '2007-01-01'
end_date = '2016-12-31'
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
start_year = start_date.year
end_year = end_date.year

coastName = f'nzd_0100_Average_Value'
output_dir = f'../derived_data/{start_year}_{end_year}/nzd0100/'

os.makedirs(output_dir, exist_ok=True)
print(df_nzd.iloc[:,-1])
df_nzd[coastName] = df_nzd.iloc[:,1:-1].mean(axis=1)
nzd_filter = df_nzd.iloc[:, [0, -1]]  # 只保留第一列（日期）和最后一列（平均值）
#
# nzd_filter.to_csv(os.path.join(output_dir, f'nzd0132_transects_average.csv'), index=False)
nzd_file = pd.read_csv("../raw_data/nzd group.CSV")
rows = nzd_file.iloc[0,:]
print(rows)
