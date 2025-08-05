import pandas as pd
import glob
import os
import re

# 文件夹路径，替换为你的路径
folder_path = '../derived_data/'

# 获取所有csv文件路径
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 存储结果的DataFrame列表
merged_df = None
common_cols = ['nzd_name', 'orientation', 'coordinate', 'location', 'y', 'x', 'quadrant']

for file in csv_files:
    filename = os.path.basename(file)
    match = re.search(r'(\d{4}_\d{4})', filename)
    if not match:
        print(f"跳过未找到年份区间的文件: {filename}")
        continue
    year_range = match.group(1)  # 提取例如 1999_2024

    df = pd.read_csv(file)

    # 仅保留p-value <= 0.05的行
    df_filtered = df[df['p-value'] <= 0.05]

    # 仅保留需要的列
    df_filtered = df_filtered[['nzd_name', 'Max_lag', 'Max_Correlation']].copy()
    df_filtered.rename(columns={
        'Max_lag': f'Max_lag_{year_range}',
        'Max_Correlation': f'Max_Correlation_{year_range}'
    }, inplace=True)
    # 添加来源信息（可选）
    # df_selected['source_file'] = os.path.basename(file)

    if merged_df is None:
        merged_df = df_filtered
    else:
        merged_df = pd.merge(merged_df, df_filtered, on='nzd_name', how='outer')

if merged_df is not None:
    merged_df.to_csv('../derived_data/merged_by_year_range.csv', index=False)
