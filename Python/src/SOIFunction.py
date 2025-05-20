import pandas as pd
import matplotlib.pyplot as plt
from smooth_algorithms.smoothn import smoothn

# def plot_soi_data(csv_path):
#     """
#     读取 SOI 数据并绘制折线图。
#
#     参数：
#     - file_path: str，TXT 数据文件路径
#     - start_year: int，起始年份
#     - end_year: int，结束年份
#     """
#     # 重新读取 CSV
#     df = pd.read_csv(csv_path)
#
#     # 2. 将数据转换成长格式
#     df_melted = df.melt(id_vars=["YEAR"], var_name="Month", value_name="Value")
#
#     # 3. 确保月份按正确顺序排列, 修改月份为category类型，具有顺序特性
#     month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
#     df_melted["Month"] = pd.Categorical(df_melted["Month"], categories=month_order, ordered=True)
#
#     # 4. 创建时间序列列并进行排序，通过sort_values进行排序
#     df_melted["Date"] = pd.to_datetime(df_melted["YEAR"].astype(str) + "-" + df_melted["Month"].astype(str),format="%Y-%b")
#     df_melted = df_melted.sort_values(by="Date")
#     filepath = '../derived_data/SOI_data_filtered.csv'
#     df_melted.to_csv("../derived_data/SOI_data_filtered.csv", index=False)
#
#     return filepath
#     # 8.画图操作
    # plt.figure(figsize=(16, 5))
    # plt.plot(df_melted["Date"], soi_smooth, marker='o', linestyle='-', color='b', label="Value")
    # plt.xlabel("Year")
    # plt.ylabel("Value")
    # plt.title(f"Monthly Data from {start_date} to {end_date}")
    # plt.xticks(rotation=45)  # X轴标签旋转，防止重叠
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Plot the line chart separately
    #获得月份
    # months = data.columns[1:]
    # # 筛选指定年份区间的数据
    # filtered_data = data[(data["YEAR"] >= start_year) & (data["YEAR"] <= end_year)].reset_index(drop=True)
    # num_rows = len(filtered_data)

    # # 如果没有符合条件的数据
    # if num_rows == 0:
    #     print(f"没有 {start_year}-{end_year} 之间的数据！")
    #     return
    # # 绘制折线图
    # if num_rows == 1:
    #     fig = plt.figure(figsize=(8, 6))
    #     values = filtered_data.iloc[0, 1:].values
    #     plt.plot(months, values, label=filtered_data["YEAR"])
    #     plt.ylabel('Value')
    #     plt.title(f"SOI Data for {filtered_data['YEAR']}")
    #     plt.legend()
    # else:
    #     fig, axes = plt.subplots(num_rows, 1, figsize=(12, 14))  # 多行子图
    #     for index, row in filtered_data.iterrows():
    #         axes[index].plot(months, row[1:], label=row["YEAR"])
    #         axes[index].set_title(f"SOI Data for {row['YEAR']}")
    #         axes[index].legend(loc='best')
    #         axes[index].set_ylabel("Value")
    #         axes[index].grid(True, linestyle="--", alpha=0.6)  # 添加网格线

    # plt.tight_layout()
    # plt.show()
