from IPython.core.pylabtools import figsize

from DataProcess import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from smooth_algorithms.smoothn import smoothn
filepath = '../derived_data/Combined_raw_nzd_188_soi.csv'
coastline_month_avg = pd.read_csv(filepath)
coast_average = f'nzd_188_Average_Value'
coastline_month_avg["Year-Month"] = pd.to_datetime(coastline_month_avg["Year-Month"])
coastline_month_avg["Year"] = coastline_month_avg["Year-Month"].dt.year
coastline_month_avg["Month"] = coastline_month_avg["Year-Month"].dt.month
# MEI calculation method
# OND_data = coastline_month_avg[coastline_month_avg["Month"].isin([10, 11, 12])]
# SOI_OND_mean = OND_data.groupby("Year")["Value"].mean().reset_index()
# soi_std = SOI_OND_mean["Value"].std()
# def classify_enso(soi, std):
#     if soi > 0.5 * std:
#         return "La Niña"
#     elif soi < -0.5 * std:
#         return "El Niño"
#     else:
#         return "Neutral"
#
# SOI_OND_mean["ENSO_phase"] = SOI_OND_mean["Value"].apply(lambda x: classify_enso(x, soi_std))

# SOI calculation method

def classify_enso(soi):
    if soi >= 1:
        return "La Niña"
    elif soi <= -1:
        return "El Niño"
    else:
        return "Neutral"

coastline_month_avg["ENSO_phase"] = coastline_month_avg["Value"].apply(lambda x: classify_enso(x))
print(coastline_month_avg)
# Smooth the soi data
coastline_smooth_avg = coastline_month_avg[coast_average].mean()
coastline_month_avg["Bias"] = coastline_month_avg[coast_average] - coastline_smooth_avg
coast_smooth_bias = smoothn(coastline_month_avg["Bias"].values)[0]
s = smoothn(coastline_month_avg["Bias"].values)[1]
coastline_month_avg["Bias_smooth"] = coast_smooth_bias
print(f"Original parameter s = {s}")
print(coastline_month_avg.head())
# Combined_data = pd.merge(SOI_OND_mean[["Year", "ENSO_phase"]],coastline_month_avg, on = "Year", how = "inner")
# Combined_data.to_csv(f'../derived_data/Combined_{filename}_bias.csv', index=False)
Lanina_phrase = coastline_month_avg[(coastline_month_avg["Year-Month"]>="2008-01-01") & (coastline_month_avg["Year-Month"]<="2009-12-31")]
Elnino_phrase = coastline_month_avg[(coastline_month_avg["Year-Month"]>="2015-01-01") & (coastline_month_avg["Year-Month"]<="2016-12-31")]
print(f"La Niña phrase : {Lanina_phrase["Year-Month"].unique()}")
print(f"El Niño phrase : {Elnino_phrase["Year-Month"].unique()}")
coastline_month_avg.to_csv()
numerical_data = coastline_month_avg.select_dtypes(include=['float64'])
# correlation = numerical_data.corr()# 计算皮尔逊相关系数
x = numerical_data["Bias_smooth"].values
y = numerical_data['Smooth_Value'].values
cross_corr = np.correlate(x - np.mean(x), y - np.mean(y), mode="full") / (np.std(x) * np.std(y) * len(x))
# 找到最大相关性
lag = np.argmax(cross_corr) - (len(x) - 1)
print("=======================================================")
print("Maximum correlation lag:", lag)
print("Maximum cross-correlation coefficient:",max(cross_corr))

# Plot the graph for Bias, SOI value in ENSO phrase
fig, axes = plt.subplots(2,2,sharey=False, figsize = (16, 8))
plt.grid(True,linestyle = "-", alpha = 0.5, color = "black")

axes[0,0].plot(coastline_month_avg["Year-Month"], coastline_month_avg["Bias"], data = coastline_month_avg, label = "Coastline bias", color = "black")
axes[0,0].set_title("Bias change over time")
axes[0,0].set_xlabel("Year")
axes[0,0].set_ylabel("Bias")

axes[0,1].plot(coastline_month_avg["Year-Month"], coastline_month_avg["Smooth_Value"], data=coastline_month_avg, color="r")
axes[0,1].set_ylabel("Soi change over time")
axes[0,1].set_xlabel("Year")
axes[0,1].set_title("The smooth of bias and SOI change over time")
for i in range(1, len(coastline_month_avg)):
    x1, x2 = coastline_month_avg["Year-Month"].iloc[i - 1], coastline_month_avg["Year-Month"].iloc[i]
    y1, y2 = coastline_month_avg["Smooth_Value"].iloc[i - 1], coastline_month_avg["Smooth_Value"].iloc[i]

    # 如果 y1 和 y2 同号
    if y1 > 0 and y2 > 0:
        axes[0,1].fill_between([x1, x2], [y1, y2], 0, color='lightcoral')
    elif y1 < 0 and y2 < 0:
        axes[0,1].fill_between([x1, x2], [y1, y2], 0, color='lightblue')

    # 如果跨越了 0（插值点计算）
    elif y1 * y2 < 0:
        # 插值得到穿过 y=0 的交点横坐标
        ratio = abs(y1) / (abs(y1) + abs(y2))
        x0 = x1 + (x2 - x1) * ratio

        # 分成两段填色
        if y1 > 0:
            axes[0,1].fill_between([x1, x0], [y1, 0], 0, color='lightcoral')
            axes[0,1].fill_between([x0, x2], [0, y2], 0, color='lightblue')
        else:
            axes[0,1].fill_between([x1, x0], [y1, 0], 0, color='lightblue')
            axes[0,1].fill_between([x0, x2], [0, y2], 0, color='lightcoral')

# ax2 = axes[0,1].twinx()
# ax2.plot(coastline_month_avg["Year-Month"], coastline_month_avg["Bias_smooth"], data = coastline_month_avg, label = "Coastline bias", color = "black")
# ax2.set_ylabel("Bias change over time")
# color_map = {
#     'El Niño' : 'red',
#     'La Niña' : 'blue',
#     'Neutral' : 'gray'
# }
# colors = coastline_month_avg['ENSO_phase'].map(color_map).tolist()
# print(colors)
axes[1,0].plot(Lanina_phrase["Year-Month"],
              Lanina_phrase["Bias_smooth"], data = Lanina_phrase)
axes[1,0].set_ylabel("Bias change")
axes[1,0].set_title("Bias change during Lanina phrase")
axes[1,0].set_xlabel("Year")

axes[1,1].plot(Elnino_phrase["Year-Month"],
              Elnino_phrase["Bias_smooth"], data = Elnino_phrase)
axes[1,1].set_ylabel("Bias change ")
axes[1,1].set_title("Bias change during Elnino phrase")
axes[1,1].set_xlabel("Year")

for ax in axes.flat:
    ax.grid(True, linestyle = "-", alpha = 0.5, color = "black")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
