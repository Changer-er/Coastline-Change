import glob
import numpy as np
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from DataProcess import *

coastal_list, quadrant, nzd_file = extract_filename()

series_dict = {}
all_dates = []
start_year = int(input("Start year (e.g., 2010): ").strip())
end_year   = int(input("End year   (e.g., 2023): ").strip())

# 1) 读入每条海岸线
for i, CoastlineName in coastal_list.items():

    filepath = f'../derived_data/Results/{start_year}_{end_year}/{CoastlineName}/{CoastlineName}_monthly.csv'

    df = pd.read_csv(filepath)
    d = pd.to_datetime(df["Year-Month"] + "-01")  # 'YYYY-MM' -> 日期
    s = pd.Series(df["monthly_anomaly"].astype(float).values, index=d).sort_index() # reindex by data, get data series "s"
    series_dict[CoastlineName] = s # add data into dictionary
    all_dates.append(s.index) # Generate data list

# 2) 全局月时间轴
start = min([idx.min() for idx in all_dates])
end   = max([idx.max() for idx in all_dates])
full_index = pd.date_range(start, end, freq="MS")

# 3) 对齐 + 缺失处理（插值 + 兜底填充）
X_list, names = [], []
for name, s in series_dict.items():

    # 数据按照full time排列，s有原来的数据保留，没有原来的数据为Nan
    s2 = s.reindex(full_index)

    # 先做时间插值（只在内部缺口有效） 按“真实天数比例”分配权重, 由于时间间隔一样，所以和 method = linear 一样
    s2 = s2.interpolate(method="time", limit_direction="both")

    # 仍有缺失的话（比如整段缺），用该序列均值兜底
    if s2.isna().any():
        s2 = s2.fillna(s2.mean())

    X_list.append(s2.values)
    names.append(name)

X = np.vstack(X_list)  # shape: (n_transects, n_months)

# 4) 选择：按形状聚类（每条序列 z-score）
X_shape = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

# 5) 试不同 k，用 silhouette 选一个“相对合理”的 k
best_k, best_sc = None, -1
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    lab = km.fit_predict(X_shape)
    sc = silhouette_score(X_shape, lab)
    print("k=", k, "silhouette=", round(sc, 3))
    if sc > best_sc:
        best_k, best_sc = k, sc

print("Best k:", best_k, "score:", best_sc)

# 6) 用最佳 k 重新拟合，得到每条海岸线所属簇
km = KMeans(n_clusters=best_k, random_state=0, n_init="auto")
labels = km.fit_predict(X_shape)

result = pd.DataFrame({"transect": names, "cluster": labels})
print(result.sort_values("cluster"))
