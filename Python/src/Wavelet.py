import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataProcess import *
from Crosscorrelation import *

# -----------------------------
# 1) 读取 + 对齐 + 插值
# -----------------------------
def load_align_interpolate(
        coastal_csv, SOI_csv, value,
        date_col="Year-Month",
        freq="MS", # 月初频率：Month Start
        interpolate_method="random_linear_noise", # "linear" 或 "random_linear_noise"
        seed=1
):

    X = pd.read_csv(coastal_csv)
    Y = pd.read_csv(SOI_csv)
    X = X.iloc[:, 0:2]
    Y = Y.iloc[:, 0:2]
    X[date_col] = pd.to_datetime(X[date_col])
    Y[date_col] = pd.to_datetime(Y[date_col])

    full_dates = pd.date_range(Y[date_col].min(), Y[date_col].max(),freq=freq)
    base = pd.DataFrame(full_dates, columns=["Year-Month"])


    base = base.merge(Y, how="left", on=date_col)
    base = base.merge(X, how="left", on=date_col)
    base = base.rename(
        columns={
            "monthly_anomaly": "coastline_change",
            "Value": "SOI"
        }
    )
    base["Year-Month"] = base["Year-Month"].dt.strftime("%Y/%m")
    x_col = "coastline_change"
    if interpolate_method == "linear":
        base[x_col] = base[x_col].interpolate(method="linear", limit_direction="both")

    elif interpolate_method == "random_linear_noise":
        rng = np.random.default_rng(seed)
        missing = base[x_col].isna()

        baseline = base[x_col].interpolate(method="linear", limit_direction="both")

        # 相邻观测差分的波动作为噪声依据
        obs = base.loc[~missing, x_col].values  # 原始观测点
        if len(obs) >= 3:
            sigma = np.nanstd(np.diff(obs))  # 用相邻观测变化估计波动
        else:
            sigma = np.nanstd(obs)

        if not np.isfinite(sigma) or sigma == 0:
            sigma = np.nanstd(baseline.values)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0

        noise = rng.normal(0, sigma, size=missing.sum())
        baseline.loc[missing] = baseline.loc[missing].values + noise
        base[x_col] = baseline

    else:
        raise ValueError("interpolate_method must be 'linear' or 'random_linear_noise'")

    base_std = base.copy()
    cols = [ "SOI","coastline_change"]

    base_std[cols] = (base_std[cols] - base_std[cols].mean()) / base_std[cols].std(ddof=0)
    print(base_std[cols].mean())
    print(base_std[cols].std(ddof=0))

    return base_std

names, quadrant, nzd_file = extract_filename()
start_date = '2010'
end_date = '2014'
soi_path = f'../derived_data/Results/{start_date}_{end_date}/SOI_smooth.csv'
i = 1
for index, coastline in names.items():
    coastal_path = f'../derived_data/Results/{start_date}_{end_date}/{coastline}/{coastline}_monthly.csv'

    base_std = load_align_interpolate(coastal_path,soi_path, coastline)

    wavelet_dir = f"../wavelet_data/{start_date}_{end_date}/"
    os.makedirs(wavelet_dir, exist_ok=True)
    base_std.to_csv(f"{wavelet_dir}/{coastline}_wavelet.csv", index=False)

