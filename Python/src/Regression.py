import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import openpyxl
import matplotlib.pyplot as plt
df = pd.read_excel("../Trend_data/nzd0121.xlsx")
df90 = df[['dates', 'nzd0121-0090']].copy()
df90 = df90.dropna(subset=['nzd0121-0090'])

df90['dates'] = pd.to_datetime(df90['dates'])

# 假设有 DataFrame: df，包含 time 和 coastline
X = (df90['dates']  - df90['dates'].min()).dt.days.values / 365
X = X.reshape(-1, 1)

df90['nzd0121-0090'] = df90['nzd0121-0090'] - df90['nzd0121-0090'].mean()
y = df90['nzd0121-0090'].values

model = LinearRegression().fit(X, y)
slope = model.coef_[0]        # 斜率
intercept = model.intercept_  # 截距
trend = model.predict(X)
detrend = y - trend

df90['nzd0121-0090'] = detrend

plt.figure(figsize=(12, 8))
plt.plot(df90['dates'], df90['nzd0121-0090'], color='red', marker = 'o')
plt.title("nzd change over time")
plt.xlabel("Year-Month")
plt.ylabel("nzd0121-0090")
plt.grid(True)
plt.show()
print(f"斜率: {slope}")
print(f"截距: {intercept}")
