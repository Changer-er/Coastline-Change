import os.path
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2010'
end_date = '2014'
nzd_file = pd.read_csv(f"../derived_data/Results/{start_date}_{end_date}/Coastline_summary.csv")
quadrant_values = sorted(nzd_file["quadrant"].unique())
directions_values = sorted(nzd_file["location"].unique())

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
axs = ax.flatten()
output = f"../derived_data/Figure/{start_date}_{end_date}_Summary/"
os.makedirs(output, exist_ok=True)
# 绘制不同象限的结果
for i, quad in enumerate(quadrant_values):
    data_q = nzd_file[nzd_file["quadrant"] == quad]
    data_q = data_q[data_q["p-value"] <= 0.05]

    axs[i].plot(data_q["nzd_name"], data_q["Max_Correlation"], marker='o', markersize=5)
    axs[i].set_title(f"Quadrant {quad}", pad=15)
    axs[i].set_ylabel("Max Correlation")
    axs[i].tick_params(axis='x', rotation=45)
    for x,y,z in zip(data_q["nzd_name"], data_q["Max_Correlation"], data_q["Max_lag"]):
        axs[i].annotate(f'r={y:.2f}\n lag={int(z)}',
                        xy=(x, y),
                        xytext=(0, 2),
                        textcoords='offset points',
                        horizontalalignment='center')
plt.tight_layout()
os.path.join(output,"Coastline_quadrant.png")
fig.savefig(os.path.join(output,"Coastline_quadrant.png"), dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(12, 8))
axs = ax.flatten()

# 绘制不同方向海岸的互相关性
for i, dirs in enumerate(directions_values):
    data_dirs = nzd_file[nzd_file["location"] == dirs]
    data_dirs = data_dirs[data_dirs["p-value"] <= 0.05]

    axs[i].plot(data_dirs["nzd_name"], data_dirs["Max_Correlation"], marker='o', markersize = 5)
    axs[i].set_title(f"Direction {dirs}", pad=15)
    axs[i].set_ylabel("Max Correlation")
    axs[i].tick_params(axis='x', rotation=45)
    for x, y, z in zip(data_dirs["nzd_name"], data_dirs["Max_Correlation"], data_dirs["Max_lag"]):
        axs[i].annotate(f'r={y:.2f}\n lag={int(z)}',
                        xy=(x, y),
                        xytext=(0, 2),
                        textcoords='offset points',
                        horizontalalignment='center')
plt.tight_layout()

fig.savefig(os.path.join(output,"Coastline_location.png"), dpi=300, bbox_inches='tight')
plt.show()

# 绘制不同方向海岸的lag
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
axs = ax.flatten()

for i, dirs in enumerate(directions_values):
    data_dirs = nzd_file[nzd_file["location"] == dirs]
    data_dirs = data_dirs[data_dirs["p-value"] <= 0.05]

    axs[i].plot(data_dirs["nzd_name"], data_dirs["Max_lag"], marker='o', markersize = 5)
    axs[i].set_title(f"Direction {dirs}", pad=15)
    axs[i].set_ylabel("Max lag")
    axs[i].tick_params(axis='x', rotation=45)
    for x, z, y in zip(data_dirs["nzd_name"], data_dirs["Max_Correlation"], data_dirs["Max_lag"]):
        axs[i].annotate(f'lag={int(y)}',
                        xy=(x, y),
                        xytext=(0, 2),
                        textcoords='offset points',
                        horizontalalignment='center')
plt.tight_layout()

fig.savefig(os.path.join(output,"Coastline_lag.png"), dpi=300, bbox_inches='tight')
plt.show()