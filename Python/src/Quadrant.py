import pandas as pd
import matplotlib.pyplot as plt

start_date = '1999'
end_date = '2024'
nzd_file = pd.read_csv(f"../derived_data/nzd_results_{start_date}_{end_date}.csv")
quadrant_values = sorted(nzd_file["quadrant"].unique())

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
axs = ax.flatten()

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

fig.savefig(f"../fig/Coastline_{start_date}_{end_date}.png", dpi=300, bbox_inches='tight')

plt.show()