import pandas as pd
import matplotlib.pyplot as plt

start_date = '2011'
end_date = '2013'
nzd_file = pd.read_csv(f"../derived_data/nzd_results_{start_date}_{end_date}.csv")
quadrant_values = sorted(nzd_file["quadrant"].unique())

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
axs = ax.flatten()

for i, quad in enumerate(quadrant_values):
    data_q = nzd_file[nzd_file["quadrant"] == quad]
    axs[i].plot(data_q["nzd_name"], data_q["Max_Correlation"], marker='o', markersize=5)
    axs[i].set_title(f"Quadrant {quad}")
    axs[i].set_ylabel("Max Correlation")
    axs[i].tick_params(axis='x', rotation=45)
    for x,y,z in zip(data_q["nzd_name"], data_q["Max_Correlation"], data_q["p-value"]):
        axs[i].annotate(f'{y:.2f}',
                        xy=(x, y),
                        xytext=(0, 5),
                        textcoords='offset points',
                        horizontalalignment='center')


fig.savefig(f"../fig/Coastline_{start_date}_{end_date}.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()