import numpy as np
from scipy.stats import pearsonr

def calc_cross_corr(x, y):
    n = len(x)
    lags = range(0, 4)
    r_list = []
    p_list = []

    x = np.array(x)
    y = np.array(y)

    for lag in lags:
        if lag >= n:
            r_list.append(np.nan)
            p_list.append(np.nan)
            continue

        x_seg = x[lag:]
        y_seg = y[0: n - lag]

        if len(x_seg) < 2:
            r_list.append(np.nan)
            p_list.append(np.nan)
            continue

        r, p_value = pearsonr(x_seg, y_seg)

        r_list.append(r)
        p_list.append(p_value)

    max_index = np.nanargmax(np.abs(r_list))
    max_lag = lags[max_index]
    max_corr = r_list[max_index]
    max_p = p_list[max_index]

    return max_lag, max_corr, max_p,  np.array(list(lags)),np.array(r_list), np.array(p_list)