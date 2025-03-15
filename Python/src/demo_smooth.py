from smooth_algorithms.smoothn import smoothn
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from smooth_algorithms.smoothn import smoothn, test1


# x = np.linspace(0, 100, 2 ** 8)
# y = cos(x / 10) + (x / 50) ** 2 + np.random.rand(size(x)) / 10
# y[[70, 75, 80]] = [5.5, 5, 6]
# z = smoothn(y)[0]  # Regular smoothing
# [z_smoothed,s] = smoothn(y)
# zr = smoothn(y, isrobust=True)[0]  # Robust smoothing
# print([z,s])
# # plt.plot(y, 'o', label='Original data')
# plt.plot(z, '-', label='reg Smoothed')
# plt.plot(zr, '--', label='robust Smoothed')
# plt.legend()
# plt.show()

t = np.linspace(0, 2 * np.pi, 1000)
x = 2 * np.cos(t) * (1 - np.cos(t)) + np.random.randn(len(t)) * 0.1
y = 2 * np.sin(t) * (1 - np.cos(t)) + np.random.randn(len(t)) * 0.1
z = smoothn(x + 1j * y, s = 40)[0]
s = smoothn(x + 1j * y)[1]
print(s)
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'r.', label="Noisy Data", alpha=0.5)
plt.plot(np.real(z), np.imag(z), 'k', linewidth=2, label="Smoothed Curve")
plt.axis('equal')  # 让坐标轴保持比例
plt.tight_layout()
plt.legend()
plt.show()