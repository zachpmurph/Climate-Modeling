import csv

import numpy as np
import matplotlib.pyplot as plt

# Units: meters and minutes throughout
L = 10.0
T_final = 300.0
S0 = 0.05
n0 = 0.05
g = 35316.0   # 9.81 m/s^2 converted: 9.81 * 60^2 = 35316 m/min^2


def r(x, t):
    if 0 <= t < 50:
        return 0.00002 * np.ones(len(x))
    return np.zeros(len(x))


def run_model(L, T_final, record_interval=1.0, h_init=None, q_init=None):
    raise NotImplementedError


def save_time_series_csv(result, path):
    raise NotImplementedError


if __name__ == "__main__":
    pass
