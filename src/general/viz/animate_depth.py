"""View a kinematic wave depth-vs-time table as an animation.

Reads a CSV produced by general.solvers.linear_advection.save_time_series_csv
(header row: t, x_0, x_1, ...; one row per recorded time) and animates
u(x) over time, one frame per recorded timestamp.

Usage:
    python src/general/viz/animate_depth.py [path/to/timeseries.csv]

With no argument, reads data/linear_advection_timeseries.csv.
"""
import csv
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PATH = "data/linear_advection_timeseries.csv"
DEFAULT_PATH = "data/saint_venant_1d_timeseries.csv"


def load_time_series(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        x = np.array([float(v) for v in header[1:]])
        times = []
        u_history = []
        for row in reader:
            times.append(float(row[0]))
            u_history.append([float(v) for v in row[1:]])
    return x, np.array(times), np.array(u_history)


def animate(x, times, u_history):
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, u_history[0])
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, u_history.max() * 1.1)
    title = ax.set_title(f"t = {times[0]:.2f}")

    def update(frame):
        line.set_ydata(u_history[frame])
        title.set_text(f"t = {times[frame]:.2f}")
        return line, title

    return animation.FuncAnimation(
        fig, update, frames=len(times), interval=150, repeat=True
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    x, times, u_history = load_time_series(path)
    # Keep a reference to anim -- FuncAnimation stops if garbage collected.
    anim = animate(x, times, u_history)
    plt.show()
