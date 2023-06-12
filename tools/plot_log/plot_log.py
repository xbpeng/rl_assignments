import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce

import sys
sys.path.append(os.getcwd())

import tools.util.plot_util as plot_util

files = [
    "output/log.txt",
]

draw_band = True
x_key = "Samples"
y_key = "Test_Return"
plot_title = "BC - Cheetah"
std_key = None

def filter_data(x, window_size):
    n = len(x)
    filter_n = n // window_size
    x = x[:filter_n * window_size]
    x = np.reshape(x, [filter_n, window_size])
    filter_x = np.mean(x, axis=-1)
    return filter_x

plt.figure(figsize=(5.5 * 0.8, 4 * 0.8))

min_x = np.inf
max_x = -np.inf
filter_window_size = 1

for f, file_group in enumerate(files):
    if not isinstance(file_group, list):
        if os.path.isdir(file_group):
            files = os.listdir(file_group)
            files = filter(lambda f: "log" in f, files)
            file_group = list(map(lambda f: file_group + "/" + f, files))
        else:
            file_group = [file_group]

    x_data = []
    y_data = []
    std_data = []
    num_files = len(file_group)

    for file in file_group:
        with open(file, "r") as file_data:
            clean_lines = [line.replace(",", "\t") for line in file_data]
            data = np.genfromtxt(clean_lines, delimiter=None, dtype=None, names=True)
        
        data_x_key = x_key
        data_y_key = y_key
        curr_window_size = filter_window_size

        if data_x_key in data.dtype.names and data_y_key in data.dtype.names:
            xs = data[data_x_key]
            ys = data[data_y_key]
            xs = filter_data(xs, curr_window_size)
            ys = filter_data(ys, curr_window_size)

            stds = None
            if std_key is not None and std_key in data.dtype.names:
                stds = data[std_key]
                stds = filter_data(stds, curr_window_size)
                std_data.append(stds)

            x_data.append(xs)
            y_data.append(ys)

    label = os.path.basename(file_group[0])
    label = os.path.splitext(label)[0]

    line_col = None
    curr_min_x, curr_max_x, _, _ = plot_util.plot_line(x_data, y_data, std_data, label, color=line_col,
                                                      draw_band=draw_band)

    min_len = int(reduce(lambda x, y: np.minimum(x, len(y)), x_data, np.inf))
    x_final = x_data[0][min_len - 1]
    y_final = np.array([y[min_len - 1] for y in y_data])
    print("Final value: {:.2f}, {:.5f} +/- {:.5f}".format(x_final, np.mean(y_final), np.std(y_final)))
    
    min_x = min(curr_min_x, min_x)
    max_x = max(curr_max_x, max_x)


ax = plt.gca()

plt.xlabel(x_key)
plt.ylabel(y_key)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.grid(linestyle='dotted')
ax.xaxis.grid(True)
ax.yaxis.grid(True)

plt.legend()
plt.title(plot_title)
plt.tight_layout()

plt.show()