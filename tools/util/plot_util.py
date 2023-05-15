import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

def plot_line(x_data, y_data, std_data=[], label='', line_style=None, color=None, draw_band=True):
    x_data = x_data if isinstance(x_data, list) else [x_data] 
    y_data = y_data if isinstance(y_data, list) else [y_data] 
    std_data = std_data if isinstance(std_data, list) else [std_data] 

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    min_len = int(reduce(lambda x, y: np.minimum(x, len(y)), x_data, np.inf))

    if draw_band:
        x_data = list(map(lambda x: x[:min_len], x_data))
        y_data = list(map(lambda x: x[:min_len], y_data))
        std_data = list(map(lambda x: x[:min_len], std_data))
    
        xs = np.mean(x_data, axis=0)
        ys = np.mean(y_data, axis=0)
        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)

        stds = None
        if (len(y_data) > 1):
            stds = np.std(y_data, axis=0)
        elif (len(std_data) > 0):
            stds = np.mean(std_data, axis=0)

        curr_line = plt.plot(xs, ys, label=label, linestyle=line_style, color=color)

        if stds is not None:
            plt.fill_between(xs, ys - stds, ys + stds, alpha=0.25, linewidth=0,
                             facecolor=curr_line[0].get_color(), label='_nolegend_')
    else:
        for i in range(len(x_data)):
            alpha = 0.8 if (len(x_data) > 1) else 1.0
            xs = x_data[i]
            ys = y_data[i]
            
            min_x = np.minimum(np.min(xs), min_x)
            max_x = np.maximum(np.max(xs), max_x)
            min_y = np.minimum(np.min(ys), min_y)
            max_y = np.maximum(np.max(ys), max_y)

            if (i == 0):
                curr_line = plt.plot(xs, ys, label=label, alpha=alpha, linestyle=line_style, color=color)
            else:
                curr_line = plt.plot(xs, ys, color=curr_line[0].get_color(), linestyle=line_style, label='_nolegend_', alpha=alpha)

            if len(std_data) > 0:
                stds = std_data[i]
                plt.fill_between(xs, ys - stds, ys + stds, alpha=0.25, linewidth=0,
                                 facecolor=curr_line[0].get_color(), label='_nolegend_')

    return min_x, max_x, min_y, max_y