import numpy as np

RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi/ 180.0
INVALID_IDX = -1

def lerp(x, y, t):
    return (1 - t) * x + t * y

def log_lerp(x, y, t):
    return np.exp(lerp(np.log(x), np.log(y), t))

def flatten(arr_list):
    return np.concatenate([np.reshape(a, [-1]) for a in arr_list], axis=0)

def flip_coin(p):
    rand_num = np.random.binomial(1, p, 1)
    return rand_num[0] == 1

def add_average(avg0, count0, avg1, count1):
	total = count0 + count1
	return (float(count0) / total) * avg0 + (float(count1) / total) * avg1

def smooth_step(x):
    smooth_x = x * x * x * (x * (x * 6 - 15) + 10)
    return smooth_x