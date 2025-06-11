import numpy as np


def minmax_normalization(x):
    x_min = min(x)
    x_max = max(x)
    x_norm = []
    for e in x:
        if x_max != 0:
            x_norm.append((e-x_min)/(x_max-x_min))
        else: 
            x_norm.append(0)
    return x_norm
