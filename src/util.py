import numpy as np


def minmax_normalization(x):
    x_min = min(x)
    x_max = max(x)
    return [(e-x_min)/(x_max-x_min) for e in x]
