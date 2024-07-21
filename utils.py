
import matplotlib.pyplot as plt
import params
import pandas as pd
import numpy as np
def py_ang(point_a, point_b):
    if len(np.ravel(point_a)) == 2:
        x = point_a[0]
        y = point_a[1]
    else:
        x = [row[0] for row in point_a]
        y = [row[1] for row in point_a]

    ang_a = np.arctan2(y, x)

    if len(np.ravel(point_b)) == 2:
        x = point_b[0]
        y = point_b[1]
    else:
        x = [row[0] for row in point_b]
        y = [row[1] for row in point_b]

    ang_b = np.arctan2(y, x)
    return np.rad2deg((ang_a - ang_b) % (2 * np.pi))

def get_angle(A, B, C, centered_filtered, pos=None):
    coords_ids = params.coords_ids
    A = str(coords_ids[params.keys.index(A)])
    B = str(coords_ids[params.keys.index(B)])
    C = str(coords_ids[params.keys.index(C)])
    p_A = np.array([centered_filtered.loc[:, "x" + A].values, centered_filtered.loc[:, "y" + A].values]).T
    p_B = np.array([centered_filtered.loc[:, "x" + B].values, centered_filtered.loc[:, "y" + B].values]).T
    p_C = np.array([centered_filtered.loc[:, "x" + C].values, centered_filtered.loc[:, "y" + C].values]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
    return py_ang(p_BA, p_BC)

def min_max(min, max, hL, hR):
    min = np.minimum(min, np.min(hL))
    min = np.minimum(min, np.min(hR))
    max = np.maximum(max, np.max(hL))
    max = np.maximum(max, np.max(hR))
    return min, max

def calc(df, min, max):
    histogram_L = get_angle("left hip", "left knee", "left ankle", df)
    histogram_R = get_angle("left wrist", "left elbow", "left shoulder", df)
    histogram_knees = get_angle("left knee", "left ankle", "left heel", df)
    min, max = min_max(min, max, histogram_L, histogram_R)
    min, max = min_max(min, max, histogram_knees, histogram_knees)
    return histogram_L, histogram_R, histogram_knees, min, max