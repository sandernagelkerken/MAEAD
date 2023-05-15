import numpy as np
import pandas as pd

def get_peak_indices(curve):
    max_val = max(curve)
    for i in range(len(curve)):
        if curve[i] == max_val:
            peak_index = i
    peak_indices = [peak_index]
    diff = -1
    i = 1
    while diff < 0:
        peak_indices.append(peak_index+i)
        i += 1
        diff = curve[peak_index+i] - curve[peak_index+i-1]
    diff = 1
    i = -1
    while diff > 0:
        peak_indices.append(peak_index+i)
        i -= 1
        diff = curve[peak_index+i+1] - curve[peak_index+i]
    return np.sort(peak_indices)

def change_peak_height(curve, ratio):
    peak_indices = get_peak_indices(curve)
    new_curve = curve.copy()
    for i in peak_indices:
        new_curve[i] = curve[i]*ratio
    return new_curve

def add_standard_deviation(curve, sd, ratio):
    new_curve = curve.copy()
    for i in range(len(curve)):
        new_curve[i] = new_curve[i]+sd[i]*ratio
    return new_curve

def widen_peak(curve, ratio):
    if ratio < 1:
        raise ValueError("Ratio cannot be smaller than 1.")
    max_val = max(curve)
    for i in range(len(curve)):
        if curve[i] == max_val:
            peak_index = i
    new_curve = curve.copy()
    peak_indices = get_peak_indices(curve)
    start_peak = min(peak_indices)
    end_peak = max(peak_indices)
    peak = []
    for i in peak_indices:
        peak.append(curve[i])
    new_indices = np.linspace(start_peak,end_peak, int(len(peak_indices)*ratio))
    new_peak = np.interp(new_indices, peak_indices, peak)
    max_val = max(new_peak)
    for i in range(len(new_peak)):
        if new_peak[i] == max_val:
            new_peak_index = i
    for i in range(len(new_peak)):
        try:
            new_curve[peak_index-new_peak_index+i] = new_peak[i]
        except IndexError:
            pass
    for i in range(len(curve)):
        if curve[i] > new_curve[i]:
            new_curve[i] = curve[i]
    return new_curve

def add_smooth_noise(curve, amplitude):
    x = np.linspace(0, 2*np.pi*10, len(curve))
    noise = [amplitude*i for i in np.sin(x)]
    new_curve = curve.copy()
    for i in range(len(new_curve)):
        if new_curve[i] >= 1:
            new_curve[i] += noise[i]
    return new_curve

def add_random_noise(curve, sd):
    if sd < 0:
        raise ValueError("Standard deviation cannot be negative.")
    new_curve = curve.copy()
    for i in range(len(new_curve)):
        new_curve[i] = np.random.normal(new_curve[i], sd)
    return new_curve

def change_random_point(curve, increase):
    location = np.random.randint(len(curve))
    new_curve = curve.copy()
    new_curve[location] += increase
    return new_curve

def shift_peak(curve, shifting):
    peak_indices = get_peak_indices(curve)
    new_curve = curve.copy()
    replaced_values = []
    shifting = int(shifting)
    for i in peak_indices:
        if i+shifting not in peak_indices:
            replaced_values.append(curve[i+shifting])
        new_curve[i+shifting] = curve[i]
    if shifting > 0:
        for i in range(len(replaced_values)):
            new_curve[min(peak_indices)+i] = replaced_values[i]
    elif shifting < 0:
        for i in range(len(replaced_values)):
            new_curve[max(peak_indices)-i] = replaced_values[i]
    return new_curve

def shift_all(curve, shifting):
    for i in range(len(curve)):
        if curve[i] > 1:
            start_index = i
            break
    num_new_zeros = start_index - shifting
    all = curve[start_index:len(curve)]
    new_curve = [0]*num_new_zeros + list(all) + [0]*(len(curve) - num_new_zeros - len(all))
    return new_curve

def reverse_curve(curve):
    new_curve = curve.copy()
    new_curve = list(reversed(new_curve))
    return new_curve