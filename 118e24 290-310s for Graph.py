# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:32:27 2026

@author: samha
"""

"""
Combined pipeline: median baseline subtraction + 60 Hz notch + ~40 Hz low-pass.
- SNR gain: tested on NSTDB noisy records against matching clean Arrhythmia originals
- R-peak amplitude change: tested across all 48 Arrhythmia database records
"""

import numpy as np
import pandas as pd
import wfdb
import os
from scipy.signal import medfilt, iirnotch, butter, filtfilt
import matplotlib.pyplot as plt

#%matplotlib QT

# --- Filtering functions ---

def to_odd_samples(fs, ms):
    n = int(round(fs * ms / 1000))
    return n if n % 2 == 1 else n + 1

def median_baseline_subtract(signal, fs, win1_ms=200, win2_ms=600):
    w1 = to_odd_samples(fs, win1_ms)
    w2 = to_odd_samples(fs, win2_ms)
    pass1 = medfilt(signal, kernel_size=w1)
    baseline = medfilt(pass1, kernel_size=w2)
    cleaned = signal - baseline
    return cleaned, baseline

def notch_filter(signal, fs, notch_freq=60.0, Q=30.0):
    nyquist = fs / 2.0
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)

def low_pass_filter(signal, fs, cutoff=40.0, order=4):
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def full_pipeline(signal, fs):
    """Median baseline subtraction -> 60 Hz notch -> ~40 Hz low-pass."""
    stage1, baseline = median_baseline_subtract(signal, fs)
    stage2 = notch_filter(stage1, fs, notch_freq=60.0, Q=30.0)
    stage3 = low_pass_filter(stage2, fs, cutoff=40.0, order=4)
    return stage3

# --- Metric functions, unchanged ---

valid_beat_symbols = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
                       'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}

def r_peak_amplitude_change_pct(raw, cleaned, r_peak_samples, fs,
                                  pre_window_ms=100, min_amp=1e-3):
    valid = r_peak_samples[(r_peak_samples >= 0) & (r_peak_samples < len(raw))]
    pre_samples = int(round(fs * pre_window_ms / 1000))
    raw_heights, clean_heights = [], []
    for peak in valid:
        start = max(0, peak - pre_samples - 20)
        end = max(0, peak - 20)
        if end <= start:
            continue
        raw_local_baseline = np.median(raw[start:end])
        clean_local_baseline = np.median(cleaned[start:end])
        raw_heights.append(raw[peak] - raw_local_baseline)
        clean_heights.append(cleaned[peak] - clean_local_baseline)
    raw_heights = np.array(raw_heights)
    clean_heights = np.array(clean_heights)
    safe = np.abs(raw_heights) > min_amp
    raw_heights, clean_heights = raw_heights[safe], clean_heights[safe]
    pct_change = (clean_heights - raw_heights) / np.abs(raw_heights) * 100
    return pct_change, np.mean(pct_change), np.median(pct_change), np.std(pct_change)

def true_snr_db(clean, noisy_or_cleaned):
    noise = noisy_or_cleaned - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# --- Paths ---
"""You will have to change the path to the stress test database and arrhythmia database to match your pc"""

nstdb_dir = r"C:\Users\samha\OneDrive - Queen's University Belfast\QUB PHYSICS\Level 3\PHY3008\mit-bih-noise-stress-test-database-1.0.0"
arrhythmia_dir = r"C:\Users\samha\OneDrive - Queen's University Belfast\QUB PHYSICS\Level 3\PHY3008\mit-bih-arrhythmia-database-1.0.0"

# ========================================================================
# PART 1: SNR gain, using NSTDB noisy records vs matching Arrhythmia originals
# ========================================================================




# --- Load record 118e24 ---

nstdb_dir = r"C:\Users\samha\OneDrive - Queen's University Belfast\QUB PHYSICS\Level 3\PHY3008\mit-bih-noise-stress-test-database-1.0.0"

record_path = os.path.join(nstdb_dir, "118e24")
record = wfdb.rdrecord(record_path)
fs = record.fs

idx = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
raw_signal = record.p_signal[:, idx]

# --- Apply the full pipeline to the whole signal first ---
cleaned_signal = full_pipeline(raw_signal, fs)

# --- Select just the 100-120 s window for plotting ---
t_start, t_end = 290, 310
sample_start = int(t_start * fs)
sample_end = int(t_end * fs)

t = np.arange(sample_start, sample_end) / fs
raw_window = raw_signal[sample_start:sample_end]
cleaned_window = cleaned_signal[sample_start:sample_end]

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(t, raw_window, color='blue', label='Raw ECG', linewidth=1)
plt.plot(t, cleaned_window, color='green', label='Filtered ECG', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Record 118e24: Raw vs Filtered ECG (290–310 s)')
plt.legend()
plt.tight_layout()
plt.show()