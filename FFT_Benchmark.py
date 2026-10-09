# .........................................................................
# BIOMEDICAL SIGNAL PROCESSING: CLINICAL ECG ANALYSIS
# .........................................................................

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# 1. DATA ACQUISITION
# -----------------------------------------------------------------------------
# Load clinical dataset from the MIT-BIH Arrhythmia Database (Record '100')
data = wfdb.rdrecord('100')
ecg_raw = data.p_signal[:, 0]  # Extracting Lead 0
fs = 360  # Sampling frequency in Hz

# Define a 5-second analysis window
start_index = 0
end_index = 5 * fs
time_axis = np.arange(start_index, end_index) / fs

plt.figure(figsize=(10, 4))
plt.plot(time_axis, ecg_raw[start_index:end_index], color='gray')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Time (Seconds)")
plt.ylabel("Voltage (mV)")
plt.title("Time-Domain ECG Signal (MIT-BIH Record 100)")
plt.show()

# -----------------------------------------------------------------------------
# 2. SIGNAL CONDITIONING & FREQUENCY ANALYSIS (FFT)
# -----------------------------------------------------------------------------

# [ENGINEERING RATIONALE]: DC Offset Removal
# Medical signals frequently exhibit baseline wander. Failing to remove this
# mean voltage creates a massive zero-Hertz artifact during the Fourier Transform,
# which can severely mask crucial physiological frequencies.
mean_voltage = np.mean(ecg_raw[start_index:end_index])
ecg_zero_mean = ecg_raw[start_index:end_index] - mean_voltage

# Compute Fast Fourier Transform (FFT)
ecg_fft = fft(ecg_zero_mean)
ecg_freq_axis = fftfreq(len(ecg_zero_mean), d=1/fs)

# [ENGINEERING RATIONALE]: Nyquist Theorem Application
# Real physiological signals yield symmetric FFT outputs. We extract only the
# positive frequency spectrum up to the Nyquist limit (fs/2).
window_size = end_index - start_index
freqs_positive = ecg_freq_axis[:window_size//2]
magnitude = np.abs(ecg_fft)[:window_size//2]

plt.figure(figsize=(10, 4))
plt.plot(freqs_positive, magnitude, color='navy')

# Clinical Frequency Spectrum Guide:
# 0.5 - 4 Hz: T-wave and baseline wander
# 8 - 20 Hz: QRS complex (Primary heartbeat spike)
# 50/60 Hz: Powerline interference (Grid noise)
plt.xlim(0, 60)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("ECG Frequency Spectrum Analysis")
plt.show()

# -----------------------------------------------------------------------------
# 3. ALGORITHMIC FEATURE EXTRACTION (R-PEAK DETECTION)
# -----------------------------------------------------------------------------

# Intelligent R-peak extraction:
# Utilizing amplitude (>0.4 mV) and temporal distance (>3 samples) thresholds
# to strictly reject secondary physiological waves (T/P waves) and high-frequency noise.
peaks_filtered, _ = find_peaks(
    ecg_raw[start_index:end_index],
    height=0.4,
    distance=3
)
print(f"Algorithm extracted {len(peaks_filtered)} valid R-peaks.")

plt.figure(figsize=(10, 4))
plt.plot(ecg_raw[start_index:end_index],
         label='Original ECG Signal', color='gray', alpha=0.7)
plt.plot(peaks_filtered, ecg_raw[start_index:end_index][peaks_filtered],
         "x", color='red', markersize=10, markeredgewidth=3, label='Verified R-peaks')
plt.axhline(y=0.4, color='green', linestyle='--',
            label='Height Threshold (0.4 mV)')

plt.title("Intelligent R-Peak Detection Algorithm")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (mV)")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# -----------------------------------------------------------------------------
# 4. PHYSIOLOGICAL METRICS (HEART RATE)
# -----------------------------------------------------------------------------

# Calculate R-R intervals and dynamically convert to Beats Per Minute (BPM)
RR_intervals_sec = np.diff(peaks_filtered) / fs
instantaneous_bpm = 60 / RR_intervals_sec

print(f"R-R Intervals (Seconds): {np.round(RR_intervals_sec, 3)}")
print(f"Instantaneous Heart Rate (BPM): {np.round(instantaneous_bpm, 1)}")
print(f"Average Heart Rate: {np.round(np.mean(instantaneous_bpm), 1)} BPM")
