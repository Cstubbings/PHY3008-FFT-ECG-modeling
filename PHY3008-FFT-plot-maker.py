import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

from scipy.signal import find_peaks
from scipy.fft import fft, ifft, fftfreq


# ==================================================
# SETTINGS
# ==================================================



input_folder = (
    r"C:\Users\conno\OneDrive - Queen's University Belfast\Level 3\PHY 3008\mit-bih-noise-stress-test-database-1.0.0"
)

output_folder = (
    r"C:\Users\conno\OneDrive - Queen's University Belfast"
    r"\Level 3\PHY 3008\FFT_Output"
)

patient_record = "119e_6"

base_record = patient_record.split("e")[0]

if base_record not in ["118", "119"]:
    raise ValueError(
        f"No reference record defined for {base_record}"
    )

clean_record = base_record

fs = 360

start_time = 290
duration = 20
channel = 0  # use channel 0 first

# ==================================================
# FFT FILTER
# ==================================================

def fft_filter(signal, fs,
               lowcut=0.5,
               highcut=40):

    spectrum = fft(signal)

    freqs = fftfreq(
        len(signal),
        d=1/fs
    )

    mask = (
        (np.abs(freqs) >= lowcut)
        &
        (np.abs(freqs) <= highcut)
    )

    filtered_spectrum = spectrum * mask

    filtered_signal = np.real(
        ifft(filtered_spectrum)
    )

    return filtered_signal


# ==================================================
# TRUE SNR
# ==================================================

def calculate_snr(reference_signal,
                  test_signal):

    ref = (
        reference_signal -
        np.mean(reference_signal)
    )

    test = (
        test_signal -
        np.mean(test_signal)
    )

    # Normalise amplitude
    ref = ref / np.std(ref)
    test = test / np.std(test)

    noise = test - ref

    signal_power = np.mean(
        ref**2
    )

    noise_power = np.mean(
        noise**2
    )

    if noise_power == 0:
        return np.inf

    return 10*np.log10(
        signal_power /
        noise_power
    )

# ==================================================
# R PEAK DETECTION
# ==================================================

def detect_r_peaks(signal, fs):

    threshold = (
        np.mean(signal)
        + 1.2*np.std(signal)
    )

    peaks, _ = find_peaks(
        signal,
        height=threshold,
        distance=int(0.45*fs),
        prominence=0.5
    )

    return peaks


# ==================================================
# PEAK AMPLITUDE CHANGE
# ==================================================

def peak_amplitude_change(
        clean,
        filtered,
        peaks):

    if len(peaks) == 0:
        return np.nan

    clean_peaks = []
    filtered_peaks = []

    window = 10

    for p in peaks:

        clean_peak = np.max(
            np.abs(
                clean[
                    max(0, p-window):
                    min(len(clean),
                        p+window)
                ]
            )
        )

        filtered_peak = np.max(
            np.abs(
                filtered[
                    max(0, p-window):
                    min(len(filtered),
                        p+window)
                ]
            )
        )

        clean_peaks.append(clean_peak)
        filtered_peaks.append(filtered_peak)

    clean_peaks = np.array(clean_peaks)
    filtered_peaks = np.array(filtered_peaks)

    return np.mean(
        np.abs(
            filtered_peaks -
            clean_peaks
        ) / clean_peaks
    ) * 100

# ==================================================
# FFT PLOTTING
# ==================================================

def plot_fft(signal,
             save_path,
             title):

    signal = signal - np.mean(signal)

    fft_values = fft(signal)

    freqs = fftfreq(
        len(signal),
        d=1/fs
    )

    mask = freqs > 0

    magnitude = np.abs(
        fft_values[mask]
    )

    magnitude_db = (
        20*np.log10(
            magnitude + 1e-12
        )
    )

    plt.figure(figsize=(14,5))

    plt.plot(
        freqs[mask],
        magnitude_db,
        linewidth=1.5
    )

    plt.xlim(0,60)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

# ==================================================
# LOAD RECORDS
# ==================================================

print(f"\nLoading clean record: {clean_record}")

clean_path = os.path.join(
    input_folder,
    clean_record
)

noisy_path = os.path.join(
    input_folder,
    patient_record
)

print("Clean path:", clean_path)
print("Noisy path:", noisy_path)

clean_data = wfdb.rdrecord(clean_path)
noisy_data = wfdb.rdrecord(noisy_path)

print("Both records loaded successfully")

# Channel selection
ecg_clean_full = clean_data.p_signal[:, channel]
ecg_noisy_full = noisy_data.p_signal[:, channel]

print(
    "Clean shape:",
    clean_data.p_signal.shape
)

print(
    "Noisy shape:",
    noisy_data.p_signal.shape
)

print(
    "Clean samples:",
    len(ecg_clean_full)
)

print(
    "Noisy samples:",
    len(ecg_noisy_full)
)


# ==================================================
# DIAGNOSTIC INFO
# ==================================================

print("\n===== RECORD INFORMATION =====")

print(
    "Clean shape:",
    clean_data.p_signal.shape
)

print(
    "Noisy shape:",
    noisy_data.p_signal.shape
)

print(
    "Clean samples:",
    len(ecg_clean_full)
)

print(
    "Noisy samples:",
    len(ecg_noisy_full)
)

print("==============================\n")


# ==================================================
# EXTRACT SEGMENT
# ==================================================

start_idx = int(start_time * fs)

end_idx = int(
    (start_time + duration) * fs
)

ecg_clean = ecg_clean_full[
    start_idx:end_idx
]

ecg_noisy = ecg_noisy_full[
    start_idx:end_idx
]

time = (
    np.arange(
        len(ecg_clean)
    ) / fs
) + start_time


# ==================================================
# FFT FILTERING
# ==================================================

ecg_filtered = fft_filter(
    ecg_noisy,
    fs
)


# ==================================================
# SNR CALCULATIONS
# ==================================================

snr_before = calculate_snr(
    ecg_clean,
    ecg_noisy
)

snr_after = calculate_snr(
    ecg_clean,
    ecg_filtered
)

snr_gain = (
    snr_after -
    snr_before
)


# ==================================================
# PEAK ANALYSIS
# ==================================================

peaks = detect_r_peaks(
    ecg_clean,
    fs
)
plt.figure(figsize=(14,5))

plt.plot(time, ecg_clean)

plt.scatter(
    time[peaks],
    ecg_clean[peaks],
    color='red',
    s=40
)

plt.title(
    f"Detected Peaks = {len(peaks)}"
)

plt.grid(True)
plt.show()
peak_change = peak_amplitude_change(
    ecg_clean,
    ecg_filtered,
    peaks
)


# ==================================================
# OUTPUT FOLDER
# ==================================================

save_folder = os.path.join(
    output_folder,
    patient_record
)

os.makedirs(
    save_folder,
    exist_ok=True
)


# ==================================================
# CLEAN VS NOISY
# ==================================================

plot_clean = ecg_clean
plot_noisy = ecg_noisy - 5

plt.figure(figsize=(14,5))

plt.plot(
    time,
    plot_clean,
    label="Clean ECG"
)

plt.plot(
    time,
    plot_noisy,
    label="Noisy ECG (-5 mV)"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")

plt.title(
    f"Input SNR = {snr_before:.2f} dB"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_folder,
        "clean_vs_noisy.png"
    ),
    dpi=300
)

plt.close()


# ==================================================
# FILTERED ECG
# ==================================================

plot_filtered = ecg_filtered - 5

plt.figure(figsize=(14,5))

plt.plot(
    time,
    plot_clean,
    linewidth=2,
    label="Clean ECG"
)

plt.plot(
    time,
    plot_filtered,
    linewidth=1.5,
    label="FFT Filtered ECG (-5 mV)"
)

plt.scatter(
    time[peaks],
    plot_clean[peaks],
    color="red",
    s=25,
    label="R Peaks"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")

plt.title(
    f"Output SNR = {snr_after:.2f} dB\n"
    f"SNR Gain = {snr_gain:.2f} dB\n"
    f"Peak Change = {peak_change:.2f}%"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_folder,
        "filtered_ecg.png"
    ),
    dpi=300
)

plt.close()


# ==================================================
# FFT PLOTS
# ==================================================

plot_fft(
    ecg_noisy,
    os.path.join(
        save_folder,
        "fft_before.png"
    ),
    "FFT Spectrum - Noisy ECG"
)

plot_fft(
    ecg_filtered,
    os.path.join(
        save_folder,
        "fft_after.png"
    ),
    "FFT Spectrum - FFT Filtered ECG"
)


# ==================================================
# SAVE RESULTS
# ==================================================

results = pd.DataFrame({

    "Metric": [
        "Input SNR (dB)",
        "Output SNR (dB)",
        "SNR Improvement (dB)",
        "R Peak Amplitude Change (%)",
        "Detected R Peaks"
    ],

    "Value": [
        snr_before,
        snr_after,
        snr_gain,
        peak_change,
        len(peaks)
    ]

})

results.to_csv(
    os.path.join(
        save_folder,
        "results.csv"
    ),
    index=False
)


# ==================================================
# FINAL RESULTS
# ==================================================

print("\n========== RESULTS ==========")

print(
    f"Input SNR       : {snr_before:.2f} dB"
)

print(
    f"Output SNR      : {snr_after:.2f} dB"
)

print(
    f"SNR Improvement : {snr_gain:.2f} dB"
)

print(
    f"Detected Peaks  : {len(peaks)}"
)

print(
    f"Peak Change     : {peak_change:.2f} %"
)

print("=============================")
plt.figure(figsize=(12,4))
plt.plot(ecg_noisy[:1000])
plt.title("Raw ECG")

plt.figure(figsize=(12,4))
plt.plot(ecg_filtered[:1000])
plt.title("FFT Filtered ECG")

plt.show()