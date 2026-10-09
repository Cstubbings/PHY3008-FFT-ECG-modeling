"""
Stage 1B - ECG Cleaning
0.5 Hz Zero-Phase Butterworth High-Pass Filter

Dataset:
    MIT-BIH Arrhythmia Database

Purpose:
    Remove low-frequency baseline wander while preserving ECG morphology
    and R-peak amplitude/timing.

Outputs:
    1. Raw vs filtered ECG plot
    2. Mean absolute R-peak amplitude change (%)
    3. Baseline-wander reduction (dB)
    4. Results appended to shared_results.csv


"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing MIT-BIH files:
# e.g. 100.dat, 100.hea, 100.atr
DATA_FOLDER = "mitdb"

# MIT-BIH record to test
RECORD_NAME = "100"

# ECG channel to use
CHANNEL = 0

# Butterworth settings
CUTOFF_HZ = 0.5
FILTER_ORDER = 4

# Length of ECG shown in comparison plot
PLOT_DURATION_SECONDS = 10

# Shared results file
RESULTS_FILE = "shared_results.csv"


# ============================================================
# 1. LOAD ECG DATA
# ============================================================

def load_ecg_record(data_folder, record_name, channel=0):
    """
    Load one ECG channel from an MIT-BIH record.

    Parameters
    ----------
    data_folder : str
        Folder containing MIT-BIH files.

    record_name : str
        Record number, e.g. "100".

    channel : int
        ECG channel index.

    Returns
    -------
    ecg : ndarray
        ECG signal in physical units (mV).

    fs : float
        Sampling frequency in Hz.

    signal_name : str
        Name of ECG channel.

    record : wfdb.Record
        Complete WFDB record object.
    """

    record_path = os.path.join(data_folder, record_name)

    record = wfdb.rdrecord(record_path)

    fs = record.fs

    ecg = record.p_signal[:, channel]

    signal_name = record.sig_name[channel]

    return ecg, fs, signal_name, record


# ============================================================
# 2. LOAD EXPERT R-PEAK ANNOTATIONS
# ============================================================

def load_r_peak_annotations(data_folder, record_name):
    """
    Load expert beat annotations from the MIT-BIH .atr file.

    The 'sample' values give the sample locations of annotated beats.

    Returns
    -------
    r_peaks : ndarray
        Sample indices corresponding to expert beat annotations.
    """

    annotation_path = os.path.join(data_folder, record_name)

    annotations = wfdb.rdann(
        annotation_path,
        "atr"
    )

    return annotations.sample


# ============================================================
# 3. STAGE 1B - BUTTERWORTH HIGH-PASS FILTER
# ============================================================

def butterworth_highpass(
    ecg,
    fs,
    cutoff=0.5,
    order=4
):
    """
    Apply a zero-phase Butterworth high-pass filter.

    A cutoff of 0.5 Hz is used to suppress low-frequency
    baseline wander.

    Second-order sections (SOS) are used for numerical stability.

    Zero-phase filtering is achieved using sosfiltfilt(),
    which applies the filter forwards and backwards.

    Parameters
    ----------
    ecg : ndarray
        Raw ECG signal.

    fs : float
        Sampling frequency in Hz.

    cutoff : float
        High-pass cutoff frequency in Hz.

    order : int
        Butterworth filter order.

    Returns
    -------
    filtered_ecg : ndarray
        Filtered ECG signal.
    """

    sos = butter(
        N=order,
        Wn=cutoff,
        btype="highpass",
        fs=fs,
        output="sos"
    )

    filtered_ecg = sosfiltfilt(
        sos,
        ecg
    )

    return filtered_ecg


# ============================================================
# 4. R-PEAK AMPLITUDE PRESERVATION
# ============================================================

def calculate_r_peak_amplitude_change(
    raw_ecg,
    filtered_ecg,
    r_peaks
):
    """
    Measure how much the filter changes amplitudes at
    expert-annotated R-peak locations.

    Returns the mean absolute percentage change.

    Parameters
    ----------
    raw_ecg : ndarray
        Original ECG.

    filtered_ecg : ndarray
        Filtered ECG.

    r_peaks : ndarray
        Expert beat locations.

    Returns
    -------
    mean_change : float
        Mean absolute R-peak amplitude change (%).
    """

    changes = []

    for peak in r_peaks:

        # Safety check
        if peak >= len(raw_ecg):
            continue

        raw_amplitude = raw_ecg[peak]
        filtered_amplitude = filtered_ecg[peak]

        # Avoid dividing by values extremely close to zero
        if abs(raw_amplitude) < 1e-8:
            continue

        percentage_change = (
            abs(filtered_amplitude - raw_amplitude)
            / abs(raw_amplitude)
        ) * 100

        changes.append(percentage_change)

    if len(changes) == 0:
        return np.nan

    return np.mean(changes)


# ============================================================
# 5. ESTIMATE BASELINE-WANDER REDUCTION
# ============================================================

def estimate_baseline_wander_reduction(
    raw_ecg,
    filtered_ecg
):
    """
    Estimate how strongly low-frequency baseline content
    has been removed.

    The difference between the raw and filtered signals
    represents the low-frequency component removed by the
    high-pass filter.

    This is reported in dB as a baseline-wander reduction
    measure.

    NOTE:
    This is NOT a true SNR measurement because the true
    noise-free ECG is unknown for ordinary MIT-BIH records.

    True SNR gain should later be measured using controlled
    noise experiments / the Noise Stress Test Database.

    Returns
    -------
    reduction_db : float
        Estimated baseline reduction in dB.
    """

    removed_baseline = raw_ecg - filtered_ecg

    raw_rms = np.sqrt(
        np.mean(raw_ecg ** 2)
    )

    baseline_rms = np.sqrt(
        np.mean(removed_baseline ** 2)
    )

    if baseline_rms == 0:
        return np.nan

    reduction_db = 20 * np.log10(
        raw_rms / baseline_rms
    )

    return reduction_db


# ============================================================
# 6. PLOT RAW VS FILTERED ECG
# ============================================================

def plot_comparison(
    raw_ecg,
    filtered_ecg,
    fs,
    record_name,
    signal_name,
    duration=10
):
    """
    Plot a short section of raw and filtered ECG.
    """

    number_samples = int(
        duration * fs
    )

    number_samples = min(
        number_samples,
        len(raw_ecg)
    )

    time = (
        np.arange(number_samples)
        / fs
    )

    plt.figure(
        figsize=(13, 6)
    )

    plt.plot(
        time,
        raw_ecg[:number_samples],
        label="Raw ECG",
        alpha=0.7
    )

    plt.plot(
        time,
        filtered_ecg[:number_samples],
        label="0.5 Hz Butterworth high-pass",
        linewidth=1
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Amplitude (mV)"
    )

    plt.title(
        f"MIT-BIH Record {record_name} - "
        f"{signal_name}\n"
        "Raw vs 0.5 Hz Zero-Phase Butterworth High-Pass"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. SAVE RESULT USING GROUP'S SHARED FORMAT
# ============================================================

def save_result(
    record_name,
    method,
    metric,
    value,
    filename=RESULTS_FILE
):
    """
    Save one metric using the shared project format:

        record, method, metric, value
    """

    new_result = pd.DataFrame(
        {
            "record": [record_name],
            "method": [method],
            "metric": [metric],
            "value": [value]
        }
    )

    # Create file if it does not already exist
    if not os.path.exists(filename):

        new_result.to_csv(
            filename,
            index=False
        )

    else:

        new_result.to_csv(
            filename,
            mode="a",
            header=False,
            index=False
        )


# ============================================================
# 8. RUN STAGE 1B EXPERIMENT
# ============================================================

def main():

    print("=" * 60)
    print("STAGE 1B")
    print("0.5 Hz ZERO-PHASE BUTTERWORTH HIGH-PASS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load ECG
    # --------------------------------------------------------

    raw_ecg, fs, signal_name, record = load_ecg_record(
        DATA_FOLDER,
        RECORD_NAME,
        CHANNEL
    )

    print()
    print(f"Record: {RECORD_NAME}")
    print(f"Channel: {signal_name}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Number of samples: {len(raw_ecg)}")

    duration = len(raw_ecg) / fs

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Load expert annotations
    # --------------------------------------------------------

    r_peaks = load_r_peak_annotations(
        DATA_FOLDER,
        RECORD_NAME
    )

    print(
        f"Expert beat annotations: {len(r_peaks)}"
    )

    # --------------------------------------------------------
    # Apply Stage 1B filter
    # --------------------------------------------------------

    filtered_ecg = butterworth_highpass(
        raw_ecg,
        fs,
        cutoff=CUTOFF_HZ,
        order=FILTER_ORDER
    )

    print()
    print("Filter applied successfully.")

    print(
        f"Cutoff frequency: {CUTOFF_HZ} Hz"
    )

    print(
        f"Butterworth order: {FILTER_ORDER}"
    )

    print(
        "Filtering method: zero-phase forward/backward"
    )

    # --------------------------------------------------------
    # Evaluate R-peak amplitude preservation
    # --------------------------------------------------------

    amplitude_change = (
        calculate_r_peak_amplitude_change(
            raw_ecg,
            filtered_ecg,
            r_peaks
        )
    )

    print()
    print(
        "Mean absolute R-peak amplitude change: "
        f"{amplitude_change:.3f}%"
    )

    # --------------------------------------------------------
    # Estimate baseline-wander reduction
    # --------------------------------------------------------

    baseline_reduction = (
        estimate_baseline_wander_reduction(
            raw_ecg,
            filtered_ecg
        )
    )

    print(
        "Estimated baseline-wander reduction: "
        f"{baseline_reduction:.3f} dB"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    method_name = (
        "Butterworth HP 0.5 Hz zero-phase"
    )

    save_result(
        RECORD_NAME,
        method_name,
        "R_peak_amplitude_change_percent",
        amplitude_change
    )

    save_result(
        RECORD_NAME,
        method_name,
        "baseline_wander_reduction_db",
        baseline_reduction
    )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Display visual comparison
    # --------------------------------------------------------

    plot_comparison(
        raw_ecg,
        filtered_ecg,
        fs,
        RECORD_NAME,
        signal_name,
        duration=PLOT_DURATION_SECONDS
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
