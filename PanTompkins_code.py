import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from scipy.signal import freqz


# ============================================================
# USER SETTINGS
# ============================================================

FS = 360

INPUT_FOLDER = r"C:\Users\conno\OneDrive - Queen's University Belfast\Level 3\PHY 3008\mit-bih-arrhythmia-database-1.0.0"

OUTPUT_FOLDER = r"C:\Users\conno\OneDrive - Queen's University Belfast\Level 3\PHY 3008\PanTompkins_Output"

PROCESS_ALL_RECORDS = True

SINGLE_RECORD = "100"

MATCH_TOLERANCE_MS = 50

MATCH_TOLERANCE = int(
    MATCH_TOLERANCE_MS * FS / 1000
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

if PROCESS_ALL_RECORDS:

    RECORDS = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".dat")
    ])

else:

    RECORDS = [SINGLE_RECORD]


# ============================================================
# FILTERING
# ============================================================

def bandpass_filter(ecg, fs):

    nyq = fs / 2

    b, a = butter(
        3,
        [5 / nyq, 15 / nyq],
        btype="band"
    )

    return filtfilt(b, a, ecg)


def differentiate(signal):

    return np.diff(
        signal,
        prepend=signal[0]
    )


def moving_window_integration(signal, fs):

    window = int(0.150 * fs)

    kernel = np.ones(window) / window

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


# ============================================================
# PAN TOMPKINS
# ============================================================

def pan_tompkins_detector(ecg, fs=360):

    filtered = bandpass_filter(
        ecg,
        fs
    )

    differentiated = differentiate(
        filtered
    )

    squared = differentiated ** 2

    integrated = moving_window_integration(
        squared,
        fs
    )

    candidate_peaks, _ = find_peaks(
        integrated,
        distance=int(0.20 * fs)
    )

    init_len = min(
        len(integrated),
        2 * fs
    )

    spki = 0.25 * np.max(
        integrated[:init_len]
    )

    npki = np.mean(
        integrated[:init_len]
    )

    threshold = npki + 0.25 * (spki - npki)

    detected = []

    rr_intervals = []

    for peak in candidate_peaks:

        peak_value = integrated[peak]

        if peak_value > threshold:

            if len(detected) > 0:

                rr = peak - detected[-1]

                if rr < int(0.36 * fs):

                    current_slope = np.max(
                        np.abs(
                            differentiated[
                                max(0, peak - 10):
                                min(len(differentiated), peak + 10)
                            ]
                        )
                    )

                    previous_peak = detected[-1]

                    previous_slope = np.max(
                        np.abs(
                            differentiated[
                                max(0, previous_peak - 10):
                                min(len(differentiated), previous_peak + 10)
                            ]
                        )
                    )

                    if current_slope < 0.5 * previous_slope:

                        npki = (
                            0.125 * peak_value
                            + 0.875 * npki
                        )

                        threshold = (
                            npki
                            + 0.25 * (spki - npki)
                        )

                        continue

            detected.append(peak)

            spki = (
                0.125 * peak_value
                + 0.875 * spki
            )

            if len(detected) >= 2:

                rr_intervals.append(
                    detected[-1]
                    - detected[-2]
                )

        else:

            npki = (
                0.125 * peak_value
                + 0.875 * npki
            )

        threshold = (
            npki
            + 0.25 * (spki - npki)
        )

    # ========================================================
    # SEARCH BACK
    # ========================================================

    if len(rr_intervals) > 0 and len(detected) > 0:

        mean_rr = np.mean(rr_intervals)

        improved = [detected[0]]

        for i in range(1, len(detected)):

            prev_peak = improved[-1]

            curr_peak = detected[i]

            gap = curr_peak - prev_peak

            if gap > 1.66 * mean_rr:

                missed = candidate_peaks[
                    (candidate_peaks > prev_peak)
                    &
                    (candidate_peaks < curr_peak)
                ]

                best_peak = None
                best_value = 0

                for p in missed:

                    value = integrated[p]

                    if (
                        value > 0.5 * threshold
                        and value > best_value
                    ):

                        best_peak = p
                        best_value = value

                if best_peak is not None:

                    improved.append(best_peak)

            improved.append(curr_peak)

        detected = sorted(improved)

    # ========================================================
    # RELOCATE TO TRUE R PEAK
    # ========================================================

    corrected = []

    search_window = int(
        0.10 * fs
    )

    for peak in detected:

        left = max(
            0,
            peak - search_window
        )

        right = min(
            len(filtered),
            peak + search_window
        )

        segment = filtered[left:right]

        if len(segment) == 0:
            continue

        true_peak = (
            left
            + np.argmax(
                np.abs(segment)
            )
        )

        corrected.append(true_peak)

    corrected = np.array(
        sorted(
            set(corrected)
        )
    )

    return corrected


# ============================================================
# EVALUATION
# ============================================================

def evaluate_detector(
    detected_peaks,
    reference_peaks,
    tolerance
):

    tp = 0

    matched_ref = set()

    for detected in detected_peaks:

        best_match = None
        best_error = tolerance + 1

        for idx, ref_peak in enumerate(reference_peaks):

            if idx in matched_ref:
                continue

            error = abs(
                detected - ref_peak
            )

            if error <= tolerance:

                if error < best_error:

                    best_error = error
                    best_match = idx

        if best_match is not None:

            tp += 1
            matched_ref.add(best_match)

    fp = len(detected_peaks) - tp

    fn = len(reference_peaks) - tp

    if (2 * tp + fp + fn) > 0:

        f1 = (
            2 * tp
            /
            (2 * tp + fp + fn)
        )

    else:

        f1 = np.nan

    return tp, fp, fn, f1


# ============================================================
# PLOTTING
# ============================================================

def save_stage_plots(ecg, record_name):

    filtered = bandpass_filter(
        ecg,
        FS
    )

    differentiated = differentiate(
        filtered
    )

    squared = differentiated ** 2

    integrated = moving_window_integration(
        squared,
        FS
    )

    # --------------------------------------------------------

    nyq = FS / 2

    b, a = butter(
        3,
        [5 / nyq, 15 / nyq],
        btype="band"
    )

    w, h = freqz(
        b,
        a,
        worN=2048
    )

    plt.figure(figsize=(8,4))

    plt.plot(
        w * FS / (2*np.pi),
        20*np.log10(
            np.maximum(
                np.abs(h),
                1e-12
            )
        )
    )

    plt.title("Bandpass Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(-80, 5)
    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{record_name}_FilterResponse.png"
        )
    )

    plt.close()

    # --------------------------------------------------------

    N = min(
        5000,
        len(ecg)
    )

    fig, axs = plt.subplots(
        5,
        1,
        figsize=(12,10)
    )

    axs[0].plot(ecg[:N])
    axs[0].set_title("Raw ECG")

    axs[1].plot(filtered[:N])
    axs[1].set_title("Bandpassed")

    axs[2].plot(differentiated[:N])
    axs[2].set_title("Differentiated")

    axs[3].plot(squared[:N])
    axs[3].set_title("Squared")

    axs[4].plot(integrated[:N])
    axs[4].set_title("Integrated")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{record_name}_ProcessingStages.png"
        )
    )

    plt.close()


# ============================================================
# BATCH PROCESSING
# ============================================================

summary_rows = []

valid_symbols = {
    'N','L','R','A','a',
    'J','S','V','F',
    'e','j','E'
}

for record_name in RECORDS:

    print(f"\nProcessing {record_name}...")

    try:

        record_path = (
            f"{INPUT_FOLDER}\\{record_name}"
        )

        record = wfdb.rdrecord(
            record_path
        )

        ecg = record.p_signal[:,0]

        if record_name == "100":

            save_stage_plots(
                ecg,
                record_name
            )

        r_peaks = pan_tompkins_detector(
            ecg,
            FS
        )

        peak_file = (
            f"{OUTPUT_FOLDER}"
            f"\\peaks_PT_{record_name}.csv"
        )

        pd.DataFrame(
            r_peaks
        ).to_csv(
            peak_file,
            header=False,
            index=False
        )

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )

        reference_peaks = np.array([
            annotation.sample[i]
            for i, sym in enumerate(
                annotation.symbol
            )
            if sym in valid_symbols
        ])

        tp, fp, fn, f1 = evaluate_detector(
            r_peaks,
            reference_peaks,
            MATCH_TOLERANCE
        )

        summary_rows.append({

            "Record":
                record_name,

            "Detected_Beats":
                len(r_peaks),

            "Reference_Beats":
                len(reference_peaks),

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "F1":
                f1,

            "F1_%":
                f1 * 100

        })

        print(
            f"Detected Beats = {len(r_peaks)}"
        )

        print(
            f"F1 = {f1:.4f}"
        )

    except Exception as e:

        print(
            f"FAILED: {e}"
        )

# ============================================================
# SUMMARY SHEET
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

if len(summary_df) > 0:

    total_tp = summary_df["TP"].sum()

    total_fp = summary_df["FP"].sum()

    total_fn = summary_df["FN"].sum()

    overall_f1 = (
        2 * total_tp
        /
        (
            2 * total_tp
            + total_fp
            + total_fn
        )
    )

    summary_df.loc[
        len(summary_df)
    ] = {

        "Record": "OVERALL",

        "Detected_Beats":
            summary_df[
                "Detected_Beats"
            ].sum(),

        "Reference_Beats":
            summary_df[
                "Reference_Beats"
            ].sum(),

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,

        "F1": overall_f1,

        "F1_%":
            overall_f1 * 100

    }

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "PanTompkins_Summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n================================")
print("Batch Processing Complete")
print("================================")
print(f"Summary saved to:\n{summary_file}")
