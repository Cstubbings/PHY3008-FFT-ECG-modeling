"""Stage 3b - Test runner (PHY3008).

Runs every  record x noise x SNR x input (raw / cleaned) x detector
combination, scores each run with Davide's Stage 2a scorer and appends one
row per run to a shared results CSV.

Put this file in the ecg-stage2a project folder (next to run_stage2a.py) and
run, for example:

    python run_stage3b.py ^
        --mixtures-folder ..\\stage3\\noisy ^
        --database-folder ..\\mit-bih-arrhythmia-database-1.0.0 ^
        --pt-function pan_tompkins:pan_tompkins_detector ^
        --include-clean-reference

Detectors (method codes):
    BASE     stage2a.detectors.base_detector  (fixed 0.4 mV threshold)
    TUNED    stage2a.detectors.tuned_detector (prominence from run_settings.json)
    PT       Connor's pan_tompkins_detector(signal, sample_rate), via --pt-function
    STANDIN  plain find_peaks placeholder, for testing the loop only

Default detectors: BASE and TUNED, plus PT when --pt-function is given.

The run is resumable: rows already present in the output file are skipped.
"""

import argparse
import csv
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, medfilt, sosfiltfilt

NOISES = ("bw", "ma", "em")
INPUTS = ("raw", "clean")
NAME_PATTERN = re.compile(r"^(\d+)_(bw|ma|em)_(-?\d+)$")

COLUMNS = [
    "record", "method", "input", "noise", "snr_db",
    "tp", "fp", "fn", "se", "ppv", "f1",
    "mean_timing_error_ms", "mean_signed_timing_error_ms",
    "expert_beats", "detected_peaks", "runtime_s",
]


# ----------------------------------------------------------------------
# Stage 1 cleaning filters
# ----------------------------------------------------------------------

def _odd(fs, ms):
    n = int(round(fs * ms / 1000))
    return n if n % 2 == 1 else n + 1


def stage1_combined(signal, fs):
    """Median baseline subtraction -> 60 Hz notch -> 40 Hz low-pass
    (from '118e24 290-310s for Graph.py')."""
    base = medfilt(medfilt(signal, _odd(fs, 200)), _odd(fs, 600))
    x = signal - base
    b, a = iirnotch(60.0 / (fs / 2), 30.0)
    x = filtfilt(b, a, x)
    b, a = butter(4, 40.0 / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def stage1_highpass(signal, fs):
    """0.5 Hz zero-phase Butterworth high-pass (Stage 1B)."""
    sos = butter(4, 0.5, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


STAGE1_FILTERS = {"combined": stage1_combined, "highpass": stage1_highpass}


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------

def standin_detector(signal, fs):
    x = np.asarray(signal, dtype=float)
    prominence = 0.4 * (np.percentile(x, 99.5) - np.median(x))
    peaks, _ = find_peaks(x, distance=int(0.2 * fs), prominence=prominence)
    return peaks


def read_tuned_prominence(stage2a_folder, override):
    if override is not None:
        return override
    settings = Path(stage2a_folder) / "results" / "run_settings.json"
    if settings.is_file():
        return float(json.loads(settings.read_text())["tuned_prominence"])
    print("WARNING: run_settings.json not found; using TUNED prominence 0.5")
    return 0.5


def build_detectors(names, prominence, pt_function):
    """Return {method_code: f(signal, fs) -> sample indices}."""
    detectors = {}
    for name in names:
        if name == "BASE":
            from stage2a.detectors import base_detector
            detectors[name] = lambda s, fs, f=base_detector: f(s)
        elif name == "TUNED":
            from stage2a.detectors import tuned_detector
            detectors[name] = (
                lambda s, fs, f=tuned_detector: f(s, fs, prominence))
        elif name == "PT":
            if not pt_function:
                raise SystemExit("PT requested but --pt-function not given "
                                 "(e.g. pan_tompkins:pan_tompkins_detector).")
            module, _, func = pt_function.partition(":")
            detectors[name] = getattr(importlib.import_module(module), func)
        elif name == "STANDIN":
            detectors[name] = standin_detector
    return detectors


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def discover_mixtures(folder):
    found = set()
    for header in Path(folder).glob("*.hea"):
        m = NAME_PATTERN.match(header.stem)
        if m:
            found.add((m.group(1), m.group(2), int(m.group(3))))
    return sorted(found, key=lambda t: (t[0], t[1], -t[2]))


def load_mixture(folder, name):
    """Read a 3a mixture (MLII, 360 Hz WFDB) -> (signal, fs)."""
    import wfdb
    from stage2a.data import choose_channel
    rec = wfdb.rdrecord(str(Path(folder) / name))
    channel, _ = choose_channel(rec.sig_name)
    return np.asarray(rec.p_signal[:, channel], dtype=float), float(rec.fs)


# ----------------------------------------------------------------------
# Results file
# ----------------------------------------------------------------------

def key_of(record, method, input_name, noise, snr):
    return (str(record), method, input_name, noise, "" if snr is None else str(snr))


def load_done(output_file):
    done = set()
    if output_file.is_file():
        with open(output_file, newline="") as f:
            for r in csv.DictReader(f):
                done.add(key_of(r["record"], r["method"], r["input"],
                                r["noise"], r["snr_db"]))
    return done


def append_row(output_file, row):
    new_file = not output_file.is_file()
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def print_summary(output_file):
    import pandas as pd
    table = pd.read_csv(output_file)
    table["snr_db"] = table["snr_db"].fillna(999)
    pivot = table.pivot_table(index=["noise", "method", "input"],
                              columns="snr_db", values="f1", aggfunc="mean")
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    pivot.columns = ["clean" if c == 999 else f"{int(c)} dB" for c in pivot.columns]
    print("\nMean F1 across records (clean = un-noised reference):")
    print(pivot.round(3).to_string())


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def read_arguments():
    p = argparse.ArgumentParser(description="Stage 3b test runner.")
    p.add_argument("--mixtures-folder", required=True, type=Path,
                   help="3a output, files named {record}_{noise}_{snr}.")
    p.add_argument("--database-folder", required=True, type=Path,
                   help="MIT-BIH Arrhythmia Database (clean signals + .atr).")
    p.add_argument("--stage2a-folder", type=Path,
                   default=Path(__file__).resolve().parent,
                   help="Folder containing the stage2a package "
                        "(default: this file's folder).")
    p.add_argument("--output-file", type=Path,
                   default=Path("results/stage3_results.csv"))
    p.add_argument("--detectors", nargs="+",
                   choices=["BASE", "TUNED", "PT", "STANDIN"], default=None)
    p.add_argument("--pt-function", default=None, metavar="module:function")
    p.add_argument("--tuned-prominence", type=float, default=None,
                   help="Override; default is read from run_settings.json.")
    p.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=INPUTS)
    p.add_argument("--noises", nargs="+", default=list(NOISES), choices=NOISES)
    p.add_argument("--records", nargs="+", default=None)
    p.add_argument("--stage1-filter", default="combined",
                   choices=list(STAGE1_FILTERS))
    p.add_argument("--include-clean-reference", action="store_true",
                   help="Also score the un-noised record (noise='none').")
    return p.parse_args()


def main():
    args = read_arguments()
    sys.path.insert(0, str(args.stage2a_folder.resolve()))

    from stage2a.data import find_database_folder, load_record
    from stage2a.scoring import score_detections

    names = args.detectors or (["BASE", "TUNED"] +
                               (["PT"] if args.pt_function else []))
    prominence = read_tuned_prominence(args.stage2a_folder,
                                       args.tuned_prominence)
    detectors = build_detectors(names, prominence, args.pt_function)
    clean_filter = STAGE1_FILTERS[args.stage1_filter]
    print(f"Detectors: {', '.join(detectors)}"
          f"{f' (TUNED prominence {prominence})' if 'TUNED' in detectors else ''}")
    print(f"Stage 1 filter: {args.stage1_filter}")

    mixtures = [m for m in discover_mixtures(args.mixtures_folder)
                if m[1] in args.noises
                and (not args.records or m[0] in args.records)]
    if not mixtures:
        raise SystemExit("No mixtures named {record}_{noise}_{snr} found.")

    jobs = [(r, n, s) for r, n, s in mixtures]
    if args.include_clean_reference:
        jobs += [(r, "none", None) for r in sorted({m[0] for m in mixtures})]

    database = find_database_folder(args.database_folder)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output_file)
    references = {}

    total = len(jobs) * len(args.inputs) * len(detectors)
    count = new_rows = 0
    started = time.time()

    for record_name, noise, snr in jobs:
        if record_name not in references:
            references[record_name] = load_record(database, record_name)
        ref = references[record_name]
        fs = ref.sample_rate

        if noise == "none":
            raw = ref.signal
        else:
            name = f"{record_name}_{noise}_{snr}"
            raw, mix_fs = load_mixture(args.mixtures_folder, name)
            if mix_fs != fs or len(raw) != len(ref.signal):
                raise ValueError(
                    f"{name}: fs/length ({mix_fs}, {len(raw)}) does not match "
                    f"clean record ({fs}, {len(ref.signal)})")

        signals = {"raw": raw}
        if "clean" in args.inputs:
            signals["clean"] = clean_filter(raw, fs)

        for input_name in args.inputs:
            signal = signals[input_name]
            for method, detector in detectors.items():
                count += 1
                key = key_of(record_name, method, input_name, noise, snr)
                if key in done:
                    continue

                t0 = time.perf_counter()
                peaks = np.asarray(detector(signal, fs))
                runtime = time.perf_counter() - t0
                peaks = np.unique(peaks.astype(int))
                peaks = peaks[(peaks >= 0) & (peaks < len(signal))]

                s = score_detections(ref.annotations, peaks, fs)
                row = {
                    "record": record_name, "method": method,
                    "input": input_name, "noise": noise,
                    "snr_db": "" if snr is None else snr,
                    "tp": s.true_positives, "fp": s.false_positives,
                    "fn": s.false_negatives, "se": round(s.sensitivity, 5),
                    "ppv": round(s.positive_predictive_value, 5),
                    "f1": round(s.f1_score, 5),
                    "mean_timing_error_ms": round(s.mean_timing_error_ms, 3),
                    "mean_signed_timing_error_ms":
                        round(s.mean_signed_timing_error_ms, 3),
                    "expert_beats": len(ref.annotations),
                    "detected_peaks": len(peaks),
                    "runtime_s": round(runtime, 4),
                }
                append_row(args.output_file, row)
                new_rows += 1
                print(f"[{count}/{total}] {record_name} {noise:>4} "
                      f"{'' if snr is None else snr:>3} {input_name:>5} "
                      f"{method:<7} F1={row['f1']:.3f}")

    print(f"\nDone in {time.time() - started:.0f} s: {new_rows} new rows, "
          f"{count - new_rows} already present.")
    print(f"Results: {args.output_file.resolve()}")
    print_summary(args.output_file)


if __name__ == "__main__":
    main()
