"""Running and saving the complete Stage 2a experiment."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
  BASE_DISTANCE_SAMPLES,
  BASE_HEIGHT_MV,
  DEVELOPMENT_RECORDS,
  MATCHING_TOLERANCE_MS,
  PROMINENCE_VALUES,
  TUNED_DISTANCE_MS,
)
from .data import find_database_folder, list_record_names, load_record
from .detectors import base_detector, tuned_detector
from .scoring import score_detections


def score_row(record, method, detected_peaks, prominence=None):
  """Create one row in the shared Stage 2 results format."""

  scores = score_detections(
    record.annotations,
    detected_peaks,
    record.sample_rate,
  )

  return {
    "record": record.record_name,
    "method": method,
    "partition": (
      "development"
      if record.record_name in DEVELOPMENT_RECORDS
      else "evaluation"
    ),
    "input": "raw",
    "noise": "none",
    "snr_db": np.nan,
    "tp": scores.true_positives,
    "fp": scores.false_positives,
    "fn": scores.false_negatives,
    "se": scores.sensitivity,
    "ppv": scores.positive_predictive_value,
    "f1": scores.f1_score,
    "mean_timing_error_ms": scores.mean_timing_error_ms,
    "mean_signed_timing_error_ms": scores.mean_signed_timing_error_ms,
    "expert_beats": len(record.annotations),
    "detected_peaks": len(detected_peaks),
    "channel": record.channel_name,
    "channel_number": record.channel_number,
    "sample_rate_hz": record.sample_rate,
    "prominence": prominence,
  }


def save_peaks(output_folder, method, record_name, detected_peaks):
  """Save sample indices using the agreed one-column CSV format."""

  peak_folder = Path(output_folder) / "peak_files"
  peak_folder.mkdir(parents=True, exist_ok=True)
  peak_file = peak_folder / f"peaks_{method}_{record_name}.csv"
  pd.DataFrame({"sample": detected_peaks}).to_csv(peak_file, index=False)


def tune_prominence(database_folder, output_folder):
  """Choose prominence using only the declared development records."""

  sweep_rows = []

  for record_name in DEVELOPMENT_RECORDS:
    record = load_record(database_folder, record_name)

    for prominence in PROMINENCE_VALUES:
      detected_peaks = tuned_detector(
        record.signal,
        record.sample_rate,
        prominence,
      )
      row = score_row(
        record,
        "TUNED_DEVELOPMENT",
        detected_peaks,
        prominence,
      )
      sweep_rows.append(row)

  sweep_table = pd.DataFrame(sweep_rows)
  sweep_file = Path(output_folder) / "prominence_sweep.csv"
  sweep_table.to_csv(sweep_file, index=False)

  mean_scores = (
    sweep_table.groupby("prominence", as_index=False)["f1"]
    .mean()
    .sort_values(["f1", "prominence"], ascending=[False, True])
  )

  return float(mean_scores.iloc[0]["prominence"]), sweep_table


def summarise_methods(results_table):
  """Calculate pooled counts and metrics for each method."""

  summary_rows = []

  for scope in ("all_records", "evaluation_only"):
    if scope == "evaluation_only":
      scoped_results = results_table.query("partition == 'evaluation'")
    else:
      scoped_results = results_table

    for method, method_rows in scoped_results.groupby("method"):
      true_positives = int(method_rows["tp"].sum())
      false_positives = int(method_rows["fp"].sum())
      false_negatives = int(method_rows["fn"].sum())
      sensitivity = true_positives / (true_positives + false_negatives)
      ppv = true_positives / (true_positives + false_positives)
      f1_score = 2 * sensitivity * ppv / (sensitivity + ppv)

      summary_rows.append({
        "scope": scope,
        "method": method,
        "record_count": method_rows["record"].nunique(),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "pooled_se": sensitivity,
        "pooled_ppv": ppv,
        "pooled_f1": f1_score,
        "mean_record_f1": method_rows["f1"].mean(),
        "standard_deviation_record_f1": method_rows["f1"].std(ddof=1),
        "mean_timing_error_ms": method_rows["mean_timing_error_ms"].mean(),
      })

  return pd.DataFrame(summary_rows)


def write_settings(output_folder, database_folder, prominence):
  """Record the settings needed to reproduce the analysis."""

  settings = {
    "database_folder": str(database_folder),
    "record_count": 48,
    "matching_tolerance_ms": MATCHING_TOLERANCE_MS,
    "base_height_mv": BASE_HEIGHT_MV,
    "base_distance_samples": BASE_DISTANCE_SAMPLES,
    "tuned_distance_ms": TUNED_DISTANCE_MS,
    "tuned_prominence": prominence,
    "development_records": list(DEVELOPMENT_RECORDS),
    "prominence_values_tested": list(PROMINENCE_VALUES),
    "timing_error_definition": "mean absolute detection minus annotation",
    "lead_rule": "MLII where available; otherwise first recorded channel",
    "baseline_source_repository": (
      "https://github.com/Cstubbings/PHY3008-FFT-ECG-modeling.git"
    ),
    "baseline_source_commit": "a99ba65",
    "baseline_source_file": "FFT_Benchmark.py, lines 75-79",
    "python_version": platform.python_version(),
    "package_versions": {
      package: version(package)
      for package in ("numpy", "pandas", "scipy", "wfdb")
    },
  }

  settings_file = Path(output_folder) / "run_settings.json"
  settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def run_complete_analysis(data_folder, output_folder):
  """Tune TUNED, run BASE and TUNED, then save all required files."""

  database_folder = find_database_folder(data_folder)
  output_folder = Path(output_folder).expanduser().resolve()
  output_folder.mkdir(parents=True, exist_ok=True)

  record_names = list_record_names(database_folder)
  if len(record_names) != 48:
    raise ValueError(f"Expected 48 MIT-BIH records, found {len(record_names)}")

  print("Choosing the TUNED prominence on development records...")
  prominence, unused_sweep_table = tune_prominence(
    database_folder,
    output_folder,
  )
  print(f"Selected prominence: {prominence:.2f}")

  result_rows = []

  for number, record_name in enumerate(record_names, start=1):
    print(f"Processing record {record_name} ({number}/48)")
    record = load_record(database_folder, record_name)

    base_peaks = base_detector(record.signal)
    tuned_peaks = tuned_detector(
      record.signal,
      record.sample_rate,
      prominence,
    )

    save_peaks(output_folder, "BASE", record_name, base_peaks)
    save_peaks(output_folder, "TUNED", record_name, tuned_peaks)

    result_rows.append(score_row(record, "BASE", base_peaks))
    result_rows.append(score_row(
      record,
      "TUNED",
      tuned_peaks,
      prominence,
    ))

  results_table = pd.DataFrame(result_rows)
  results_table.to_csv(output_folder / "stage2a_scores.csv", index=False)

  method_summary = summarise_methods(results_table)
  method_summary.to_csv(
    output_folder / "stage2a_method_summary.csv",
    index=False,
  )

  write_settings(output_folder, database_folder, prominence)
  return results_table, method_summary, prominence
