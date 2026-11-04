"""Score peak CSV files produced by Stage 2b or another detector."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stage2a.data import find_database_folder, list_record_names, load_record
from stage2a.pipeline import score_row


PROJECT_FOLDER = Path(__file__).resolve().parent
DEFAULT_DATA_FOLDER = PROJECT_FOLDER.parent / "mit-bih-arrhythmia-database-1.0.0"


def read_arguments():
  """Read the method, peak folder, data folder and output filename."""

  parser = argparse.ArgumentParser(
    description="Score external peak files with the Stage 2a harness.",
  )
  parser.add_argument("--method", required=True, help="Method code, e.g. PT.")
  parser.add_argument("--peak-folder", required=True, type=Path)
  parser.add_argument("--data-folder", type=Path, default=DEFAULT_DATA_FOLDER)
  parser.add_argument("--output-file", required=True, type=Path)
  return parser.parse_args()


def load_peak_file(peak_file):
  """Load and validate one agreed-format peak file."""

  table = pd.read_csv(peak_file)

  if list(table.columns) != ["sample"]:
    raise ValueError(
      f"{peak_file.name} must contain exactly one column called sample"
    )

  numeric_samples = pd.to_numeric(table["sample"], errors="raise")

  if numeric_samples.isna().any():
    raise ValueError(f"{peak_file.name} contains a missing sample")
  if not np.all(numeric_samples == np.floor(numeric_samples)):
    raise ValueError(f"{peak_file.name} contains a non-integer sample")
  if (numeric_samples < 0).any():
    raise ValueError(f"{peak_file.name} contains a negative sample")

  return np.sort(numeric_samples.to_numpy(dtype=int))


def main():
  """Score all 48 external files and save one combined table."""

  arguments = read_arguments()
  method = arguments.method.upper()
  database_folder = find_database_folder(arguments.data_folder)
  record_names = list_record_names(database_folder)
  result_rows = []

  for record_name in record_names:
    peak_file = arguments.peak_folder / f"peaks_{method}_{record_name}.csv"

    if not peak_file.is_file():
      raise FileNotFoundError(f"Missing required peak file: {peak_file}")

    record = load_record(database_folder, record_name)
    detected_peaks = load_peak_file(peak_file)

    if len(detected_peaks) and detected_peaks[-1] >= len(record.signal):
      raise ValueError(f"{peak_file.name} contains a sample outside the record")

    result_rows.append(score_row(record, method, detected_peaks))

  output_file = arguments.output_file.expanduser().resolve()
  output_file.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(result_rows).to_csv(output_file, index=False)

  print(f"Scored {len(result_rows)} {method} records")
  print(f"Saved: {output_file}")


if __name__ == "__main__":
  main()
