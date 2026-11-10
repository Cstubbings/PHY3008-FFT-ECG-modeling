"""Run Davide's PHY3008 Stage 2a analysis.

This file is deliberately small: the calculation methods are kept in the
stage2a folder so that they can be checked and tested separately.
"""

import argparse
from pathlib import Path

from stage2a.pipeline import run_complete_analysis


PROJECT_FOLDER = Path(__file__).resolve().parent
DEFAULT_DATA_FOLDER = PROJECT_FOLDER.parent / "mit-bih-arrhythmia-database-1.0.0"
DEFAULT_OUTPUT_FOLDER = PROJECT_FOLDER / "results"


def read_arguments():
  """Read optional data and output folders from the command line."""

  parser = argparse.ArgumentParser(
    description="Run the Stage 2a BASE and TUNED R-peak detectors.",
  )
  parser.add_argument(
    "--data-folder",
    type=Path,
    default=DEFAULT_DATA_FOLDER,
    help="Folder containing the MIT-BIH Arrhythmia Database.",
  )
  parser.add_argument(
    "--output-folder",
    type=Path,
    default=DEFAULT_OUTPUT_FOLDER,
    help="Folder used for peak CSVs and score tables.",
  )
  return parser.parse_args()


def main():
  """Run the complete analysis and print a short summary."""

  arguments = read_arguments()
  results, summary, prominence = run_complete_analysis(
    arguments.data_folder,
    arguments.output_folder,
  )

  print("\nStage 2a analysis complete")
  print(f"Records scored: {results['record'].nunique()}")
  print(f"TUNED prominence: {prominence:.2f}\n")
  print(summary.to_string(index=False))


if __name__ == "__main__":
  main()
