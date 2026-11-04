"""Loading functions for the MIT-BIH Arrhythmia Database."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .constants import BEAT_SYMBOLS


@dataclass
class ECGRecord:
  """The information needed to detect and score peaks in one record."""

  record_name: str
  signal: np.ndarray
  sample_rate: float
  channel_name: str
  channel_number: int
  annotations: np.ndarray


def find_database_folder(data_folder):
  """Find the folder containing the 48 MIT-BIH record headers."""

  data_folder = Path(data_folder).expanduser().resolve()

  if not data_folder.exists():
    raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

  possible_folders = {header.parent for header in data_folder.rglob("*.hea")}

  for folder in sorted(possible_folders):
    records = list_record_names(folder)
    if len(records) == 48:
      return folder

  raise FileNotFoundError(
    "Could not find the 48 MIT-BIH records beneath "
    f"{data_folder}"
  )


def list_record_names(data_folder):
  """Return records which have header, signal and annotation files."""

  data_folder = Path(data_folder)
  record_names = []

  for header in sorted(data_folder.glob("*.hea")):
    record_path = header.with_suffix("")
    has_signal = record_path.with_suffix(".dat").is_file()
    has_annotations = record_path.with_suffix(".atr").is_file()

    if has_signal and has_annotations:
      record_names.append(record_path.name)

  return record_names


def choose_channel(channel_names):
  """Use MLII when available, otherwise use the first recorded channel."""

  normalised_names = [name.upper().strip() for name in channel_names]

  if "MLII" in normalised_names:
    channel_number = normalised_names.index("MLII")
  else:
    channel_number = 0

  return channel_number, channel_names[channel_number]


def keep_beat_annotations(samples, symbols):
  """Remove rhythm changes, comments and other non-beat annotations."""

  kept_samples = [
    sample
    for sample, symbol in zip(samples, symbols)
    if symbol in BEAT_SYMBOLS
  ]

  return np.asarray(kept_samples, dtype=int)


def load_record(data_folder, record_name):
  """Load one ECG signal and its expert beat annotations."""

  record_path = Path(data_folder) / str(record_name)
  record = wfdb.rdrecord(str(record_path))
  annotation = wfdb.rdann(str(record_path), "atr")

  if record.p_signal is None:
    raise ValueError(f"Record {record_name} does not contain a physical signal")

  channel_number, channel_name = choose_channel(record.sig_name)
  signal = np.asarray(record.p_signal[:, channel_number], dtype=float)
  annotations = keep_beat_annotations(annotation.sample, annotation.symbol)

  return ECGRecord(
    record_name=str(record_name),
    signal=signal,
    sample_rate=float(record.fs),
    channel_name=channel_name,
    channel_number=channel_number,
    annotations=annotations,
  )
