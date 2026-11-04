"""The BASE and TUNED R-peak detectors."""

import numpy as np
from scipy.signal import find_peaks

from .constants import (
  BASE_DISTANCE_SAMPLES,
  BASE_HEIGHT_MV,
  TUNED_DISTANCE_MS,
)


def base_detector(signal):
  """Reproduce the fixed detector from the shared GitHub repository.

  The repository uses a fixed 0.4 mV height and a distance of three
  samples. These settings are intentionally preserved, even though the
  three-sample gap is not physiologically realistic.
  """

  peaks, unused_properties = find_peaks(
    np.asarray(signal, dtype=float),
    height=BASE_HEIGHT_MV,
    distance=BASE_DISTANCE_SAMPLES,
  )

  return peaks.astype(int)


def tuned_detector(signal, sample_rate, prominence):
  """Detect peaks using physiological spacing and signal prominence."""

  minimum_distance = round(TUNED_DISTANCE_MS * sample_rate / 1000.0)

  peaks, unused_properties = find_peaks(
    np.asarray(signal, dtype=float),
    distance=minimum_distance,
    prominence=prominence,
  )

  return peaks.astype(int)
