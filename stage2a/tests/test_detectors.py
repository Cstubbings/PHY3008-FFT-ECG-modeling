"""Checks that the BASE detector remains faithful to the GitHub code."""

import numpy as np

from stage2a.detectors import base_detector


def test_base_uses_fixed_height_and_three_sample_distance():
  signal = np.zeros(20)
  signal[2] = 0.39
  signal[5] = 0.50
  signal[7] = 0.80
  signal[12] = 0.60

  peaks = base_detector(signal)

  # Sample 2 is below 0.4 mV. Of samples 5 and 7, find_peaks keeps the
  # larger because they are less than three samples apart.
  assert np.array_equal(peaks, np.array([7, 12]))
