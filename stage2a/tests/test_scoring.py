"""Checks for the Stage 2a scoring harness."""

import numpy as np
import pytest

from stage2a.data import keep_beat_annotations
from stage2a.scoring import score_detections


SAMPLE_RATE = 360.0
ANNOTATIONS = np.array([100, 300, 500, 700])


def test_annotations_scored_against_themselves_are_perfect():
  result = score_detections(ANNOTATIONS, ANNOTATIONS, SAMPLE_RATE)

  assert result.true_positives == 4
  assert result.false_positives == 0
  assert result.false_negatives == 0
  assert result.f1_score == pytest.approx(1.0)
  assert result.mean_timing_error_ms == pytest.approx(0.0)


def test_small_shift_keeps_matches_and_reports_timing_error():
  shifted = ANNOTATIONS + 20
  result = score_detections(ANNOTATIONS, shifted, SAMPLE_RATE)

  assert result.f1_score == pytest.approx(1.0)
  assert result.mean_timing_error_ms == pytest.approx(55.5556, rel=1e-4)
  assert result.mean_signed_timing_error_ms == pytest.approx(55.5556, rel=1e-4)


def test_shift_beyond_150_ms_loses_matches():
  shifted = ANNOTATIONS + 55
  result = score_detections(ANNOTATIONS, shifted, SAMPLE_RATE)

  assert result.true_positives == 0
  assert result.false_positives == 4
  assert result.false_negatives == 4
  assert result.f1_score == pytest.approx(0.0)


def test_thinning_reduces_sensitivity_but_not_ppv():
  thinned = ANNOTATIONS[::2]
  result = score_detections(ANNOTATIONS, thinned, SAMPLE_RATE)

  assert result.sensitivity == pytest.approx(0.5)
  assert result.positive_predictive_value == pytest.approx(1.0)
  assert result.f1_score == pytest.approx(2 / 3)


def test_false_and_duplicate_peaks_are_not_reused():
  detections = np.array([100, 100, 300, 500, 700, 900])
  result = score_detections(ANNOTATIONS, detections, SAMPLE_RATE)

  assert result.true_positives == 4
  assert result.false_positives == 2
  assert result.false_negatives == 0


def test_only_the_specified_beat_symbols_are_kept():
  samples = np.array([10, 20, 30, 40, 50, 60])
  symbols = ["N", "+", "V", "~", "?", "Q"]

  kept = keep_beat_annotations(samples, symbols)

  assert np.array_equal(kept, np.array([10, 30, 60]))
