"""One-to-one comparison of detected peaks with expert annotations."""

from dataclasses import asdict, dataclass

import numpy as np

from .constants import MATCHING_TOLERANCE_MS


@dataclass
class ScoringResult:
  """Performance values for one set of detected peaks."""

  true_positives: int
  false_positives: int
  false_negatives: int
  sensitivity: float
  positive_predictive_value: float
  f1_score: float
  mean_timing_error_ms: float
  mean_signed_timing_error_ms: float

  def as_dictionary(self):
    return asdict(self)


def safe_divide(numerator, denominator):
  """Return zero when a metric has no valid denominator."""

  if denominator == 0:
    return 0.0

  return numerator / denominator


def prepare_samples(samples, name):
  """Check peak samples and put them into ascending order."""

  samples = np.asarray(samples, dtype=int)

  if samples.ndim != 1:
    raise ValueError(f"{name} must be a one-dimensional list of samples")
  if np.any(samples < 0):
    raise ValueError(f"{name} cannot contain negative sample numbers")

  return np.sort(samples)


def possible_matches(annotations, detections, tolerance_samples):
  """Create every possible annotation/detection pair within tolerance."""

  candidates = []

  for detection_number, detection in enumerate(detections):
    first = np.searchsorted(
      annotations,
      detection - tolerance_samples,
      side="left",
    )
    last = np.searchsorted(
      annotations,
      detection + tolerance_samples,
      side="right",
    )

    for annotation_number in range(first, last):
      timing_error = detection - annotations[annotation_number]
      candidates.append((
        abs(int(timing_error)),
        detection_number,
        annotation_number,
        int(timing_error),
      ))

  # The closest available pairs are assigned first. This prevents one
  # annotation or detection from being counted more than once.
  candidates.sort(key=lambda match: (match[0], match[1], match[2]))
  return candidates


def match_peaks(annotations, detections, tolerance_samples):
  """Perform nearest, one-to-one peak matching."""

  used_annotations = set()
  used_detections = set()
  timing_errors = []

  for unused_distance, detection_number, annotation_number, error in (
    possible_matches(annotations, detections, tolerance_samples)
  ):
    annotation_used = annotation_number in used_annotations
    detection_used = detection_number in used_detections

    if not annotation_used and not detection_used:
      used_annotations.add(annotation_number)
      used_detections.add(detection_number)
      timing_errors.append(error)

  return np.asarray(timing_errors, dtype=float)


def score_detections(
  annotations,
  detections,
  sample_rate,
  tolerance_ms=MATCHING_TOLERANCE_MS,
):
  """Calculate TP, FP, FN, sensitivity, PPV, F1 and timing error."""

  if sample_rate <= 0:
    raise ValueError("sample_rate must be greater than zero")
  if tolerance_ms < 0:
    raise ValueError("tolerance_ms cannot be negative")

  annotations = prepare_samples(annotations, "annotations")
  detections = prepare_samples(detections, "detections")
  tolerance_samples = round(tolerance_ms * sample_rate / 1000.0)
  timing_errors = match_peaks(
    annotations,
    detections,
    tolerance_samples,
  )

  true_positives = len(timing_errors)
  false_positives = len(detections) - true_positives
  false_negatives = len(annotations) - true_positives

  sensitivity = safe_divide(
    true_positives,
    true_positives + false_negatives,
  )
  positive_predictive_value = safe_divide(
    true_positives,
    true_positives + false_positives,
  )
  f1_score = safe_divide(
    2 * sensitivity * positive_predictive_value,
    sensitivity + positive_predictive_value,
  )

  if true_positives:
    errors_ms = timing_errors * 1000.0 / sample_rate
    mean_timing_error_ms = float(np.mean(np.abs(errors_ms)))
    mean_signed_timing_error_ms = float(np.mean(errors_ms))
  else:
    mean_timing_error_ms = float("nan")
    mean_signed_timing_error_ms = float("nan")

  return ScoringResult(
    true_positives=true_positives,
    false_positives=false_positives,
    false_negatives=false_negatives,
    sensitivity=sensitivity,
    positive_predictive_value=positive_predictive_value,
    f1_score=f1_score,
    mean_timing_error_ms=mean_timing_error_ms,
    mean_signed_timing_error_ms=mean_signed_timing_error_ms,
  )
