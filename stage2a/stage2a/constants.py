"""Constants used throughout the Stage 2a analysis."""

# MIT-BIH was recorded at 360 samples per second.
DEFAULT_SAMPLE_RATE = 360.0

# A detection is correct when it is within 150 ms of an expert annotation.
MATCHING_TOLERANCE_MS = 150.0

# These are the beat symbols specified in the Stage 2 and 3 work split.
# Non-beat marks such as +, ~, | and " are deliberately excluded.
BEAT_SYMBOLS = {
  "N", "L", "R", "A", "a", "J", "S", "V",
  "F", "e", "j", "E", "/", "f", "Q",
}

# Original fixed-threshold detector from the shared GitHub repository.
BASE_HEIGHT_MV = 0.4
BASE_DISTANCE_SAMPLES = 3

# A physiological minimum gap of 200 ms is used for the tuned detector.
TUNED_DISTANCE_MS = 200.0

# The prominence is chosen on this small development subset, then frozen.
DEVELOPMENT_RECORDS = ("100", "105", "200", "203")
PROMINENCE_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
