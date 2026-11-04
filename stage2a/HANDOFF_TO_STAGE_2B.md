# Handoff to Stage 2b: Pan-Tompkins

Davide has completed the common scoring harness and the `BASE` and `TUNED`
detectors. Stage 2b should implement Pan-Tompkins as method code `PT`, save its
detected sample indices and use this same scorer.

Using one scorer matters: different annotation filters or matching rules would
make the detector F1 values impossible to compare fairly.

## What is ready to reuse

The project is located at:

```text
C:\Users\david\Documents\professionalSkills\ecg-stage2a
```

- `stage2a/data.py` finds all records, selects MLII and loads annotations.
- `stage2a/constants.py` defines the accepted beats and 150 ms tolerance.
- `stage2a/scoring.py` performs nearest one-to-one matching and metrics.
- `score_external_peaks.py` scores a complete folder of PT peak files.
- `results/stage2a_scores.csv` contains BASE and TUNED comparisons.

## Required detector interface

The detector should accept a full ECG signal and sampling rate, and return a
one-dimensional array of full-record integer sample indices:

```python
def pan_tompkins_detector(signal, sample_rate):
  # Band-pass, derivative, squaring, moving-window integration,
  # adaptive thresholds and search-back.
  return detected_peak_samples
```

Do not return seconds, indices relative to a short window or indices that
still contain uncorrected filter delay.

## Loading the same signals

Run from the `ecg-stage2a` folder and use:

```python
from stage2a.data import (
  find_database_folder,
  list_record_names,
  load_record,
)

database_folder = find_database_folder(
  r"C:\Users\david\Documents\professionalSkills\mit-bih-arrhythmia-database-1.0.0"
)

for record_name in list_record_names(database_folder):
  record = load_record(database_folder, record_name)
  detected_peaks = pan_tompkins_detector(
    record.signal,
    record.sample_rate,
  )
```

The loader uses MLII except for records 102 and 104, which lack MLII and use
V5. `record.annotations` is the reference answer and must not be used to move
individual detections manually.

## Peak-file format

Create exactly 48 files named:

```text
peaks_PT_100.csv
peaks_PT_101.csv
...
peaks_PT_234.csv
```

Each file contains only the `sample` column:

```csv
sample
77
370
663
```

Example saving code:

```python
import pandas as pd

pd.DataFrame({"sample": detected_peak_samples}).to_csv(
  output_file,
  index=False,
)
```

## Scoring all PT files

Once the 48 files are together, run:

```powershell
cd C:\Users\david\Documents\professionalSkills\ecg-stage2a

..\ecg-stage5\.venv\Scripts\python.exe score_external_peaks.py `
  --method PT `
  --peak-folder "C:\path\to\PT\peak_files" `
  --output-file "results\stage2b_PT_scores.csv"
```

The command requires all 48 files and produces 48 comparable rows using the
same accepted beat symbols, +/-54-sample tolerance, one-to-one matching, TP,
FP, FN, sensitivity, PPV, F1 and mean absolute timing error.

## Pan-Tompkins delay warning

Band-pass filtering, differentiation and moving-window integration introduce
delay. The final result should identify the R peak in the original ECG, not
only a delayed integration maximum. A common approach is to use the integrated
signal to locate a candidate region, then search near the corresponding point
in the band-passed or original ECG for the actual R peak.

Document the compensation method. Do not rely on the 150 ms scoring tolerance
to hide a known systematic delay because timing error is also reported.

## Completion checklist

- [ ] PT runs automatically on all 48 records.
- [ ] Exactly 48 `peaks_PT_<record>.csv` files exist.
- [ ] Each CSV contains only sorted integer `sample` values.
- [ ] Every value is within the full record length.
- [ ] Filter/integration delay is compensated or clearly documented.
- [ ] The common scorer finishes and produces 48 result rows.
- [ ] Poor-performing records are identified, not hidden by an overall mean.

Return the PT source code, 48 peak files, scored results table, method settings
and a short note about any unusual records. Do not overwrite Davide's
`stage2a_scores.csv`; Stage 2c can combine the separate result tables.
