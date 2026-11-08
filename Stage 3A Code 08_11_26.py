from pathlib import Path
import csv
import numpy as np
import wfdb

FS = 360
CLEAN_RECORDS = ["100", "101", "118", "119"]
EM_RECORDS = ["118", "119"]
NOISE_TYPES = ["bw", "ma"]
SNR_LEVELS = [24, 18, 12, 6, 0, -6]

MITDB_DIR = Path(r"C:\Users\samha\OneDrive - Queen's University Belfast\QUB PHYSICS\Level 3\PHY3008\mit-bih-arrhythmia-database-1.0.0")
NSTDB_DIR = Path(r"C:\Users\samha\OneDrive - Queen's University Belfast\QUB PHYSICS\Level 3\PHY3008\mit-bih-noise-stress-test-database-1.0.0")
OUTPUT_DIR = Path(__file__).resolve().parent / "stage3" / "noisy"
SUMMARY_FILE = OUTPUT_DIR / "snr_check.csv"


def read_mlii(record_path):
    """Read the MLII channel from a WFDB record."""
    record = wfdb.rdrecord(str(record_path))

    if record.fs != FS:
        raise ValueError(
            f"{record_path}: expected {FS} Hz, found {record.fs} Hz"
        )

    try:
        channel = record.sig_name.index("MLII")
    except ValueError as exc:
        raise ValueError(
            f"{record_path}: MLII channel not found; "
            f"available channels are {record.sig_name}"
        ) from exc

    if record.p_signal is None:
        raise ValueError(f"{record_path}: no physical signal data found")

    return record.p_signal[:, channel].astype(np.float64)


def read_noise(record_path):
    """Read the first channel from a noise-only WFDB record."""
    record = wfdb.rdrecord(str(record_path))

    if record.fs != FS:
        raise ValueError(
            f"{record_path}: expected {FS} Hz, found {record.fs} Hz"
        )

    if record.p_signal is None:
        raise ValueError(f"{record_path}: no physical signal data found")

    return record.p_signal[:, 0].astype(np.float64)


def fit_noise_length(noise, target_length):
    """Trim noise to the clean record length, repeating only if necessary."""
    if len(noise) == 0:
        raise ValueError("Noise record is empty")

    if len(noise) < target_length:
        repeats = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeats)

    return noise[:target_length]


def add_noise_at_snr(clean, noise, target_snr_db):
    """Scale noise to the requested power-based SNR and mix it with clean ECG."""
    noise = fit_noise_length(noise, len(clean))
    noise = noise - np.mean(noise)

    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)

    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("Signal and noise must both have non-zero power")

    target_ratio = 10 ** (target_snr_db / 10)
    scale = np.sqrt(signal_power / (noise_power * target_ratio))
    scaled_noise = noise * scale

    return clean + scaled_noise


def write_mlii_record(record_name, signal):
    """Write a one-channel WFDB record in mV."""
    wfdb.wrsamp(
        record_name=record_name,
        fs=FS,
        units=["mV"],
        sig_name=["MLII"],
        p_signal=signal.reshape(-1, 1),
        fmt=["16"],
        adc_gain=[1000],
        baseline=[0],
        write_dir=str(OUTPUT_DIR),
    )

    # Read back the quantized output so the SNR check includes file-writing effects.
    saved = wfdb.rdrecord(str(OUTPUT_DIR / record_name)).p_signal[:, 0]
    return saved.astype(np.float64)


def measured_snr_db(clean, noisy):
    """Measure SNR from the clean signal and the added-noise residual."""
    residual = noisy - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(residual**2)

    if noise_power <= 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def em_source_name(record_name, snr_db):
    """NSTDB naming: e.g. 118e24, 118e06, 118e00, 118e_6."""
    snr_part = f"{snr_db:02d}" if snr_db >= 0 else f"_{abs(snr_db)}"
    return f"{record_name}e{snr_part}"


def output_name(record_name, noise_type, snr_db):
    return f"{record_name}_{noise_type}_{snr_db}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    clean_signals = {
        record_name: read_mlii(MITDB_DIR / record_name)
        for record_name in CLEAN_RECORDS
    }

    noise_signals = {
        noise_type: read_noise(NSTDB_DIR / noise_type)
        for noise_type in NOISE_TYPES
    }

    # Generate baseline-wander and muscle-noise mixtures.
    for record_name, clean in clean_signals.items():
        for noise_type, noise in noise_signals.items():
            for snr_db in SNR_LEVELS:
                mixed = add_noise_at_snr(clean, noise, snr_db)
                name = output_name(record_name, noise_type, snr_db)
                saved = write_mlii_record(name, mixed)
                measured = measured_snr_db(clean, saved)

                results.append({
                    "record": record_name,
                    "noise": noise_type,
                    "target_snr_db": snr_db,
                    "measured_snr_db": round(float(measured), 3),
                    "output_record": name,
                })

    # Copy the database's existing electrode-motion mixtures into the same
    # output naming scheme, and check their SNR against the clean recordings.
    for record_name in EM_RECORDS:
        clean = clean_signals[record_name]

        for snr_db in SNR_LEVELS:
            source_name = em_source_name(record_name, snr_db)
            source_signal = read_mlii(NSTDB_DIR / source_name)

            if len(source_signal) != len(clean):
                raise ValueError(
                    f"{source_name}: length differs from clean record {record_name}"
                )

            em_noise = source_signal - clean
            em_signal = add_noise_at_snr(clean, em_noise, snr_db)
            name = output_name(record_name, "em", snr_db)
            saved = write_mlii_record(name, em_signal)
            measured = measured_snr_db(clean, saved)

            results.append({
                "record": record_name,
                "noise": "em",
                "target_snr_db": snr_db,
                "measured_snr_db": round(float(measured), 3),
                "output_record": name,
            })

    with SUMMARY_FILE.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "record",
                "noise",
                "target_snr_db",
                "measured_snr_db",
                "output_record",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        error = abs(row["measured_snr_db"] - row["target_snr_db"])
        status = "OK" if error <= 0.5 else "CHECK"
        print(
            f'{row["output_record"]}: '
            f'target {row["target_snr_db"]} dB, '
            f'measured {row["measured_snr_db"]} dB [{status}]'
        )

    print(f"\nSNR summary written to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()