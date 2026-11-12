from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.95
NOISE = {
    'bw': 'Baseline wander',
    'ma': 'Muscle noise',
    'em': 'Electrode motion'
}

METHODS = ('BASE', 'PT')
INPUTS = ('raw', 'clean')


def find_results():
    folder = Path(__file__).resolve().parent
    filename = 'stage3_results_with_PT.csv'

    matches = sorted(
        folder.rglob(filename),
        key=lambda p: len(p.parts)
    )

    if not matches and Path.cwd() != folder:
        matches = sorted(
            Path.cwd().rglob(filename),
            key=lambda p: len(p.parts)
        )

    if matches:
        return matches[0], folder

    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    chosen = filedialog.askopenfilename(
        title='Select Stage 3 results CSV',
        filetypes=[('CSV files', '*.csv')]
    )
    root.destroy()

    if not chosen:
        raise FileNotFoundError('No results CSV selected.')

    return Path(chosen), folder


csv_path, project = find_results()
output = project / 'stage3c_results'
output.mkdir(exist_ok=True)

results = pd.read_csv(csv_path, dtype={'record': str})

results = results[
    results['noise'].isin(NOISE)
    & results['method'].isin(METHODS)
    & results['input'].isin(INPUTS)
].copy()

results['snr_db'] = pd.to_numeric(results['snr_db'])
results['f1'] = pd.to_numeric(results['f1'])

keys = ['record', 'noise', 'snr_db', 'input']

if results.duplicated(keys + ['method']).any():
    raise ValueError('Duplicate experimental results.')

base = set(map(
    tuple,
    results.loc[results.method == 'BASE', keys].to_numpy()
))

pt = set(map(
    tuple,
    results.loc[results.method == 'PT', keys].to_numpy()
))

if not base or base != pt:
    raise ValueError('BASE and PT experimental conditions do not match.')


# 1. Mean F1 against SNR

curves = (
    results.groupby(
        ['noise', 'snr_db', 'method', 'input'],
        as_index=False
    )
    .agg(
        mean_f1=('f1', 'mean'),
        std_f1=('f1', 'std'),
        n_records=('record', 'nunique')
    )
)

curves.to_csv(output / 'f1_summary.csv', index=False)


# 2. Cleaning effect

clean = results.pivot(
    index=['record', 'noise', 'snr_db', 'method'],
    columns='input',
    values='f1'
).dropna(subset=['raw', 'clean']).reset_index()

clean['delta_f1'] = clean['clean'] - clean['raw']
clean['effect'] = 'Cleaning'
clean['comparison'] = clean['method']


# 3. Detector effect

compare = results.pivot(
    index=['record', 'noise', 'snr_db', 'input'],
    columns='method',
    values='f1'
).dropna(subset=['BASE', 'PT']).reset_index()

compare['delta_f1'] = compare['PT'] - compare['BASE']
compare['effect'] = 'Detector'
compare['comparison'] = compare['input']

effects = (
    pd.concat([clean, compare], ignore_index=True)
    .groupby(
        ['noise', 'snr_db', 'effect', 'comparison'],
        as_index=False
    )
    .agg(
        mean_delta_f1=('delta_f1', 'mean'),
        n_records=('record', 'nunique')
    )
)

effects.to_csv(output / 'effect_summary.csv', index=False)


# 4. Breakdown SNR calculation

def crossing(group):
    group = group.sort_values('snr_db')

    x = group['snr_db'].to_numpy()
    y = group['mean_f1'].to_numpy()

    if np.all(y >= THRESHOLD):
        return np.nan, f'Below or at {x.min():g} dB (not observed)'

    if np.all(y < THRESHOLD):
        return np.nan, f'Above {x.max():g} dB (not reached)'

    found = []

    for i in range(len(x) - 1):
        if y[i] < THRESHOLD <= y[i + 1]:
            snr = (
                x[i]
                + (THRESHOLD - y[i])
                * (x[i + 1] - x[i])
                / (y[i + 1] - y[i])
            )
            found.append(snr)

    if not found:
        return np.nan, 'No upward crossing'

    status = 'Interpolated'

    if len(found) > 1 or np.any(np.diff(y) < -0.001):
        status += ' (non-monotonic curve)'

    return float(max(found)), status


breakdowns = []

for (noise, method, input_type), group in curves.groupby(
    ['noise', 'method', 'input']
):
    snr, status = crossing(group)

    breakdowns.append({
        'noise': noise,
        'method': method,
        'input': input_type,
        'breakdown_snr_db': snr,
        'status': status,
        'n_records': int(group['n_records'].min())
    })

pd.DataFrame(breakdowns).to_csv(
    output / 'breakdown_snr.csv',
    index=False
)


# 5. Plotting function


def plot_chart(filename, title, data,
               y_col, series_col, y_label, horizontal):

    if filename == '02_cleaning_effect.png':

        fig, axes = plt.subplots(
            2, 3, figsize=(14, 7),
            sharex=True, sharey='row'
        )

        for j, (noise, name) in enumerate(NOISE.items()):
            for i, (method, color) in enumerate([
                ('BASE', 'tab:blue'),
                ('PT', 'tab:green')
            ]):
                ax = axes[i, j]

                part = data[
                    (data['noise'] == noise)
                    & (data[series_col] == method)
                ].sort_values('snr_db')

                ax.plot(
                    part['snr_db'],
                    part[y_col],
                    marker='o',
                    color=color,
                    linewidth=2
                )

                ax.axhline(0, color='gray', linestyle=':')
                ax.set_title(f'{name} - {method}')
                ax.set_xlabel('SNR (dB)')
                ax.grid(alpha=0.25)

                if i == 1:
                    ax.ticklabel_format(
                        axis='y',
                        style='plain',
                        useOffset=False
                    )

        axes[0, 0].set_ylabel('BASE: Change in F1')
        axes[1, 0].set_ylabel('PT: Change in F1 (zoomed)')

        fig.suptitle(title)
        fig.tight_layout()

    else:

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4.5),
            sharey=True
        )

        if filename == '01_f1_vs_snr.png':
            styles = {
                'BASE / raw': dict(
                    color='tab:orange', marker='o'
                ),
                'BASE / clean': dict(
                    color='tab:blue', marker='s',
                    linestyle='--'
                ),
                'PT / raw': dict(
                    color='tab:red', marker='x'
                ),
                'PT / clean': dict(
                    color='tab:green', marker='D',
                    linestyle='--', zorder=5
                )
            }
        else:
            styles = {
                'raw': dict(color='tab:orange', marker='o'),
                'clean': dict(color='tab:blue', marker='s')
            }

        for ax, (noise, name) in zip(axes, NOISE.items()):

            part = data[data['noise'] == noise]

            for label, style in styles.items():

                series = part[
                    part[series_col] == label
                ].sort_values('snr_db')

                if not series.empty:
                    ax.plot(
                        series['snr_db'],
                        series[y_col],
                        label=label,
                        **style
                    )

            ax.axhline(
                horizontal, color='gray', linestyle=':'
            )

            ax.set_title(name)
            ax.set_xlabel('SNR (dB)')
            ax.grid(alpha=0.25)

            if filename == '01_f1_vs_snr.png':
                ax.set_ylim(0, 1.05)

        axes[0].set_ylabel(y_label)

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles, labels,
            loc='upper center',
            ncol=4,
            bbox_to_anchor=(0.5, 1.03)
        )

        fig.suptitle(title, y=1.14)
        fig.tight_layout(rect=(0, 0, 1, 0.90))

    fig.savefig(
        output / filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)



# 6. F1 versus SNR graph

curves['line'] = (
    curves['method'] + ' / ' + curves['input']
)

plot_chart(
    '01_f1_vs_snr.png',
    'R-peak detection: F1 vs SNR',
    curves,
    'mean_f1',
    'line',
    'Mean F1 score',
    THRESHOLD
)


# 7. Cleaning effect graph

clean_effects = effects[
    effects['effect'] == 'Cleaning'
].copy()

plot_chart(
    '02_cleaning_effect.png',
    'Stage 1 cleaning effect: cleaned minus raw',
    clean_effects,
    'mean_delta_f1',
    'comparison',
    'Change in F1',
    0
)


# 8. BASE versus Pan-Tompkins graph

detector_effects = effects[
    effects['effect'] == 'Detector'
].copy()

plot_chart(
    '03_detector_gap.png',
    'Detector effect: Pan-Tompkins minus BASE',
    detector_effects,
    'mean_delta_f1',
    'comparison',
    'Change in F1',
    0
)

print('Stage 3c completed:', output)
