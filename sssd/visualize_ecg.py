# AI Generated class to visualize ECG samples

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


def visualize_ecg_npy(
    folder="generated_ecg",
    index=0,
    sample_idx=0,
    fs=100,
    figsize=(12, 8),
    save_path=None,
    tight_layout=True,
    show=False,
    gain=2.0,  # Adjust for readability
    label_scores=None,  # optional dict of {label_name: score}
    top_k: int = 10,
):
    """
    Visualize a generated ECG sample as a 12-lead standard 3x4 layout.
    - Use actual signal amplitudes; no normalization.
    - Y-axis grid: major=1.0 mV, minor=0.1 mV; symmetric global y-limits rounded to 1.0 mV.
    - Show y-axis ticks/labels only once on the top-left subplot (mV numeric scale).
    - Show x-axis ticks/labels only on the bottom row for each column, in milliseconds.
    - Duration shown is 10 seconds (0..10000 ms).
    """
    path = os.path.join(folder, f"{index}_samples.npy")
    data = np.load(path)  # (batch, 12, L)
    assert data.ndim == 3 and data.shape[1] == 12, f"Unexpected shape: {data.shape}"

    ecg = data[sample_idx]  # (12, L)
    L = ecg.shape[1]
    # Use provided fs to compute seconds; expect ~10s total
    t_sec = np.arange(L) / fs
    # For display, we want exactly 10 seconds on the axis
    total_sec = 10.0
    # If needed, stretch time for display only
    if t_sec[-1] > 0:
        scale = total_sec / t_sec[-1]
    else:
        scale = 1.0
    t_plot = t_sec * scale

    # Generator lead order: [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]
    gen_names = ["I","II","V1","V2","V3","V4","V5","V6","III","aVR","aVL","aVF"]

    # Standard 2x6 layout order: [[I, aVR, V1, V4], [II, aVL, V2, V5], [III, aVF, V3, V6]]
    layout_order = ["aVL","V1","I","V2","aVR","V3","II","V4","aVF","V5","III","V6"]
    # layout_order = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]
    idx_map = [gen_names.index(n) for n in layout_order]

    ecg_reordered = ecg[idx_map]
    names_reordered = layout_order

    # Apply optional gain (no normalization)
    signals = ecg_reordered * (gain if gain else 1.0)

    # Global symmetric y-limits rounded to nearest 1.0 mV (min ±1.0 mV)
    y_major = 1.0
    y_minor = 0.1
    y_max = float(np.max(np.abs(signals)))
    if not np.isfinite(y_max) or y_max == 0.0:
        y_max = y_major
    y_max = max(y_major, np.ceil(y_max / y_major) * y_major)

    # Create 6x2 grid; share x by column and y by row (we still set same ylim for all)
    fig, axes = plt.subplots(6, 2, figsize=figsize, sharex='col', sharey='row')
    axes = axes.reshape(6, 2)

    # Time grid constants (paper-like 200/40 ms)
    major_xtick_sec = 0.2  # 200 ms
    minor_xtick_sec = 0.04  # 40 ms

    def ms_formatter(x, pos):
        return f"{int(round(x*1000))}"

    rows, cols = 6, 2

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            ax = axes[r, c]
            ax.plot(t_plot, signals[i], color="black", linewidth=0.5)
            ax.set_title(names_reordered[i], fontsize=10, loc="left")

            # Limits
            ax.set_xlim(0, total_sec)
            ax.set_ylim(-y_max, y_max)

            # Grid: set both time (x) and amplitude (y) grids
            ax.grid(which="major", axis="both", color="#ffb3b3", linewidth=0.4)
            ax.grid(which="minor", axis="both", color="#ffd6d6", linewidth=0.1)
            # X (time) locators
            ax.xaxis.set_major_locator(MultipleLocator(major_xtick_sec))
            ax.xaxis.set_minor_locator(MultipleLocator(minor_xtick_sec))
            # Y (mV) locators
            ax.yaxis.set_major_locator(MultipleLocator(y_major))
            ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

            # Y-axis tick labels only on the very top-left subplot
            if c == 0:
                # Let Matplotlib show numeric labels at 1.0 mV increments
                pass
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            # X-axis ticks/labels only on bottom row, in ms
            if r == rows - 1:
                ax.xaxis.set_major_formatter(FuncFormatter(ms_formatter))
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

    # Single figure-level y-axis unit label
    fig.text(0.02, 0.5, "mV", rotation=90, va='center', ha='center', fontsize=10)

    # Optional labels panel below figure
    if isinstance(label_scores, dict) and len(label_scores) > 0:
        # Only show labels with score exactly 1.0
        items = [(name, score) for name, score in label_scores.items() if float(score) == 1.0]
        items = sorted(items, key=lambda kv: kv[0])  # stable order
        if top_k is not None:
            items = items[:top_k]
        if items:
            text_lines = [f"{name}: {score:.3f}" for name, score in items]
            text = "\n".join(text_lines)
            fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=9, family='monospace')

    if tight_layout:
        plt.tight_layout(rect=[0.04, 0.05 if label_scores else 0.0, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig

if __name__ == "__main__":
    pass 
