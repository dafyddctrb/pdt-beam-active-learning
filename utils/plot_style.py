"""
Shared plotting style utilities for academic paper aesthetics.
"""
import os
import matplotlib.pyplot as plt


def plots_enabled() -> bool:
    """False when PDT_NO_PLOTS=1: skip matplotlib PNG output (the web app plots from JSON)."""
    return os.environ.get("PDT_NO_PLOTS") != "1"

# ─── Colour constants ────────────────────────────────────────────────────────
PRED_COLOR  = "#5B9BD5"   # steel blue   – predictions / posteriors / AL points
TRUE_COLOR  = "#37474F"   # dark charcoal – true / ground-truth values / seed pts
MEAS_COLOR  = "#d62728"   # red          – observations / measurements / LSF
PRIOR_COLOR = "#bcbcbc"   # grey         – prior distributions / sample clouds
TRUTH_COLOR = "#ffcc00"   # gold         – ground-truth (E, I) highlight (star)


def _presentation_default(presentation):
    """Resolve presentation mode: explicit arg wins, else the PLOT_PRESENTATION
    env var (so a single `--presentation` flag flips every plot in the run)."""
    if presentation is None:
        return os.environ.get("PLOT_PRESENTATION", "0").strip().lower() in (
            "1", "true", "yes", "on")
    return bool(presentation)


def set_academic_style(presentation=None):
    """Unified matplotlib style for every comparison plot.

    - paper (default): serif (Times New Roman), compact sizes.
    - presentation:    Arial sans-serif, larger sizes for slides.
    - presentation=None: read the ``PLOT_PRESENTATION`` env var (default paper).

    All figures inherit these rcParams and use *relative* font sizes
    ('small'/'large'), so flipping this one switch restyles titles, labels and
    legends consistently across the whole plot family. Returns the resolved mode.
    """
    presentation = _presentation_default(presentation)
    base = 15 if presentation else 11
    plt.rcParams.update({
        'font.family': 'sans-serif' if presentation else 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': base,
        'axes.titlesize': base + 1,
        'axes.titleweight': 'bold',
        'axes.labelsize': base,
        'xtick.labelsize': base - 2,
        'ytick.labelsize': base - 2,
        'legend.fontsize': base - 2,
        'figure.titlesize': base + 4,
        'figure.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    return presentation


def legend_below(fig, ncol=4, fontsize=16):
    """
    Collect unique legend entries from all axes, remove per-axis legends, and
    place a single legend just outside the figure bottom.

    Works with savefig(bbox_inches='tight') to capture the legend.
    """
    handles, labels, seen = [], [], set()
    for ax in fig.get_axes():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith("_"):
                handles.append(h)
                labels.append(l)
                seen.add(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()
    fig.tight_layout()
    if handles:
        import math
        n = len(handles)
        nrows = math.ceil(n / ncol)
        balanced_ncol = math.ceil(n / nrows)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            bbox_transform=fig.transFigure,
            ncol=balanced_ncol,
            frameon=True, framealpha=0.9, edgecolor="0.8",
            fontsize=fontsize,
        )
