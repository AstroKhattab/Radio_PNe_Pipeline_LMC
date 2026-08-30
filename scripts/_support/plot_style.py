"""Consistent publication styling; does not alter scientific calculations."""

import matplotlib


def apply_paper_style():
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.alpha": 0.20,
        "lines.linewidth": 1.6,
    })
