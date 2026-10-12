"""Shared plotting helpers: styled histograms, scatterplots, and theme setup."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_theme():
    sns.set_theme(
        context="notebook", 
        style="ticks", 
        palette="deep",
        font="sans-serif", 
        font_scale=1, 
        color_codes=True, 
        rc=None)


def plot_error_histogram(
    series, 
    title, 
    xlabel, 
    out_path, 
    percentiles=(1, 99),
    ylabel="Frequency", 
    dpi=120, 
    save_dpi=300
):
    """
    Percentile-bounded histogram: bounds are computed on whichever `series`
    is passed in, so a pre-filtered subset (e.g. only occupied cells) gets
    bounds matching that subset rather than the full column.
    """
    clean = series.dropna()
    lo_pct, hi_pct = percentiles

    lower_bound = np.percentile(clean, lo_pct) if lo_pct is not None else clean.min()
    upper_bound = np.percentile(clean, hi_pct) if hi_pct is not None else clean.max()

    bins = int(upper_bound + abs(min(lower_bound, 0)))
    bins = max(20, min(bins, 100))

    filtered = clean[(clean >= lower_bound) & (clean <= upper_bound)]

    plt.figure(figsize=(8, 6), dpi=dpi)
    plt.hist(filtered, bins=bins, edgecolor="0", zorder=2)
    plt.title(title, fontsize=12, color="0")
    plt.xlabel(xlabel, fontsize=12, color="0")
    plt.ylabel(ylabel, fontsize=12, color="0")
    plt.xticks(fontsize=10, color="0.3")
    plt.yticks(fontsize=10, color="0.3")
    plt.grid(axis="y", alpha=0.75, zorder=0)
    sns.despine()
    plt.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()


def plot_scatter(
        x, 
        y, 
        xlabel, 
        ylabel, 
        title, 
        out_path, 
        xlim=None, 
        ylim=None,
        r2=None, 
        dpi=120, 
        save_dpi=300, 
        diagonal=True
    ):
    """
    Scatterplot with an optional x=y reference line and R^2 annotation box.
    """
    plt.figure(figsize=(8, 7), dpi=dpi)
    plt.scatter(x, y, color="darkblue", marker=".", s=10, zorder=2)

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if diagonal and xlim and ylim:
        plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], "k--", label="x = y")

    plt.title(title, fontsize=12, color="0")
    plt.xlabel(xlabel, fontsize=12, color="0")
    plt.ylabel(ylabel, fontsize=12, color="0")
    plt.xticks(fontsize=10, color="0.3")
    plt.yticks(fontsize=10, color="0.3")
    plt.axhline(0, color="0", zorder=1)
    plt.grid(axis="y", alpha=0.75, zorder=0)
    sns.despine()

    if r2 is not None:
        plt.text(0.05, 0.95, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
                  fontsize=12, color="red", ha="left", va="top",
                  bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"))

    plt.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()