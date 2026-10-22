"""Plot helpers (one colour-blind-safe style, Okabe-Ito palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
HAM_COLOR, SPAM_COLOR = PALETTE[0], PALETTE[3]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e5e5e5"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
        "legend.frameon": False,
    })


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_length_distribution(texts, labels, path: Path) -> None:
    set_style()
    lengths = np.array([len(t) for t in texts])
    labels = np.asarray(labels)
    upper = np.percentile(lengths, 99)
    bins = np.linspace(0, upper, 40)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for lab, name, color in ((0, "ham", HAM_COLOR), (1, "spam", SPAM_COLOR)):
        ax.hist(lengths[labels == lab], bins=bins, density=True, histtype="stepfilled",
                alpha=0.35, color=color, edgecolor=color, linewidth=1.5, label=name)
    ax.set(title="Message length by class", xlabel="Characters per message", ylabel="Density")
    ax.legend()
    _save(fig, path)


def plot_pr_curves(y_true, scores: dict[str, np.ndarray], path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for color, (name, s) in zip(PALETTE, scores.items()):
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax.plot(rec, prec, color=color, label=name)
    ax.set(title="Precision-recall curves (test set)", xlabel="Recall (spam caught)",
           ylabel="Precision", ylim=(0.0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.grid(False)
    ax.imshow(cm, cmap="Blues")
    labels = ["ham", "spam"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def plot_top_terms(spam_terms, ham_terms, path: Path) -> None:
    """Each argument: list of (term, weight). Two panels of horizontal bars."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, terms, color, title in (
        (axes[0], ham_terms, HAM_COLOR, "Strongest 'ham' terms"),
        (axes[1], spam_terms, SPAM_COLOR, "Strongest 'spam' terms"),
    ):
        names = [t for t, _ in terms][::-1]
        weights = [abs(w) for _, w in terms][::-1]
        ax.barh(names, weights, color=color)
        ax.set(title=title, xlabel="|logistic-regression weight|")
        ax.grid(axis="y", visible=False)
    _save(fig, path)
