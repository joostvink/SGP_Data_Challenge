"""Small plotting helpers shared by the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

TEAM_COLOURS = {
    "AUS": "#f2b600",
    "BRA": "#1a9641",
    "CAN": "#7f7f7f",
    "DEN": "#d7191c",
    "ESP": "#e66101",
    "GBR": "#1f3b99",
    "GER": "#111111",
    "ITA": "#2b9fd9",
    "NZL": "#5e3c99",
    "SUI": "#c2185b",
    "USA": "#8c6d31",
}


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "font.size": 9,
        }
    )


def plot_course(ax, course, labels=True, boundary=True, mark_colour="k"):
    """Draw boundary, marks and gates of a course in local metres."""
    if boundary and "Boundary" in course.limits:
        b = course.limits["Boundary"]
        ax.plot(*np.vstack([b, b[:1]]).T, color="0.6", lw=1, ls="--", zorder=1)
    seen = set()
    for wp in course.waypoints:
        if wp.is_gate:
            ax.plot(wp.xy[:, 0], wp.xy[:, 1], color=mark_colour, lw=0.8, alpha=0.5, zorder=2)
        for name, (x, y) in zip(wp.mark_names, wp.xy):
            ax.plot(x, y, "o", color=mark_colour, ms=4, zorder=3)
            if labels and (name, round(x), round(y)) not in seen:
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=mark_colour)
                seen.add((name, round(x), round(y)))
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    return ax


def label_lines(ax, frame, colours=TEAM_COLOURS, fmt="{}", min_gap=None):
    """Write each column's name at the right-hand end of its line, nudged apart so labels do not overlap."""
    ends = {c: (frame[c].dropna().index[-1], frame[c].dropna().iloc[-1]) for c in frame if frame[c].notna().any()}
    lo, hi = ax.get_ylim()
    gap = min_gap if min_gap is not None else abs(hi - lo) / 28
    placed = []
    for name, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_text = y if not placed or y - placed[-1] >= gap else placed[-1] + gap
        placed.append(y_text)
        ax.annotate(fmt.format(name), (x, y_text), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=colours.get(name, "k"))
