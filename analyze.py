"""Turn results into the figures and table for your deck.

    python analyze.py --out results/latest

Produces in <out>/figures/:
    fig1_scenario.png    link quality over the trip + which link each strategy used
    fig2_comparison.png  mean +/- 95% CI of the key metrics, all strategies
    fig3_mos.png         per-second VoIP MOS over the trip (seed 0)
    results_table.md     the numbers, ready to paste
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical order is fixed per strategy (never cycled) - reference dataviz palette.
COLORS = {"bbm": "#2a78d6", "threshold": "#eb6834", "redundant": "#1baf7a", "predictive_linear": "#e87ba4", "predictive": "#4a3aa7"}
LABELS = {"bbm": "Break-before-make (today)", "threshold": "Threshold MBB + migration",
          "redundant": "MPTCP redundant", "predictive_linear": "Predictive, linear forecast",
          "predictive": "Predictive, Kalman (ours)"}
SHORT = {"bbm": "BBM", "threshold": "Threshold", "redundant": "Redundant", "predictive_linear": "Linear", "predictive": "Kalman"}
LINK_LABELS = {"wired": "Wired", "wifi": "Wi-Fi", "cellular_5g": "5G", "sat_leo": "LEO sat", "sat_geo": "GEO sat"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig_scenario(out, strategies, figdir):
    tl0 = pd.read_csv(out / f"timeline_{strategies[0]}.csv")
    links = [c[2:] for c in tl0.columns if c.startswith("q_")]
    n = len(links) + 1
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.15 * n + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(links) + [1.3]})
    for ax, ln in zip(axes, links):
        ax.fill_between(tl0.t, tl0[f"q_{ln}"], color="#8f8e89", alpha=0.35, linewidth=0)
        ax.plot(tl0.t, tl0[f"q_{ln}"], color="#52514e", linewidth=1.2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel(LINK_LABELS.get(ln, ln), rotation=0, ha="right", va="center")
    axes[0].set_title("Link quality along the trip (0 = unusable, 1 = excellent) and the link each strategy used")

    ax = axes[-1]
    y = {ln: i for i, ln in enumerate(links)}
    for k, s in enumerate(strategies):
        tl = pd.read_csv(out / f"timeline_{s}.csv")
        yy = tl.primary.map(y).astype(float) + (k - (len(strategies) - 1) / 2) * 0.14
        ax.step(tl.t, yy, where="post", color=COLORS.get(s), linewidth=2, label=LABELS.get(s, s))
    ax.set_yticks(range(len(links)))
    ax.set_yticklabels([LINK_LABELS.get(l, l) for l in links])
    ax.invert_yaxis()
    ax.set_ylabel("in use", rotation=0, ha="right", va="center")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=len(strategies), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_scenario.png", dpi=160)
    plt.close(fig)


def fig_comparison(df, strategies, figdir):
    metrics = [
        ("voip_interruption_s", "Voice interruption (s)", "lower is better"),
        ("voip_bad_seconds", "Seconds with MOS < 3.1", "lower is better"),
        ("video_stall_s", "Video stall time (s)", "lower is better"),
        ("voip_mos", "Call quality (MOS)", "higher is better"),
        ("overhead_pct", "Duplicate traffic (%)", "lower is better"),
        ("energy_kj", "Radio energy (kJ)", "lower is better"),
    ]
    metrics = [m for m in metrics if m[0] in df]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (col, title, note) in zip(axes.flat, metrics):
        g = df.groupby("strategy")[col]
        means = g.mean().reindex(strategies)
        errs = g.apply(ci95).reindex(strategies)
        xs = np.arange(len(strategies))
        ax.bar(xs, means, yerr=errs, width=0.62, color=[COLORS[s] for s in strategies],
               edgecolor=SURF, linewidth=2, capsize=3, error_kw={"ecolor": INK2, "elinewidth": 1})
        for x, m in zip(xs, means):
            ax.text(x, m, f"{m:.2f}" if m < 10 else f"{m:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT.get(s, s) for s in strategies], fontsize=8.5)
        ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, (means + errs).max() * 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in strategies]
    fig.legend(handles, [LABELS[s] for s in strategies], loc="lower center", ncol=len(strategies), frameon=False)
    fig.suptitle(f"Mean over {df.seed.nunique()} runs (error bars = 95% confidence interval)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(figdir / "fig2_comparison.png", dpi=160)
    plt.close(fig)


def fig_mos(out, strategies, figdir, events):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for s in strategies:
        ps = pd.read_csv(out / f"persec_{s}.csv").sort_values("sec")
        ax.plot(ps.sec, ps.voip_mos, color=COLORS[s], linewidth=2, label=LABELS[s])
    ax.axhline(3.1, color=INK2, linewidth=1, linestyle="--")
    ax.text(1, 3.13, "MOS 3.1: many users dissatisfied", fontsize=8, color=INK2)
    for t, txt in events:
        if "roam" in txt:
            ax.axvline(t, color=GRID, linewidth=1)
    ax.set_ylim(1, 4.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("VoIP MOS (per second)")
    ax.set_title("Call quality second by second (seed 0); thin vertical lines = Wi-Fi AP roams")
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_mos.png", dpi=160)
    plt.close(fig)


def table(df, strategies, figdir):
    cols = ["voip_mos", "voip_bad_seconds", "voip_interruption_s", "voip_longest_gap_s", "video_frozen_pct",
            "video_stall_s", "handovers", "overhead_pct", "energy_kj"]
    cols = [c for c in cols if c in df]
    g = df.groupby("strategy")[cols]
    m, c = g.mean().reindex(strategies), g.agg(ci95).reindex(strategies)
    lines = ["| strategy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s in strategies:
        lines.append(f"| {s} | " + " | ".join(f"{m.loc[s, k]:.2f} ± {c.loc[s, k]:.2f}" for k in cols) + " |")
    txt = "\n".join(lines) + f"\n\nRuns per strategy: {df.seed.nunique()}. Physical coverage hole: {df.no_coverage_s.iloc[0]:.1f} s.\n"
    (figdir / "results_table.md").write_text(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latest")
    a = ap.parse_args()
    out = Path(a.out)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    df = pd.read_csv(out / "summary.csv")
    strategies = [s for s in COLORS if s in set(df.strategy)] + [s for s in df.strategy.unique() if s not in COLORS]
    for s in strategies:
        COLORS.setdefault(s, "#e87ba4")
        LABELS.setdefault(s, s)
    ev = pd.read_csv(out / "events.csv") if (out / "events.csv").exists() else pd.DataFrame(columns=["t", "event"])
    fig_scenario(out, strategies, figdir)
    fig_comparison(df, strategies, figdir)
    fig_mos(out, strategies, figdir, list(ev.itertuples(index=False, name=None)))
    print(table(df, strategies, figdir))
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
