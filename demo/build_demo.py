"""Build the interactive demo page from real simulator output.

    python demo/build_demo.py                         # seed 0, default scenario
    python demo/build_demo.py --seed 3 --scenario rak_office_to_site

Runs every strategy once on the same trip (same seed = same radio conditions),
packs the timeline, call quality and handover log into JSON, and inlines it
into demo/template.html -> demo/edge_demo.html.

The output is ONE self-contained file: open it in any browser, no server and
no internet needed (good for the NSTI stand). Re-run this script after any
change to the strategies or link profiles and the page shows the new behaviour.

If results/latest/summary.csv exists (from run.py), its 20-seed averages are
shown in the "Across 20 runs" panel; otherwise that panel is hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emu.model import DT, Scenario, load_profiles, load_scenario  # noqa: E402
from emu.sim import Simulator  # noqa: E402
from emu.strategies import STRATEGIES  # noqa: E402

STEP = 0.1  # demo resolution (s)
LINK_NAMES = {"wired": "Wired", "wifi": "Wi-Fi", "cellular_5g": "5G", "sat_leo": "LEO satellite", "sat_geo": "GEO satellite"}
STRAT_INFO = {
    "bbm": ("Break-before-make", "Today's default: fixed priority, reconnects after every switch"),
    "threshold": ("Threshold switching", "Leaves a weak link after 0.3 s; the session migrates"),
    "redundant": ("MPTCP redundant", "Sends every packet on two links at once"),
    "predictive_linear": ("Predictive, linear forecast", "Our controller with the original trend forecast"),
    "predictive": ("Predictive (ours)", "Kalman forecast of link quality; prepares the next link early"),
}
# The page shows the existing mechanisms against our final controller.
# predictive_linear is an ablation: it stays in run.py/analyze.py, not on the demo.
DEMO_STRATEGIES = ["bbm", "threshold", "redundant", "predictive"]


def per_step_ok(df, app, n_steps, frame_level=False):
    """Fraction (0-100) of this app's packets/frames delivered on time per STEP."""
    d = df[df.app == app]
    if d.empty:
        return None
    if frame_level:
        d = d.groupby("frame").agg(t_send=("t_send", "min"), on_time=("on_time", "all")).reset_index()
    b = (d.t_send / STEP).astype(int).clip(0, n_steps - 1)
    tot = np.bincount(b, minlength=n_steps)
    ok = np.bincount(b, weights=d.on_time.astype(float), minlength=n_steps)
    frac = np.where(tot > 0, ok / np.maximum(tot, 1), 1.0)
    return [int(round(100 * f)) for f in frac]


def zone_events(scn, cfg):
    """Times when the device enters/leaves dead zones and obstructions."""
    ev = []
    pos, t = scn.pos, scn.t

    def crossings(s0, s1, enter, leave):
        inside = (pos >= s0) & (pos <= s1)
        ch = np.flatnonzero(np.diff(inside.astype(int)))
        for i in ch:
            ev.append((round(float(t[i + 1]), 1), enter if inside[i + 1] else leave))

    for s0, s1, _ in cfg.get("cellular_5g", {}).get("shadow_zones", []):
        if s0 > 100:
            crossings(s0, s1, "Enters a mountain pass: 5G signal fades out", "Leaves the pass: 5G signal returns")
    obs = cfg.get("sat_leo", {}).get("obstructions", [])
    for s0, s1 in obs:
        if 100 < s0 < 2300:
            crossings(s0, s1, "Canyon walls block the view of the sky", "Out of the canyon: satellites visible again")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rak_office_to_site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", default=str(ROOT / "results" / "latest" / "summary.csv"))
    ap.add_argument("--out", default=str(ROOT / "demo" / "edge_demo.html"))
    a = ap.parse_args()

    cfg = load_scenario(a.scenario)
    prof = load_profiles()
    base = Scenario(cfg, prof, seed=a.seed)
    links = list(base.links)
    dur = base.duration
    n = int(round(dur / STEP)) + 1
    idx = (np.arange(n) * STEP / DT).round().astype(int).clip(0, len(base.t) - 1)

    data = {
        "scenario": cfg.get("name", a.scenario),
        "seed": a.seed,
        "duration": dur,
        "step": STEP,
        "links": [{"id": l, "name": LINK_NAMES.get(l, l), "qMin": prof[l]["q_min"],
                   "attachMs": prof[l]["attach_ms"], "powerW": prof[l].get("power_w", 0)} for l in links],
        "pos": [round(float(p), 1) for p in base.pos[idx]],
        "speed": [round(abs(float(v)) * 3.6) for v in base.speed[idx]],
        "q": {l: [int(round(100 * base.links[l].q[i])) for i in idx] for l in links},
        "geometry": {
            "length": float(base.pos.max()),
            "phases": cfg.get("phases", []),
            "aps": [s for s, _ in cfg.get("wifi", {}).get("access_points", [])],
            "cells": [s for s, _ in cfg.get("cellular_5g", {}).get("cells", [])],
            "deadZones": [[s0, s1] for s0, s1, _ in cfg.get("cellular_5g", {}).get("shadow_zones", []) if s0 > 100],
            "skyBlocked": [[s0, s1] for s0, s1 in cfg.get("sat_leo", {}).get("obstructions", [])],
        },
        "strategies": [],
    }

    events = [(float(t), txt) for t, txt in cfg.get("narration", [])]
    events += zone_events(base, cfg)
    for l in links:
        for t, txt in base.links[l].events:
            if "roam" in txt:
                a_, b_ = txt.split()[-1].split("->")
                events.append((round(t, 1), f"Wi-Fi hands over from access point {a_[2:]} to {b_[2:]}"))
    data["events"] = sorted(events)
    data["noCoverageS"] = round(base.no_coverage_s(), 1)

    for s in [x for x in DEMO_STRATEGIES if x in STRATEGIES]:
        scn = Scenario(cfg, prof, seed=a.seed)
        res, df, tl, per_sec, ho = Simulator(scn, STRATEGIES[s](), seed=a.seed).run()
        # timeline is sampled every 100 ms by the controller -> align to our steps
        # "no link" must survive the forward-fill below, so give it a real value first
        tl["primary"] = tl["primary"].where(tl["primary"].notna(), "__none__")
        tl = tl.set_index((tl.t / STEP).round().astype(int))
        tl = tl[~tl.index.duplicated()].reindex(range(n)).ffill().bfill()
        prim = [links.index(p) if isinstance(p, str) and p in links else -1 for p in tl.primary]
        states = ["".join(str(tl.at[i, f"state_{l}"])[0] for l in links) for i in range(n)]
        name, blurb = STRAT_INFO.get(s, (s, ""))
        data["strategies"].append({
            "id": s, "name": name, "blurb": blurb, "ours": s == "predictive",
            "primary": prim,
            "states": states,
            "voice": per_step_ok(df, "voip", n),
            "video": per_step_ok(df, "video", n, frame_level=True),
            "mos": [round(m, 2) for _, m in per_sec.get("voip_mos", [])],
            "handovers": [[round(t, 1), links.index(new) if new else -1, reason] for t, old, new, reason in ho],
            "run": {k: round(float(v), 2) for k, v in res.items() if isinstance(v, (int, float, np.floating))},
        })
        print(f"{s:10s} voice interruption {res['voip_interruption_s']:.1f} s, handovers {res['handovers']}")

    sp = Path(a.summary)
    if sp.exists():
        sm = pd.read_csv(sp)
        g = sm.groupby("strategy")
        cols = ["voip_interruption_s", "voip_bad_seconds", "video_stall_s", "overhead_pct", "energy_kj", "handovers"]
        data["summary"] = {
            "runs": int(sm.seed.nunique()),
            "metrics": {s: {c: [round(float(g[c].mean()[s]), 2),
                                round(float(1.96 * g[c].std(ddof=1)[s] / np.sqrt(g[c].count()[s])), 2)]
                            for c in cols if c in sm} for s in sm.strategy.unique()},
        }

    tpl = (ROOT / "demo" / "template.html").read_text()
    payload = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("/*__DATA__*/null", payload)
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
