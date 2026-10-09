"""Run experiments: every strategy x N seeds on one scenario.

    python run.py                                   # default: all strategies, 20 seeds
    python run.py --seeds 5 --strategies bbm predictive
    python run.py --scenario my_scenario --out results/my_run

Outputs (in --out):
    summary.csv          one row per (strategy, seed) with every metric
    timeline_<s>.csv     seed-0 timeline: primary link + link quality every 100 ms
    persec_<s>.csv       seed-0 per-second VoIP MOS and video frozen fraction
    handovers_<s>.csv    seed-0 handover log (time, from, to, reason)
Then run:  python analyze.py --out <same folder>
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from emu.model import Scenario, load_profiles, load_scenario
from emu.sim import Simulator
from emu.strategies import STRATEGIES


def one(job):
    scen_name, strat, seed, out_dir = job
    scn = Scenario(load_scenario(scen_name), load_profiles(), seed=seed)
    res, df, tl, per_sec, ho = Simulator(scn, STRATEGIES[strat](), seed=seed).run()
    if seed == 0:
        out = Path(out_dir)
        tl.to_csv(out / f"timeline_{strat}.csv", index=False)
        ps = pd.DataFrame(per_sec.get("voip_mos", []), columns=["sec", "voip_mos"])
        if per_sec.get("video_frozen"):
            ps = ps.merge(pd.DataFrame(per_sec["video_frozen"], columns=["sec", "video_frozen"]), on="sec", how="outer")
        ps.to_csv(out / f"persec_{strat}.csv", index=False)
        pd.DataFrame(ho, columns=["t", "from", "to", "reason"]).to_csv(out / f"handovers_{strat}.csv", index=False)
        events = [(t, txt) for lt in scn.links.values() for t, txt in lt.events]
        pd.DataFrame(sorted(events), columns=["t", "event"]).to_csv(out / "events.csv", index=False)
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rak_office_to_site")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.scenario, s, seed, str(out)) for s in a.strategies for seed in range(a.seeds)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = pool.map(one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    (out / "run.json").write_text(json.dumps(vars(a), indent=2))

    cols = ["voip_mos", "voip_bad_seconds", "voip_interruption_s", "video_frozen_pct",
            "video_stall_s", "handovers", "overhead_pct", "energy_kj"]
    cols = [c for c in cols if c in df]
    print(f"{len(jobs)} runs in {time.time() - t0:.1f}s  ->  {out}/summary.csv")
    print(f"(physical coverage hole in this scenario: {df.no_coverage_s.iloc[0]:.1f} s — no strategy can avoid it)\n")
    print(df.groupby("strategy")[cols].mean().reindex(a.strategies).round(2).to_string())


if __name__ == "__main__":
    main()
