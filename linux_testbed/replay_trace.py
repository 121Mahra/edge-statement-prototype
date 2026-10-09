"""Replay a simulated trip onto the real Linux testbed.

Reads the link-quality timeline produced by run.py (results/<run>/timeline_*.csv),
converts quality -> delay / jitter / loss / rate with the SAME model as the
simulator (emu/model.py:link_params), and applies it every 100 ms with tc netem
on the veth pairs created by setup.sh. Real applications running inside the
`cli` / `srv` namespaces then experience the trip.

    sudo python3 linux_testbed/replay_trace.py results/latest/timeline_bbm.csv
    python3 linux_testbed/replay_trace.py results/latest/timeline_bbm.csv --dry-run | head

Example experiment (two terminals):
    sudo ip netns exec srv iperf3 -s
    sudo ip netns exec cli mptcpize run iperf3 -c 10.0.2.2 -u -b 2.5M -t 200 -i 1   # MPTCP
    sudo ip netns exec cli iperf3 -c 10.0.2.2 -t 200 -i 1                             # plain TCP
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emu.model import link_params, load_profiles  # noqa: E402

IFACE = {"wired": "wired", "wifi": "wifi", "cellular_5g": "5g", "sat_leo": "leo", "sat_geo": "geo"}


def netem_args(p):
    if not p["up"]:
        return "delay 1ms loss 100%"
    rate = max(int(p["bw_kbps"]), 64)
    return (f"delay {p['delay_ms']:.1f}ms {p['jitter_ms']:.1f}ms distribution normal "
            f"loss {100 * p['loss']:.2f}% rate {rate}kbit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--dry-run", action="store_true", help="print tc commands instead of running them")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 replays faster than real time")
    a = ap.parse_args()

    prof = load_profiles()
    tl = pd.read_csv(a.timeline)
    links = [c[2:] for c in tl.columns if c.startswith("q_") and c[2:] in IFACE]
    t0 = time.time()
    for _, row in tl.iterrows():
        batch = {"cli": [], "srv": []}
        for ln in links:
            args = netem_args(link_params(prof[ln], row[f"q_{ln}"]))
            batch["cli"].append(f"qdisc replace dev c-{IFACE[ln]} root netem {args}")
            batch["srv"].append(f"qdisc replace dev s-{IFACE[ln]} root netem {args}")
        if a.dry_run:
            print(f"# t={row.t:.1f}s")
            for ns, cmds in batch.items():
                for c in cmds:
                    print(f"ip netns exec {ns} tc {c}")
            continue
        for ns, cmds in batch.items():
            subprocess.run(["ip", "netns", "exec", ns, "tc", "-batch", "-"],
                           input="\n".join(cmds) + "\n", text=True, check=True)
        # keep wall-clock in sync with the trace
        wait = row.t / a.speed - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
        if int(row.t * 10) % 50 == 0:
            print(f"t={row.t:6.1f}s  " + "  ".join(f"{IFACE[l]}={row[f'q_{l}']:.2f}" for l in links), flush=True)


if __name__ == "__main__":
    main()
