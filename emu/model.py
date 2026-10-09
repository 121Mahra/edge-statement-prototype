"""Scenario + link-quality model.

Turns the YAML scenario (where the user is, at what speed, which APs / cells /
satellites exist) into a quality trace q(t) in [0, 1] for every link, sampled
every DT seconds. Everything downstream (packet loss, delay, bandwidth,
whether a link is up) is derived from these traces.

Read this file first: it is the "physics" of the testbed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DT = 0.01  # trace resolution: 10 ms

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_profiles(path=ROOT / "config" / "links.yaml"):
    return load_yaml(path)


def load_scenario(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = ROOT / "config" / "scenarios" / f"{name_or_path}.yaml"
    return load_yaml(p)


@dataclass
class LinkTrace:
    name: str
    profile: dict
    q: np.ndarray                  # quality per time step
    signal_dbm: np.ndarray | None  # RSSI/RSRP-like value (None for satellite/wired)
    events: list = field(default_factory=list)  # (time_s, text) e.g. AP roams

    def q_at(self, t: float) -> float:
        i = int(t / DT)
        if i < 0:
            i = 0
        elif i >= len(self.q):
            i = len(self.q) - 1
        return float(self.q[i])


class Scenario:
    """Mobility + all link traces for one run (one random seed)."""

    def __init__(self, scenario: dict, profiles: dict, seed: int = 0):
        self.cfg = scenario
        self.profiles = profiles
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.duration = float(scenario["duration_s"])
        self.t = np.arange(0.0, self.duration + DT, DT)

        wp = np.array(scenario["waypoints"], dtype=float)
        self.pos = np.interp(self.t, wp[:, 0], wp[:, 1])          # metres along route
        self.speed = np.gradient(self.pos, DT)                     # m/s, sign = direction

        self.links: dict[str, LinkTrace] = {}
        for name in scenario["links"]:
            prof = profiles[name]
            builder = {
                "ap": self._build_ap,
                "cell": self._build_cell,
                "sat": self._build_sat,
                "dock": self._build_dock,
            }[prof["kind"]]
            self.links[name] = builder(name, prof, scenario.get(name, {}))

    # ---------------------------------------------------------------- helpers
    def _spatial_shadowing(self, std_db: float, corr_m: float = 20.0) -> np.ndarray:
        """Log-normal shadowing that depends on WHERE you are (not on time),
        so standing still gives a stable signal and moving fast changes it fast."""
        lo, hi = self.pos.min() - corr_m, self.pos.max() + corr_m
        grid = np.arange(lo, hi + corr_m, corr_m)
        field_ = self.rng.normal(0.0, std_db, size=len(grid))
        slow = np.interp(self.pos, grid, field_)
        fast = self.rng.normal(0.0, 1.0, size=len(self.t))  # small fast fading
        return slow + fast

    @staticmethod
    def _to_q(sig_dbm, best, worst):
        return np.clip((sig_dbm - worst) / (best - worst), 0.0, 1.0)

    @staticmethod
    def _path_signal(pos, s_node, offset, tx_dbm, n):
        d = np.sqrt((pos - s_node) ** 2 + offset ** 2)
        return tx_dbm - 10.0 * n * np.log10(np.maximum(d, 1.0))

    def _zones_mask(self, zones):
        m = np.zeros_like(self.pos, dtype=bool)
        for z in zones or []:
            m |= (self.pos >= z[0]) & (self.pos <= z[1])
        return m

    # --------------------------------------------------------------- builders
    def _build_ap(self, name, prof, cfg):
        """Wi-Fi: the device associates to one AP at a time and roams (with a
        short outage) when another AP is `roam_hysteresis_db` stronger."""
        aps = cfg.get("access_points", [])
        per_ap = np.array([
            self._path_signal(self.pos, s, off, prof["tx_dbm"], prof["path_loss_exp"]) for s, off in aps
        ])
        shadow = self._spatial_shadowing(prof["shadowing_db"])
        hyst = prof.get("roam_hysteresis_db", 6)
        gap_steps = int(prof.get("roam_gap_ms", 0) / 1000 / DT)

        cur = int(np.argmax(per_ap[:, 0]))
        sig = np.empty_like(self.t)
        events = []
        roam_until = -1
        for i in range(len(self.t)):
            best = int(np.argmax(per_ap[:, i]))
            # roam only if another AP is usable AND clearly better (hysteresis)
            if (best != cur and per_ap[best, i] > prof["rssi_worst"]
                    and per_ap[best, i] > per_ap[cur, i] + hyst):
                events.append((float(self.t[i]), f"wifi roam AP{cur + 1}->AP{best + 1}"))
                cur = best
                roam_until = i + gap_steps
            sig[i] = per_ap[cur, i]
            if i < roam_until:
                sig[i] = -200.0  # re-association outage
        sig = sig + shadow
        q = self._to_q(sig, prof["rssi_best"], prof["rssi_worst"])
        return LinkTrace(name, prof, q, sig, events)

    def _build_cell(self, name, prof, cfg):
        cells = cfg.get("cells", [])
        per_cell = np.array([
            self._path_signal(self.pos, s, off, prof["tx_dbm"], prof["path_loss_exp"]) for s, off in cells
        ])
        sig = per_cell.max(axis=0)  # simplification: always on the best cell
        ramp = cfg.get("shadow_ramp_m", 150.0)  # terrain shadow fades in/out over this distance
        for s0, s1, att in cfg.get("shadow_zones", []):
            inside = np.clip(np.minimum(self.pos - s0, s1 - self.pos) / ramp + 1.0, 0.0, 1.0)
            if s0 <= 0:  # zone starting at the route origin (e.g. indoors) has no ramp on that side
                inside = np.clip((s1 - self.pos) / ramp + 1.0, 0.0, 1.0) * (self.pos >= s0)
            sig = sig - att * inside
        sig = sig + self._spatial_shadowing(prof["shadowing_db"], corr_m=50.0)
        q = self._to_q(sig, prof["rssi_best"], prof["rssi_worst"])
        return LinkTrace(name, prof, q, sig)

    def _build_sat(self, name, prof, cfg):
        q = prof["nominal_q"] + self.rng.normal(0, 0.02, size=len(self.t))
        q[self._zones_mask(cfg.get("obstructions"))] = 0.0
        events = []
        period = prof.get("reconfig_period_s", 0)
        gap = int(prof.get("reconfig_gap_ms", 0) / 1000 / DT)
        if period and gap:
            phase = self.rng.uniform(0, period)
            for t0 in np.arange(phase, self.duration, period):
                i = int(t0 / DT)
                q[i:i + gap] = np.minimum(q[i:i + gap], 0.05)
                events.append((float(t0), f"{name} satellite reconfiguration"))
        return LinkTrace(name, prof, np.clip(q, 0, 1), None, events)

    def _build_dock(self, name, prof, cfg):
        q = np.zeros_like(self.t)
        for a, b in cfg.get("docked", []):
            q[(self.t >= a) & (self.t < b)] = 1.0
        return LinkTrace(name, prof, q, None)


    def no_coverage_s(self) -> float:
        """Seconds where NO link at all is usable: a physical coverage hole.
        Report it next to interruption time so reviewers see what is avoidable."""
        usable = np.zeros_like(self.t, dtype=bool)
        for lt in self.links.values():
            usable |= lt.q >= lt.profile["q_min"]
        return float((~usable).sum() * DT)


# ------------------------------------------------------------------ link math
def link_params(prof: dict, q: float) -> dict:
    """Map quality q -> packet-level parameters (see comments in links.yaml)."""
    up = q >= prof["q_min"]
    delay = prof["base_delay_ms"] + (1.0 - q) * prof["extra_delay_ms"]
    loss = prof["loss_floor"] + prof["loss_max"] * max(0.0, (prof["q_good"] - q) / prof["q_good"])
    bw = prof["bw_max_kbps"] * q
    return {"up": up, "delay_ms": delay, "jitter_ms": prof["jitter_ms"], "loss": min(loss, 1.0), "bw_kbps": bw}
