"""Discrete-event, packet-level simulator.

    apps  --packets-->  strategy.paths_for()  -->  interfaces/links  -->  receiver
                              ^                                             |
                              +---- observations (q, RTT, loss) <-- feedback+

One run = one scenario + one strategy + one random seed. Everything is
deterministic for a given seed, so results are reproducible and you can run
many seeds to get confidence intervals.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque

import numpy as np
import pandas as pd

from . import metrics as M
from .model import DT, Scenario, link_params

TICK = DT               # 10 ms: interface state machine
CTRL_TICK = 0.100       # 100 ms: strategy decision period (like a measurement report)
MEAS_NOISE = 0.03       # std of quality measurement noise
DETACH_AFTER = 0.5      # s below q_min before the OS declares "link lost"
MAX_QUEUE = 0.2         # s of queued data per link before tail drop
SESSION_RTTS = 3        # TCP + TLS + app resume after a break-before-make switch


class Interface:
    """One network interface. States: down -> attaching -> up."""

    def __init__(self, name, trace):
        self.name = name
        self.trace = trace
        self.prof = trace.profile
        self.state = "down"
        self.want = False
        self.attach_done = 0.0
        self.below_since = None
        self.tx_free = 0.0
        self.on_time_s = 0.0
        self.srtt = None      # smoothed RTT (s) from feedback
        self.loss_ewma = 0.0

    def params(self, t):
        return link_params(self.prof, self.trace.q_at(t))

    def step(self, t):
        q = self.trace.q_at(t)
        usable = q >= self.prof["q_min"]
        if self.state == "up":
            if usable:
                self.below_since = None
            else:
                self.below_since = self.below_since if self.below_since is not None else t
                if t - self.below_since >= self.prof.get("detach_ms", DETACH_AFTER * 1000) / 1000:
                    self.state, self.below_since = "down", None
            if not self.want:
                self.state = "down"
        elif self.state == "attaching":
            if not usable or not self.want:
                self.state = "down"
            elif t >= self.attach_done:
                self.state = "up"
        else:  # down
            if self.want and usable:
                self.state = "attaching"
                self.attach_done = t + self.prof["attach_ms"] / 1000.0
        if self.state != "down":
            self.on_time_s += TICK


class Ctx:
    """What a strategy is allowed to see and do."""

    def __init__(self, sim):
        self._sim = sim
        self.names = list(sim.ifaces)
        self.primary = None
        self.t = 0.0

    def profile(self, name):
        return self._sim.ifaces[name].prof

    def state(self, name):
        return self._sim.ifaces[name].state

    def is_up(self, name):
        return self._sim.ifaces[name].state == "up"

    def want_up(self, name, on=True):
        self._sim.ifaces[name].want = on

    def switch_primary(self, name, reason=""):
        if name == self.primary:
            return
        if name is not None and not self.is_up(name):
            return
        self._sim.on_switch(self.primary, name, reason)
        self.primary = name


class Simulator:
    def __init__(self, scenario: Scenario, strategy, seed=0, video_kbps=2500, fps=30):
        self.scn = scenario
        self.strategy = strategy
        self.rng = np.random.default_rng(seed + 1000)
        self.ifaces = {n: Interface(n, tr) for n, tr in scenario.links.items()}
        self.ctx = Ctx(self)
        self.video_kbps, self.fps = video_kbps, fps
        self.apps = scenario.cfg.get("apps", ["voip", "video"])

        self._q = []
        self._cnt = itertools.count()
        self.pkts = []            # [app, seq, frame, size, t_send, n_paths, t_arrive]
        self.outstanding = {n: deque() for n in self.ifaces}   # (t_send, pkt_id)
        self.acked = set()
        self.handovers = []
        self.session_ready_at = 0.0
        self.bytes_offered = 0
        self.bytes_sent = 0
        self.bytes_dup = 0
        self.timeline = []       # per CTRL_TICK: t, primary, q per link, state per link

    # ------------------------------------------------------------ event queue
    def at(self, t, fn, *args):
        heapq.heappush(self._q, (t, next(self._cnt), fn, args))

    def run(self):
        end = self.scn.duration
        self.strategy.on_start(self.ctx)
        # at t=0 the device is already connected to whatever it wants and has
        for itf in self.ifaces.values():
            if itf.want and itf.params(0)["up"]:
                itf.state = "up"
        self.strategy.on_tick(0.0, self.observe(0.0), self.ctx)

        self.at(0.0, self.tick)
        self.at(CTRL_TICK, self.ctrl_tick)
        if "voip" in self.apps:
            self.at(0.0, self.gen_voip, 0)
        if "video" in self.apps:
            self.at(0.0, self.gen_video, 0)

        while self._q:
            t, _, fn, args = heapq.heappop(self._q)
            if t > end + 1.0:
                break
            self.ctx.t = t
            fn(t, *args)
        return self.results()

    # ------------------------------------------------------------ periodic
    def tick(self, t):
        for itf in self.ifaces.values():
            itf.step(t)
        if self.ctx.primary and not self.ctx.is_up(self.ctx.primary):
            self.on_switch(self.ctx.primary, None, "primary link lost")
            self.ctx.primary = None
        if t + TICK <= self.scn.duration:
            self.at(t + TICK, self.tick)

    def observe(self, t):
        obs = {}
        for n, itf in self.ifaces.items():
            q = itf.trace.q_at(t)
            obs[n] = {
                "q": float(np.clip(q + self.rng.normal(0, MEAS_NOISE), 0, 1)),
                "state": itf.state,
                "srtt_ms": itf.srtt * 1000 if itf.srtt else None,
                "loss": itf.loss_ewma,
            }
        return obs

    def ctrl_tick(self, t):
        # loss detection: packets not acked within max(0.3 s, 3*srtt)
        for n, itf in self.ifaces.items():
            to = max(0.3, 3 * (itf.srtt or 0.1))
            dq = self.outstanding[n]
            while dq and t - dq[0][0] > to:
                ts, pid = dq.popleft()
                if (pid, n) not in self.acked:
                    itf.loss_ewma = 0.9 * itf.loss_ewma + 0.1
                    self.strategy.on_feedback(t, n, False, None)
        obs = self.observe(t)
        self.strategy.on_tick(t, obs, self.ctx)
        self.timeline.append((t, self.ctx.primary,
                              *[self.ifaces[n].trace.q_at(t) for n in self.ifaces],
                              *[self.ifaces[n].state for n in self.ifaces]))
        if t + CTRL_TICK <= self.scn.duration:
            self.at(t + CTRL_TICK, self.ctrl_tick)

    # ------------------------------------------------------------ switching
    def on_switch(self, old, new, reason):
        self.handovers.append((self.ctx.t, old, new, reason))
        if new is not None and getattr(self.strategy, "session_mode", "migrate") == "reconnect":
            # break-before-make: the transport session must be rebuilt on the new path
            rtt = 2 * self.ifaces[new].params(self.ctx.t)["delay_ms"] / 1000
            self.session_ready_at = self.ctx.t + SESSION_RTTS * rtt

    # ------------------------------------------------------------ traffic
    def gen_voip(self, t, seq):
        self.send(t, "voip", seq, -1, 200)
        nt = t + M.VOIP_INTERVAL
        if nt < self.scn.duration:
            self.at(nt, self.gen_voip, seq + 1)

    def gen_video(self, t, frame):
        size = int(self.video_kbps * 1000 / 8 / self.fps)
        if frame % (2 * self.fps) == 0:
            size *= 3  # key frame every 2 s
        n = int(np.ceil(size / 1200))
        for k in range(n):
            self.send(t, "video", frame * 1000 + k, frame, min(1200, size - k * 1200))
        nt = t + 1.0 / self.fps
        if nt < self.scn.duration:
            self.at(nt, self.gen_video, frame + 1)

    def send(self, t, app, seq, frame, size):
        pid = len(self.pkts)
        rec = [app, seq, frame, size, t, 0, np.nan]
        self.pkts.append(rec)
        self.bytes_offered += size
        if t < self.session_ready_at:
            return  # transport session still re-connecting
        paths = self.strategy.paths_for(t, app, self.ctx) or []
        rec[5] = len(paths)
        for i, n in enumerate(paths):
            if self.transmit(t, pid, n, size) and i > 0:
                self.bytes_dup += size

    def transmit(self, t, pid, name, size):
        itf = self.ifaces[name]
        if itf.state != "up":
            return False
        self.bytes_sent += size
        p = itf.params(t)
        self.outstanding[name].append((t, pid))
        if not p["up"] or self.rng.random() < p["loss"]:
            return True
        start = max(t, itf.tx_free)
        if start - t > MAX_QUEUE:
            return True  # queue overflow (sent, but dropped in the queue)
        itf.tx_free = start + size * 8 / (max(p["bw_kbps"], 1.0) * 1000)
        arrive = itf.tx_free + (p["delay_ms"] + abs(self.rng.normal(0, p["jitter_ms"]))) / 1000
        self.at(arrive, self.receive, pid, name, t, p["delay_ms"] / 1000)
        return True

    def receive(self, t, pid, name, t_send, back_delay):
        rec = self.pkts[pid]
        if np.isnan(rec[6]) or t < rec[6]:
            rec[6] = t  # keep first copy (duplicates are discarded)
        self.at(t + back_delay, self.feedback, pid, name, t_send)

    def feedback(self, t, pid, name, t_send):
        itf = self.ifaces[name]
        rtt = t - t_send
        itf.srtt = rtt if itf.srtt is None else 0.875 * itf.srtt + 0.125 * rtt
        itf.loss_ewma *= 0.9
        self.acked.add((pid, name))
        self.strategy.on_feedback(t, name, True, rtt)

    # ------------------------------------------------------------ results
    def results(self):
        df = pd.DataFrame(self.pkts, columns=["app", "seq", "frame", "size", "t_send", "n_paths", "t_arrive"])
        lat = df.t_arrive - df.t_send
        deadline = np.where(df.app == "voip", M.VOIP_DEADLINE, M.VIDEO_DEADLINE)
        df["on_time"] = lat.notna() & (lat <= deadline)

        out = {"strategy": self.strategy.name, "seed": self.scn.seed}
        per_sec = {}
        if "voip" in self.apps:
            vm, per_sec["voip_mos"] = M.voip_metrics(df, self.scn.duration)
            out.update(vm)
        if "video" in self.apps:
            vi, per_sec["video_frozen"] = M.video_metrics(df, self.fps)
            out.update(vi)
        out["handovers"] = sum(1 for h in self.handovers if h[2] is not None)
        out["overhead_pct"] = 100.0 * self.bytes_dup / max(self.bytes_offered, 1)
        out["no_coverage_s"] = self.scn.no_coverage_s()  # nobody can fix these seconds
        out["energy_kj"] = sum(i.on_time_s * i.prof.get("power_w", 0) for i in self.ifaces.values()) / 1000
        for n, i in self.ifaces.items():
            out[f"on_s_{n}"] = round(i.on_time_s, 1)

        names = list(self.ifaces)
        tl = pd.DataFrame(self.timeline, columns=["t", "primary", *[f"q_{n}" for n in names],
                                                  *[f"state_{n}" for n in names]])
        return out, df, tl, per_sec, self.handovers
