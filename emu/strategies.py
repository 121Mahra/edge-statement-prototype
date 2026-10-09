"""Handover strategies.

Every strategy answers three questions, over and over:
  1. Which interfaces should be switched on (warm)?     -> ctx.want_up()
  2. Which one carries the traffic (primary)?            -> ctx.switch_primary()
  3. For this packet, which path(s) do we send it on?   -> paths_for()

Baselines (existing mechanisms you compare against):
  bbm        : what a typical device does today. Fixed priority
               (wired > Wi-Fi > 5G > satellite), switches only when the link
               is lost or a higher-priority link appears, and the app's TCP/TLS
               session has to reconnect after every switch (break-before-make).
  threshold  : policy-based make-before-break (like Wi-Fi-calling / ANDSF /
               3GPP A3-style rules): leave a link when its quality stays below a
               threshold for a time-to-trigger, with hysteresis. Session
               survives the switch (QUIC connection migration / MPTCP).
  redundant  : MPTCP "redundant" scheduler: every packet on two paths at once.
               Very robust, but doubles traffic and keeps satellite powered.

Proposed (starting point for YOUR idea):
  predictive : predicts each link's quality a few seconds ahead, scores links by
               predicted user experience (E-model MOS) minus energy cost,
               pre-warms the next link BEFORE it is needed, and duplicates only
               the critical app (voice) only during the risky handover window.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .metrics import emodel_mos
from .model import link_params

PRIORITY = ["wired", "wifi", "cellular_5g", "sat_leo", "sat_geo"]
TERRESTRIAL = ["wired", "wifi", "cellular_5g"]


def ordered(ctx):
    return [n for n in PRIORITY if n in ctx.names]


class Strategy:
    name = "base"
    session_mode = "migrate"   # "reconnect" = break-before-make transport

    def on_start(self, ctx):
        pass

    def on_tick(self, t, obs, ctx):
        pass

    def paths_for(self, t, app, ctx):
        return [ctx.primary] if ctx.primary else []

    def on_feedback(self, t, link, ok, rtt):
        pass


# ---------------------------------------------------------------- baseline 1
class BreakBeforeMake(Strategy):
    name = "bbm"
    session_mode = "reconnect"

    def __init__(self, sat_fallback_after=2.0):
        self.sat_fallback_after = sat_fallback_after
        self.no_terr_since = None

    def on_start(self, ctx):
        for n in TERRESTRIAL:
            if n in ctx.names:
                ctx.want_up(n)

    def on_tick(self, t, obs, ctx):
        terr_up = [n for n in TERRESTRIAL if n in ctx.names and ctx.is_up(n)]
        # satellite only as a manual-ish last resort
        if not terr_up:
            self.no_terr_since = self.no_terr_since if self.no_terr_since is not None else t
            if t - self.no_terr_since >= self.sat_fallback_after and "sat_leo" in ctx.names:
                ctx.want_up("sat_leo")
        else:
            self.no_terr_since = None
            if ctx.primary not in ("sat_leo", "sat_geo") and "sat_leo" in ctx.names:
                ctx.want_up("sat_leo", False)
        # highest-priority interface that is up becomes the default route
        for n in ordered(ctx):
            if ctx.is_up(n):
                if n != ctx.primary:
                    ctx.switch_primary(n, "priority/default route")
                break


# ---------------------------------------------------------------- baseline 2
class ThresholdMBB(Strategy):
    name = "threshold"

    def __init__(self, ttt=0.3, hyst=0.10):
        self.ttt, self.hyst = ttt, hyst
        self.bad_since = None
        self.better_since = {}

    def on_start(self, ctx):
        for n in TERRESTRIAL:
            if n in ctx.names:
                ctx.want_up(n)

    def on_tick(self, t, obs, ctx):
        good = lambda n: obs[n]["q"] >= ctx.profile(n)["q_good"]
        # warm the LEO backup when every terrestrial link is weak
        best_terr = max((obs[n]["q"] for n in TERRESTRIAL if n in ctx.names), default=0)
        if "sat_leo" in ctx.names:
            if best_terr < 0.25:
                ctx.want_up("sat_leo")
            elif best_terr > 0.45 and ctx.primary not in ("sat_leo", "sat_geo"):
                ctx.want_up("sat_leo", False)
        if "sat_geo" in ctx.names:
            ctx.want_up("sat_geo", not any(ctx.is_up(n) for n in ordered(ctx) if n != "sat_geo"))

        cur = ctx.primary
        up = [n for n in ordered(ctx) if ctx.is_up(n)]
        if not up:
            return
        if cur is None or not ctx.is_up(cur):
            ctx.switch_primary(next((n for n in up if good(n)), up[0]), "current lost")
            return
        # leave current if below its threshold for time-to-trigger
        if not good(cur):
            self.bad_since = self.bad_since if self.bad_since is not None else t
            if t - self.bad_since >= self.ttt:
                cand = [n for n in up if n != cur and good(n)]
                if cand:
                    ctx.switch_primary(cand[0], "below threshold")
                    self.bad_since = None
                    return
        else:
            self.bad_since = None
        # return to a higher-priority link once it is clearly good (hysteresis)
        for n in up:
            if n == cur:
                break
            if obs[n]["q"] >= ctx.profile(n)["q_good"] + self.hyst:
                self.better_since.setdefault(n, t)
                if t - self.better_since[n] >= self.ttt:
                    ctx.switch_primary(n, "higher priority is good")
                    self.better_since.clear()
                    return
            else:
                self.better_since.pop(n, None)


# ---------------------------------------------------------------- baseline 3
class Redundant(ThresholdMBB):
    name = "redundant"

    def on_start(self, ctx):
        super().on_start(ctx)
        if "sat_leo" in ctx.names:
            ctx.want_up("sat_leo")

    def on_tick(self, t, obs, ctx):
        super().on_tick(t, obs, ctx)
        if "sat_leo" in ctx.names:
            ctx.want_up("sat_leo")  # always keep a second path ready

    def paths_for(self, t, app, ctx):
        up = [n for n in ordered(ctx) if ctx.is_up(n)]
        if ctx.primary in up:
            up.remove(ctx.primary)
            return [ctx.primary] + up[:1]
        return up[:2]


# ---------------------------------------------------------------- proposed
class Predictive(Strategy):
    """Anticipatory, QoE-aware, energy-aware handover.

    >>> This is where your own idea goes. The three methods marked HOOK are
    >>> the natural places to change: how you PREDICT, how you SCORE a link,
    >>> and when you consider the moment RISKY enough to duplicate packets.
    """
    name = "predictive"

    def __init__(self, horizon=1.5, window=30, margin=0.15, energy_weight=0.008,
                 dup_apps=("voip",), risk_mos=3.8, sat_warm_mos=3.6, min_dwell=3.0,
                 predictor="kalman"):
        self.predictor = predictor
        self.min_dwell = min_dwell   # anti ping-pong: stay >= this long unless link is dying
        self.horizon = horizon
        self.window = window
        self.margin = margin
        self.energy_weight = energy_weight
        self.dup_apps = set(dup_apps)
        self.risk_mos = risk_mos
        self.sat_warm_mos = sat_warm_mos
        self.hist = {}
        self.pred = {}
        self.score = {}
        self.last_switch = -99.0
        self.better_count = 0
        self.sat_good_since = None
        self.risky = False

    # ---------------------------------------------------------------- HOOK 1
    def predict_linear(self, name, h):
        """Original baseline: least-squares linear trend over the recent window."""
        ys = np.asarray(self.hist[name], dtype=float)
        if len(ys) < 5:
            return float(ys[-1])
        xs = np.arange(len(ys)) * 0.1
        (slope, icpt), res, *_ = np.polyfit(xs, ys, 1, full=True)
        sigma2 = (res[0] if len(res) else 0.0) / max(len(ys) - 2, 1)
        se = np.sqrt(sigma2 / ((xs - xs.mean()) ** 2).sum())
        if abs(slope) < 2 * se:
            slope = 0.0
            icpt = ys.mean()
        now = icpt + slope * xs[-1]
        return float(np.clip(now + slope * h, 0, 1))

    def predict(self, name, h):
        """Kalman forecast of link quality h seconds ahead.

        State = [quality, quality_rate].  The filter explicitly models the
        simulator's 100 ms noisy measurements, smoothing noise while retaining
        genuine quality trends.  Lower rate process noise prevents handover
        flapping from short-lived measurement changes.
        """
        ys = np.asarray(self.hist[name], dtype=float)
        if len(ys) < 2:
            return float(ys[-1])

        dt = 0.1
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        R = 0.03 ** 2
        Q = np.array([[2.0e-5, 0.0], [0.0, 1.0e-4]])

        rate0 = (ys[-1] - ys[-2]) / dt
        x = np.array([ys[-1], rate0], dtype=float)
        P = np.diag([R, 0.05])

        for z in ys[1:]:
            x = F @ x
            P = F @ P @ F.T + Q
            # index the single element explicitly: float() on a 1-element
            # array is an error in NumPy >= 2.x (only a warning in 1.x)
            innovation = float(z) - float((H @ x)[0])
            S = float((H @ P @ H.T)[0, 0] + R)
            K = (P @ H.T / S).reshape(2)
            x = x + K * innovation
            P = (np.eye(2) - np.outer(K, H.reshape(2))) @ P

        return float(np.clip(x[0] + x[1] * h, 0, 1))

    # ---------------------------------------------------------------- HOOK 2
    def link_score(self, ctx, name, q):
        """Expected user experience on this link at quality q (higher = better).
        Current: VoIP MOS from the E-model using the link's delay/loss, minus an
        energy penalty, minus a penalty if bandwidth can't carry the video."""
        prof = ctx.profile(name)
        p = link_params(prof, q)
        if not p["up"]:
            return 1.0
        mos = emodel_mos(p["delay_ms"], 100 * p["loss"])
        if p["bw_kbps"] < 1.5 * 2500:
            mos -= 0.5
        return mos - self.energy_weight * prof.get("power_w", 0)

    # ---------------------------------------------------------------- HOOK 3
    def is_risky(self, t, ctx):
        """Should we duplicate critical packets right now?"""
        cur = ctx.primary
        if cur is None:
            return True
        return (self.score.get(cur, 1) < self.risk_mos        # predicted to get bad
                or t - self.last_switch < 1.0)                 # just switched

    # ---------------------------------------------------------------- logic
    def on_start(self, ctx):
        for n in TERRESTRIAL:
            if n in ctx.names:
                ctx.want_up(n)

    def on_tick(self, t, obs, ctx):
        for n in ctx.names:
            self.hist.setdefault(n, deque(maxlen=self.window)).append(obs[n]["q"])
            q_now = self.predict(n, 0.0)          # de-noised current quality
            self.pred[n] = self.predict(n, self.horizon)
            # conservative: judge a link by the worse of now and predicted
            self.score[n] = self.link_score(ctx, n, min(q_now, self.pred[n]))

        # --- anticipatory warm-up of satellite (it needs seconds to attach)
        terr_best = max((self.score[n] for n in TERRESTRIAL if n in ctx.names), default=1.0)
        if "sat_leo" in ctx.names:
            h_warm = ctx.profile("sat_leo")["attach_ms"] / 1000 + 1.0
            terr_future = max((self.link_score(ctx, n, self.predict(n, h_warm))
                               for n in TERRESTRIAL if n in ctx.names), default=1.0)
            if min(terr_best, terr_future) < self.sat_warm_mos:
                ctx.want_up("sat_leo")
                self.sat_good_since = None
            elif ctx.primary != "sat_leo":
                self.sat_good_since = self.sat_good_since if self.sat_good_since is not None else t
                if t - self.sat_good_since > 10:
                    ctx.want_up("sat_leo", False)
        if "sat_geo" in ctx.names:
            ctx.want_up("sat_geo", not any(ctx.is_up(n) for n in ctx.names if n != "sat_geo"))

        # --- choose primary among links that are already up (make-before-break)
        up = [n for n in ctx.names if ctx.is_up(n)]
        if not up:
            return
        best = max(up, key=lambda n: self.score[n])
        cur = ctx.primary
        if cur is None or not ctx.is_up(cur):
            ctx.switch_primary(best, "current lost")
            self.last_switch = t
        elif best != cur:
            imminent = self.pred[cur] < 1.5 * ctx.profile(cur)["q_min"]
            if self.score[best] > self.score[cur] + self.margin:
                self.better_count += 1
            else:
                self.better_count = 0
            dwell_ok = t - self.last_switch >= self.min_dwell
            if (imminent and self.score[best] >= self.sat_warm_mos) or (dwell_ok and self.better_count >= 2):
                ctx.switch_primary(best, "predicted loss" if imminent else "better predicted QoE")
                self.last_switch = t
                self.better_count = 0
        self.risky = self.is_risky(t, ctx)

    def paths_for(self, t, app, ctx):
        if ctx.primary is None:
            return []
        paths = [ctx.primary]
        if self.risky and app in self.dup_apps:
            others = [n for n in ctx.names if n != ctx.primary and ctx.is_up(n)]
            if others:
                paths.append(max(others, key=lambda n: self.score.get(n, 0)))
        return paths


class PredictiveLinear(Predictive):
    """Same controller as Predictive, but restores the original linear predictor."""

    name = "predictive_linear"

    def predict(self, name, h):
        return self.predict_linear(name, h)


STRATEGIES = {
    "bbm": BreakBeforeMake,
    "threshold": ThresholdMBB,
    "redundant": Redundant,
    "predictive_linear": PredictiveLinear,
    "predictive": Predictive,
}
