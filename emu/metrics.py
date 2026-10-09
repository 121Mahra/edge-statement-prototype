"""Quality-of-Experience metrics.

VoIP  : ITU-T G.107 E-model (simplified) -> R-factor -> MOS (1..4.5)
Video : frozen frames and stalls (a frame that is not fully delivered before
        its playout deadline is "frozen"; >= 3 frozen frames in a row = stall)
Both  : service interruption time = total time with no usable media arriving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOIP_INTERVAL = 0.020
VOIP_DEADLINE = 0.400      # packets later than this miss the jitter buffer
VIDEO_DEADLINE = 0.400     # live / interactive video playout deadline
STALL_MIN_FRAMES = 3


def emodel_mos(owd_ms: float, loss_pct: float, codec_ie=0.0, bpl=25.1) -> float:
    """G.107 E-model, G.711 with packet-loss concealment by default.
    owd_ms  : mean one-way network delay of delivered packets
    loss_pct: % of packets lost OR late."""
    if np.isnan(owd_ms):
        return 1.0
    d = owd_ms + 60.0  # + jitter buffer (40 ms) + codec/packetisation (20 ms)
    i_d = 0.024 * d + 0.11 * (d - 177.3) * (d > 177.3)
    ie_eff = codec_ie + (95.0 - codec_ie) * loss_pct / (loss_pct + bpl)
    r = 93.2 - i_d - ie_eff
    if r <= 0:
        return 1.0
    if r >= 100:
        return 4.5
    return 1 + 0.035 * r + 7e-6 * r * (r - 60) * (100 - r)


def interruption(t_send_ok: np.ndarray, interval: float, t_start: float, t_end: float, factor=3.0):
    """Total and longest gap (s) between consecutive on-time deliveries,
    counting only gaps longer than `factor` packet intervals."""
    ts = np.concatenate([[t_start], np.sort(t_send_ok), [t_end]])
    gaps = np.diff(ts)
    bad = gaps[gaps > factor * interval]
    return float((bad - interval).sum()), float(bad.max() - interval if len(bad) else 0.0), int(len(bad))


def voip_metrics(df: pd.DataFrame, duration: float) -> dict:
    v = df[df.app == "voip"]
    ok = v.on_time
    owd = (v.t_arrive - v.t_send)[ok] * 1000
    loss_pct = 100.0 * (1 - ok.mean())
    tot, longest, n = interruption(v.t_send[ok].values, VOIP_INTERVAL, 0.0, duration)

    # per-second MOS timeline
    sec = (v.t_send // 1).astype(int)
    per_sec = []
    for s, g in v.groupby(sec):
        g_ok = g.on_time
        o = ((g.t_arrive - g.t_send)[g_ok] * 1000).mean() if g_ok.any() else np.nan
        per_sec.append((s, emodel_mos(o, 100.0 * (1 - g_ok.mean()))))
    mos_ts = np.array([m for _, m in per_sec])
    return {
        "voip_loss_pct": loss_pct,
        "voip_owd_ms": float(owd.mean()) if len(owd) else float("nan"),
        "voip_owd_p95_ms": float(np.percentile(owd, 95)) if len(owd) else float("nan"),
        "voip_mos": emodel_mos(float(owd.mean()) if len(owd) else np.nan, loss_pct),
        "voip_mos_min_1s": float(mos_ts.min()),
        "voip_bad_seconds": int((mos_ts < 3.1).sum()),   # MOS < 3.1 ~ "many users dissatisfied"
        "voip_interruption_s": tot,
        "voip_longest_gap_s": longest,
        "voip_gaps": n,
    }, per_sec


def video_metrics(df: pd.DataFrame, fps: float) -> dict:
    v = df[df.app == "video"]
    if v.empty:
        return {}, []
    # a frame is OK if every packet of it arrived before the deadline
    frames = v.groupby("frame").agg(t=("t_send", "min"), ok=("on_time", "all")).sort_index()
    frozen = ~frames.ok.values
    stalls, stall_frames, run = 0, 0, 0
    for f in np.append(frozen, False):
        if f:
            run += 1
        else:
            if run >= STALL_MIN_FRAMES:
                stalls += 1
                stall_frames += run
            run = 0
    sec = (frames.t // 1).astype(int)
    per_sec = [(int(s), float(1 - g.ok.mean())) for s, g in frames.groupby(sec)]
    return {
        "video_frozen_pct": 100.0 * frozen.mean(),
        "video_stalls": stalls,
        "video_stall_s": stall_frames / fps,
    }, per_sec
