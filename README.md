# EDGE Seamless Connectivity Testbed

Prototype for ATRC Advanced Technology Pioneers 2026, EDGE Group track:
*"How might we keep a user's experience unbroken as they move between Wi-Fi, mobile, satellite and wired networks?"*

A user on a voice call and a live video stream travels from an office desk in RAK to a remote site. On the way they go through a wired dock, Wi-Fi roaming between access points, 5G with a mountain dead zone, and LEO and GEO satellite. The testbed replays that trip packet by packet under different handover strategies and measures what the user actually experiences.

## 1. Tools

| Layer | Tool | Why | Needed on |
|---|---|---|---|
| Language | **Python 3.10+** | Everything in this repo | any OS |
| Libraries | **numpy, pandas, matplotlib, pyyaml** | Maths, results tables, figures, config | any OS |
| Simulator | **This repo (`emu/`)** — discrete-event, packet-level | Fast (≈1 s per run), reproducible, 20 seeds → confidence intervals | any OS |
| Real emulation (optional, stronger submission) | **Linux** (Ubuntu 22.04/24.04, or a VM / WSL2) | Network namespaces and tc are Linux-only | Linux |
| | **iproute2** (`ip netns`, `ip mptcp`, `tc netem`) | One virtual cable per technology, with delay/loss/rate applied from the trace | Linux |
| | **MPTCP** (in the kernel ≥ 5.6) + `mptcpize` (from `mptcpd`) | Existing multipath mechanism to compare against | Linux |
| | **iperf3** | Throughput / loss while the trip replays | Linux |
| Optional extras | Wireshark / tcpdump, `aioquic` (QUIC connection migration) | Show packets moving between paths in your video | Linux |
| Deck | Your usual slides tool + the PNGs in `results/*/figures/` | 10-slide PDF + 1–2 min video | – |

Install: `pip install -r requirements.txt`. For the Linux part: `sudo apt install iproute2 iperf3 mptcpd`.

## 2. Run it (2 commands)

```bash
python run.py                   # 4 strategies x 20 seeds  (~1 min)
python analyze.py               # figures + results table -> results/latest/figures/
```

`python run.py --seeds 5 --strategies bbm predictive` for quick iterations while you develop.

## 3. How it works

```
config/links.yaml          what each technology is like (delay, loss, bandwidth, attach time, power)
config/scenarios/*.yaml    the trip: waypoints (speed + direction), APs, cells, dead zones, obstructions
        │
emu/model.py               the "physics": position -> signal -> quality q(t) in [0,1] per link
        │                  (path loss, shadowing, Wi-Fi AP roaming, LEO reconfiguration gaps)
emu/sim.py                 packets: apps -> strategy -> interface -> link -> receiver -> feedback
        │                  interface state machine: down -> attaching -> up (attach takes time!)
emu/strategies.py          the handover brains (baselines + ours)
emu/metrics.py             what the user feels: VoIP MOS (ITU-T G.107 E-model), interruption, video stalls
        │
run.py  -> analyze.py      experiments over many seeds -> figures + table for the deck
linux_testbed/             same trip replayed on real Linux interfaces with tc netem + MPTCP
demo/                      build_demo.py -> edge_demo.html, an interactive replay of the trip; cover_1024x576.png
```

**Read these files in this order:** `config/links.yaml` → `config/scenarios/rak_office_to_site.yaml` → `emu/model.py` → `emu/sim.py` → `emu/strategies.py` → `emu/metrics.py`.

### Strategies compared

| Name | What it is | Existing mechanism it represents |
|---|---|---|
| `bbm` | Fixed priority (wired > Wi-Fi > 5G > satellite). Switches only when the link dies or a higher-priority link appears. App session must reconnect (3 RTT) after every switch. | Today's default OS routing + plain TCP/TLS |
| `threshold` | Leave a link when quality < threshold for a time-to-trigger; return with hysteresis. Session survives (migration). | 3GPP A3-style rules, Wi-Fi calling policies, QUIC connection migration |
| `redundant` | Every packet on two paths. | MPTCP redundant scheduler |
| `predictive` | **Starting point for your idea.** Predicts quality 1.5 s ahead (only trusting statistically significant trends), scores links by predicted MOS minus energy cost, pre-warms satellite before it's needed, and duplicates voice only while a handover is risky. | Ours |

### Metrics

| Metric | Meaning |
|---|---|
| `voip_interruption_s` | Total time the call had no usable audio (gaps > 60 ms) |
| `voip_bad_seconds` | Seconds with MOS < 3.1 |
| `voip_mos` | Overall call quality (1–4.5) |
| `video_stall_s`, `video_frozen_pct` | Frames not delivered before their 400 ms playout deadline |
| `overhead_pct` | Extra bytes sent as duplicates |
| `energy_kj` | Radio energy (satellite terminals draw tens of watts) |
| `no_coverage_s` | Seconds when no network at all exists. Nobody can fix these, so report them separately. |

## 4. Results (20 seeds, mean)

| Strategy | Voice interruption (s) | Bad seconds | Video stall (s) | Handovers | Duplicate traffic | Energy (kJ) |
|---|---|---|---|---|---|---|
| bbm | 12.2 | 20.8 | 15.5 | 7.1 | 0% | 1.4 |
| threshold | 5.4 | 8.4 | 6.5 | 7.8 | 0% | 2.0 |
| redundant | 4.6 | 5.3 | 4.7 | 8.4 | 87% | 6.3 |
| predictive_linear | 4.9 | 6.3 | 6.0 | 10.8 | 0.2% | 3.4 |
| **predictive (Kalman)** | **4.7** | **5.3** | **5.8** | **11.8** | **0.2%** | **3.8** |

About 2.9 s of every run is a physical coverage hole that no strategy can avoid.

**The story so far:** the predictive controller with the Kalman forecast matches MPTCP redundant on poor-quality seconds (5.3) and comes within 0.1 s on interruption, while sending 0.2% duplicate traffic instead of 87% and using 60% of the energy. It still switches more often (about 12 handovers vs 8) and freezes more video than redundant. Those gaps are the next things to work on.


## 5. Interactive demo page (for the video and the NSTI stand)

```bash
python run.py                  # 20-seed results for the comparison panel (skip if already run)
python demo/build_demo.py      # runs every strategy on the trip -> demo/edge_demo.html
```

Open `demo/edge_demo.html` in any browser. It's one self-contained file and needs no server; with no internet it falls back to system fonts.

- **Replay:** press Play (or Space). Choose 1×, 5×, 10× or 20× speed; at 5× the whole trip takes 40 s, which suits a 1–2 minute video. Use ←/→ to step 1 s and Home to restart.
- **Seek:** drag anywhere on the timeline to jump to that moment.
- **Follow a strategy:** click its row to see which networks it has warmed up and the reasons behind each switch it makes.
- **Kiosk mode:** add `#autoplay` to the address (`edge_demo.html#autoplay`) and it plays in a loop.
- **Keep it in sync:** the page is built from the simulator's output. After you change a strategy, link profile or scenario, run `build_demo.py` again.
- **Captions:** the trip captions come from `narration:` in the scenario YAML. Wi-Fi roams, the 5G dead zone and the canyon are detected automatically.

## 6. Honest limitations 

- Link parameters are published ballparks, not measurements. 
- The route is 1-D, and 5G always uses the best cell. Inter-cell 5G handovers aren't modelled, but Wi-Fi AP roaming is.
- Quality is measurable for every link, even when an interface is off. Real devices must scan for Wi-Fi or power a satellite terminal to measure it.
- The "reconnect = 3 RTT" and "migration = free" assumptions are simplifications of TCP+TLS and QUIC/MPTCP.

## 7. AI prediction hook: Kalman filter

The `Predictive` strategy uses a constant-velocity Kalman filter in `emu/strategies.py → Predictive.predict()` instead of the original least-squares linear trend. The filter tracks:

- link quality `q`
- its rate of change `dq/dt`

It models the simulator's 100 ms measurement interval and its 0.03 measurement noise. A low process noise on the rate keeps short-lived noise from causing handover flapping. The original forecast is kept as `predictive_linear` for a like-for-like comparison.

```bash
python run.py --seeds 20 --strategies predictive_linear predictive --out results/kalman_vs_linear
```

### Kalman vs linear, 20 paired runs (same seeds, so each trip is identical)

| Measure | Linear | Kalman | Kalman − linear (95% CI) | Kalman better in |
|---|---:|---:|---|---|
| Voice interruption (s) | 4.87 | **4.70** | −0.17 (−0.22 to −0.13) | 20 of 20 runs |
| Seconds with MOS < 3.1 | 6.25 | **5.30** | −0.95 (−1.42 to −0.48) | 13 of 20 (6 equal) |
| Call quality (MOS) | 4.10 | **4.11** | +0.015 (+0.010 to +0.021) | 19 of 20 |
| Video stall (s) | 5.96 | 5.83 | −0.12 (−0.31 to +0.06), not significant | 11 of 20 |
| Handovers | **10.75** | 11.80 | +1.05 (+0.05 to +2.05) | 3 of 20 (9 equal) |
| Energy (kJ) | **3.41** | 3.76 | +0.35 (+0.15 to +0.55) | 3 of 20 |

The Kalman forecast gives a small but consistent gain for voice: less interruption in every one of the 20 runs, and about 15% fewer seconds of poor call quality. It pays for that with about one extra handover per trip and 10% more radio energy, because it reacts earlier and powers the satellite on more often. Video stall time doesn't change significantly. The 3-run development results that first suggested the gain are kept in `results/teammate/paired3/`. `results/teammate/kalman3/` came from an earlier version of the filter and doesn't match the current code.



The video is built from the current results, so rerun it after any change to the strategies. It has six parts: title, problem, approach, a replay of the trip with captions, results over 20 runs, and next steps. The headline numbers are read from `results/latest/summary.csv`.

The video is silent. `video/narration.md` has a voice-over script with the time each line starts. Record it on a phone and add it in any video editor (CapCut, iMovie, Clipchamp). Without the brand fonts installed it falls back to system fonts; pass `--fonts <folder with @fontsource packages>` to use Barlow and IBM Plex Mono.
