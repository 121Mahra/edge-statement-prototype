"""Render the 1-2 minute demo video from the real simulator output.

    python video/make_video.py                         # -> video/unbroken_demo.mp4
    python video/make_video.py --fonts <dir>           # optional: @fontsource folder for the brand fonts
    python video/make_video.py --probe 131             # just save one frame of the demo at t=131 s

Needs: playwright (Chromium) and imageio-ffmpeg (pip install imageio-ffmpeg).
Run demo/build_demo.py first: the replay frames come from demo/edge_demo.html,
and the headline numbers from results/latest/summary.csv, so the video always
matches the current results. The video is silent; narration.md (written next to
it) has a script timed to each scene for a voice-over.
"""
from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1280, 720, 25
FADE = 0.4  # seconds of cross-fade between scenes

LINK = {"Wired": "#8a8f97", "Wi-Fi": "#3987e5", "5G": "#1baf7a", "Satellite": "#eb6834"}


def font_css(fonts: str | None) -> str:
    if not fonts:
        return ""
    f = Path(fonts).resolve()
    faces = [("Barlow Semi Condensed", "barlow-semi-condensed", w) for w in (500, 600, 700)]
    faces += [("Barlow", "barlow", w) for w in (400, 500, 600)]
    faces += [("IBM Plex Mono", "ibm-plex-mono", w) for w in (400, 500)]
    return "".join(
        f'@font-face{{font-family:"{fam}";font-weight:{w};src:url(file://{f}/{d}/files/{d}-latin-{w}-normal.woff2)}}'
        for fam, d, w in faces)


def numbers():
    d = pd.read_csv(ROOT / "results" / "latest" / "summary.csv").groupby("strategy").mean(numeric_only=True)
    bbm, red, ours = d.loc["bbm"], d.loc["redundant"], d.loc["predictive"]
    return {
        "bbm_int": bbm.voip_interruption_s, "ours_int": ours.voip_interruption_s, "red_int": red.voip_interruption_s,
        "cut": 100 * (1 - ours.voip_interruption_s / bbm.voip_interruption_s),
        "ours_dup": ours.overhead_pct, "red_dup": red.overhead_pct,
        "energy_less": 100 * (1 - ours.energy_kj / red.energy_kj),
        "ours_ho": ours.handovers, "base_ho": (d.loc["bbm"].handovers, d.loc["threshold"].handovers),
        "runs": int(pd.read_csv(ROOT / "results" / "latest" / "summary.csv").seed.nunique()),
        "hole": d.no_coverage_s.iloc[0],
    }


def cards_html(n, fonts_css):
    chips = "".join(f'<span class="chip"><i style="background:{c}"></i>{k}</span>' for k, c in LINK.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{fonts_css}
*{{box-sizing:border-box}} body{{margin:0;background:#000}}
.s{{width:{W}px;height:{H}px;background:#12161c;color:#fff;position:relative;overflow:hidden;padding:84px 96px;font-family:"Barlow",system-ui,sans-serif}}
.k{{font:600 18px "Barlow Semi Condensed",sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#9aa3ad}}
h1{{font:700 168px/1 "Barlow Semi Condensed",sans-serif;margin:28px 0 0;letter-spacing:-.01em}}
h2{{font:700 58px/1.08 "Barlow Semi Condensed",sans-serif;margin:22px 0 0;max-width:1000px;text-wrap:balance}}
p{{font:400 27px/1.42 "Barlow",sans-serif;color:#d5dae0;max-width:980px;margin:26px 0 0}}
.sub{{font:500 34px/1.3 "Barlow",sans-serif;color:#d5dae0;max-width:900px;margin-top:26px}}
.chips{{display:flex;gap:14px;margin-top:40px;align-items:center}}
.chip{{display:inline-flex;align-items:center;gap:10px;font:500 22px "Barlow",sans-serif;padding:8px 16px;border:1px solid #39404a;border-radius:999px;color:#e6e9ed}}
.chip i{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.arrow{{color:#6f7883;font-size:24px}}
.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-top:52px}}
.step{{border-top:3px solid #3987e5;padding-top:18px}}
.step b{{display:block;font:700 30px "Barlow Semi Condensed",sans-serif}}
.step span{{display:block;font:400 22px/1.38 "Barlow",sans-serif;color:#c3c9d0;margin-top:10px}}
.nums{{display:grid;grid-template-columns:repeat(3,1fr);gap:40px;margin-top:56px}}
.num b{{display:block;font:700 92px/1 "Barlow Semi Condensed",sans-serif;color:#7ee2b8}}
.num span{{display:block;font:500 24px/1.35 "Barlow",sans-serif;color:#e6e9ed;margin-top:14px}}
.num small{{display:block;font:400 19px/1.35 "Barlow",sans-serif;color:#9aa3ad;margin-top:8px}}
.foot{{position:absolute;left:96px;right:96px;bottom:56px;font:400 19px "Barlow",sans-serif;color:#7d8794}}
ul{{margin:40px 0 0;padding:0;list-style:none}} li{{font:500 31px/1.35 "Barlow",sans-serif;color:#e6e9ed;padding:16px 0;border-top:1px solid #2a2f37}}
li span{{color:#7ee2b8;font-family:"IBM Plex Mono",monospace;font-size:24px;margin-right:18px}}
</style></head><body>
<div class="s" id="title"><div class="k">ATRC Advanced Technology Pioneers 2026 &middot; EDGE Group challenge</div>
<h1>Unbroken</h1><div class="sub">Predictive handover that keeps a call alive across wired, Wi-Fi, 5G and satellite networks</div>
<div class="foot">Prototype demo</div></div>

<div class="s" id="problem"><div class="k">The problem</div>
<h2>Every change of network is a chance for the call to drop.</h2>
<p>A field engineer on a video call leaves an office in Ras Al Khaimah, drives up a mountain road and reaches a remote site. In 200 seconds the device crosses a cable, three Wi-Fi access points, 5G and satellite. Most devices wait for a link to fail before they switch, so the call breaks each time.</p>
<div class="chips">{chips.replace('</span><span', '</span><span class="arrow">&rarr;</span><span')}</div></div>

<div class="s" id="idea"><div class="k">Our approach</div>
<h2>Unbroken switches on predicted call quality, before the user notices.</h2>
<div class="steps">
<div class="step"><b>1. Forecast</b><span>A Kalman filter projects each link's quality 1.5 s ahead.</span></div>
<div class="step"><b>2. Score</b><span>Each link is ranked by the call quality it will give (ITU&#8209;T E-model), minus its power cost.</span></div>
<div class="step"><b>3. Prepare</b><span>The satellite terminal is powered up before it is needed.</span></div>
<div class="step"><b>4. Protect</b><span>Voice is sent on two links only in risky moments.</span></div>
</div><div class="foot">Runs on the device every 100 ms. No change to the networks.</div></div>

<div class="s" id="results"><div class="k">Results &middot; {n['runs']} runs of the same simulated trip</div>
<h2>Close to full redundancy, at a fraction of its cost.</h2>
<div class="nums">
<div class="num"><b>{n['cut']:.0f}%</b><span>less audio interruption than today's default switching</span><small>{n['bbm_int']:.1f} s down to {n['ours_int']:.1f} s per trip</small></div>
<div class="num"><b>{n['ours_dup']:.1f}%</b><span>extra data for that result</span><small>MPTCP redundant sends {n['red_dup']:.0f}% extra for {n['red_int']:.1f} s</small></div>
<div class="num"><b>{n['energy_less']:.0f}%</b><span>less radio energy than redundant multipath</span><small>the satellite is powered only when forecast</small></div>
</div>
<div class="foot">Trade-off we are working on: about {n['ours_ho']:.0f} switches per trip, against {n['base_ho'][0]:.0f} to {n['base_ho'][1]:.0f} for the baselines. {n['hole']:.1f} s of the trip has no coverage for any strategy.</div></div>

<div class="s" id="next"><div class="k">If shortlisted</div>
<h2>From simulation to real interfaces.</h2>
<ul>
<li><span>01</span>Drive a real RAK route on Wi-Fi and 5G, and replay the measured trip</li>
<li><span>02</span>Carry a live voice and video call over real Multipath TCP on our Linux testbed</li>
<li><span>03</span>Let visitors at the NSTI stand hear both calls side by side</li>
</ul>
<div class="foot">Simulated prototype &middot; link parameters from published ranges &middot; team Unbroken</div></div>
</body></html>"""


# Demo replay plan: (trip_from_s, trip_to_s, seconds_of_video, caption). A from == to segment is a hold.
REPLAY = [
    (0, 20, 4.5, "Same simulated trip, four strategies. Docked at the desk on a call."),
    (20, 62, 6.0, "Unplugged and walking: Wi-Fi hands over between access points."),
    (62, 100, 6.0, "Leaving the building: Wi-Fi fades and 5G takes over."),
    (100, 124, 3.0, "Mountain road at 100 km/h."),
    (124, 129.5, 7.5, "5G fades in the pass. Unbroken sees it coming and moves to satellite early."),
    (129.5, 129.5, 6.0, "Today's default has lost the call. Unbroken switched at 126 s, before 5G failed."),
    (129.5, 160, 5.0, "Canyon walls block the sky: for a few seconds no network works at all."),
    (160, 200, 4.0, "At the site, everyone returns to Wi-Fi."),
]
TIMELINE_HOLD = (6.0, "The whole trip. Red marks under each row are moments with no audio.")

SCENES = [("title", 8.0), ("problem", 16.0), ("idea", 17.0), ("REPLAY", None), ("results", 15.0), ("next", 11.0)]


async def render(out: Path, fonts: str | None, probe: float | None):
    from playwright.async_api import async_playwright
    import imageio_ffmpeg

    n = numbers()
    build = out.parent / "build"
    build.mkdir(parents=True, exist_ok=True)
    fcss = font_css(fonts)
    (build / "cards.html").write_text(cards_html(n, fcss))
    demo_css = fcss + """
      .wrap{zoom:0.76;padding-top:10px} header{display:none}
      #cap{position:fixed;left:0;right:0;bottom:0;background:#12161c;color:#fff;padding:20px 40px 24px;
           font:500 27px/1.3 "Barlow",system-ui,sans-serif;z-index:9}
      #cap b{font-family:"IBM Plex Mono",monospace;font-weight:500;color:#7ee2b8;margin-right:16px;font-size:22px}"""

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        cards = await browser.new_page(viewport={"width": W, "height": H})
        await cards.goto((build / "cards.html").as_uri())
        await cards.evaluate("document.fonts.ready")
        demo = await browser.new_page(viewport={"width": W, "height": H})
        await demo.goto((ROOT / "demo" / "edge_demo.html").as_uri())
        await demo.add_style_tag(content=demo_css)
        await demo.evaluate("document.body.insertAdjacentHTML('beforeend','<div id=cap></div>')")
        await demo.evaluate("document.fonts.ready")
        await demo.wait_for_timeout(300)

        async def demo_frame(t, caption):
            await demo.evaluate("([t, c]) => { edgeDemo.seek(t); document.getElementById('cap').innerHTML = c; }",
                                [t, f"<b>{t:5.1f} s</b>{caption}"])
            return Image.open(io.BytesIO(await demo.screenshot(type="jpeg", quality=92))).convert("RGB")

        if probe is not None:
            (await demo_frame(probe, "probe")).save(out.with_suffix(".probe.png"))
            await browser.close()
            return

        writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libx264",
                                             pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
        writer.send(None)
        prev = None
        clock = 0.0
        script = []

        def emit(img, seconds):
            nonlocal prev
            arr = np.asarray(img)
            frames = int(round(seconds * FPS))
            if prev is not None:  # cross-fade from the previous scene's last frame
                k = min(int(FADE * FPS), frames)
                for i in range(k):
                    a = (i + 1) / (k + 1)
                    writer.send(np.ascontiguousarray((prev * (1 - a) + arr * a).astype(np.uint8)))
                frames -= k
            for _ in range(frames):
                writer.send(np.ascontiguousarray(arr))
            prev = arr.astype(np.float32)

        for scene, secs in SCENES:
            if scene != "REPLAY":
                img = Image.open(io.BytesIO(await cards.locator(f"#{scene}").screenshot())).convert("RGB")
                emit(img, secs)
                script.append((clock, scene, secs))
                clock += secs
                continue
            await demo.evaluate("scrollTo(0,0)")
            first = True
            for a, b, secs, cap in REPLAY:
                script.append((clock, "replay: " + cap, secs))
                frames = int(round(secs * FPS))
                for i in range(frames):
                    t = a + (b - a) * (i / max(frames - 1, 1)) if b != a else a
                    img = await demo_frame(t, cap)
                    if first:
                        emit(img, 1 / FPS)  # fade into the replay on its first frame
                        first = False
                    else:
                        arr = np.asarray(img)
                        writer.send(np.ascontiguousarray(arr))
                        prev = arr.astype(np.float32)
                clock += secs
            # whole-trip timeline, held
            secs, cap = TIMELINE_HOLD
            await demo.evaluate("document.getElementById('h-tl').closest('section').scrollIntoView({block:'start'})")
            await demo.evaluate("scrollBy(0,-12)")
            emit(await demo_frame(200, cap), secs)
            script.append((clock, "replay: " + cap, secs))
            clock += secs
        writer.close()
        await browser.close()
    return script, n


# Voice-over, keyed by scene name or by the replay caption it plays over.
NARRATION = {
    "title": "We're team Unbroken. We keep a call alive across wired, Wi-Fi, 5G and satellite networks.",
    "problem": "Picture a field engineer on a call, driving from an office in Ras Al Khaimah to a remote site. Most devices wait for a link to fail before switching, so the call breaks each time.",
    "idea": "Unbroken looks ahead. A Kalman filter forecasts each link's quality one and a half seconds ahead. Links are ranked by the call quality they'll give, the satellite powers up early, and only voice is duplicated when it's risky.",
    REPLAY[0][3]: "This is our prototype, a simulator we built. It replays one trip for four strategies, with each call's status on the right.",
    REPLAY[2][3]: "Leaving the building, every strategy moves from Wi-Fi to 5G.",
    REPLAY[4][3]: "In the pass, Unbroken forecasts 5G fading and moves the call to satellite before it fails.",
    REPLAY[5][3]: "Today's default waits for the link to die, and loses the call.",
    REPLAY[6][3]: "In the canyon, no network works for a few seconds, for any strategy.",
    TIMELINE_HOLD[1]: "The whole trip: red marks show moments with no audio.",
    "results": "Across {runs} runs, Unbroken cut audio interruption by {cut:.0f} percent versus today's default. That's close to full redundancy, with only {dup:.1f} percent extra data and {en:.0f} percent less energy.",
    "next": "If shortlisted, we'll test on a real route in Ras Al Khaimah and let visitors hear the difference live.",
}


def write_narration(path: Path, script, n):
    lines = ["# Narration script (read over the video)\n",
             "Read at a calm pace, about 130 words a minute. Each line starts at the time shown.\n"]
    for t, label, _ in script:
        key = label[len("replay: "):] if label.startswith("replay: ") else label
        if key in NARRATION:
            txt = NARRATION[key].format(runs=n["runs"], cut=n["cut"], dup=n["ours_dup"], en=n["energy_less"])
            lines.append(f"**{int(t // 60)}:{t % 60:04.1f}** {txt}\n")
    total = script[-1][0] + script[-1][2]
    lines.append(f"\nVideo length: {int(total // 60)}:{total % 60:04.1f}\n")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "video" / "unbroken_demo.mp4"))
    ap.add_argument("--fonts", default=None, help="folder containing @fontsource packages (optional)")
    ap.add_argument("--probe", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(render(out, a.fonts, a.probe))
    if res:
        script, n = res
        write_narration(out.parent / "narration.md", script, n)
        print(f"wrote {out} and narration.md")


if __name__ == "__main__":
    main()
