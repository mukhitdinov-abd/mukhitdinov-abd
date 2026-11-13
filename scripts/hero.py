"""Generate the static profile assets:

  assets/hero-{light,dark}.svg   animated header (a speech encoder-decoder writes the tagline, magenta smoke)
  assets/link-<name>-{light,dark}.svg   minimal link pills

Pure SVG + CSS/SMIL, no JS, so GitHub renders it.  Stdlib only:

    python3 scripts/hero.py
"""
import math
import random
from pathlib import Path

from palette import THEMES

NAME = "Abdul-Aziz"
ROLE = "AI/ML Engineer"
# The network "generates" one of these per forward pass.
# Keep each under ~25 chars so the tagline ends before the waveform (x=WAVE_X).
PHRASES = [
    "building multilingual STT",
    "fine-tuning Whisper",
    "mining speech data",
    "training ASR models",
]
# (label, url) — rendered as "~/label ↗" pills under the header.
LINKS = [
    ("linkedin", "https://linkedin.com/in/abdul-aziz-muxitdinov"),
    ("telegram", "https://t.me/capybaraml"),
]

W, H = 1000, 300
# Whisper-style speech model, left to right: audio waveform -> conv stem ->
# encoder (self-attention, xN) -> decoder (cross- + causal self-attention) -> text tokens,
# which feed back into the decoder.  Stage 0 is the waveform.
WAVE_X, WAVE_N, WAVE_DX = 596, 11, 4
STAGES = [  # (key, x, nodes)
    ("conv", 672, 5),
    ("enc1", 736, 7),
    ("enc2", 794, 7),
    ("dec", 878, 5),
    ("out", 950, 3),
]
SELF_ATTN = {"enc1": -1, "enc2": 1, "dec": -1}   # attention arcs: bulge left (-1) or right (1)
BOXES = [("encoder", "×12", 716, 818), ("decoder", "", 852, 904)]  # (label, badge, x0, x1)
BOX_Y0, BOX_Y1, LABEL_Y = 48, 208, 40
CY, DY = 128, 22           # vertical centre, spacing between nodes
STEP = 0.5                 # seconds per stage hop
OUT_STAGE = len(STAGES)    # output layer fires at OUT_STAGE * STEP
CYCLE = 3.6                # one forward pass (s)
FIRE_P = 0.55              # share of edges that carry a signal

TX, NAME_Y, ROLE_Y = 8, 118, 162
ADV = 12.04                # monospace advance at 20px
TYPE_DT, ERASE_DT = 0.045, 0.018

# Smoke wisps curling along the bottom, like the magenta haze around the cat in the
# profile picture: (baseline y, amplitude, stroke width, opacity, drift period s).
WISPS = [(266, 18, 1.0, .35, 13), (244, 24, 1.2, .5, 17), (222, 16, .9, .3, 11), (278, 12, 1.4, .6, 19),
         (206, 10, .8, .18, 23)]
SMOKE_TOP = 190


def smooth(pts):
    """Catmull-Rom through pts, as a cubic Bezier path."""
    d = f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f"C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
    return d


def wisp_path(rng, base, amp):
    """A lazy curl of smoke: a few summed sines, overscanned so drifting never shows an end."""
    waves = [(rng.uniform(.004, .009), rng.uniform(0, 6.28), 1.0),
             (rng.uniform(.012, .02), rng.uniform(0, 6.28), .45),
             (rng.uniform(.025, .04), rng.uniform(0, 6.28), .2)]
    pts = []
    for x in range(-80, W + 81, 30):
        y = base + amp * sum(a * math.sin(f * x + ph) for f, ph, a in waves)
        pts.append((x, y))
    return smooth(pts)


def nodes():
    return [[(x, CY + (i - (n - 1) / 2) * DY) for i in range(n)] for _, x, n in STAGES]


def arc(p, q, side):
    """Attention arc between two nodes of the same layer, bulging sideways."""
    (x, y1), (_, y2) = p, q
    b = side * (6 + 0.22 * abs(y2 - y1))
    return f"M{x:.1f} {y1:.1f}C{x + b:.1f} {y1:.1f} {x + b:.1f} {y2:.1f} {x:.1f} {y2:.1f}"


def network(rng):
    """Edges, signal pulses, nodes, waveform, block outlines and labels of the model."""
    layers = nodes()
    edges, attn, pulses, dots = [], [], [], []

    def pulse(d, delay, cls=""):
        c = f' class="{cls}"' if cls else ""
        pulses.append(f'<path{c} d="{d}" pathLength="100" style="animation-delay:{delay:.2f}s"/>')

    # waveform: bars breathing like live audio, feeding the conv stem
    bars = []
    for i in range(WAVE_N):
        h = 8 + 34 * math.sin(math.pi * (i + .5) / WAVE_N) * rng.uniform(.45, 1)
        bars.append(f'<rect x="{WAVE_X + i * WAVE_DX:.1f}" y="{CY - h / 2:.1f}" width="2" height="{h:.1f}" rx="1" '
                    f'style="animation-duration:{rng.uniform(.7, 1.3):.2f}s;animation-delay:-{rng.uniform(0, 1):.2f}s"/>')
    tap = (WAVE_X + WAVE_N * WAVE_DX + 6, CY)
    for q in layers[0]:
        d = curve(tap, q)
        edges.append(f'<path d="{d}"/>')
        pulse(d, rng.uniform(0, .08))

    # dense projections between stages (enc -> dec is cross-attention)
    for li in range(len(layers) - 1):
        for p in layers[li]:
            for q in layers[li + 1]:
                d = curve(p, q)
                edges.append(f'<path d="{d}"/>')
                if rng.random() < FIRE_P:
                    pulse(d, (li + 1) * STEP + rng.uniform(0, .1))

    # self-attention inside a layer, fired just after the layer lights up
    for li, (key, _, _) in enumerate(STAGES):
        side = SELF_ATTN.get(key)
        if not side:
            continue
        layer = layers[li]
        for i in range(len(layer)):
            for j in range(i + 1, min(i + 4, len(layer))):
                if rng.random() < .55:
                    d = arc(layer[i], layer[j], side)
                    attn.append(f'<path d="{d}"/>')
                    if rng.random() < .6:
                        pulse(d, (li + 1) * STEP + .12 + rng.uniform(0, .1), "a")

    # autoregressive loop: emitted token goes back into the decoder
    (ox, oy), (dx, dy) = layers[-1][-1], layers[-2][-1]
    loop = f"M{ox:.1f} {oy:.1f}C{ox:.1f} {BOX_Y1 + 18},{dx:.1f} {BOX_Y1 + 18},{dx:.1f} {dy:.1f}"
    pulse(loop, OUT_STAGE * STEP + .15, "a")

    for li, layer in enumerate(layers):
        delay = (li + 1) * STEP
        for (x, y) in layer:
            dots.append(
                f'<circle class="halo" cx="{x:.1f}" cy="{y:.1f}" r="6" style="animation-delay:{delay:.2f}s"/>'
                f'<circle class="core" cx="{x:.1f}" cy="{y:.1f}" r="4" style="animation-delay:{delay:.2f}s"/>'
            )

    boxes, labels = [], []
    for label, badge, x0, x1 in BOXES:
        boxes.append(f'<rect x="{x0}" y="{BOX_Y0}" width="{x1 - x0}" height="{BOX_Y1 - BOX_Y0}" rx="10"/>')
        b = f' <tspan class="badge">{badge}</tspan>' if badge else ""
        labels.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{LABEL_Y}">{label}{b}</text>')
    labels.append(f'<text x="{WAVE_X + (WAVE_N - 1) * WAVE_DX / 2 + 1:.0f}" y="{LABEL_Y}">audio</text>')
    labels.append(f'<text x="{STAGES[0][1]}" y="{LABEL_Y}">conv</text>')
    labels.append(f'<text x="{STAGES[-1][1]}" y="{LABEL_Y}">text</text>')

    return f'''<g class="boxes">{"".join(boxes)}</g>
<g class="labels">{"".join(labels)}</g>
<path class="loop" d="{loop}"/>
<g class="edges">{"".join(edges)}</g>
<g class="attn">{"".join(attn)}</g>
<g class="wave">{"".join(bars)}</g>
<g class="pulses" filter="url(#glow)">{"".join(pulses)}</g>
<g filter="url(#glow)">{"".join(dots)}</g>'''


def curve(p, q):
    (x1, y1), (x2, y2) = p, q
    xm = (x1 + x2) / 2
    return f"M{x1:.1f} {y1:.1f}C{xm:.1f} {y1:.1f} {xm:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"


def discrete(attr, values, times, total):
    """SMIL discrete animation; times in seconds, first must be 0."""
    kt = ";".join(f"{t / total:.4f}" for t in times)
    vs = ";".join(str(v) for v in values)
    return (f'<animate attributeName="{attr}" dur="{total}s" repeatCount="indefinite" '
            f'calcMode="discrete" keyTimes="{kt}" values="{vs}"/>')


def tagline(prefix_len):
    """Per-character typing of PHRASES, synced to the output layer firing."""
    total = CYCLE * len(PHRASES)
    out_t = OUT_STAGE * STEP + 0.05          # output layer fires
    base = TX + prefix_len * ADV
    chars, cursor_events = [], []
    for k, phrase in enumerate(PHRASES):
        n = len(phrase)
        start = k * CYCLE + out_t
        erase_start = (k + 1) * CYCLE + out_t - 0.12 - n * ERASE_DT
        for i, ch in enumerate(phrase):
            on = (start + i * TYPE_DT) % total
            off = (erase_start + (n - 1 - i) * ERASE_DT) % total
            cursor_events += [(on, base + (i + 1) * ADV), (off, base + i * ADV)]
            if ch == " ":
                continue
            if on < off:
                anim = discrete("opacity", [0, 1, 0], [0, on, off], total)
            else:
                anim = discrete("opacity", [1, 0, 1], [0, off, on], total)
            esc = {"<": "&lt;", "&": "&amp;", ">": "&gt;"}.get(ch, ch)
            chars.append(f'<text class="tok" x="{base + i * ADV:.1f}" y="{ROLE_Y}" opacity="0">{esc}{anim}</text>')
    cursor_events.sort()
    # state at t=0 is the end of the loop: last phrase fully typed
    xs = [base + len(PHRASES[-1]) * ADV] + [x for _, x in cursor_events]
    ts = [0] + [t for t, _ in cursor_events]
    cursor = discrete("x", [f"{x:.1f}" for x in xs], ts, total)
    end_x = base + max(len(p) for p in PHRASES) * ADV
    return "".join(chars), cursor, end_x


def build(t):
    rng = random.Random(7)
    hop = STEP / CYCLE * 100  # % of the cycle a signal spends travelling
    net = network(rng)

    prefix = f"> {ROLE} · "
    xs = " ".join(f"{TX + i * ADV:.1f}" for i in range(len(prefix)))
    toks, cursor_anim, tag_end = tagline(len(prefix))
    srng = random.Random(11)
    wisps, paths = [], []
    for i, (base, amp, sw, op, period) in enumerate(WISPS):
        d = wisp_path(srng, base, amp)
        paths.append(d)
        wisps.append(f'<path class="wisp" d="{d}" pathLength="100" style="stroke-width:{sw};stroke-opacity:{op};'
                     f'animation-delay:{.2 + i * .25:.2f}s,0s;animation-duration:2.8s,{period}s"/>')
    # a spark of colour rides the brightest wisp, drifting with it
    gi = max(range(len(WISPS)), key=lambda i: WISPS[i][3])
    glow_wisp, glow_period = paths[gi], WISPS[gi][4]
    stops = (f'<stop offset="0" stop-color="{t["a1"]}"/><stop offset=".5" stop-color="{t["am"]}"/>'
             f'<stop offset="1" stop-color="{t["a2"]}"/>')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{NAME} — {ROLE}">
<title>{NAME} — {ROLE}</title>
<defs>
  <linearGradient id="g" gradientUnits="userSpaceOnUse" x1="{WAVE_X}" y1="0" x2="{STAGES[-1][1]}" y2="0">{stops}</linearGradient>
  <linearGradient id="tg" gradientUnits="userSpaceOnUse" x1="{TX}" y1="0" x2="{tag_end:.0f}" y2="0">{stops}</linearGradient>
  <linearGradient id="rg" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{W}" y2="0">{stops}</linearGradient>
  <linearGradient id="rf" gradientUnits="userSpaceOnUse" x1="0" y1="{SMOKE_TOP}" x2="0" y2="{H}">
    <stop offset="0" stop-color="{t['a1']}" stop-opacity="0"/><stop offset="1" stop-color="{t['a1']}" stop-opacity=".14"/>
  </linearGradient>
  <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".12" stop-color="#fff"/>
    <stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
  </linearGradient>
  <mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="2.2" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<style>
  .name {{ font: 600 56px -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
          fill: {t['fg']}; letter-spacing: -1.5px; opacity: 0; animation: rise .9s cubic-bezier(.2,.7,.2,1) .1s forwards; }}
  .role, .tok {{ font: 400 20px ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace; }}
  .role {{ fill: {t['muted']}; opacity: 0; animation: rise .9s cubic-bezier(.2,.7,.2,1) .35s forwards; }}
  .role .p, .tok, .cursor {{ fill: url(#tg); }}
  .cursor {{ animation: blink 1.1s steps(1) infinite; }}
  .edges path {{ fill: none; stroke: {t['muted']}; stroke-opacity: .16; stroke-width: 1; }}
  .pulses path {{ fill: none; stroke: url(#g); stroke-width: 1.8; stroke-linecap: round;
                 stroke-dasharray: 14 200; stroke-dashoffset: 14; animation: flow {CYCLE}s linear infinite; }}
  .attn path {{ fill: none; stroke: {t['a1']}; stroke-opacity: .22; stroke-width: .9; }}
  .pulses path.a {{ stroke-width: 1.4; }}
  .loop {{ fill: none; stroke: {t['muted']}; stroke-opacity: .35; stroke-width: 1; stroke-dasharray: 2 3; }}
  .boxes rect {{ fill: {t['a1']}; fill-opacity: .035; stroke: {t['muted']}; stroke-opacity: .3; stroke-width: 1; stroke-dasharray: 3 4; }}
  .labels text {{ font: 400 10px ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace;
                 fill: {t['muted']}; text-anchor: middle; letter-spacing: .5px; }}
  .labels .badge {{ fill: {t['am']}; }}
  .boxes, .labels, .loop {{ opacity: 0; animation: show 1.2s ease .6s forwards; }}
  .wave rect {{ fill: url(#g); transform-box: fill-box; transform-origin: center;
               animation: listen 1s ease-in-out infinite alternate; }}
  .core {{ fill: {t['node']}; animation: fire {CYCLE}s ease-out infinite; }}
  .halo {{ fill: none; stroke: url(#g); stroke-width: 1.5; opacity: 0; transform-box: fill-box; transform-origin: center;
          animation: ring {CYCLE}s ease-out infinite; }}
  .wisp {{ fill: none; stroke: {t['a1']}; stroke-linecap: round; stroke-dasharray: 100; stroke-dashoffset: 100;
          animation-name: draw, drift; animation-timing-function: cubic-bezier(.4,0,.2,1), ease-in-out;
          animation-fill-mode: forwards, none; animation-iteration-count: 1, infinite; animation-direction: normal, alternate; }}
  .drifter {{ animation: drift 1s ease-in-out infinite alternate; }}
  .haze {{ fill: url(#rf); opacity: 0; animation: show 1.6s ease 1.6s forwards; }}
  .smoke-glow {{ fill: none; stroke: url(#rg); stroke-width: 2; stroke-linecap: round;
                stroke-dasharray: 16 300; stroke-dashoffset: 16; animation: sweep 9s cubic-bezier(.5,0,.5,1) 3s infinite; }}
  @keyframes rise  {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
  @keyframes show  {{ to {{ opacity: 1; }} }}
  @keyframes draw  {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes drift {{ from {{ transform: translateX(-28px); }} to {{ transform: translateX(28px); }} }}
  @keyframes sweep {{ 0% {{ stroke-dashoffset: 16; }} 60%, 100% {{ stroke-dashoffset: -100; }} }}
  @keyframes listen {{ from {{ transform: scaleY(.3); }} to {{ transform: scaleY(1); }} }}
  @keyframes blink {{ 0%, 50% {{ opacity: 1; }} 50.01%, 100% {{ opacity: 0; }} }}
  @keyframes flow  {{ 0% {{ stroke-dashoffset: 14; }} {hop:.1f}% {{ stroke-dashoffset: -100; }} 100% {{ stroke-dashoffset: -100; }} }}
  @keyframes fire  {{ 0% {{ fill: {t['am']}; }} 18% {{ fill: {t['node']}; }} 100% {{ fill: {t['node']}; }} }}
  @keyframes ring  {{ 0% {{ opacity: .9; transform: scale(.7); }} 20% {{ opacity: 0; transform: scale(2.4); }} 100% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; opacity: 1 !important; stroke-dashoffset: 0 !important; }}
    .pulses, .smoke-glow, .halo {{ display: none; }}
  }}
</style>
<g mask="url(#m)">
  <rect class="haze" y="{SMOKE_TOP}" width="{W}" height="{H - SMOKE_TOP}"/>
  {"".join(wisps)}
  <g class="drifter" style="animation-duration:{glow_period}s"><path class="smoke-glow" d="{glow_wisp}" pathLength="100" filter="url(#glow)"/></g>
</g>
<text class="name" x="{TX - 2}" y="{NAME_Y}">{NAME}</text>
<text class="role" x="{xs}" y="{ROLE_Y}"><tspan class="p">&gt;</tspan> {ROLE} ·</text>
<g>{toks}</g>
<rect class="cursor" x="0" y="{ROLE_Y - 16}" width="10" height="20" rx="1">{cursor_anim}</rect>
{net}
</svg>
'''


def pill(label, t):
    text = f"~/{label}"
    w = round(28 + (len(text) + 2) * 8.43)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 32" width="{w}" height="32" role="img" aria-label="{label}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="{t['a1']}"/><stop offset=".5" stop-color="{t['am']}"/><stop offset="1" stop-color="{t['a2']}"/>
</linearGradient></defs>
<rect x=".75" y=".75" width="{w - 1.5}" height="30.5" rx="15.25" fill="none" stroke="{t['muted']}" stroke-opacity=".35" stroke-width="1.5"/>
<text x="14" y="21" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace" font-size="14">
<tspan fill="{t['muted']}">~/</tspan><tspan fill="{t['fg']}">{label}</tspan><tspan fill="url(#g)"> ↗</tspan></text>
</svg>
'''


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        (out / f"hero-{name}.svg").write_text(build(theme))
        for label, _ in LINKS:
            (out / f"link-{label}-{name}.svg").write_text(pill(label, theme))
    print("wrote", sorted(p.name for p in out.glob("*.svg")))
