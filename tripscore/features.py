"""Step 2: from cached YOLO sidecars compute risk events per segment (pure python).

Event types
  closing_in   lead vehicle in our corridor with looming time-to-collision (tau) below threshold
  vru_conflict person/cyclist near our path closing in (tau or proximity+growth)
Plus per-segment context: agent density, share of time following a lead vehicle safely.
tau = h / (dh/dt) on the box height; a standard 'looming' proxy that works while the camera itself moves.
"""
import json
import os
import sys

DATA = os.path.expanduser("~/trip-risk-data")
SIDE = os.path.join(DATA, "side")
VEH = {"car", "truck", "bus", "motorcycle"}
VRU = {"person", "bicycle"}
W, H = 1920, 1080

TAU_VEH = 2.0      # seconds
TAU_VRU = 2.4
MIN_RUN = 10       # consecutive frames (~0.27s at 30fps)
WIN = 15           # frames for slope fit


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def in_corridor(b, wide=False):
    """Bottom-centre of the box inside a trapezoid in front of the camera."""
    cx, by = (b[0] + b[2]) / 2, b[3]
    if by < 0.50 * H:
        return False
    f = (by - 0.50 * H) / (0.5 * H)           # 0 at horizon band, 1 at bottom
    half = (0.07 + 0.30 * f) * W * (1.35 if wide else 1.0)
    return abs(cx - W / 2) <= half


def on_heading(b, k):
    """Constant-bearing test: object stays within k box-widths of the heading line (image centre)."""
    w = max(1, b[2] - b[0])
    return abs((b[0] + b[2]) / 2 - W / 2) <= k * w + 0.06 * W


def track(frames, classes):
    tracks, nid = [], 0
    for t, dets in frames:
        cur = [d[2] for d in dets if d[0] in classes and d[1] >= 0.35]
        used = set()
        for tr in tracks:
            if tr["miss"] > 5:
                continue
            best, bi = 0.25, None
            for i, b in enumerate(cur):
                if i in used:
                    continue
                v = iou(tr["last"], b)
                if v > best:
                    best, bi = v, i
            if bi is not None:
                used.add(bi)
                tr["pts"].append((t, cur[bi]))
                tr["last"], tr["miss"] = cur[bi], 0
            else:
                tr["miss"] += 1
        for i, b in enumerate(cur):
            if i not in used:
                tracks.append({"id": nid, "pts": [(t, b)], "last": b, "miss": 0})
                nid += 1
    return [t for t in tracks if len(t["pts"]) >= WIN]


def slope(ts, hs):
    n = len(ts)
    mt, mh = sum(ts) / n, sum(hs) / n
    den = sum((x - mt) ** 2 for x in ts)
    return 0.0 if den == 0 else sum((x - mt) * (y - mh) for x, y in zip(ts, hs)) / den


def tau_series(tr):
    pts = tr["pts"]
    out = []
    for i in range(0, len(pts) - WIN + 1):
        w = pts[i:i + WIN]
        ts = [p[0] for p in w]
        hs = [p[1][3] - p[1][1] for p in w]
        s = slope(ts, hs)
        h = hs[-1]
        if s > 5 and h > 35:                    # growing, not tiny
            out.append((ts[-1], h / s, w[-1][1]))
        else:
            out.append((ts[-1], 99.0, w[-1][1]))
    return out


def sustained(series, thr, corridor_fn):
    best, run, run_start = None, 0, 0
    for k, (t, tau, b) in enumerate(series):
        if tau < thr and corridor_fn(b):
            if run == 0:
                run_start = k
            run += 1
            if run >= MIN_RUN and (best is None or tau < best[1]):
                best = (t, tau, b, run_start, k)
        else:
            run = 0
    return best


def analyse(seg):
    p = os.path.join(SIDE, os.path.basename(seg["source"]) + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    frames = d["frames"]
    nfr = max(1, len(frames))
    agents = sum(1 for _, dets in frames for x in dets if x[0] in VEH | VRU) / nfr
    events = []
    vt = track(frames, VEH)
    lead_frames = safe_frames = 0
    seen_lead = False
    for tr in vt:
        series = tau_series(tr)
        in_c = [s for s in series if in_corridor(s[2])]
        lead_frames += len(in_c)
        safe_frames += sum(1 for s in in_c if s[1] > 6)
        ev = sustained(series, TAU_VEH, lambda b: in_corridor(b) and on_heading(b, 0.9))
        if ev:
            t, tau, b, k0, k1 = ev
            pts = tr["pts"][k0: k1 + WIN]
            events.append({"type": "closing_in", "tau": round(tau, 2), "t_in_seg": round(t, 2),
                           "sev": round(min(3.0, TAU_VEH / max(tau, 0.3)), 2),
                           "track": [[round(q[0], 2)] + [int(v) for v in q[1]] for q in pts[::3]]})
    pt = track(frames, VRU)
    for tr in pt:
        series = tau_series(tr)
        ev = sustained(series, TAU_VRU, lambda b: in_corridor(b, wide=True) and on_heading(b, 1.6))
        if ev:
            t, tau, b, k0, k1 = ev
            pts = tr["pts"][k0: k1 + WIN]
            events.append({"type": "vru_conflict", "tau": round(tau, 2), "t_in_seg": round(t, 2),
                           "sev": round(min(3.0, 1.2 * TAU_VRU / max(tau, 0.3)), 2),
                           "track": [[round(q[0], 2)] + [int(v) for v in q[1]] for q in pts[::3]]})
    events.sort(key=lambda e: -e["sev"])
    return {"source": seg["source"], "trip": seg["trip"], "t": seg["t"], "dur": seg["dur"],
            "agents": round(agents, 2), "lead_frames": lead_frames, "safe_lead_frames": safe_frames,
            "nframes": nfr, "events": events[:3]}


if __name__ == "__main__":
    index = json.load(open(os.path.join(DATA, "index.json")))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out = []
    for trip, segs in index.items():
        if only and trip != only:
            continue
        for s in segs:
            r = analyse(s)
            if r:
                r["caption"] = s["caption"]
                r["chunk"] = s["chunk"]
                out.append(r)
    json.dump(out, open(os.path.join(DATA, "features%s.json" % ("_" + only if only else "")), "w"))
    evs = [(e["sev"], r["trip"], r["t"], e["type"], e["tau"], r["source"].split("/")[-1][-40:]) for r in out for e in r["events"]]
    evs.sort(reverse=True)
    print("segments", len(out), "events", len(evs))
    for e in evs[:15]:
        print(e)
    byt = {}
    for r in out:
        d = byt.setdefault(r["trip"], [0, 0, 0.0])
        d[0] += 1
        d[1] += len(r["events"])
        d[2] += r["agents"]
    for t, d in sorted(byt.items()):
        print(t, "segs", d[0], "events", d[1], "mean agents", round(d[2] / d[0], 1))
