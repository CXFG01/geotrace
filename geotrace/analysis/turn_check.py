"""Independent check: turns seen in the video vs turns in the route reconstructed from street-name signs only."""
import json
import math
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
res = json.load(open(os.path.join(D, "results.json")))
geo = json.load(open(os.path.join(D, "geo_pass.json")))
idx = json.load(open(os.path.join(D, "index.json")))


def xy(p, k=math.cos(math.radians(43.67))):
    return ((p[1] + 79.4) * 111320 * k, (p[0] - 43.67) * 110540)


def bearing(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def dh(a, b):
    return (b - a + 180) % 360 - 180


def route_heading_change(route, p0, p1):
    """Signed (clockwise=+) heading change along the route between the points nearest p0 and p1."""
    R = [xy(q) for q in route]
    P0, P1 = xy(p0), xy(p1)
    i0 = int(np.argmin([math.dist(r, P0) for r in R]))
    i1 = int(np.argmin([math.dist(r, P1) for r in R]))
    if i0 > i1:
        i0, i1 = i1, i0
    i0 = max(0, i0 - 1)
    i1 = min(len(R) - 1, i1 + 1)
    # resample sub-polyline every ~12 m
    pts = [R[i0]]
    for r in R[i0 + 1:i1 + 1]:
        if math.dist(pts[-1], r) >= 12:
            pts.append(r)
    if len(pts) < 3:
        return 0.0
    hs = [bearing(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    cum, best = 0.0, 0.0
    for a, b in zip(hs[:-1], hs[1:]):
        cum += dh(a, b)
        if abs(cum) > abs(best):
            best = cum
    return best


tot = {"turns": 0, "match": 0, "wrong": 0, "none": 0, "base_n": 0, "base_turn": 0}
for t in res["trips"]:
    segs = idx[t["id"]]
    pos = t["pos"]
    rows = {"turns": 0, "match": 0, "wrong": 0, "none": 0, "base_n": 0, "base_turn": 0}
    for i in range(1, len(segs) - 1):
        if pos[i - 1][2] or pos[i][2] or pos[i + 1][2]:
            continue          # only where the route actually covers the trip
        g = geo.get(segs[i]["source"], {})
        m = g.get("maneuver")
        change = route_heading_change(t["route"], pos[i - 1][:2], pos[i + 1][:2])
        if m in ("left_turn", "right_turn"):
            rows["turns"] += 1
            want = 1 if m == "right_turn" else -1
            if abs(change) < 35:
                rows["none"] += 1
            elif change * want > 0:
                rows["match"] += 1
            else:
                rows["wrong"] += 1
        elif m in ("straight", "none"):
            rows["base_n"] += 1
            rows["base_turn"] += abs(change) >= 35
    print(t["id"], rows)
    for k in tot:
        tot[k] += rows[k]
print("TOTAL", tot)
print("observed turns whose direction the route reproduces: %d/%d (%.0f%%); route shows no turn: %d; opposite turn: %d" % (
    tot["match"], tot["turns"], 100 * tot["match"] / max(1, tot["turns"]), tot["none"], tot["wrong"]))
print("false-turn rate where no turn was observed: %.0f%% of %d segments" % (100 * tot["base_turn"] / max(1, tot["base_n"]), tot["base_n"]))
