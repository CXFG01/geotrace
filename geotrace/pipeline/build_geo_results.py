"""Assemble results.json for the GeoTrace app from chain_results.json, pf_curves.json, geo_pass.json and index.json."""
import json
import math
import os

import numpy as np

import geo_match as gm
from anchor_chain import cum_dist, SEG

D = gm.D
idx = json.load(open(os.path.join(D, "index.json")))
geo = json.load(open(os.path.join(D, "geo_pass.json")))
chain = json.load(open(os.path.join(D, "chain_results.json")))
pf = json.load(open(os.path.join(D, "pf_curves.json")))
G = gm.Graph()


def hav(a, b):
    R = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def thin(pts, min_d=14):
    out = [pts[0]]
    for p in pts[1:]:
        if hav(out[-1], p) >= min_d:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def interp(poly, f):
    if len(poly) == 1:
        return poly[0]
    seg = [hav(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]
    tot = sum(seg) or 1.0
    target = f * tot
    acc = 0.0
    for i, s in enumerate(seg):
        if acc + s >= target:
            r = (target - acc) / (s or 1.0)
            return [poly[i][0] + (poly[i + 1][0] - poly[i][0]) * r, poly[i][1] + (poly[i + 1][1] - poly[i][1]) * r]
        acc += s
    return poly[-1]


out = {"meta": {"title": "GeoTrace", "city": "Toronto", "method": "Cosmos3-Reason reads signs, motion and turns from dashcam video; an anchor-chain solver matches them to the OpenStreetMap road network",
                "bbox": [43.63, -79.47, 43.72, -79.33]}, "trips": []}
for trip in sorted(idx):
    c = chain[trip]
    segs = idx[trip]
    obs = gm.observations(segs, geo, G)
    cd = cum_dist(obs)
    acc = c["accepted"]
    # display names
    for a in acc:
        i = int(round(a["idx"]))
        disp = None
        for j in range(max(0, i - 3), min(len(obs), i + 4)):
            for rn in obs[j]["raw_names"]:
                if G.match_name(rn) == a["name"]:
                    disp = rn
                    break
            if disp:
                break
        a["display"] = disp or a["name"].title()
    # legs from route + marks
    route = c["route"]
    legs = []
    for (start, ia, ib, ln) in c["route_marks"]:
        legs.append((ia, ib, route[start:start + ln]))
    pos = []
    for i in range(len(segs)):
        p = None
        flag = "route"
        for ia, ib, poly in legs:
            if ia <= i <= ib and ib > ia:
                f = (cd[i] - cd[int(round(ia))]) / max(cd[int(round(ib))] - cd[int(round(ia))], 1.0)
                p = interp(poly, min(max(f, 0), 1))
                break
        if p is None:
            if acc and i < acc[0]["idx"]:
                p, flag = acc[0]["latlon"], "before first anchor"
            elif acc:
                p, flag = acc[-1]["latlon"], "after last anchor"
            else:
                p, flag = [43.67, -79.40], "none"
        pos.append([round(p[0], 5), round(p[1], 5), 0 if flag == "route" else 1])
    r = thin(route) if route else []
    # per-segment evidence strings
    ev = []
    for i, o in enumerate(obs):
        g = geo.get(segs[i]["source"], {})
        ev.append({"n": o["raw_names"][:3], "b": (g.get("businesses") or [])[:2], "m": {"left_turn": "L", "right_turn": "R", "u_turn": "U"}.get(o["maneuver"], "S"),
                   "tram": 1 if o["tram"] else 0, "mv": (o["moving"] or "normal")[0]})
    curves = pf[trip]["curves"]
    thin_idx = list(range(0, len(obs), 4)) + [len(obs) - 1]
    out["trips"].append({
        "id": trip, "label": trip.replace("set", "Trip "), "minutes": round(len(segs) * SEG / 60, 1),
        "segs": [s["source"] for s in segs],
        "route": [[round(a, 5), round(b, 5)] for a, b in r],
        "anchors": [{"name": a["display"], "t": a["t"], "latlon": a["latlon"], "reads": a["reads"]} for a in acc],
        "rejected": [{"name": x["name"].title(), "t": x["idx"] * SEG} for x in c["rejected"]],
        "dominant": c["dominant"],
        "pos": pos, "ev": ev,
        "loo": c.get("loo_summary"), "loo_detail": c.get("loo", []),
        "ambiguity": {"x": [i * SEG for i in thin_idx], "turns": [curves["turns only"][i] for i in thin_idx], "turns_tram": [curves["turns + tram tracks"][i] for i in thin_idx]},
        "turns": pf[trip]["turns"],
        "reads": c["n_reads"], "n_anchors": c["n_anchors"],
    })
    print(trip, "route pts", len(r), "anchors", len(acc), "loo", c.get("loo_summary", {}).get("time_median_m"))
p = os.path.join(D, "results.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
print("results.json bytes", os.path.getsize(p))
