"""Independent validation: businesses read in the video (never used in matching) -> Nominatim -> distance to the reconstructed position."""
import json
import math
import os
import re
import time
from collections import defaultdict

import requests

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
H = {"User-Agent": "geotrace-hackathon/0.1 (you@example.com)"}
res = json.load(open(os.path.join(D, "results.json")))
cache_p = os.path.join(D, "poi_cache.json")
cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
geo = json.load(open(os.path.join(D, "geo_pass.json")))
idx = json.load(open(os.path.join(D, "index.json")))
BAD = re.compile(r"^(n/?a|none|sign|store|shop|parking|restaurant|cafe|the|\d+)$", re.I)


def hav(a, b):
    R = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def seg_dist(p, a, b):
    k = math.cos(math.radians(p[0]))
    ax, ay = (a[1] - p[1]) * 111320 * k, (a[0] - p[0]) * 110540
    bx, by = (b[1] - p[1]) * 111320 * k, (b[0] - p[0]) * 110540
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = max(0, min(1, -(ax * dx + ay * dy) / L)) if L else 0
    return math.hypot(ax + t * dx, ay + t * dy)


def route_dist(p, route):
    return min(seg_dist(p, route[i], route[i + 1]) for i in range(len(route) - 1))


def lookup(name):
    if name in cache:
        return cache[name]
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=H, timeout=25,
                     params={"q": name + ", Toronto", "format": "json", "limit": 8, "viewbox": "-79.47,43.72,-79.33,43.63", "bounded": 1})
    out = [[float(x["lat"]), float(x["lon"]), x.get("display_name", "")[:80]] for x in r.json()] if r.status_code == 200 else []
    cache[name] = out
    json.dump(cache, open(cache_p, "w"))
    time.sleep(1.1)
    return out


total = {"n": 0, "w150": 0, "w300": 0, "w600": 0}
for t in res["trips"]:
    trip = t["id"]
    cnt = defaultdict(list)
    for i, s in enumerate(idx[trip]):
        g = geo.get(s["source"], {})
        for b in g.get("businesses") or []:
            b = b.strip()
            if len(b) >= 4 and not BAD.match(b):
                cnt[b].append(i)
    names = sorted(cnt, key=lambda n: -len(cnt[n]))[:30]
    rows = []
    for n in names:
        pts = lookup(n)
        if not pts:
            continue
        i = cnt[n][0]
        pos = t["pos"][i][:2]
        best = min(pts, key=lambda p: route_dist(p[:2], t["route"]))
        d_route = route_dist(best[:2], t["route"])
        d_time = min(hav(pos, p[:2]) for p in pts)
        rows.append({"name": n, "reads": len(cnt[n]), "t": i * 5, "latlon": [best[0], best[1]], "dist_route_m": round(d_route), "dist_time_m": round(d_time), "matches": len(pts)})
    t["pois"] = rows
    ok = [r for r in rows if r["matches"] <= 6]
    s = {"n": len(ok), "w150": sum(r["dist_time_m"] <= 150 for r in ok), "w300": sum(r["dist_time_m"] <= 300 for r in ok), "w600": sum(r["dist_time_m"] <= 600 for r in ok),
         "route300": sum(r["dist_route_m"] <= 300 for r in ok)}
    t["poi_summary"] = s
    for k in total:
        total[k] += s.get(k, 0)
    # a true-place example for the claim box: nearest POI hit on the route
    good = [r for r in rows if r["dist_route_m"] <= 120 and r["matches"] <= 3]
    t["example_true"] = good[0]["name"] + ", Toronto" if good else None
    print(trip, "queried", len(names), "geocoded", len(rows), s, "example:", t["example_true"], flush=True)
print("TOTAL", total)
json.dump(res, open(os.path.join(D, "results.json"), "w"), separators=(",", ":"))
print("bytes", os.path.getsize(os.path.join(D, "results.json")))
