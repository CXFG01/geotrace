"""Fetch drivable roads + tram lines for downtown-west Toronto from Overpass in tiles. Saves osm_raw.json (merged)."""
import json
import os
import sys
import time

import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
H = {"User-Agent": "geotrace-hackathon/0.1 (you@example.com)"}
S, W, N, E = 43.630, -79.470, 43.720, -79.330
rows, cols = 3, 3
HW = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"


def query(s, w, n, e):
    return (f'[out:json][timeout:120];(way["highway"~"^({HW})$"]["access"!="private"]({s},{w},{n},{e});'
            f'way["railway"="tram"]({s},{w},{n},{e}););out body geom;')



import hashlib


def fetch_box(s, w, n, e, depth=0):
    key = hashlib.md5(f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}".encode()).hexdigest()[:10]
    p = os.path.join(OUT, f"box_{key}.json")
    if os.path.exists(p):
        return [json.load(open(p))]
    for attempt in range(2 if depth < 2 else 5):
        try:
            r = requests.post("https://overpass.openstreetmap.fr/api/interpreter", data={"data": query(s, w, n, e)}, headers=H, timeout=150)
            if r.status_code == 200:
                d = r.json()
                json.dump(d, open(p, "w"))
                print("box ok", depth, round(s, 3), round(w, 3), len(d["elements"]), flush=True)
                return [d]
            print("box", depth, "status", r.status_code, flush=True)
        except Exception as ex:  # noqa: BLE001
            print("box err", str(ex)[:60], flush=True)
        time.sleep(8)
    if depth >= 3:
        return []
    ms, mw = (s + n) / 2, (w + e) / 2
    out = []
    for (a, b, c, d2) in ((s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)):
        out += fetch_box(a, b, c, d2, depth + 1)
    return out


rows, cols = 3, 3
docs = []
for i in range(rows):
    for j in range(cols):
        docs += fetch_box(S + (N - S) * i / rows, W + (E - W) * j / cols, S + (N - S) * (i + 1) / rows, W + (E - W) * (j + 1) / cols)
nodes, ways = {}, {}
for d in docs:
    for el in d["elements"]:
        if el["type"] == "way" and "geometry" in el:
            for nid, g in zip(el["nodes"], el["geometry"]):
                if g:
                    nodes[nid] = (g["lat"], g["lon"])
            ways[el["id"]] = {"nodes": el["nodes"], "tags": el.get("tags", {})}
json.dump({"nodes": {str(k): v for k, v in nodes.items()}, "ways": {str(k): v for k, v in ways.items()}}, open(os.path.join(OUT, "osm_raw.json"), "w"))
print("saved", len(nodes), len(ways), flush=True)
