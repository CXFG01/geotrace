"""Build a junction-to-junction directed road graph from osm_raw.json (numpy only)."""
import json
import math
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
LAT0, LON0 = 43.67, -79.40
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110540


def xy(lat, lon):
    return ((lon - LON0) * KX, (lat - LAT0) * KY)


TYPES = {"street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd", "drive", "dr", "lane", "ln", "court", "ct", "crescent", "cres",
         "place", "pl", "terrace", "terr", "way", "trail", "gardens", "gdns", "square", "sq", "parkway", "pkwy", "circle", "cir", "east", "west",
         "north", "south", "e", "w", "n", "s", "the", "of"}


def core(name):
    t = re.sub(r"[^a-z0-9' ]", " ", (name or "").lower())
    toks = [w for w in t.split() if w not in TYPES]
    return " ".join(toks)


def bearing(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360   # x east, y north -> 0=N


def main():
    raw = json.load(open(os.path.join(D, "osm_raw.json")))
    nodes = {int(k): v for k, v in raw["nodes"].items()}
    ways = {int(k): v for k, v in raw["ways"].items()}
    roads, trams = {}, []
    for wid, w in ways.items():
        t = w["tags"]
        if t.get("railway") == "tram":
            trams.append(w)
        elif "highway" in t and t["highway"] != "service" and t.get("area") != "yes":
            roads[wid] = w
    use = defaultdict(int)
    for w in roads.values():
        ns = w["nodes"]
        for n in ns:
            use[n] += 1
        use[ns[0]] += 1
        use[ns[-1]] += 1
    junction = {n for n, c in use.items() if c >= 2}
    edges = []   # directed
    for wid, w in roads.items():
        t = w["tags"]
        ow = t.get("oneway")
        fwd, bwd = True, True
        if ow in ("yes", "true", "1") or t.get("junction") == "roundabout":
            bwd = False
        elif ow == "-1":
            fwd = False
        ns = [n for n in w["nodes"] if n in nodes]
        if len(ns) < 2:
            continue
        seg = [ns[0]]
        pieces = []
        for n in ns[1:]:
            seg.append(n)
            if n in junction:
                pieces.append(seg)
                seg = [n]
        for p in pieces:
            pts = [xy(*nodes[n]) for n in p]
            ln = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
            if ln < 1:
                continue
            nm = t.get("name", "")
            for (a, b, pp, pn, ok) in ((p[0], p[-1], pts, p, fwd), (p[-1], p[0], pts[::-1], p[::-1], bwd)):
                if not ok:
                    continue
                k = min(len(pp) - 1, 3)
                edges.append({"u": a, "v": b, "len": ln, "name": nm, "core": core(nm), "hw": t.get("highway"), "oneway": ow in ("yes", "true", "1", "-1"),
                              "b_out": bearing(pp[0], pp[min(1, len(pp) - 1)] if math.dist(pp[0], pp[1]) > 6 else pp[k]),
                              "b_in": bearing(pp[-2] if math.dist(pp[-1], pp[-2]) > 6 else pp[-1 - k], pp[-1]),
                              "pts": pp, "node_ids": pn})
    # tram flag: edge midpoint within 14 m of a tram node
    tn = np.array([xy(*nodes[n]) for w in trams for n in w["nodes"] if n in nodes]) if trams else np.zeros((0, 2))
    tree = cKDTree(tn) if len(tn) else None
    for e in edges:
        mid = np.array(e["pts"][len(e["pts"]) // 2])
        e["tram"] = bool(tree is not None and tree.query(mid)[0] < 14)
    jn = sorted({e["u"] for e in edges} | {e["v"] for e in edges})
    pos = {n: xy(*nodes[n]) for n in jn}
    out = defaultdict(list)
    for i, e in enumerate(edges):
        out[e["u"]].append(i)
    # undirected neighbour count -> true intersections
    nb = defaultdict(set)
    for e in edges:
        nb[e["u"]].add(e["v"])
        nb[e["v"]].add(e["u"])
    G = {"edges": edges, "out": dict(out), "pos": pos, "deg": {n: len(nb[n]) for n in jn}, "latlon": {n: nodes[n] for n in jn}, "LAT0": LAT0, "LON0": LON0}
    pickle.dump(G, open(os.path.join(D, "graph.pkl"), "wb"))
    names = defaultdict(int)
    for e in edges:
        names[e["core"]] += 1
    print("edges", len(edges), "junction nodes", len(jn), "tram edges", sum(e["tram"] for e in edges), "named cores", len(names))
    print([k for k in ("bathurst", "bloor", "wellesley", "nassau", "denison", "barton", "spadina", "dundas", "college", "queen", "king") if k in names])


if __name__ == "__main__":
    main()
