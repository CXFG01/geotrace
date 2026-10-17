"""Leave-one-anchor-out evaluation: hide a street name, localise, then measure how far the matched position at the time the
name was read is from the nearest real junction of that street."""
import json
import os
import sys

import numpy as np

import geo_match as gm

D = gm.D


def pos_at(G, res, t):
    return G.position(np.array([res["path_e"][t]]), np.array([res["path_s"][t]]))[0]


def evaluate(G, obs, n=30000, seed=3, skip=("bathurst",), min_gap=3):
    # anchors: (name, first segment index) for names with at least one read; skip dominant 'current road' names
    seen = {}
    for i, o in enumerate(obs):
        for nm in o["names"]:
            if nm in skip:
                continue
            seen.setdefault(nm, []).append(i)
    rows = []
    for nm, segs in seen.items():
        t = segs[0]
        r = gm.run(G, obs, n=n, seed=seed, drop_names=(nm,))
        p = pos_at(G, r, t)
        d, _ = G.street_tree[nm].query(p)
        # also for the full (non-hidden) run
        r2 = gm.run(G, obs, n=n, seed=seed)
        d2, _ = G.street_tree[nm].query(pos_at(G, r2, t))
        rows.append({"name": nm, "seg": t, "t": obs[t]["t"], "reads": len(segs), "hidden_err_m": round(float(d)), "visible_err_m": round(float(d2))})
    return rows


if __name__ == "__main__":
    trip = sys.argv[1]
    idx = json.load(open(os.path.join(D, "index.json")))
    geo = json.load(open(os.path.join(D, "geo_pass.json")))
    G = gm.Graph()
    obs = gm.observations(idx[trip], geo, G)
    rows = evaluate(G, obs)
    for r in sorted(rows, key=lambda r: r["seg"]):
        print(r)
    h = [r["hidden_err_m"] for r in rows]
    v = [r["visible_err_m"] for r in rows]
    if rows:
        print(trip, "anchors", len(rows), "hidden median", int(np.median(h)), "within300m", sum(x <= 300 for x in h), "/", len(h), "| visible median", int(np.median(v)))
