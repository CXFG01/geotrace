"""Motion energy per segment (mean abs frame difference over the forward scene) + test against sign-to-sign distances."""
import json
import math
import os
import sys

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NF, W, H = 40, 160, 90
res = json.load(open(os.path.join(D, "results.json")))
geo = json.load(open(os.path.join(D, "geo_pass.json")))
idx = json.load(open(os.path.join(D, "index.json")))
SPEED = {"stopped": 0.0, "slow": 4.5, "normal": 10.5, "fast": 15.0}


def hav(a, b):
    R = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def energy(trip):
    arr = np.memmap(os.path.join(D, f"frames_{trip}.bin"), dtype=np.uint8, mode="r")
    n = arr.size // (NF * W * H)
    arr = arr.reshape(n, NF, H, W)
    out = []
    for i in range(n):
        F = arr[i].astype(np.float32)
        d = np.abs(np.diff(F[:, 18:78, 10:150], axis=0)).mean(axis=(1, 2))
        out.append(float(np.median(d)))
    return np.array(out)


if __name__ == "__main__":
    trips = sys.argv[1:]
    E_all, V_all, L_all, N_all = [], [], [], []
    for trip in trips:
        t = next(x for x in res["trips"] if x["id"] == trip)
        E = energy(trip)
        np.save(os.path.join(D, f"energy_{trip}.npy"), E)
        mv = [geo.get(s["source"], {}).get("moving") or "normal" for s in idx[trip]]
        by = {}
        for e, m in zip(E, mv):
            by.setdefault(m, []).append(e)
        print(trip, {k: (len(v), round(float(np.mean(v)), 2)) for k, v in by.items()})
        # legs between consecutive accepted anchors: shortest-path length vs sum of per-segment signals
        route = t["route"]
        pos = t["pos"]
        anchors = t["anchors"]
        for a, b in zip(anchors[:-1], anchors[1:]):
            i0, i1 = int(round(a["t"] / 5)), int(round(b["t"] / 5))
            if i1 - i0 < 2:
                continue
            L = 0.0
            for j in range(i0, i1):
                L += hav(pos[j][:2], pos[j + 1][:2])
            # route length between the anchors is better than interpolated positions: use anchor latlon polyline distance along route
            ia = int(np.argmin([hav(a["latlon"], q) for q in route]))
            ib = int(np.argmin([hav(b["latlon"], q) for q in route]))
            if ia > ib:
                ia, ib = ib, ia
            Lr = sum(hav(route[k], route[k + 1]) for k in range(ia, ib))
            E_all.append(float(E[i0:i1].sum()))
            V_all.append(sum(SPEED.get(m, 10.5) * 5 for m in mv[i0:i1]))
            L_all.append(Lr)
            N_all.append(i1 - i0)
    E_all, V_all, L_all = map(np.array, (E_all, V_all, L_all))
    ok = L_all > 60
    for name, X in (("nominal VLM speed model", V_all), ("frame-difference energy", E_all)):
        x, y = X[ok], L_all[ok]
        k = float((x * y).sum() / (x * x).sum())
        r = np.corrcoef(x, y)[0, 1]
        rel = np.abs(k * x - y) / y
        print(f"{name}: legs {ok.sum()}  correlation r={r:.2f}  scale k={k:.3f}  median relative error {np.median(rel):.0%}  p80 {np.percentile(rel, 80):.0%}")
