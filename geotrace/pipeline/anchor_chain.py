"""Robust anchor-chain localisation.

Each street-name sign read is an anchor that can sit at any junction of that street. A dynamic program picks, for every trip, the
largest chain of anchors whose junctions are reachable from each other in network distance consistent with the elapsed time and the
observed motion states; reads that do not fit are rejected as outliers (VLM misreads / hallucinations).
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import geo_match as gm
from build_graph import LAT0, LON0, KX, KY

D = gm.D
SPEED = {"stopped": 0.0, "slow": 4.5, "normal": 10.5, "fast": 15.0}
OUTLIER = -2.2
SIGMA = 0.55
SEG = 5.0


def ll(x, y):
    return [round(LAT0 + y / KY, 5), round(LON0 + x / KX, 5)]


class Net:
    def __init__(self, G):
        self.G = G
        E = G.E
        ids = sorted({e["u"] for e in E} | {e["v"] for e in E})
        self.nid = {n: i for i, n in enumerate(ids)}
        self.ids = ids
        self.pos = np.array([G.G["pos"][n] for n in ids])
        rows = [self.nid[e["u"]] for e in E]
        cols = [self.nid[e["v"]] for e in E]
        w = [e["len"] for e in E]
        best = {}
        for r, c, l, k in zip(rows, cols, w, range(len(E))):
            if (r, c) not in best or l < best[(r, c)][0]:
                best[(r, c)] = (l, k)
        self.edge_of = {k: v[1] for k, v in best.items()}
        rr = [k[0] for k in best]
        cc = [k[1] for k in best]
        ww = [v[0] for v in best.values()]
        self.A = csr_matrix((ww, (rr, cc)), shape=(len(ids), len(ids)))
        self._rows = {}
        self._pred = {}
        # junction nodes per street core
        by = defaultdict(set)
        dom = defaultdict(set)
        for e in E:
            if not e["core"]:
                continue
            for n in (e["u"], e["v"]):
                dom[e["core"]].add(self.nid[n])
                if G.G["deg"].get(n, 0) >= 3:
                    by[e["core"]].add(self.nid[n])
        self.cands = {k: sorted(v) for k, v in by.items()}
        self.on_road = {k: v for k, v in dom.items()}

    def dist_row(self, i):
        if i not in self._rows:
            d, p = dijkstra(self.A, directed=True, indices=i, return_predecessors=True)
            self._rows[i], self._pred[i] = d.astype(np.float32), p
        return self._rows[i]

    def route(self, a, b):
        self.dist_row(a)
        p = self._pred[a]
        nodes = [b]
        while nodes[-1] != a:
            q = p[nodes[-1]]
            if q < 0:
                return None
            nodes.append(int(q))
        nodes.reverse()
        pts = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            e = self.G.E[self.edge_of[(u, v)]]
            seg = e["pts"]
            pts += [ll(*q) for q in seg[:-1]]
        pts.append(ll(*self.pos[nodes[-1]]))
        return pts


def build_anchors(obs, dominant):
    """Collapse runs of the same name (gap<=2 segments) into single anchors with a mean segment index."""
    anchors = []
    last = {}
    for i, o in enumerate(obs):
        for nm in o["names"]:
            if nm in dominant:
                continue
            if nm in last and i - anchors[last[nm]]["segs"][-1] <= 2:
                anchors[last[nm]]["segs"].append(i)
            else:
                anchors.append({"name": nm, "segs": [i]})
                last[nm] = len(anchors) - 1
    for a in anchors:
        a["idx"] = float(np.mean(a["segs"]))
        a["reads"] = len(a["segs"])
    anchors.sort(key=lambda a: a["idx"])
    return anchors


def cum_dist(obs):
    c = [0.0]
    for o in obs:
        c.append(c[-1] + SPEED.get(o["moving"], 10.5) * SEG)
    return np.array(c)


def solve(N, obs, anchors, dominant_names, cd, skip_idx=None):
    A = [a for k, a in enumerate(anchors) if k != skip_idx and a["name"] in N.cands]
    n = len(A)
    if n == 0:
        return [], []
    cand = [np.array(N.cands[a["name"]]) for a in A]
    # emission: slight preference for nodes on the dominant road, and for multi-read anchors
    def em(k):
        a = A[k]
        base = np.zeros(len(cand[k]))
        for dn in dominant_names:
            on = N.on_road.get(dn, set())
            base += np.array([0.0 for c in cand[k]])
        return base + 0.3 * min(a["reads"], 3)
    E = [em(k) for k in range(n)]
    best = [None] * n
    arg = [None] * n
    for i in range(n):
        score = np.full(len(cand[i]), OUTLIER * i) + E[i]
        back = [(-1, -1)] * len(cand[i])
        for j in range(i):
            exp = max(cd[int(round(A[i]["idx"]))] - cd[int(round(A[j]["idx"]))], 40.0)
            for mi, m in enumerate(cand[j]):
                dr = N.dist_row(m)[cand[i]]
                ok = np.isfinite(dr)
                ratio = np.log(np.maximum(dr, 25.0) / exp)
                tr = np.where(ok, -0.5 * (ratio / SIGMA) ** 2, -30.0)
                # also reachable in time at a max plausible 18 m/s
                tmax = (A[i]["idx"] - A[j]["idx"]) * SEG * 18.0
                tr = np.where(dr > tmax + 80, tr - 15.0, tr)
                cand_score = best[j][mi] + OUTLIER * (i - j - 1) + tr + E[i]
                upd = cand_score > score
                if upd.any():
                    score = np.where(upd, cand_score, score)
                    for ci in np.where(upd)[0]:
                        back[ci] = (j, mi)
        best[i], arg[i] = score, back
    # end: choose best (i, node) with tail outliers
    tot = [(best[i].max() + OUTLIER * (n - 1 - i), i, int(best[i].argmax())) for i in range(n)]
    _, i, ni = max(tot)
    chain = []
    while i >= 0:
        chain.append((i, ni))
        i, ni = arg[i][ni]
    chain.reverse()
    used = {k for k, _ in chain}
    out = [{"anchor": A[k], "node": int(cand[k][ci])} for k, ci in chain]
    rejected = [A[k] for k in range(n) if k not in used]
    return out, rejected


def analyse(N, trip, obs_all, with_loo=True):
    obs = obs_all
    tot = defaultdict(int)
    for o in obs:
        for nm in o["names"]:
            tot[nm] += 1
    dominant = {nm for nm, c in tot.items() if c >= 15}
    anchors = build_anchors(obs, dominant)
    cd = cum_dist(obs)
    chain, rej = solve(N, obs, anchors, dominant, cd)
    res = {"trip": trip, "dominant": sorted(dominant), "n_reads": sum(len(a["segs"]) for a in anchors), "n_anchors": len(anchors),
           "accepted": [{"name": c["anchor"]["name"], "idx": c["anchor"]["idx"], "t": c["anchor"]["idx"] * SEG, "latlon": ll(*N.pos[c["node"]]), "node": c["node"], "reads": c["anchor"]["reads"]} for c in chain],
           "rejected": [{"name": a["name"], "idx": a["idx"], "reads": a["reads"]} for a in rej]}
    # route
    route, marks = [], []
    for a, b in zip(chain[:-1], chain[1:]):
        r = N.route(a["node"], b["node"])
        if r is None:
            continue
        marks.append((len(route), a["anchor"]["idx"], b["anchor"]["idx"], len(r)))
        route += r
    res["route"] = route
    res["route_marks"] = marks
    if with_loo and len(chain) >= 3:
        errs = []
        for k, c in enumerate(chain):
            a = c["anchor"]
            ai = anchors.index(a)
            ch2, _ = solve(N, obs, anchors, dominant, cd, skip_idx=ai)
            if len(ch2) < 2:
                continue
            # route through remaining chain, remembering the polyline of each leg
            legs = []
            pts = []
            for p, q in zip(ch2[:-1], ch2[1:]):
                r = N.route(p["node"], q["node"])
                if r:
                    legs.append((p["anchor"]["idx"], q["anchor"]["idx"], r))
                    pts += r
            if not pts:
                continue
            P = np.array([[(lo - LON0) * KX, (la - LAT0) * KY] for la, lo in pts])
            tree = gm.cKDTree(P)
            nodes_xy = N.pos[N.cands[a["name"]]]
            d_route = float(min(tree.query(nodes_xy)[0]))
            # time-aligned: interpolate along the leg that brackets the hidden anchor (else n/a)
            d_time = None
            for i0, i1, r in legs:
                if i0 <= a["idx"] <= i1 and i1 > i0:
                    f = (cd[int(round(a["idx"]))] - cd[int(round(i0))]) / max(cd[int(round(i1))] - cd[int(round(i0))], 1.0)
                    R = np.array([[(lo - LON0) * KX, (la - LAT0) * KY] for la, lo in r])
                    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
                    target = f * seglen[-1]
                    k2 = int(np.searchsorted(seglen, target))
                    k2 = min(max(k2, 0), len(R) - 1)
                    d_time = float(min(np.linalg.norm(nodes_xy - R[k2], axis=1)))
                    break
            errs.append({"name": a["name"], "idx": a["idx"], "min_dist_m": round(d_route), "time_aligned_m": None if d_time is None else round(d_time)})
        res["loo"] = errs
        if errs:
            v = [e["min_dist_m"] for e in errs]
            tv = [e["time_aligned_m"] for e in errs if e["time_aligned_m"] is not None]
            res["loo_summary"] = {"n": len(v), "median_m": int(np.median(v)), "within150": sum(x <= 150 for x in v), "within300": sum(x <= 300 for x in v),
                                  "time_n": len(tv), "time_median_m": int(np.median(tv)) if tv else None, "time_within300": sum(x <= 300 for x in tv),
                                  "time_within600": sum(x <= 600 for x in tv)}
    return res, chain


if __name__ == "__main__":
    idx = json.load(open(os.path.join(D, "index.json")))
    geo = json.load(open(os.path.join(D, "geo_pass.json")))
    G = gm.Graph()
    N = Net(G)
    trips = sys.argv[1:] or sorted(idx)
    out = {}
    for trip in trips:
        obs = gm.observations(idx[trip], geo, G)
        res, chain = analyse(N, trip, obs)
        out[trip] = res
        print(trip, "dominant", res["dominant"], "reads", res["n_reads"], "anchors", res["n_anchors"], "accepted", len(res["accepted"]), "rejected", len(res["rejected"]))
        print("   accepted:", [(a["name"], int(a["t"])) for a in res["accepted"]])
        print("   rejected:", [(a["name"], int(a["idx"] * SEG)) for a in res["rejected"]])
        print("   loo:", res.get("loo_summary"))
    json.dump(out, open(os.path.join(D, "chain_results.json"), "w"))
