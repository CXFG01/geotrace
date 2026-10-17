"""Particle-filter map matching of a dashcam trip onto the OSM junction graph.

Evidence per 5 s segment: street-name signs (anchors), net heading change (maneuver), tram tracks, one-way, motion state.
"""
import difflib
import json
import math
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from build_graph import core, xy, LAT0, LON0, KX, KY

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
SPEED = {"stopped": 0.0, "slow": 4.5, "normal": 10.5, "fast": 15.0}
SEG = 5.0


class Graph:
    def __init__(self):
        G = pickle.load(open(os.path.join(D, "graph.pkl"), "rb"))
        self.G = G
        E = G["edges"]
        self.E = E
        self.n = len(E)
        self.len = np.array([e["len"] for e in E])
        self.u = np.array([e["u"] for e in E])
        self.v = np.array([e["v"] for e in E])
        self.b_in = np.array([e["b_in"] for e in E])
        self.b_out = np.array([e["b_out"] for e in E])
        self.tram = np.array([e["tram"] for e in E])
        self.oneway = np.array([e["oneway"] for e in E])
        self.core = [e["core"] for e in E]
        self.pu = np.array([G["pos"][e["u"]] for e in E])
        self.pv = np.array([G["pos"][e["v"]] for e in E])
        out = G["out"]
        # reverse edge lookup
        rev = {}
        for i, e in enumerate(E):
            rev.setdefault((e["u"], e["v"]), []).append(i)
        K = 8
        self.OUT = -np.ones((self.n, K), dtype=np.int64)
        self.cnt = np.zeros(self.n, dtype=np.int64)
        self.REV = -np.ones(self.n, dtype=np.int64)
        for i, e in enumerate(E):
            cands = [j for j in out.get(e["v"], []) if E[j]["v"] != e["u"]]
            r = [j for j in out.get(e["v"], []) if E[j]["v"] == e["u"]]
            if r:
                self.REV[i] = r[0]
            if not cands:
                cands = r
            cands = cands[:K]
            self.OUT[i, :len(cands)] = cands
            self.cnt[i] = len(cands)
        # junction nodes per street core (true intersections only: degree>=3)
        by = defaultdict(set)
        for e in E:
            if G["deg"].get(e["u"], 0) >= 3 and e["core"]:
                by[e["core"]].add(e["u"])
            if G["deg"].get(e["v"], 0) >= 3 and e["core"]:
                by[e["core"]].add(e["v"])
        self.street_nodes = {k: np.array([G["pos"][n] for n in v]) for k, v in by.items()}
        self.street_tree = {k: cKDTree(v) for k, v in self.street_nodes.items()}
        self.names = sorted(self.street_nodes)

    def match_name(self, text):
        c = core(text)
        if not c or len(c) < 3:
            return None
        if c in self.street_tree:
            return c
        m = difflib.get_close_matches(c, self.names, n=1, cutoff=0.86)
        return m[0] if m else None

    def core_mask(self, nm):
        if not hasattr(self, "_cm"):
            self._cm = {}
        if nm not in self._cm:
            self._cm[nm] = np.array([c == nm for c in self.core])
        return self._cm[nm]

    def position(self, e, s):
        f = np.clip(s / np.maximum(self.len[e], 1e-6), 0, 1)[:, None]
        return self.pu[e] * (1 - f) + self.pv[e] * f


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def observations(trip_segs, geo, G):
    obs = []
    for s in trip_segs:
        r = geo.get(s["source"], {})
        names = []
        for t in r.get("street_names") or []:
            m = G.match_name(t)
            if m and m not in names:
                names.append(m)
        obs.append({"names": names, "raw_names": r.get("street_names") or [], "tram": bool(r.get("tram_tracks")), "moving": r.get("moving") or "normal",
                    "maneuver": r.get("maneuver") or "straight", "road": r.get("road") or "unknown", "t": s["t"], "ok": "parse_error" not in r and "error" not in r})
    # persistence: a name read in >=3 of the 5 neighbouring segments is a strong anchor
    total = defaultdict(int)
    for o in obs:
        for nm in o["names"]:
            total[nm] += 1
    for i, o in enumerate(obs):
        o["strength"] = {}
        for nm in o["names"]:
            c = sum(1 for j in range(max(0, i - 2), min(len(obs), i + 3)) if nm in obs[j]["names"])
            o["strength"][nm] = (c, total[nm])
    return obs


def run(G, obs, n=40000, seed=0, use_names=True, use_turns=True, use_tram=True, drop_names=(), keep_hist=True, name_tol=120.0, straight_p=0.7, yaw=None, sig_yaw=30.0, name_pen=None, speed_mult=1.0, pins=None, pin_tol=140.0):
    rng = np.random.default_rng(seed)
    N = n
    p = G.len / G.len.sum()
    e = rng.choice(G.n, size=N, p=p)
    s = rng.random(N) * G.len[e]
    scale = np.exp(rng.normal(0, 0.35, N))
    T = len(obs)
    H_e = np.zeros((T, N), dtype=np.int32) if keep_hist else None
    H_s = np.zeros((T, N), dtype=np.float32) if keep_hist else None
    H_par = np.zeros((T, N), dtype=np.int32) if keep_hist else None
    perp, mass = [], []
    clouds = {}
    w = np.ones(N) / N
    for t, o in enumerate(obs):
        sp = SPEED.get(o["moving"], 10.5) * SEG * speed_mult
        scale = scale * np.exp(rng.normal(0, 0.10, N))
        d = sp * scale * np.exp(rng.normal(0, 0.3, N))
        dh = np.zeros(N)
        rem = d.copy()
        cur_e, cur_s = e.copy(), s.copy()
        for _ in range(14):
            left = G.len[cur_e] - cur_s
            go = rem > left
            if not go.any():
                break
            idx = np.where(go)[0]
            ce = cur_e[idx]
            cnt = G.cnt[ce]
            has = cnt > 0
            r = np.floor(rng.random(len(idx)) * np.maximum(cnt, 1)).astype(np.int64)
            cand = G.OUT[ce]
            ang = np.abs(angdiff(G.b_out[np.where(cand >= 0, cand, 0)], G.b_in[ce][:, None]))
            ang = np.where(cand >= 0, ang, 999.0)
            sidx = np.argmin(ang, axis=1)
            r = np.where(rng.random(len(idx)) < straight_p, sidx, r)
            nxt = np.where(has, G.OUT[ce, r], ce)
            turn = np.where(has, angdiff(G.b_out[nxt], G.b_in[ce]), 0.0)
            dh[idx] += turn
            rem[idx] -= left[idx]
            cur_e[idx] = nxt
            cur_s[idx] = 0.0
        cur_s = np.minimum(cur_s + rem, G.len[cur_e])
        e, s = cur_e, cur_s
        # ---- weights
        lw = np.zeros(N)
        if use_turns and o["ok"]:
            man = o["maneuver"]
            if man == "left_turn":      # bearings run clockwise, so a left turn is a negative heading change
                lw += np.where((dh < -45) & (dh > -140), 0.0, math.log(0.06))
            elif man == "right_turn":
                lw += np.where((dh > 45) & (dh < 140), 0.0, math.log(0.06))
            elif man == "u_turn":
                lw += np.where(np.abs(dh) > 150, 0.0, math.log(0.05))
            else:
                lw += np.where(np.abs(dh) < 35, 0.0, math.log(0.10))
        if pins and t in pins:
            pos_p = G.position(e, s)
            dmin = np.min(np.linalg.norm(pos_p[:, None, :] - pins[t][None, :, :], axis=2), axis=1)
            lw += np.where(dmin < pin_tol, 0.0, math.log(0.03))
        if yaw is not None:
            lw += -0.5 * ((dh - yaw[t]) / sig_yaw) ** 2
        if use_tram and o["ok"]:
            if o["tram"]:
                lw += np.where(G.tram[e], 0.0, math.log(0.2))
            else:
                lw += np.where(G.tram[e], math.log(0.6), 0.0)
        if o["ok"] and o["road"] == "one_way":
            lw += np.where(G.oneway[e], 0.0, math.log(0.7))
        elif o["ok"] and o["road"] == "two_way":
            lw += np.where(G.oneway[e], math.log(0.7), 0.0)
        names = [x for x in o["names"] if x not in drop_names]
        if use_names and names:
            pos = G.position(e, s)
            lik = np.zeros(N)
            for nm in names:
                dist, _ = G.street_tree[nm].query(pos)
                c, tot = o["strength"].get(nm, (1, 1))
                if tot >= 15:      # dominant name = the road being driven: must actually be on it (or at its junction)
                    matched = G.core_mask(nm)[e] | (dist < 40)
                    pen = 0.03
                else:
                    matched = (dist < name_tol) | G.core_mask(nm)[e]
                    pen = (name_pen if name_pen is not None else (0.2 if c >= 3 else 0.5))
                lik = np.maximum(lik, np.where(matched, 1.0, pen))
            lw += np.log(lik)
        w = w * np.exp(lw - lw.max())
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if keep_hist:
            H_e[t], H_s[t] = e, s
        # summary: cell perplexity
        pos = G.position(e, s)
        cell = (np.floor(pos[:, 0] / 120).astype(np.int64) * 100003 + np.floor(pos[:, 1] / 120).astype(np.int64))
        uc, inv = np.unique(cell, return_inverse=True)
        cm = np.bincount(inv, weights=w)
        cm = cm[cm > 0]
        perp.append(float(np.exp(-np.sum(cm * np.log(cm)))))
        mass.append(float(cm.max()))
        if t in (0, 1, 2, 4, 8, 16, 32, 64, 128, T - 1) or t % 20 == 0:
            pick = rng.choice(N, size=min(500, N), p=w)
            pp = pos[pick]
            clouds[t] = [[round(LAT0 + y / KY, 5), round(LON0 + x / KX, 5)] for x, y in pp[::1]]
        # resample
        if ess < 0.5 * N or t == T - 1:
            u0 = rng.random()
            cs = np.cumsum(w)
            par = np.searchsorted(cs, (np.arange(N) + u0) / N)
            par = np.minimum(par, N - 1)
            e, s, scale = e[par], s[par], scale[par]
            w = np.ones(N) / N
            if keep_hist:
                H_par[t] = par
        else:
            if keep_hist:
                H_par[t] = np.arange(N)
    res = {"perplexity": perp, "top_mass": mass, "clouds": clouds}
    if keep_hist:
        # backtrack mode lineage: pick final particle in the heaviest cell
        fin = np.arange(N)
        pos = G.position(H_e[T - 1], H_s[T - 1])
        cell = (np.floor(pos[:, 0] / 150).astype(np.int64) * 100003 + np.floor(pos[:, 1] / 150).astype(np.int64))
        uc, inv = np.unique(cell, return_inverse=True)
        top = np.argmax(np.bincount(inv))
        cand = np.where(inv == top)[0]
        k = int(cand[0])
        path_e, path_s = [0] * T, [0.0] * T
        for t in range(T - 1, -1, -1):
            path_e[t], path_s[t] = int(H_e[t][k]), float(H_s[t][k])
            k = int(H_par[t][k])
        res["path_e"], res["path_s"] = path_e, path_s
        res["final_cell_mass"] = float(np.bincount(inv, weights=np.ones(N) / N)[top])
    return res
