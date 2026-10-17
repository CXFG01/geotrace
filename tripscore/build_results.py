"""Step 4 (v2, fault-aware): detector events + Cosmos review + Cosmos fault attribution -> per-trip scores -> results.json

Penalty only for driver-attributable events. Weight = severity x response factor
  poor 1.0, adequate 0.25 (minor / handled), good 0. Hazards caused by others or lawful
  pedestrian crossings handled well earn credit instead of penalty.
"""
import json
import math
import os

DATA = os.path.expanduser("~/trip-risk-data")
feats = json.load(open(os.path.join(DATA, "features.json")))
conf = json.load(open(os.path.join(DATA, "confirm.json")))
conf2 = json.load(open(os.path.join(DATA, "confirm2.json")))
conf3 = json.load(open(os.path.join(DATA, "nemotron.json")))
DISPUTED_W = 0.3
CLOSE_CALL_W = 0.4

RESP_W = {"poor": 1.0, "adequate": 0.25, "good": 0.0}
LABELS = {"following_too_close": "Following too close", "late_braking": "Late braking", "failed_to_yield": "Failure to yield",
          "signal_violation": "Signal violation", "close_call": "Close call", "unsafe_maneuver": "Unsafe maneuver", "speeding": "Speeding"}
K = 3.0                  # score = 100 * (1 - exp(-R / K))
FOLLOW_W = 2.5           # weight of close-following exposure (share of following time that is not 'safe')
CTX_MEAN = sum(r["agents"] for r in feats) / len(feats)


OTHER_CUES = ("cuts in", "cut in", "cutting in", "pulls out", "pulled out", "pulling out", "suddenly appears", "suddenly steps",
              "steps into", "jaywalk", "darts", "runs out", "double-parked", "door open", "merges in", "merged in")


def blames_other(text):
    t = (text or "").lower()
    return any(k in t for k in OTHER_CUES)


def norm(e):
    e = (e or "").lower().replace(" ", "_").replace("-", "_")
    return e if e in LABELS else None


trips = {}
for r in feats:
    t = trips.setdefault(r["trip"], {"segs": [], "dur": 0.0, "agents": 0.0, "lead": 0, "safe": 0})
    t["segs"].append(r)
    t["dur"] += r["dur"]
    t["agents"] += r["agents"] * r["dur"]
    t["lead"] += r["lead_frames"]
    t["safe"] += r["safe_lead_frames"]

out = {"meta": {"camera": "pie_cam-3 (Toronto)",
                "method": "YOLO11 boxes -> tracked time-to-collision flags -> Cosmos3-Reason review -> two-model fault attribution (Cosmos3-Reason + Nemotron-VL-120B via W&B) -> score",
                "k": K, "labels": list(LABELS.values())}, "trips": []}
for name, t in sorted(trips.items()):
    minutes = t["dur"] / 60
    density = t["agents"] / t["dur"]
    ctx = 0.75 + 0.25 * density / CTX_MEAN
    events, counts = [], {v: 0 for v in LABELS.values()}
    raw = 0.0
    reviewed = flagged = handled = n_driver = n_disputed = 0
    for r in t["segs"]:
        if not r["events"]:
            continue
        flagged += 1
        e0 = r["events"][0]
        c1, c2 = conf.get(r["source"]), conf2.get(r["source"])
        if c1 is not None:
            reviewed += 1
        c3chk = conf3.get(r["source"]) or {}
        if not (c1 and c1.get("risky")):
            continue
        if not c2 or "error" in c2 or "fault" not in c2:
            if "fault" not in c3chk:
                continue
            c2 = {"fault": "none", "severity": 0, "driver_response": "good", "reason": c1.get("reason", "")}
        tt = r["t"] + e0["t_in_seg"]
        resp = c2.get("driver_response") or "adequate"
        sev = float(c2.get("severity") or 0)
        err = norm(c2.get("driver_error"))
        base = {"t": round(tt, 1), "seg": r["source"], "t_in_seg": e0["t_in_seg"], "tau": e0["tau"], "sev": sev,
                "reason": c2.get("reason") or c1.get("reason", ""), "response": resp, "gap_m": c2.get("gap_m"),
                "track": e0["track"], "caption": r.get("caption", "")[:300]}
        c3 = conf3.get(r["source"]) or {}
        if blames_other(base["reason"]):
            c2["fault"] = "other"
        n_fault = c3.get("fault")
        if n_fault == "driver" and norm(c3.get("driver_error")):
            err3 = norm(c3["driver_error"])
            resp3 = c3.get("driver_response") or "poor"
            sev3 = float(c3.get("severity") or 2)
            agree = c2["fault"] == "driver"
            if agree:
                w = sev3 * RESP_W.get(resp3, 1.0)
                raw += w
                counts[LABELS[err3]] += 1
                n_driver += 1
                events.append(dict(base, sev=sev3, response=resp3, reason=c3.get("reason") or base["reason"], status="driver_at_fault", type=err3,
                                   label=LABELS[err3], weight=round(w, 2), agree="2/2 models"))
            else:
                w = sev3 * RESP_W.get(resp3, 1.0) * CLOSE_CALL_W
                raw += w
                counts["Close call"] += 1
                n_disputed += 1
                events.append(dict(base, sev=sev3, response=resp3, reason="Close call, fault unclear. Reviewer 1: " + (c3.get("reason") or "") +
                                   " Reviewer 2 (Cosmos): " + (c2.get("reason") or "no fault found"),
                                   status="disputed", type="close_call", label="Close call", weight=round(w, 2), agree="1/2 models (fault disputed)"))
        elif c2["fault"] == "driver" and err and RESP_W.get(resp, 0.25) > 0:
            w = sev * RESP_W.get(resp, 0.25) * DISPUTED_W
            raw += w
            n_disputed += 1
            events.append(dict(base, status="disputed", type=err, label=LABELS[err], weight=round(w, 2), agree="1/2 (Cosmos only)",
                               nemotron=c3.get("reason", "")))
        else:
            handled += 1
            events.append(dict(base, reason=c3.get("reason") or base["reason"], status="handled_well", type="handled", label="Handled well", weight=0.0,
                               agree="2/2 models" if (c3 and c3.get("fault") in ("none", "other")) else "1/2"))
    safe_frac = t["safe"] / t["lead"] if t["lead"] else 1.0
    follow_term = FOLLOW_W * (1 - safe_frac)
    R = (raw / (minutes / 10) + follow_term) / ctx
    credit = min(8.0, 1.0 * handled)
    score = max(1.0, min(100.0, 100 * (1 - math.exp(-R / K)) - credit))
    events.sort(key=lambda e: ({"driver_at_fault": 0, "disputed": 1, "handled_well": 2}[e["status"]], -e["weight"], -e["sev"]))
    for e in events[10:]:
        e.pop("track", None)
        e.pop("caption", None)
    out["trips"].append({"id": name, "minutes": round(minutes, 1), "score": round(score, 1), "risk_rate": round(R, 2),
                         "density": round(density, 2), "ctx_factor": round(ctx, 2), "safe_follow_pct": round(100 * safe_frac, 1),
                         "follow_term": round(follow_term, 2), "event_rate": round(raw / (minutes / 10), 2),
                         "credit_pts": round(credit, 1), "flagged": flagged, "reviewed": reviewed, "driver_events": n_driver,
                         "handled_well": handled, "disputed": n_disputed, "counts": counts, "events": events})
    print(name, "score", round(score, 1), "R", round(R, 2), "driver", n_driver, "handled", handled, "flagged", flagged,
          {k: v for k, v in counts.items() if v}, "safe%", round(100 * safe_frac, 1), "event_rate", round(raw / (minutes / 10), 2), "follow", round(follow_term, 2))

json.dump(out, open(os.path.join(DATA, "results.json"), "w"), separators=(",", ":"))
print("results.json bytes", os.path.getsize(os.path.join(DATA, "results.json")))
