"""Step 1: list Toronto (pie_cam-3) segments, group by trip (set01..06) and cache YOLO sidecars."""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import vsslib

DATA = os.path.expanduser("~/trip-risk-data")
SIDE = os.path.join(DATA, "side")
os.makedirs(SIDE, exist_ok=True)

chunks = [c for c in vsslib.explore_all() if c.get("camera_id") == "pie_cam-3"]
trips = {}
for c in chunks:
    m = re.search(r"(set\d+)", c["filename"])
    trip = m.group(1) if m else "unknown"
    trips.setdefault(trip, []).append(c)

index = {}
for trip, cs in sorted(trips.items()):
    cs.sort(key=lambda c: c["chunk_index"] if c.get("chunk_index") is not None else int(re.search(r"chunk_(\d+)", c["filename"]).group(1)))
    segs = []
    t0 = 0.0
    for c in cs:
        for s in sorted(c["timeline"], key=lambda s: s["segment_number"]):
            segs.append({
                "trip": trip,
                "source": s["source"],
                "chunk": c["filename"],
                "original_video": c["original_video"],
                "t": round(t0 + s["segment_start_sec"], 2),
                "dur": round(s["segment_end_sec"] - s["segment_start_sec"], 2),
                "caption": s.get("reasoning_content") or "",
                "classes": s.get("object_classes"),
                "counts": s.get("object_counts"),
            })
        t0 += c["chunk_duration_sec"]
    index[trip] = segs
    print(trip, "chunks", len(cs), "segments", len(segs), "minutes", round(t0 / 60, 1))

json.dump(index, open(os.path.join(DATA, "index.json"), "w"))


def fname(src):
    return os.path.join(SIDE, os.path.basename(src) + ".json")


def fetch(seg):
    p = fname(seg["source"])
    if os.path.exists(p) and os.path.getsize(p) > 10:
        return 1
    d = vsslib.detections(seg["source"])
    if d is None:
        return 0
    slim = {"fps": d.get("fps"), "shape": d.get("video_shape"),
            "frames": [[f["time_sec"], [[x["label"], x["confidence"], x["bbox"]] for x in f["detections"]]] for f in d["frames"]]}
    json.dump(slim, open(p, "w"), separators=(",", ":"))
    return 1


allsegs = [s for v in index.values() for s in v]
only = sys.argv[1] if len(sys.argv) > 1 else None
if only:
    allsegs = [s for s in allsegs if s["trip"] == only]
with ThreadPoolExecutor(8) as ex:
    ok = sum(ex.map(fetch, allsegs))
print("sidecars cached", ok, "of", len(allsegs))
