"""VM side: for each trip, fetch every 5 s segment and dump 40 low-res gray frames (8 fps, 160x90) to one raw file per trip."""
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/vast-builders-challenge/tools/trip-risk"))
import vsslib  # noqa: E402

DATA = os.path.expanduser("~/trip-risk-data")
NF, W, H = 40, 160, 90
FS = NF * W * H


def one(src):
    url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + vsslib.token()
    with tempfile.TemporaryDirectory() as d:
        mp4 = os.path.join(d, "s.mp4")
        open(mp4, "wb").write(urllib.request.urlopen(url, timeout=120).read())
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf", f"fps=8,scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                           capture_output=True, check=True)
    b = p.stdout[:FS]
    return b + b"\x00" * (FS - len(b))


if __name__ == "__main__":
    idx = json.load(open(os.path.join(DATA, "index.json")))
    for trip in sys.argv[1:] or sorted(idx):
        segs = idx[trip]
        with ThreadPoolExecutor(8) as ex:
            blobs = list(ex.map(lambda s: one(s["source"]), segs))
        open(os.path.join(DATA, f"frames_{trip}.bin"), "wb").write(b"".join(blobs))
        print(trip, "frames written", len(blobs), flush=True)
