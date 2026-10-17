"""VM: cut excerpts of Trip 02 into one 1280x720 30fps mp4 (no audio)."""
import json, os, subprocess, sys, tempfile, urllib.parse, urllib.request
sys.path.insert(0, os.path.expanduser("~/vast-builders-challenge/tools/trip-risk"))
import vsslib
DATA = os.path.expanduser("~/trip-risk-data")
idx = json.load(open(f"{DATA}/index.json"))["set02"]
EXC = [(85, 7), (350, 7), (425, 8), (492, 6), (510, 8), (592, 6), (694, 6), (816, 8)]
out_dir = tempfile.mkdtemp()
parts = []
for k, (t0, dur) in enumerate(EXC):
    i0 = t0 // 5
    i1 = (t0 + dur - 1) // 5
    files = []
    for i in range(i0, i1 + 1):
        src = idx[i]["source"]
        url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + vsslib.token()
        p = os.path.join(out_dir, f"s{k}_{i}.mp4")
        open(p, "wb").write(urllib.request.urlopen(url, timeout=120).read())
        files.append(p)
    lst = os.path.join(out_dir, f"l{k}.txt")
    open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
    o = os.path.join(out_dir, f"e{k}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-ss", str(t0 - i0 * 5), "-t", str(dur),
                    "-vf", "scale=1280:720,fps=30", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", o], check=True)
    parts.append(o)
    print("excerpt", k, t0, dur, flush=True)
lst = os.path.join(out_dir, "all.txt")
open(lst, "w").write("".join(f"file '{f}'\n" for f in parts))
final = os.path.expanduser("~/trip-risk-data/demo_footage.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
print("done", os.path.getsize(final))
