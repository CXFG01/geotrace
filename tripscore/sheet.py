import sys, os, subprocess, urllib.parse
import vsslib
src, out = sys.argv[1], sys.argv[2]
tok = vsslib.token()
url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + tok
mp4 = "/tmp/_seg.mp4"
open(mp4, "wb").write(vsslib.get("/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") , raw=True) if False else __import__("urllib.request").request.urlopen(url, timeout=120).read())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", "fps=1,scale=640:-1,tile=5x1", "-frames:v", "1", out], check=True)
print(os.path.getsize(out))
