"""Step 3c: independent fault-attribution review with a stronger vision model on W&B Serverless Inference.
Frames are sampled from the segment (ffmpeg) and sent as images. Usage: confirm3.py <model-id> <out-name> [limit]"""
import base64
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import vsslib
from confirm2 import PROMPT

DATA = os.path.expanduser("~/trip-risk-data")
MODEL = sys.argv[1]
OUT = os.path.join(DATA, sys.argv[2] + ".json")
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 999
KEY = os.environ["WANDB_API_KEY"]
URL = "https://api.inference.wandb.ai/v1/chat/completions"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
if os.environ.get("WANDB_TEAM") and os.environ.get("WANDB_PROJECT"):
    HDR["OpenAI-Project"] = os.environ["WANDB_TEAM"] + "/" + os.environ["WANDB_PROJECT"]

PROMPT3 = PROMPT.replace("5 seconds of forward dashcam video", "5 seconds of forward dashcam video, shown as a contact sheet of 8 frames in time order (left to right, top row then bottom row, about 0.6s apart)")


def frames(src):
    url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + vsslib.token()
    with tempfile.TemporaryDirectory() as d:
        mp4 = os.path.join(d, "s.mp4")
        open(mp4, "wb").write(urllib.request.urlopen(url, timeout=120).read())
        sheet = os.path.join(d, "sheet.jpg")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", "fps=1.6,scale=640:-1,tile=4x2", "-frames:v", "1", "-q:v", "3", sheet], check=True)
        return [base64.b64encode(open(sheet, "rb").read()).decode()]


def ask(src, tokens=4000):
    content = [{"type": "text", "text": PROMPT3}]
    for b in frames(src):
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}})
    body = {"model": MODEL, "max_tokens": tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR)
    msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    txt = msg.get("content") or ""
    m = re.search(r"\{[^{}]*\}", txt.split("</think>")[-1], re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        j = {}
    j["raw"] = txt[-300:]
    if "fault" not in j and tokens < 12000:
        return ask(src, 12000)
    return j


if __name__ == "__main__":
    first = json.load(open(os.path.join(DATA, "confirm.json")))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s, v in first.items() if v.get("risky") and s not in done][:LIMIT]
    print("model", MODEL, "reviews", len(todo))

    def work(s):
        try:
            return s, ask(s)
        except urllib.error.HTTPError as e:
            return s, {"error": "%s %s" % (e.code, e.read().decode()[:160])}
        except Exception as e:  # noqa: BLE001
            return s, {"error": str(e)[:160]}

    with ThreadPoolExecutor(3) as ex:
        for s, r in ex.map(work, todo):
            done[s] = r
            json.dump(done, open(OUT, "w"))
            print(s.split("/")[-1][-30:], r.get("fault"), r.get("driver_error"), r.get("severity"), r.get("driver_response"), (r.get("reason") or r.get("error") or r.get("raw", ""))[:90])
