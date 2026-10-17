"""Step 3: confirm top candidate segments with Cosmos3-Reason (video in, JSON out). Resumable."""
import base64
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import vsslib

DATA = os.path.expanduser("~/trip-risk-data")
OUT = os.path.join(DATA, "confirm.json")
GPU = "http://166.19.38.112:8001"
AUTH = {"Authorization": "Bearer " + os.environ["GPU_BEARER_TOKEN"], "Content-Type": "application/json"}

PROMPT = (
    "You review 5 seconds of forward dashcam video for an auto insurer. Decide if the DRIVER of the camera vehicle "
    "is in a risky situation: near miss, closing too fast on a vehicle ahead, hard braking, failing to yield to a "
    "pedestrian or cyclist, or running a signal. Ignore parked cars, people on sidewalks, and normal slow traffic. "
    'Reply with ONLY JSON: {"risky":true|false,"event":"near_miss|tailgating|hard_braking|failure_to_yield|'
    'signal_violation|none","severity":0-3,"reason":"max 25 words"}'
)


def model_id():
    req = urllib.request.Request(GPU + "/v1/models", headers=AUTH)
    return json.load(urllib.request.urlopen(req, timeout=30))["data"][0]["id"]


MODEL = model_id()


def video_b64(src):
    url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + vsslib.token()
    return base64.b64encode(urllib.request.urlopen(url, timeout=120).read()).decode()


def ask(src):
    body = {"model": MODEL, "max_tokens": 200, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64(src)}}]}]}
    req = urllib.request.Request(GPU + "/v1/chat/completions", data=json.dumps(body).encode(), headers=AUTH)
    txt = json.load(urllib.request.urlopen(req, timeout=240))["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        j = {}
    j["raw"] = txt[:300]
    return j


def candidates(per_trip):
    feats = json.load(open(os.path.join(DATA, "features.json")))
    by = {}
    for r in feats:
        if r["events"]:
            by.setdefault(r["trip"], []).append((r["events"][0]["sev"], r["source"]))
    out = []
    for trip, lst in by.items():
        lst.sort(reverse=True)
        out += [s for _, s in lst[:per_trip]]
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s in candidates(n) if s not in done]
    print("to confirm", len(todo), "model", MODEL)

    def work(s):
        try:
            return s, ask(s)
        except Exception as e:  # noqa: BLE001
            return s, {"error": str(e)[:200]}

    with ThreadPoolExecutor(4) as ex:
        for s, r in ex.map(work, todo):
            done[s] = r
            json.dump(done, open(OUT, "w"))
            print(s.split("/")[-1][-34:], r.get("risky"), r.get("event"), r.get("severity"), (r.get("reason") or r.get("error") or r.get("raw", ""))[:90])
