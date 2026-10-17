"""GeoTrace pass: one Cosmos3-Reason call per 5s segment -> legible text, landmarks, tram tracks, motion + maneuver. Resumable."""
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/vast-builders-challenge/tools/trip-risk"))
import confirm as c1  # noqa: E402  (model id, auth, video_b64)

DATA = os.path.expanduser("~/trip-risk-data")
OUT = os.path.join(DATA, "geo_pass.json")

PROMPT = (
    "You are analysing 5 seconds of forward-facing dashcam video from a car driving in a city. "
    "Report ONLY what is clearly visible; never guess text you cannot read. Reply ONLY with JSON: "
    '{"street_names":["names from street-name signs only, e.g. Bathurst St"],'
    '"businesses":["shop, bank, restaurant or building names that are legible"],'
    '"landmarks":["distinctive fixed landmarks: church, bridge, tower, park, school, station, large mural, etc."],'
    '"tram_tracks":true|false,'
    '"moving":"stopped|slow|normal|fast",'
    '"maneuver":"left_turn|right_turn|straight|u_turn|none",'
    '"turn_progress":"before|during|completed|none",'
    '"at_intersection":true|false,'
    '"road":"one_way|two_way|unknown","lanes":number|null}. '
    "maneuver is what the camera vehicle itself does in these 5 seconds (left_turn/right_turn only if its heading clearly changes)."
)


def ask(src):
    body = {"model": c1.MODEL, "max_tokens": 450, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + c1.video_b64(src)}}]}]}
    req = urllib.request.Request(c1.GPU + "/v1/chat/completions", data=json.dumps(body).encode(), headers=c1.AUTH)
    txt = json.load(urllib.request.urlopen(req, timeout=240))["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        j = {}
    if not j:
        j = {"parse_error": txt[:200]}
    return j


if __name__ == "__main__":
    idx = json.load(open(os.path.join(DATA, "index.json")))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    segs = [s for t in sorted(idx) for s in idx[t]]
    todo = [s for s in segs if s["source"] not in done or "error" in done[s["source"]]]
    print("segments", len(segs), "to do", len(todo), flush=True)

    def work(s):
        try:
            return s["source"], ask(s["source"])
        except Exception as e:  # noqa: BLE001
            return s["source"], {"error": str(e)[:160]}

    n = 0
    with ThreadPoolExecutor(6) as ex:
        for src, r in ex.map(work, todo):
            done[src] = r
            n += 1
            if n % 20 == 0:
                json.dump(done, open(OUT, "w"))
                print("done", n, flush=True)
    json.dump(done, open(OUT, "w"))
    print("finished", len(done), flush=True)
