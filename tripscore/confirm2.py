"""Step 3b: fault-aware second review of Cosmos-confirmed events. Resumable."""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import confirm as c1  # reuses model id, auth, video helper

DATA = os.path.expanduser("~/trip-risk-data")
OUT = os.path.join(DATA, "confirm2.json")

PROMPT = (
    "You are an insurance claims analyst reviewing 5 seconds of forward dashcam video. The camera vehicle is 'the insured driver'. "
    "Judge ONLY the insured driver's behaviour. Pedestrians or cyclists lawfully crossing or riding where they are allowed, "
    "and traffic that slows normally, are NOT the driver's fault. If the driver slows or stops in good time, that is GOOD driving. "
    "Fault is 'driver' only if the insured driver clearly caused the hazard: speeding for conditions, following too closely, "
    "late or abrupt braking, not yielding when required, running a signal, unsafe lane or turn. Fault is 'other' if another road user "
    "created the hazard (jaywalking, cutting in, door opening, sudden stop ahead). Fault is 'none' if there is no real hazard. "
    "Severity of the driver's error: 0 none, 1 minor (slightly late reaction), 2 moderate (abrupt braking or swerve needed), "
    "3 severe (collision narrowly avoided). Estimate the closest gap in metres if visible. "
    'Reply ONLY JSON: {"fault":"driver|other|none","driver_error":"none|following_too_close|late_braking|failed_to_yield|'
    'signal_violation|unsafe_maneuver|speeding","severity":0-3,"driver_response":"good|adequate|poor","gap_m":number|null,'
    '"reason":"max 25 words"}'
)


def ask2(src):
    body = {"model": c1.MODEL, "max_tokens": 250, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + c1.video_b64(src)}}]}]}
    import urllib.request
    req = urllib.request.Request(c1.GPU + "/v1/chat/completions", data=json.dumps(body).encode(), headers=c1.AUTH)
    txt = json.load(urllib.request.urlopen(req, timeout=240))["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        j = {}
    j["raw"] = txt[:300]
    return j


if __name__ == "__main__":
    first = json.load(open(os.path.join(DATA, "confirm.json")))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [s for s, v in first.items() if v.get("risky") and s not in done]
    print("second-pass reviews", len(todo))

    def work(s):
        try:
            return s, ask2(s)
        except Exception as e:  # noqa: BLE001
            return s, {"error": str(e)[:200]}

    with ThreadPoolExecutor(4) as ex:
        for s, r in ex.map(work, todo):
            done[s] = r
            json.dump(done, open(OUT, "w"))
            print(s.split("/")[-1][-34:], r.get("fault"), r.get("driver_error"), r.get("severity"), r.get("driver_response"), r.get("gap_m"), (r.get("reason") or r.get("error") or "")[:80])
