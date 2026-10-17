"""Tiny VSS client (stdlib only). Credentials come from environment variables
(source /config/*.config first). Nothing is printed or stored."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("VSS_URL") or os.environ.get("INGRESS_URL")
USER = os.environ.get("VSS_USERNAME") or os.environ.get("USERNAME")
PASS = os.environ.get("VSS_PASSWORD") or os.environ.get("PASSWORD")

_tok = {"v": None, "t": 0}


def token(force=False):
    if force or not _tok["v"] or time.time() - _tok["t"] > 1500:
        req = urllib.request.Request(
            BASE + "/api/v1/auth/login",
            data=json.dumps({"username": USER, "password": PASS}).encode(),
            headers={"Content-Type": "application/json"},
        )
        _tok["v"] = json.load(urllib.request.urlopen(req, timeout=60))["access_token"]
        _tok["t"] = time.time()
    return _tok["v"]


def get(path, retries=3, raw=False, timeout=120):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token()})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                token(True)
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1 + i)
    raise last


def post(path, body, timeout=180):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token()},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def explore_all(scope="all"):
    out, off = [], 0
    while True:
        d = get(f"/api/v1/videos/explore?scope={scope}&limit=100&offset={off}")
        out += d["chunks"]
        if len(d["chunks"]) < 100:
            return out
        off += 100


def detections(source):
    return get("/api/v1/videos/detections?source=" + urllib.parse.quote(source, safe=""))
