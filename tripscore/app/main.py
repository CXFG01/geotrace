"""TripScore web app (stdlib only). Serves the report UI, results.json, a seekable video proxy
(JWT stays server-side) and a grounded Q&A endpoint. Credentials come from env (K8s Secret)."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import vsslib

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = json.load(open(os.path.join(HERE, "results.json")))
ALLOWED = {e["seg"] for t in RESULTS["trips"] for e in t["events"]}
PORT = int(os.environ.get("PORT", "8080"))


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path.rstrip("/") or "/"
        try:
            if p == "/health":
                return self._send(200, "ok", "text/plain")
            if p == "/":
                return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            if p == "/api/results":
                return self._send(200, json.dumps(RESULTS))
            if p == "/api/video":
                return self.video(q.get("source", [""])[0])
            if p == "/api/ask":
                return self.ask(q.get("q", [""])[0])
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # noqa: BLE001
            return self._send(500, json.dumps({"error": str(e)[:200]}))
        self._send(404, "not found", "text/plain")

    def video(self, src):
        if src not in ALLOWED:
            return self._send(403, "forbidden", "text/plain")
        url = vsslib.BASE + "/api/v1/videos/stream?source=" + urllib.parse.quote(src, safe="") + "&token=" + vsslib.token()
        req = urllib.request.Request(url)
        if self.headers.get("Range"):
            req.add_header("Range", self.headers["Range"])
        try:
            r = urllib.request.urlopen(req, timeout=60)
            code = r.status
        except urllib.error.HTTPError as e:
            r, code = e, e.code
        data = r.read()
        extra = {k: r.headers[k] for k in ("Content-Range", "Accept-Ranges") if r.headers.get(k)}
        extra.setdefault("Accept-Ranges", "bytes")
        self._send(code, data, "video/mp4", extra)

    def ask(self, question):
        if not question.strip():
            return self._send(400, json.dumps({"error": "empty question"}))
        body = {"query": question[:300], "top_k": 8, "llm_top_n": 3, "min_similarity": 0.15,
                "metadata_filters": {"camera_id": "pie_cam-3"}}
        d = vsslib.post("/api/v1/agent/search-and-answer", body)
        chunks = (d.get("evidence") or {}).get("chunks") or []
        ALLOWED.update(c["preview_source"] for c in chunks if c.get("preview_source"))
        self._send(200, json.dumps({"answer": d.get("answer", ""),
                                    "clips": [{"source": c.get("preview_source"), "start": c.get("best_match_start_sec"),
                                               "score": c.get("similarity_score")} for c in chunks[:3]]}))


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
