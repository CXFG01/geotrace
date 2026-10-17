"""GeoTrace web app (stdlib only): serves the UI, results.json and a seekable video proxy (JWT stays server-side)."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import vsslib

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = json.load(open(os.path.join(HERE, "results.json")))
ALLOWED = {s for t in RESULTS["trips"] for s in t["segs"]}
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

    def log_message(self, *a):
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
                return self._send(200, json.dumps(RESULTS), extra={"Cache-Control": "no-cache"})
            if p == "/api/video":
                return self.video(q.get("source", [""])[0])
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


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
