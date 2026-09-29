"""Ask a WaterSheep model for decisions.

  %(prog)s --model OWNER/NAME --question "Which team?" --options billing,shipping --state "..."
  %(prog)s --model OWNER/NAME --request request.json
  %(prog)s --model OWNER/NAME --file requests.jsonl
  %(prog)s --model OWNER/NAME --serve
"""
from __future__ import annotations
import argparse
import json
import sys
import time


def serve(ws, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/decisions", "/v1/systemone"):
                return self._send(404, {"error": "POST /v1/decisions"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                with lock:
                    self._send(200, ws.ask(req))
            except Exception as e:
                self._send(400, {"error": str(e)})

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("serving %s on http://127.0.0.1:%d/v1/decisions  (Ctrl+C to stop)" % (ws.name, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None, pin: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", "-r", help="JSON request file ('-' = stdin)")
    ap.add_argument("--serve", nargs="?", const=8766, type=int, metavar="PORT",
                    help="serve an HTTP endpoint")
    ap.add_argument("--question", "-q")
    ap.add_argument("--state", "-s", default="")
    ap.add_argument("--options", "-o", default="", help="comma-separated; omit for yes/no")
    ap.add_argument("--type", choices=["binary", "noul", "choice", "score", "multi"])
    ap.add_argument("--file", help="JSONL file of requests")
    ap.add_argument("--model", help="Hugging Face repo id (owner/name[@revision]), model folder or "
                                    "export name; default: newest local export")
    ap.add_argument("--device", help="cuda or cpu")
    a = ap.parse_args(argv)
    if pin:
        from .core.paths import pin_caches
        pin_caches()
    from .infer import WaterSheep
    try:
        ws = WaterSheep.load(a.model, a.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    if a.serve:
        serve(ws, a.serve)
        return 0
    if a.request:
        req = json.load(sys.stdin if a.request == "-" else open(a.request, encoding="utf-8"))
        t0 = time.time()
        res = ws.ask(req)
        res["usage"]["ms"] = round((time.time() - t0) * 1000, 1)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if a.file:
        items = [json.loads(line) for line in open(a.file, encoding="utf-8") if line.strip()]
        t0 = time.time()
        for it in items:
            r = ws.ask(it) if "questions" in it else ws.decide_many([it])[0]
            print(json.dumps(r, ensure_ascii=False))
        print("%d requests in %.2fs" % (len(items), time.time() - t0), file=sys.stderr)
        return 0
    if not a.question:
        ap.error("--question, --request, --file or --serve is required")
    opts = [o.strip() for o in a.options.split(",") if o.strip()] or None
    t0 = time.time()
    r = ws.decide(a.state, a.question, opts, a.type)
    r["ms"] = round((time.time() - t0) * 1000, 1)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(pin=True))
