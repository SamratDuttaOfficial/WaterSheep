import io
import json
import unittest
from unittest.mock import patch

from watersheep import cli


class FakeServer:
    handler = None

    def __init__(self, address, handler):
        FakeServer.handler = handler

    def serve_forever(self):
        raise KeyboardInterrupt


class FakeWaterSheep:
    name = "test-watersheep"

    def __init__(self):
        self.calls = 0

    def ask(self, request):
        self.calls += 1
        return {"ok": True, "request": request}


class BrokenWriter:
    def write(self, data):
        raise BrokenPipeError(32, "Broken pipe")


class CliServerTests(unittest.TestCase):
    def _handler(self, ws, writer):
        FakeServer.handler = None
        with patch("http.server.ThreadingHTTPServer", FakeServer), patch("builtins.print"):
            cli.serve(ws, 8766)

        handler = object.__new__(FakeServer.handler)
        body = json.dumps({"state": "hello"}).encode()
        handler.path = "/v1/decisions"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = writer
        handler.response_codes = []
        handler.send_response = handler.response_codes.append
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        return handler

    def test_client_disconnect_after_inference_is_not_reclassified_as_400(self):
        ws = FakeWaterSheep()
        handler = self._handler(ws, BrokenWriter())

        handler.do_POST()

        self.assertEqual(ws.calls, 1)
        self.assertEqual(handler.response_codes, [200])

    def test_normal_response_still_returns_json(self):
        ws = FakeWaterSheep()
        writer = io.BytesIO()
        handler = self._handler(ws, writer)

        handler.do_POST()

        self.assertEqual(handler.response_codes, [200])
        self.assertEqual(
            json.loads(writer.getvalue()),
            {"ok": True, "request": {"state": "hello"}},
        )


if __name__ == "__main__":
    unittest.main()
