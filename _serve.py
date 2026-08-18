"""Static server with caching disabled, for local development.

Python's default http.server lets the browser heuristically cache ES modules,
which makes edits appear not to take effect. This sends no-store on everything.

It also accepts POST /_shot, which writes the request body to _shot/<name>.png.
Canvas figures cannot always be captured by screenshotting the page — an
embedded browser view may decline to composite a canvas that was painted
outside a real animation frame — so a page can hand its own toBlob() output
straight to disk instead.
"""
import functools
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
SHOT_DIR = os.path.join(ROOT, "_shot")


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/_shot":
            self.send_error(404)
            return
        name = re.sub(r"[^A-Za-z0-9_.-]", "", self.headers.get("X-Shot-Name", "shot")) or "shot"
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(SHOT_DIR, exist_ok=True)
        with open(os.path.join(SHOT_DIR, name + ".png"), "wb") as fh:
            fh.write(body)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    # serve the directory this file lives in, whatever the caller's cwd is
    handler = functools.partial(NoCacheHandler, directory=ROOT)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
