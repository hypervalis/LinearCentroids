#!/usr/bin/env python3
"""Serve ``docs/`` and accept page-content saves while editing locally."""

from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))

from lce.site_content import apply_page_content


class DevSiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if self.path.startswith("/api/save-page-content"):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()
            return
        super().do_OPTIONS()

    def do_POST(self) -> None:
        if self.path != "/api/save-page-content":
            self.send_error(404, "No save API on plain http.server; use scripts/dev_site.py")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
            path = apply_page_content(payload)
            body = json.dumps({"ok": True, "path": str(path.relative_to(ROOT))}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dev server for docs/ with page save API.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DevSiteHandler)
    print(f"Serving {DOCS} at http://127.0.0.1:{args.port}/index.html")
    print("Edit with ?edit=1 — Save to file writes docs/index.html via POST /api/save-page-content")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
