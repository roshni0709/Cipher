"""Collector end-to-end against a local server: robots.txt honored,
JSON-LD extracted, offer + snapshot land in the DB."""
import functools
import http.server
import json
import threading

from pricematch.collector.snapshot import snapshot_urls

ROBOTS = "User-agent: *\nAllow: /\n"
PAGE = """<html><head><script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Test Earbuds X1",
 "offers":{"@type":"Offer","price":"1499","priceCurrency":"INR"}}
</script></head><body></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = ROBOTS if self.path == "/robots.txt" else PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/plain" if self.path == "/robots.txt" else "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


def test_snapshot_local_server(tmp_path):
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/product/1"
    saved = snapshot_urls([url], str(tmp_path / "c.db"), delay=0)
    server.shutdown()
    assert len(saved) == 1
    assert saved[0]["title"] == "Test Earbuds X1" and saved[0]["price"] == 1499.0

    from pricematch.db.store import Store

    store = Store(tmp_path / "c.db")
    hist = store.history(saved[0]["offer_id"])
    assert len(hist) == 1 and hist.iloc[0]["price"] == 1499.0
    store.close()
