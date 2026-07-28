#!/usr/bin/env python3
"""
Booth relay for the two-lane race demo.

Why this exists
---------------
The race page fires two requests at once — one through the Fermyon front door,
one straight at vLLM — and needs to read the `X-Cache` header off the first.
A browser cannot do that directly:

  * neither the Fermyon proxy nor vLLM sends CORS headers, so a page served
    from file:// or any other origin is blocked from reading the response;
  * even with `Access-Control-Allow-Origin`, a custom header like `X-Cache`
    stays invisible unless the server also sends
    `Access-Control-Expose-Headers`.

So the booth laptop runs this: a tiny stdlib-only relay that holds the
port-forwards (or the tunnel), proxies each lane, times it server-side, and
hands the page a clean JSON result with permissive CORS. It also serves
`booth-race.html` itself, so the demo runs from http:// rather than file://.

No third-party dependencies — Python 3.11 standard library only.

Usage
-----
    # Terminal 1 and 2 — reach the cluster
    kubectl port-forward -n inference svc/fermyon-svc 8082:8082
    kubectl port-forward -n inference svc/vllm-svc    8000:8000

    # Terminal 3 — the relay
    python3 demo/relay.py

    # Then open http://localhost:8099

Run `python3 demo/relay.py --replay-only` to serve the page with no cluster at
all; it falls back to replaying measured data from
`phases/phase2-prefix-cache/results/phase2_cache_benchmark.json`.

Environment overrides
---------------------
    FERMYON_URL         default http://localhost:8082
    VLLM_URL            default http://localhost:8000
    MODEL_NAME          default mistralai/Mistral-7B-Instruct-v0.2
    MAX_TOKENS          default 64
    RELAY_PORT          default 8099
    KEEPALIVE_SECONDS   default 60   (0 disables)
    UPSTREAM_TIMEOUT    default 30
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FERMYON_URL = os.environ.get("FERMYON_URL", "http://localhost:8082").rstrip("/")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
RELAY_PORT = int(os.environ.get("RELAY_PORT", "8099"))
KEEPALIVE_SECONDS = int(os.environ.get("KEEPALIVE_SECONDS", "60"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "30"))

HERE = Path(__file__).resolve().parent
PAGE_PATH = HERE / "booth-race.html"

REPLAY_ONLY = False


# ---------------------------------------------------------------------------
# Upstream call
# ---------------------------------------------------------------------------

def call_upstream(base_url: str, payload: dict) -> dict:
    """POST a chat completion and time it. Never raises — errors come back as data."""
    url = f"{base_url}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
            raw = resp.read()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            # X-Cache is only set by the Fermyon proxy; vLLM leaves it absent.
            x_cache = resp.headers.get("X-Cache", "") or ""
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return {
                "ok": True,
                "status": resp.status,
                "latency_ms": round(elapsed_ms, 2),
                "x_cache": x_cache.upper(),
                "completion": _extract_completion(parsed),
                "usage": (parsed or {}).get("usage"),
                "url": url,
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": round(elapsed_ms, 2),
            "error": f"HTTP {exc.code} from {url}",
            "url": url,
        }
    except Exception as exc:  # timeouts, connection refused, DNS, tunnel drop
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "status": 0,
            "latency_ms": round(elapsed_ms, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "url": url,
        }


def _extract_completion(parsed) -> str:
    if not isinstance(parsed, dict):
        return ""
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def build_payload(prompt: str, nonce: str) -> dict:
    """
    Compose the chat request.

    The nonce is embedded in the message content so Act 1 is a guaranteed cache
    miss without needing FLUSHDB on Valkey — the same cold-cache method used by
    benchmark/bench_cache.py. Act 2 replays the identical payload, nonce and
    all, which is what turns it into a hit. The page discloses the nonce in its
    receipts panel; it is cache-busting, not sleight of hand.
    """
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": f"{prompt}\n\n[booth-run:{nonce}]"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }


# ---------------------------------------------------------------------------
# Keepalive — the 8,173 ms cold-start spike is measured and real; don't let a
# booth visitor be the request that pays it.
# ---------------------------------------------------------------------------

def keepalive_loop() -> None:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    while True:
        time.sleep(KEEPALIVE_SECONDS)
        result = call_upstream(VLLM_URL, payload)
        if not result["ok"]:
            print(f"[keepalive] vLLM unreachable: {result.get('error')}", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class RelayHandler(BaseHTTPRequestHandler):
    server_version = "ZeroWasteBoothRelay/1.0"

    def log_message(self, fmt, *args):  # quieter console at the booth
        if self.path.startswith("/race"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ------------------------------------------------------------

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    # -- verbs --------------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html", "/booth-race.html"):
            self._serve_page()
        elif self.path.startswith("/health"):
            self._json(
                {
                    "ok": True,
                    "live": not REPLAY_ONLY,
                    "model": MODEL_NAME,
                    "fermyon_url": FERMYON_URL,
                    "vllm_url": VLLM_URL,
                    "max_tokens": MAX_TOKENS,
                }
            )
        else:
            self._json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        if REPLAY_ONLY:
            self._json({"ok": False, "error": "relay running in --replay-only mode"}, status=503)
            return

        if self.path.startswith("/race/cached"):
            base = FERMYON_URL
        elif self.path.startswith("/race/direct"):
            base = VLLM_URL
        else:
            self._json({"ok": False, "error": "not found"}, status=404)
            return

        req = self._read_json()
        prompt = (req.get("prompt") or "").strip()
        nonce = (req.get("nonce") or "").strip()
        if not prompt or not nonce:
            self._json({"ok": False, "error": "prompt and nonce are required"}, status=400)
            return

        payload = build_payload(prompt, nonce)
        result = call_upstream(base, payload)
        result["request"] = payload
        self._json(result)

    def _serve_page(self):
        try:
            body = PAGE_PATH.read_bytes()
        except OSError:
            self._json({"ok": False, "error": f"cannot read {PAGE_PATH}"}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------

def main() -> int:
    global REPLAY_ONLY

    parser = argparse.ArgumentParser(description="Booth relay for the two-lane race demo.")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="serve the page without contacting the cluster (page falls back to measured data)",
    )
    parser.add_argument("--port", type=int, default=RELAY_PORT)
    args = parser.parse_args()

    REPLAY_ONLY = args.replay_only

    if not PAGE_PATH.exists():
        print(f"error: {PAGE_PATH} not found", file=sys.stderr)
        return 1

    if not REPLAY_ONLY and KEEPALIVE_SECONDS > 0:
        threading.Thread(target=keepalive_loop, daemon=True).start()
        print(f"keepalive: pinging vLLM every {KEEPALIVE_SECONDS}s to avoid cold start")

    mode = "REPLAY-ONLY" if REPLAY_ONLY else "LIVE"
    print(f"\n  Zero-Waste booth relay — {mode}")
    print(f"  page     http://localhost:{args.port}")
    if not REPLAY_ONLY:
        print(f"  fermyon  {FERMYON_URL}")
        print(f"  vllm     {VLLM_URL}")
        print(f"  model    {MODEL_NAME}")
    print("\n  Ctrl-C to stop.\n")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), RelayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
