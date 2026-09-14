"""Python dashboard grain on the frozen BookStateView.

The C++ ``ShmRing`` publishes 448-byte slots. This module:

* decodes a ``BOOK_STATE_DTYPE`` record (file, bytes, or in-process view dict);
* serves a stdlib HTTP page with L2 ladder, BBO, spread, last trade, optional VaR.

It does not replace Person A's ring producer. Attach a raw dump of slots
(``capacity * 448`` bytes, little-endian record layout matching NumPy
``align=True`` dtype) or feed live ``view()`` dicts from replay/env.

OS shared-memory attach is best-effort (POSIX ``/dev/shm/<name>``); if the
segment is absent the dashboard runs off the in-process hub or a file ring.

POSIX-only: ``read_shm_ring_latest``/``try_attach_posix_shm`` attach the
Linux ``/dev/shm`` segment by path. They do not attach the C++ ``ShmRing``'s
Windows shared-memory mapping — on Windows use the file-ring path
(``latest_from_file_ring``) or an in-process ``view()`` hub instead.
"""
from __future__ import annotations

import json
import logging
import struct
from collections import deque
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .book_state import BOOK_STATE_DTYPE, DEPTH, empty_state
from .dashboard_execution import ExecutionConfig, run_execution_demo
from .replay import check_integrity, spread_ticks

View = Mapping[str, Any]
logger = logging.getLogger(__name__)


def view_from_record(rec: np.ndarray) -> dict[str, Any]:
    """Turn a 0-d BOOK_STATE_DTYPE record into a view-shaped dict."""
    r = rec.reshape(())
    return {
        "seq": int(r["seq"]),
        "ts_ns": int(r["ts_ns"]),
        "cum_volume": int(r["cum_volume"]),
        "last_trade_sz": int(r["last_trade_sz"]),
        "last_trade_px": int(r["last_trade_px"]),
        "last_trade_side": int(r["last_trade_side"]),
        "version": int(r["version"]),
        "bid_px": np.asarray(r["bid_px"]).copy(),
        "bid_sz": np.asarray(r["bid_sz"]).copy(),
        "bid_ct": np.asarray(r["bid_ct"]).copy(),
        "ask_px": np.asarray(r["ask_px"]).copy(),
        "ask_sz": np.asarray(r["ask_sz"]).copy(),
        "ask_ct": np.asarray(r["ask_ct"]).copy(),
    }


def record_from_view(view: View) -> np.ndarray:
    rec = empty_state()
    rec["seq"] = int(view.get("seq", 0))
    rec["ts_ns"] = int(view.get("ts_ns", 0))
    rec["cum_volume"] = int(view.get("cum_volume", 0))
    rec["last_trade_sz"] = int(view.get("last_trade_sz", 0))
    rec["last_trade_px"] = int(view.get("last_trade_px", 0))
    rec["last_trade_side"] = int(view.get("last_trade_side", 2))
    rec["version"] = int(view.get("version", 0))
    for name in ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz", "ask_ct"):
        arr = np.asarray(view[name])
        rec[name][: min(DEPTH, arr.shape[0])] = arr[:DEPTH]
    return rec


def decode_slot(buf: bytes | memoryview) -> dict[str, Any]:
    raw = bytes(buf)
    if len(raw) < BOOK_STATE_DTYPE.itemsize:
        raise ValueError(f"slot too short: {len(raw)} < {BOOK_STATE_DTYPE.itemsize}")
    rec = np.frombuffer(raw[: BOOK_STATE_DTYPE.itemsize], dtype=BOOK_STATE_DTYPE, count=1)[0]
    return view_from_record(rec)


# Rolling history / latency window lengths.
_HISTORY_MAX = 200
_LATENCY_SAMPLES_MAX = 500

# Log-spaced latency histogram bins, in nanoseconds. The last bin is open-ended.
_LATENCY_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 10_000),          # <10 µs
    (10_000, 25_000),     # 10–25 µs
    (25_000, 50_000),     # 25–50 µs
    (50_000, 100_000),    # 50–100 µs
    (100_000, 250_000),   # 100–250 µs
    (250_000, 500_000),   # 250–500 µs
    (500_000, 1_000_000), # 0.5–1 ms
    (1_000_000, 2_500_000),  # 1–2.5 ms
    (2_500_000, 5_000_000),  # 2.5–5 ms
    (5_000_000, 10_000_000), # 5–10 ms
    (10_000_000, None),      # >10 ms
)

_LATENCY_LABELS = (
    "<10µs", "10-25µs", "25-50µs", "50-100µs", "100-250µs",
    "250-500µs", "0.5-1ms", "1-2.5ms", "2.5-5ms", "5-10ms", ">10ms",
)


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


class SnapshotHub:
    """Latest book + optional risk numbers + rolling history for the HTTP layer.

    ``push()`` records a dedup-by-seq rolling sample (for sparklines) and, when told,
    a latency sample (measured feed/fill latency) into a log-spaced histogram.
    """

    def __init__(self) -> None:
        self.view: dict[str, Any] | None = None
        self.risk: dict[str, float] = {}
        self.source = "none"
        self.history: list[dict[str, Any]] = []
        self.latency: list[int] = [0] * len(_LATENCY_BINS)
        self.latency_n = 0
        self.latency_kind = "feed"  # "feed" (transport) or "fill" (execution)
        self._lat_samples: deque[float] = deque(maxlen=_LATENCY_SAMPLES_MAX)
        self._last_seq = -1

    def set_unavailable(self, source: str) -> None:
        self.view = None
        self.source = source

    # -- latency ------------------------------------------------------------
    def record_latency(self, ns: float, *, kind: str = "feed") -> None:
        """Accumulate one latency sample into the histogram + percentile store."""
        self.latency_kind = kind
        ns = float(ns)
        self.latency_n += 1
        idx = len(_LATENCY_BINS) - 1
        for i, (lo, hi) in enumerate(_LATENCY_BINS):
            if hi is None or ns < hi:
                idx = i
                break
        self.latency[idx] += 1
        self._lat_samples.append(ns)

    def _latency_json(self) -> dict[str, Any]:
        ordered = sorted(self._lat_samples)
        return {
            "kind": self.latency_kind,
            "labels": list(_LATENCY_LABELS),
            "counts": self.latency,
            "n": self.latency_n,
            "p50_ns": _percentile(ordered, 0.5),
            "p95_ns": _percentile(ordered, 0.95),
        }

    # -- snapshots ----------------------------------------------------------
    def push(
        self,
        view: View,
        *,
        source: str = "live",
        feed_latency_ns: float | None = None,
    ) -> None:
        self.view = {
            k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
            for k, v in view.items()
        }
        self.source = source
        if feed_latency_ns is not None:
            self.record_latency(feed_latency_ns, kind="feed")
        seq = int(self.view.get("seq", 0))
        if seq <= self._last_seq:
            return  # already-seen seq re-polled — don't duplicate the history sample
        self._last_seq = seq
        bid_px = np.asarray(self.view["bid_px"])
        ask_px = np.asarray(self.view["ask_px"])
        b0 = int(bid_px[0]) if len(bid_px) and bid_px[0] else 0
        a0 = int(ask_px[0]) if len(ask_px) and ask_px[0] else 0
        self.history.append(
            {
                "seq": seq,
                "spread": (a0 - b0) if b0 and a0 else None,
                "mid": (b0 + a0) / 2 if b0 and a0 else None,
                "bid0": b0,
                "ask0": a0,
                "trade_px": int(self.view.get("last_trade_px", 0) or 0),
                "trade_sz": int(self.view.get("last_trade_sz", 0) or 0),
                "trade_side": int(self.view.get("last_trade_side", 2)),
                "cum_volume": int(self.view.get("cum_volume", 0) or 0),
            }
        )
        if len(self.history) > _HISTORY_MAX:
            self.history = self.history[-_HISTORY_MAX:]

    def as_json(self) -> dict[str, Any]:
        v = self.view
        if v is None:
            return {"ok": False, "source": self.source}
        spr = spread_ticks(v)
        issues = [i.code for i in check_integrity(v)]
        bid_px = np.asarray(v["bid_px"])
        ask_px = np.asarray(v["ask_px"])
        b0 = int(bid_px[0]) if len(bid_px) and bid_px[0] else 0
        a0 = int(ask_px[0]) if len(ask_px) and ask_px[0] else 0
        return {
            "ok": True,
            "source": self.source,
            "seq": int(v.get("seq", 0)),
            "ts_ns": int(v.get("ts_ns", 0)),
            "bid_px": [int(x) for x in bid_px[:DEPTH]],
            "bid_sz": [int(x) for x in np.asarray(v["bid_sz"])[:DEPTH]],
            "ask_px": [int(x) for x in ask_px[:DEPTH]],
            "ask_sz": [int(x) for x in np.asarray(v["ask_sz"])[:DEPTH]],
            "spread": spr,
            "bid0": b0,
            "ask0": a0,
            "mid": (b0 + a0) / 2 if b0 and a0 else None,
            "cum_volume": int(v.get("cum_volume", 0)),
            "last_trade_px": int(v.get("last_trade_px", 0)),
            "last_trade_sz": int(v.get("last_trade_sz", 0)),
            "last_trade_side": int(v.get("last_trade_side", 2)),
            "history": self.history,
            "latency": self._latency_json(),
            "issues": issues,
            "risk": self.risk,
        }


def try_attach_posix_shm(name: str) -> memoryview | None:
    path = Path("/dev/shm") / name.lstrip("/")
    if not path.is_file():
        return None
    data = path.read_bytes()
    return memoryview(data)


def latest_from_file_ring(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    n = BOOK_STATE_DTYPE.itemsize
    if len(raw) < n:
        return None
    # last complete slot
    off = (len(raw) // n - 1) * n
    return decode_slot(raw[off : off + n])


# Person A ShmRing control block (shm_ring.hpp). Slot array starts immediately after.
_SHM_CTRL = struct.Struct("<QQQQQII")
_SHM_CTRL_N = 48


def read_shm_ring_latest(name: str) -> dict[str, Any] | None:
    """Newest published 448-byte slot from a live C++ ShmRing, or None."""
    path = Path("/dev/shm") / name.lstrip("/")
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) < _SHM_CTRL_N + BOOK_STATE_DTYPE.itemsize:
        return None
    write_seq, _read, _drop, cap, slot_bytes, state, _pad = _SHM_CTRL.unpack_from(data, 0)
    if state != 1 or cap == 0 or int(slot_bytes) != BOOK_STATE_DTYPE.itemsize:
        return None
    if write_seq == 0:
        return None
    idx = (int(write_seq) - 1) % int(cap)
    off = _SHM_CTRL_N + idx * int(slot_bytes)
    if off + int(slot_bytes) > len(data):
        return None
    return decode_slot(data[off : off + int(slot_bytes)])


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nexus-LOB desk</title>
<style>
body{font-family:ui-monospace,monospace;background:#0b0f14;color:#d6e0ea;margin:0;padding:24px}
h1{font-size:16px;letter-spacing:.08em;color:#8ab4d8}
table{border-collapse:collapse}
td,th{padding:3px 10px;text-align:right}
.bid{color:#6ee7b7}.ask{color:#fb7185}
.meta{color:#8b9bb4;margin:12px 0}
</style></head><body>
<h1>NEXUS-LOB · L2</h1>
<div class="meta" id="meta">loading…</div>
<table id="book"><thead><tr><th>bid sz</th><th>bid</th><th>ask</th><th>ask sz</th></tr></thead>
<tbody></tbody></table>
<script>
async function tick(){
  const r = await fetch('/api/state'); const s = await r.json();
  document.getElementById('meta').textContent =
    s.ok ? ('seq '+s.seq+'  spread '+(s.spread??'—')+'  vol '+s.cum_volume+'  src '+s.source)
         : ('no snapshot ('+s.source+')');
  const tb = document.querySelector('#book tbody'); tb.innerHTML='';
  if(!s.ok) return;
  for(let i=0;i<10;i++){
    const tr=document.createElement('tr');
    tr.innerHTML = `<td class="bid">${s.bid_sz[i]}</td><td class="bid">${s.bid_px[i]}</td>
                    <td class="ask">${s.ask_px[i]}</td><td class="ask">${s.ask_sz[i]}</td>`;
    tb.appendChild(tr);
  }
}
setInterval(tick, 400); tick();
</script></body></html>
"""


# Combined dashboard page: polished console + live desk, served at "/".
_PAGE_PATH = Path(__file__).with_name("dashboard_page.html")


def _load_page(page: bytes | None) -> bytes:
    if page is not None:
        return page
    try:
        return _PAGE_PATH.read_bytes()
    except OSError:
        return _PAGE.encode()  # tiny fallback ladder if the asset is missing


def make_handler(
    hub: SnapshotHub,
    poll: Callable[[], None] | None = None,
    page: bytes | None = None,
    *,
    execution_runner: Callable[[ExecutionConfig], dict[str, Any]] | None = run_execution_demo,
):
    execution_lock = Lock()
    feed_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 405:
                self.send_header("Allow", "POST")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode()
            self._respond(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"ok": False, "error": message})

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/state":
                try:
                    with feed_lock:
                        if poll is not None:
                            poll()
                        self._json(200, hub.as_json())
                except Exception:
                    logger.exception("Book feed request failed")
                    self._error(503, "Book feed unavailable; retry when the feed is restored.")
                return
            if path == "/api/execution":
                self._error(405, "Use POST to run a seeded synthetic execution demo.")
                return
            if path not in ("/", "/index.html"):
                self._error(404, "Not found")
                return
            body = getattr(self.server, "page", None)
            if body is None:
                body = _load_page(page)
            self._respond(200, body, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/execution":
                self._error(404, "Not found")
                return
            if execution_runner is None:
                self._error(503, "Synthetic execution telemetry is unavailable on this server.")
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                self._error(415, "Content-Type must be application/json")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self._error(411, "Content-Length is required")
                return
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
                self._error(400, "Invalid request framing")
                return
            try:
                length = int(lengths[0])
            except ValueError:
                self._error(413, "Execution request is too large")
                return
            if length > 1024:
                self._error(413, "Execution request is too large")
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                config = ExecutionConfig.from_payload(json.loads(raw))
            except TimeoutError:
                self._error(408, "Execution request timed out")
                return
            except (TypeError, ValueError, UnicodeError):
                self._error(400, "Invalid execution request: use a fair strategy and bounded integer seed, horizon and inventory.")
                return
            if not execution_lock.acquire(blocking=False):
                self._error(409, "Another synthetic execution is running; retry shortly.")
                return
            try:
                result = execution_runner(config)
                self._json(200, result)
            except Exception:
                logger.exception("Synthetic execution request failed")
                self._error(500, "Synthetic execution failed. No telemetry is available; retry the run.")
            finally:
                execution_lock.release()

    return Handler


def serve(
    hub: SnapshotHub,
    host: str = "127.0.0.1",
    port: int = 8765,
    poll: Callable[[], None] | None = None,
    page: bytes | None = None,
    *,
    execution_runner: Callable[[ExecutionConfig], dict[str, Any]] | None = run_execution_demo,
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(hub, poll, page, execution_runner=execution_runner)
    )
    httpd.page = _load_page(page)
    return httpd
