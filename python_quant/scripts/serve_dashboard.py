#!/usr/bin/env python
"""Serve the Nexus-LOB combined desk: console styling + live L2 overlay.

    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --ring /tmp/nexus_slots.bin
    PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --shm nex_aapl

Default ``--synthetic`` drives a seeded mid random-walk (a view dict matching the
frozen ``BOOK_STATE_DTYPE`` shape) so the page works without C++ shm and visibly
moves. Latency samples are the real measured wall-time of each snapshot render.
An unavailable selected ring never falls back to synthetic data. The execution
lab runs a separate seeded OrderBookEnv episode, not executions from the book feed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.book_state import DEPTH
from nexus_quant.dashboard import (
    SnapshotHub,
    latest_from_file_ring,
    read_shm_ring_latest,
    serve,
)
from nexus_quant.dashboard_execution import run_execution_demo
from nexus_quant.risk import compute_var_cvar

_SEED = 0xC0FFEE


def _synthetic_view(rng: np.random.Generator, mid: int, seq: int,
                    volume: int, last: tuple[int, int, int]) -> dict:
    """A view-shaped dict (same shape as StubOrderBook.view()) around a mid."""
    half = int(rng.choice([1, 1, 1, 2, 2, 3]))  # spread mostly 2, sometimes wider
    bp = np.zeros(DEPTH, dtype=np.int64)
    bs = np.zeros(DEPTH, dtype=np.uint64)
    bc = np.zeros(DEPTH, dtype=np.uint32)
    ap = np.zeros(DEPTH, dtype=np.int64)
    az = np.zeros(DEPTH, dtype=np.uint64)
    ac = np.zeros(DEPTH, dtype=np.uint32)
    for i in range(DEPTH):
        bp[i] = mid - half - i
        bs[i] = int(60 + rng.integers(0, 260) + (DEPTH - i) * 4)
        bc[i] = int(rng.integers(1, 9))
        ap[i] = mid + half + i
        az[i] = int(60 + rng.integers(0, 260) + (DEPTH - i) * 4)
        ac[i] = int(rng.integers(1, 9))
    px, sz, side = last
    return {
        "seq": seq, "ts_ns": seq, "cum_volume": volume,
        "last_trade_px": px, "last_trade_sz": sz, "last_trade_side": side,
        "version": seq,
        "bid_px": bp, "bid_sz": bs, "bid_ct": bc,
        "ask_px": ap, "ask_sz": az, "ask_ct": ac,
    }


def make_poller(hub: SnapshotHub, *, ring: str = "", shm: str = "") -> Callable[[], None]:
    if ring and shm:
        raise ValueError("select only one book feed")
    rng = np.random.default_rng(_SEED)
    state = {"mid": 15_000, "seq": 0, "vol": 0, "last": (0, 0, 2), "n": 0}

    def poll() -> None:
        t0 = time.perf_counter_ns()
        if ring or shm:
            source = "file-ring" if ring else "shm-ring"
            try:
                v = latest_from_file_ring(Path(ring)) if ring else read_shm_ring_latest(shm)
            except (OSError, ValueError):
                v = None
            if v is None:
                hub.set_unavailable(source)
            else:
                hub.push(v, source=source)
            return
        # ----- synthetic walk -------------------------------------------------
        st = state
        st["n"] += 1
        st["seq"] += 1
        mid_new = int(np.clip(st["mid"] + int(rng.integers(-4, 5)), 1_000, 99_000))
        st["mid"] = mid_new
        # a ~40% chance a trade prints each tick
        if rng.random() < 0.4:
            side = int(rng.integers(0, 2))                     # 0=Bid aggressor, 1=Ask
            px = mid_new + (0 if rng.random() < 0.6 else (-1 if side == 0 else 1))
            sz = int(rng.integers(2, 60))
            st["vol"] += sz
            st["last"] = (px, sz, side)
        v = _synthetic_view(rng, mid_new, st["seq"], st["vol"], st["last"])
        hub.push(v, source="synthetic", feed_latency_ns=time.perf_counter_ns() - t0)
        if st["n"] % 8 == 0:
            r = compute_var_cvar(n_paths=256, steps=32, prefer_engine=False)
            hub.risk = {"var": r.var, "cvar": r.cvar}

    return poll


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    feeds = ap.add_mutually_exclusive_group()
    feeds.add_argument("--synthetic", action="store_true", help="seeded synthetic book (default)")
    feeds.add_argument("--ring", type=str, default="")
    feeds.add_argument("--shm", type=str, default="")
    ap.add_argument("--no-execution-demo", action="store_true", help="disable synthetic execution runs")
    args = ap.parse_args()

    hub = SnapshotHub()
    poll = make_poller(hub, ring=args.ring, shm=args.shm)
    poll()
    httpd = serve(
        hub, args.host, args.port, poll,
        execution_runner=None if args.no_execution_demo else run_execution_demo,
    )
    print(f"Nexus-LOB desk http://{args.host}:{args.port}", flush=True)
    print("Execution lab: disabled" if args.no_execution_demo else
          "Execution lab: independent synthetic OrderBookEnv fills (not ring telemetry)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()