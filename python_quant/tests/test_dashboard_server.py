"""Feed selection must never substitute synthetic snapshots for a missing ring."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from nexus_quant.book_state import Side, StubOrderBook
from nexus_quant.dashboard import SnapshotHub, record_from_view


@pytest.fixture
def server_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve_dashboard.py"
    spec = importlib.util.spec_from_file_location("serve_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_file_ring_stays_unavailable_until_a_slot_arrives(server_script, tmp_path):
    path = tmp_path / "missing-ring.bin"
    hub = SnapshotHub()
    poll = server_script.make_poller(hub, ring=str(path))
    poll()
    assert hub.as_json() == {"ok": False, "source": "file-ring"}
    assert hub.history == []
    book = StubOrderBook()
    book.add(Side.Bid, 100, 10)
    path.write_bytes(record_from_view(book.view()).tobytes())
    poll()
    assert hub.as_json()["bid_px"][0] == 100
    path.unlink()
    poll()
    assert hub.as_json() == {"ok": False, "source": "file-ring"}


def test_unavailable_shm_is_not_replaced_by_synthetic_data(server_script, monkeypatch):
    monkeypatch.setattr(server_script, "read_shm_ring_latest", lambda _: None)
    hub = SnapshotHub()
    poll = server_script.make_poller(hub, shm="missing-ring")
    poll()
    assert hub.as_json() == {"ok": False, "source": "shm-ring"}
    assert not hub.history


def test_feed_read_failure_is_unavailable(server_script, monkeypatch):
    def fail(_):
        raise OSError("cannot read ring")

    monkeypatch.setattr(server_script, "read_shm_ring_latest", fail)
    hub = SnapshotHub()
    server_script.make_poller(hub, shm="ring")()
    assert hub.as_json() == {"ok": False, "source": "shm-ring"}


def test_synthetic_book_source_is_explicit(server_script):
    hub = SnapshotHub()
    server_script.make_poller(hub)()
    assert hub.as_json()["ok"] is True
    assert hub.as_json()["source"] == "synthetic"


def test_ambiguous_feed_selection_is_rejected(server_script):
    with pytest.raises(ValueError):
        server_script.make_poller(SnapshotHub(), ring="file", shm="ring")
