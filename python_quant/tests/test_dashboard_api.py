"""Exercise HTTP handlers in memory; no listening server is needed."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from nexus_quant.book_state import Side, StubOrderBook
from nexus_quant.dashboard import SnapshotHub, make_handler


class MemorySocket:
    def __init__(self, request: bytes) -> None:
        self.request = request
        self.response = bytearray()

    def makefile(self, *args):
        return io.BytesIO(self.request)

    def sendall(self, data):
        self.response.extend(data)

    def settimeout(self, timeout):
        self.timeout = timeout


def request(handler, *, path="/api/execution", method="POST", body=b"{}", headers=None):
    if headers is None:
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    raw = f"{method} {path} HTTP/1.0\r\n"
    raw += "".join(f"{key}: {value}\r\n" for key, value in headers)
    sock = MemorySocket(raw.encode("ascii") + b"\r\n" + body)
    handler(sock, ("127.0.0.1", 1234), SimpleNamespace())
    head, response_body = bytes(sock.response).split(b"\r\n\r\n", 1)
    code = int(head.split(b" ")[1])
    return code, head, response_body


def test_execution_api_runs_requested_seed_without_polling_or_mutating_book():
    book = StubOrderBook()
    book.add(Side.Bid, 100, 10)
    book.add(Side.Ask, 102, 8)
    hub = SnapshotHub()
    hub.push(book.view(), source="shm-ring")
    before = hub.as_json()
    polled = []
    handler = make_handler(hub, poll=lambda: polled.append(True))
    body = json.dumps({"seed": 12, "horizon": 4, "strategy": "adaptive_pov"}).encode()
    code, head, data = request(handler, body=body)
    assert code == 200
    assert b"Cache-Control: no-store" in head
    payload = json.loads(data)
    assert payload["source"] == "synthetic-execution"
    assert payload["config"]["seed"] == 12
    assert payload["config"]["horizon"] == 4
    assert payload["fills"]
    assert hub.as_json() == before
    assert polled == []
    assert request(handler, body=body)[2] == data


@pytest.mark.parametrize("body", [
    b"not json", b"\xff", b"null", b"[]", b'{"seed":true}',
    b'{"horizon":81}', b'{"inventory":10001}', b'{"strategy":"ppo"}',
    b'{"seed":NaN}', b'{"unknown":1}',
])
def test_api_rejects_invalid_parameters_before_running(body):
    calls = []
    handler = make_handler(SnapshotHub(), execution_runner=lambda cfg: calls.append(cfg))
    code, _, data = request(handler, body=body)
    assert code == 400
    assert json.loads(data)["ok"] is False
    assert calls == []


@pytest.mark.parametrize("headers,body,code", [
    ([("Content-Type", "text/plain"), ("Content-Length", "2")], b"{}", 415),
    ([("Content-Type", "application/json")], b"{}", 411),
    ([("Content-Type", "application/json"), ("Content-Length", "-1")], b"{}", 400),
    ([("Content-Type", "application/json"), ("Content-Length", "bad")], b"{}", 400),
    ([("Content-Type", "application/json"), ("Content-Length", "2"), ("Content-Length", "2")], b"{}", 400),
    ([("Content-Type", "application/json"), ("Content-Length", "2"), ("Transfer-Encoding", "chunked")], b"{}", 400),
    ([("Content-Type", "application/json"), ("Content-Length", "1025")], b"", 413),
    ([("Content-Type", "application/json"), ("Content-Length", "9" * 5000)], b"", 413),
    ([("Content-Type", "application/json"), ("Content-Length", "0")], b"", 400),
    ([("Content-Type", "application/json"), ("Content-Length", "10")], b"{}", 400),
])
def test_api_request_framing_errors(headers, body, code):
    calls = []
    handler = make_handler(SnapshotHub(), execution_runner=lambda cfg: calls.append(cfg))
    actual, _, data = request(handler, body=body, headers=headers)
    assert actual == code
    assert json.loads(data)["error"]
    assert calls == []


def test_request_body_timeout_returns_json_error(monkeypatch):
    class TimedBody(io.BytesIO):
        def read(self, size=-1):
            raise TimeoutError("body did not arrive")

    monkeypatch.setattr(MemorySocket, "makefile", lambda sock, *args: TimedBody(sock.request))
    code, _, data = request(make_handler(SnapshotHub()))
    assert code == 408
    assert json.loads(data)["error"] == "Execution request timed out"


def test_disabled_execution_is_honestly_unavailable():
    handler = make_handler(SnapshotHub(), execution_runner=None)
    code, _, data = request(handler)
    assert code == 503
    assert "unavailable" in json.loads(data)["error"]
    assert "fills" not in json.loads(data)


def test_execution_errors_do_not_leak_details_and_release_run_lock():
    calls = []

    def runner(config):
        calls.append(config)
        if len(calls) == 1:
            raise RuntimeError("sensitive exception details")
        return {"ok": True}

    handler = make_handler(SnapshotHub(), execution_runner=runner)
    code, _, data = request(handler)
    assert code == 500
    assert b"sensitive" not in data
    assert "No telemetry" in json.loads(data)["error"]
    assert request(handler)[0] == 200


def test_concurrent_runs_are_rejected_without_touching_feed():
    nested = []

    def runner(config):
        nested.append(request(handler))
        return {"ok": True}

    handler = make_handler(SnapshotHub(), execution_runner=runner)
    assert request(handler)[0] == 200
    assert nested[0][0] == 409
    assert "running" in json.loads(nested[0][2])["error"]


def test_book_api_keeps_no_snapshot_state_and_handles_poll_errors():
    handler = make_handler(SnapshotHub())
    code, _, data = request(handler, method="GET", path="/api/state")
    assert code == 200
    assert json.loads(data) == {"ok": False, "source": "none"}

    def fail():
        raise OSError("private path")

    handler = make_handler(SnapshotHub(), poll=fail)
    code, _, data = request(handler, method="GET", path="/api/state")
    assert code == 503
    assert b"private path" not in data
    assert "unavailable" in json.loads(data)["error"]


@pytest.mark.parametrize("method,path,code", [
    ("GET", "/api/execution", 405), ("GET", "/api/state-extra", 404),
    ("POST", "/api/state", 404), ("POST", "/api/execution-extra", 404),
])
def test_routes_are_exact_and_execution_requires_post(method, path, code):
    assert request(make_handler(SnapshotHub()), method=method, path=path)[0] == code


def test_root_and_index_serve_page_without_polling():
    polled = []
    handler = make_handler(SnapshotHub(), poll=lambda: polled.append(True), page=b"test page")
    for path in ("/", "/index.html?test=1"):
        code, _, data = request(handler, method="GET", path=path)
        assert code == 200
        assert data == b"test page"
    assert polled == []
