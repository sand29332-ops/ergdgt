"""Accessible execution UI structure and honest research references."""
from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest
from nexus_quant.baselines import FAIR_BASELINES
from nexus_quant.dashboard_execution import (
    MAX_HORIZON,
    MAX_INVENTORY,
    MAX_SEED,
    ExecutionConfig,
)

ROOT = Path(__file__).resolve().parents[2]
PAGES = (ROOT / "python_quant/nexus_quant/dashboard_page.html", ROOT / "dashboard/index.html")


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.text = []
        self.feed(source)
        self.ids = {attrs["id"]: (tag, attrs) for tag, attrs in self.elements if "id" in attrs}

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


@pytest.mark.parametrize("path", PAGES, ids=["combined", "static"])
def test_both_pages_retire_misleading_claims_and_reference_fair_study(path):
    source = path.read_text(encoding="utf-8")
    page = Page(source)
    text = " ".join(" ".join(page.text).split()).lower()
    assert "not reproduced broadly" in text
    assert "only under liquidity shocks" in text
    assert "docs/results/rl_fairness.md" in text
    assert "symmetric information" in text
    assert not re.search(r"50\.4|1\.401|2\.827|14\s*%", source)
    ids = [attrs["id"] for _, attrs in page.elements if "id" in attrs]
    assert all(count == 1 for count in Counter(ids).values())


def test_form_labels_and_bounds_match_api_configuration():
    page = Page(PAGES[0].read_text(encoding="utf-8"))
    defaults = ExecutionConfig()
    for key, low, high in (("seed", 0, MAX_SEED), ("horizon", 1, MAX_HORIZON), ("inventory", 1, MAX_INVENTORY)):
        field = "execution-" + key
        tag, attrs = page.ids[field]
        assert tag == "input"
        assert (attrs["min"], attrs["max"], attrs["step"]) == (str(low), str(high), "1")
        assert attrs["value"] == str(getattr(defaults, key))
        assert "required" in attrs
        assert any(tag == "label" and a.get("for") == field for tag, a in page.elements)
    assert [a["value"] for tag, a in page.elements if tag == "option"] == list(FAIR_BASELINES)
    assert page.ids["execution-run"][1]["type"] == "submit"
    assert page.ids["execution-controls"][0] == "fieldset"


def test_no_telemetry_is_default_and_loading_and_errors_are_announced():
    source = PAGES[0].read_text(encoding="utf-8")
    page = Page(source)
    assert "hidden" in page.ids["execution-results"][1]
    assert "hidden" not in page.ids["execution-empty"][1]
    assert page.ids["execution-status"][1]["role"] == "status"
    assert page.ids["execution-status"][1]["aria-live"] == "polite"
    assert any(tag == "noscript" for tag, _ in page.elements)
    for fragment in ("status.dataset.state='loading'", "status.dataset.state='error'", "controls.disabled=true", "controls.disabled=false"):
        assert fragment in source
    assert "method:'POST'" in source
    assert "not live shm-ring telemetry" in source
    assert "one episode is not evidence of a strategy edge" in source


def test_charts_have_accessible_titles_descriptions_and_table_alternatives():
    page = Page(PAGES[0].read_text(encoding="utf-8"))
    for field in ("execution-timeline", "execution-inventory-chart"):
        tag, attrs = page.ids[field]
        assert tag == "svg" and attrs["role"] == "img"
        assert attrs["viewbox"] == "0 0 520 240"
        labels = attrs["aria-labelledby"].split()
        assert [page.ids[label][0] for label in labels] == ["title", "desc"]
    assert page.ids["execution-fill-rows"][0] == "tbody"
    assert page.ids["execution-inventory-rows"][0] == "tbody"
    for name in ("Execution fill ledger", "Execution inventory samples"):
        assert any(a.get("aria-label") == name and a.get("tabindex") == "0" for _, a in page.elements)
    assert len([tag for tag, _ in page.elements if tag == "caption"]) >= 2
