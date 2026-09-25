# -----------------------------------------------------------------------------
# File:     tests/test_ingest_collect.py
# Purpose:  Unit tests for the law collector (HTTP mocked, no network).
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from datetime import date

import httpx2
import pytest

from lab.ingest.collect import (
    USER_AGENT,
    collect,
    decode_html,
    load_raw,
    make_client,
    sha256_of,
)
from lab.ingest.sources import LawSource

SOURCE = LawSource(id="x", name="Lei X", number="Lei 1/2000", url="https://planalto.test/x.htm")
HTML = "<p>Art. 1º Esta Lei dispõe sobre proteção.</p>".encode("cp1252")
TODAY = date(2026, 9, 25)


def client_for(status: int = 200, body: bytes = HTML) -> tuple[httpx2.Client, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(status, content=body)

    client = httpx2.Client(
        transport=httpx2.MockTransport(handler), headers={"User-Agent": USER_AGENT}
    )
    return client, seen


def test_collect_writes_raw_bytes_and_meta(tmp_path):
    client, seen = client_for()
    doc = collect(SOURCE, tmp_path, client=client, today=TODAY)

    assert (tmp_path / "x.html").read_bytes() == HTML
    meta = json.loads((tmp_path / "x.meta.json").read_text("utf-8"))
    assert meta == {
        "law": "x",
        "url": "https://planalto.test/x.htm",
        "sha256": sha256_of(HTML),
        "collected_at": "2026-09-25",
    }
    assert doc.sha256.startswith("sha256:") and len(doc.sha256) == 7 + 64
    assert seen[0].headers["User-Agent"].startswith("Mozilla/5.0")


def test_collect_http_error_writes_nothing(tmp_path):
    client, _ = client_for(status=503)
    with pytest.raises(httpx2.HTTPStatusError):
        collect(SOURCE, tmp_path / "raw", client=client, today=TODAY)
    assert not (tmp_path / "raw").exists()


def test_load_raw_roundtrip_decodes_cp1252(tmp_path):
    client, _ = client_for()
    collect(SOURCE, tmp_path, client=client, today=TODAY)
    text, doc = load_raw(tmp_path, "x")
    assert "proteção" in text
    assert doc.collected_at == TODAY


def test_load_raw_rejects_modified_html(tmp_path):
    client, _ = client_for()
    collect(SOURCE, tmp_path, client=client, today=TODAY)
    (tmp_path / "x.html").write_bytes(HTML + b"<p>editado</p>")
    with pytest.raises(ValueError, match="sha256"):
        load_raw(tmp_path, "x")


def test_decode_html_prefers_utf8():
    assert decode_html("ção".encode()) == "ção"
    assert decode_html("ção".encode("cp1252")) == "ção"


def test_make_client_identifies_project():
    with make_client() as client:
        assert "llm-lab-legislacao" in client.headers["User-Agent"]
