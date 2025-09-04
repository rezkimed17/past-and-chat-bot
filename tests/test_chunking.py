from __future__ import annotations

from app.ingest import _split_into_docs


def test_split_preserves_metadata_lines():
    text = "A\nB\nC\n\nD\nE\nF\n"
    docs = _split_into_docs(text, title="Test")
    assert len(docs) >= 1
    d0 = docs[0]
    assert d0.metadata["title"] == "Test"
    assert d0.metadata["start_line"] >= 1
    assert d0.metadata["end_line"] >= d0.metadata["start_line"]

