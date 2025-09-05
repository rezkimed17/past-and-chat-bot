from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from langchain_core.documents import Document


@dataclass
class ScoredDoc:
    doc: Document
    score: float


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\u200b", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def compute_line_spans(text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    """Compute 1-based line spans for each chunk within the original text.

    Approximates start/end line numbers by locating each chunk sequentially
    in the original text and counting newlines.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        idx = text.find(chunk, cursor)
        if idx == -1:
            # Fallback: use previous cursor as start
            idx = cursor
        start_line = text.count("\n", 0, idx) + 1
        end_line = start_line + chunk.count("\n")
        spans.append((start_line, end_line))
        cursor = idx + len(chunk)
    return spans


def format_citation(doc: Document) -> str:
    """Format a source marker like: From 'Title', lines X–Y."""
    meta = doc.metadata or {}
    title = meta.get("title") or meta.get("source") or "Notes"
    start = meta.get("start_line")
    end = meta.get("end_line")
    if start and end:
        return f"From '{title}', lines {start}–{end}"
    return f"From '{title}'"


def best_supported(docs_with_scores: Iterable[Tuple[Document, float]], threshold: float) -> List[ScoredDoc]:
    """Return docs meeting the relevance threshold with robust normalization.

    - If scores already in [0,1], use them as-is.
    - If scores in [-1,1], map via (s+1)/2.
    - Otherwise, normalize by rank (top=1.0, last=0.0).
    """
    pairs = list(docs_with_scores)
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    if all(0.0 <= s <= 1.0 for s in scores):
        normalized = [ScoredDoc(doc=d, score=s) for d, s in pairs]
    elif all(-1.0 <= s <= 1.0 for s in scores):
        normalized = [ScoredDoc(doc=d, score=(s + 1.0) / 2.0) for d, s in pairs]
    else:
        ranked = sorted(pairs, key=lambda t: t[1], reverse=True)
        n = len(ranked)
        if n == 1:
            normalized = [ScoredDoc(doc=ranked[0][0], score=1.0)]
        else:
            normalized = [
                ScoredDoc(doc=d, score=(n - i - 1) / (n - 1)) for i, (d, _) in enumerate(ranked)
            ]
    return [sd for sd in normalized if sd.score >= threshold]


def quote_snippet(doc: Document, max_chars: int = 180) -> str:
    """Return a compact quoted snippet from the document."""
    text = (doc.page_content or "").strip()
    if len(text) <= max_chars:
        return f'"{text}"'
    return f'"{text[: max_chars - 1].rstrip()}…"'


def suggest_missing_details(question: str) -> str:
    """Heuristically list up to three missing details based on the question.

    Avoids external calls; keeps output terse.
    """
    q = question.lower()
    bullets: List[str] = []
    if any(k in q for k in ["version", "firmware", "os"]):
        bullets.append("specific version or firmware level")
    if any(k in q for k in ["error", "fail", "issue", "code"]):
        bullets.append("exact error message or code")
    if any(k in q for k in ["connector", "device", "service", "api"]):
        bullets.append("which device/service and configuration")
    if "how" in q or "steps" in q:
        bullets.append("step-by-step procedure expected")
    if not bullets:
        bullets = [
            "which product/feature this refers to",
            "environment and versions",
            "exact steps, symptoms, or error details",
        ]
    return ", ".join(bullets[:3])
