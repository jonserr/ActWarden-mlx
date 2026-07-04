"""Span schema and JSONL trace ingestion (reference implementation).

Wire format: one JSON object per line, OpenTelemetry-flavored field names with
OpenInference semantic attributes. See docs/architecture/overview.md ("Data
model") and actwarden/fixtures/traces/ for examples. The C++ mirror lives in
actwarden/runtime/{include/actwarden/span.h, src/ingest.cpp}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

VALID_KINDS = ("LLM", "TOOL", "RETRIEVER", "AGENT", "CHAIN", "OTHER")


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str  # one of VALID_KINDS
    start_time_unix_nano: int
    end_time_unix_nano: int
    status_code: str  # "OK" | "ERROR"
    session_id: str
    # Typed views over the semantic attributes the governor cares about.
    tool_name: str | None = None
    model_name: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    document_scores: tuple[float, ...] = field(default_factory=tuple)

    @property
    def duration_ms(self) -> float:
        return (self.end_time_unix_nano - self.start_time_unix_nano) / 1e6

    @property
    def is_error(self) -> bool:
        return self.status_code == "ERROR"

    @classmethod
    def from_json_dict(cls, d: dict) -> "Span":
        attrs = d.get("attributes", {}) or {}
        kind = str(
            d.get("kind") or attrs.get("openinference.span.kind") or "OTHER"
        ).upper()
        if kind not in VALID_KINDS:
            kind = "OTHER"
        return cls(
            trace_id=str(d["trace_id"]),
            span_id=str(d["span_id"]),
            parent_span_id=d.get("parent_span_id") or None,
            name=str(d.get("name", "")),
            kind=kind,
            start_time_unix_nano=int(d["start_time_unix_nano"]),
            end_time_unix_nano=int(d["end_time_unix_nano"]),
            status_code=str(d.get("status_code", "OK")).upper(),
            session_id=str(d.get("session_id", d["trace_id"])),
            tool_name=attrs.get("tool.name"),
            model_name=attrs.get("llm.model_name"),
            prompt_tokens=int(attrs.get("llm.token_count.prompt", 0)),
            completion_tokens=int(attrs.get("llm.token_count.completion", 0)),
            cost_usd=float(attrs.get("actwarden.cost_usd", 0.0)),
            document_scores=tuple(
                float(s) for s in attrs.get("retrieval.document_scores", [])
            ),
        )


def read_jsonl(path: str | Path) -> list[Span]:
    """Read spans from a JSONL file. Raises on malformed required fields."""
    spans: list[Span] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(Span.from_json_dict(json.loads(line)))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: malformed span: {e}") from e
    return spans


def sort_key(span: Span) -> tuple[int, str]:
    """Canonical window ordering (must match feature_spec window.ordering)."""
    return (span.start_time_unix_nano, span.span_id)


def group_by_session(spans: Iterable[Span]) -> dict[str, list[Span]]:
    """Group spans by session_id, each group in canonical window order."""
    sessions: dict[str, list[Span]] = {}
    for s in spans:
        sessions.setdefault(s.session_id, []).append(s)
    for group in sessions.values():
        group.sort(key=sort_key)
    return sessions
