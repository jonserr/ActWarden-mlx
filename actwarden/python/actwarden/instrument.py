"""Minimal agent instrumentation: emit ActWarden JSONL spans from any Python
agent loop, no framework required. This is the "50 lines of middleware" that
makes traces real instead of synthetic.

Usage (see actwarden/examples/toy_agent.py for a runnable example):

    with TraceEmitter("trace.jsonl", session_id="sess-42") as em:
        with em.agent("research"):
            with em.llm(model_name="m-small") as h:
                ...call your model...
                h.set(prompt_tokens=350, completion_tokens=120, cost_usd=0.004)
            with em.tool("web_search", cost_usd=0.001):
                ...call your tool...   # raising marks the span ERROR

Spans nest via a stack: the enclosing span becomes parent_span_id. Exceptions
propagate (instrumentation must never swallow agent errors) but mark the span
ERROR first.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, TextIO


class SpanHandle:
    """Mutable attribute holder so values known only after the call (token
    counts, cost, retrieval scores) can be attached before the span closes."""

    def __init__(self, attrs: dict):
        self._attrs = attrs

    def set(self, **kwargs) -> None:
        mapping = {
            "model_name": "llm.model_name",
            "prompt_tokens": "llm.token_count.prompt",
            "completion_tokens": "llm.token_count.completion",
            "cost_usd": "actwarden.cost_usd",
            "tool_name": "tool.name",
            "document_scores": "retrieval.document_scores",
        }
        for key, value in kwargs.items():
            self._attrs[mapping.get(key, key)] = value


class TraceEmitter:
    def __init__(self, path: str, session_id: str, trace_id: str | None = None):
        self._file: TextIO = open(path, "a", encoding="utf-8")
        self.session_id = session_id
        self.trace_id = trace_id or f"tr-{uuid.uuid4().hex[:12]}"
        self._parents: list[str] = []
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "TraceEmitter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    # -- spans ---------------------------------------------------------------
    @contextmanager
    def span(self, name: str, kind: str, **attrs) -> Iterator[SpanHandle]:
        self._seq += 1
        span_id = f"s{self._seq:04d}"
        parent = self._parents[-1] if self._parents else None
        handle = SpanHandle({})
        handle.set(**attrs)
        self._parents.append(span_id)
        start = time.time_ns()
        error = False
        try:
            yield handle
        except BaseException:
            error = True
            raise
        finally:
            self._parents.pop()
            self._file.write(
                json.dumps(
                    {
                        "trace_id": self.trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent,
                        "name": name,
                        "kind": kind,
                        "start_time_unix_nano": start,
                        "end_time_unix_nano": time.time_ns(),
                        "status_code": "ERROR" if error else "OK",
                        "session_id": self.session_id,
                        "attributes": handle._attrs,
                    }
                )
                + "\n"
            )
            self._file.flush()

    # -- convenience wrappers ----------------------------------------------
    def agent(self, name: str = "agent"):
        return self.span(name, "AGENT")

    def llm(self, name: str = "llm", **attrs):
        return self.span(name, "LLM", **attrs)

    def tool(self, tool_name: str, **attrs):
        return self.span("tool", "TOOL", tool_name=tool_name, **attrs)

    def retrieval(self, name: str = "retrieve", **attrs):
        return self.span(name, "RETRIEVER", **attrs)
