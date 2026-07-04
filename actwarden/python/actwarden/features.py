"""Reference feature packer.

Implements actwarden/feature_spec/v1.json exactly. This is the source of truth
against which the C++ packer (actwarden/runtime/src/feature_packer.cpp) is
verified via golden vectors. Any behavior change here requires a spec version
bump and regenerated goldens.

Stdlib-only by design so parity tests run anywhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from actwarden.traces import Span, sort_key

_REPO_SPEC = Path(__file__).resolve().parents[3] / "feature_spec" / "v1.json"


def load_spec(path: str | Path = _REPO_SPEC) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    names = [f["name"] for f in spec["features"]]
    indices = [f["index"] for f in spec["features"]]
    if indices != list(range(len(indices))):
        raise ValueError("feature spec indices must be contiguous from 0")
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names in spec")
    return spec


def feature_names(spec: dict) -> list[str]:
    return [f["name"] for f in spec["features"]]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _entropy_bits(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _p95(values: Sequence[float]) -> float:
    """Element at index ceil(0.95*n)-1 of ascending sort. No interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, math.ceil(0.95 * len(s)) - 1)
    return s[idx]


class FeaturePacker:
    """Packs a chronological window of spans into the v1 feature vector."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.norm = spec["normalization"]
        self.max_spans = int(spec["window"]["max_spans"])
        self.dim = len(spec["features"])

    def pack(self, session_spans: Sequence[Span]) -> list[float]:
        """session_spans: all spans of one session in canonical order.

        Only the last window.max_spans spans are used.
        """
        spans = sorted(session_spans, key=sort_key)[-self.max_spans :]
        n = len(spans)
        out = [0.0] * self.dim
        if n == 0:
            return out

        norm = self.norm
        tools = [s for s in spans if s.kind == "TOOL"]
        llms = [s for s in spans if s.kind == "LLM"]
        retrievers = [s for s in spans if s.kind == "RETRIEVER"]

        window_start = min(s.start_time_unix_nano for s in spans)
        window_end = max(s.end_time_unix_nano for s in spans)
        window_seconds = max(
            (window_end - window_start) / 1e9, norm["min_window_seconds"]
        )

        # 0: span_count_norm
        out[0] = _clamp01(n / self.max_spans)
        # 1: tool_call_ratio
        out[1] = _clamp01(len(tools) / n)
        # 2: llm_call_ratio
        out[2] = _clamp01(len(llms) / n)
        # 3: error_rate
        n_err = sum(1 for s in spans if s.is_error)
        out[3] = _clamp01(n_err / n)
        # 4: error_run_norm
        run = longest = 0
        for s in spans:
            run = run + 1 if s.is_error else 0
            longest = max(longest, run)
        out[4] = _clamp01(longest / norm["max_error_run"])
        # 5: tool_ngram_repeat_ratio
        ng = int(norm["loop_ngram"])
        tool_seq = [s.tool_name or "" for s in tools]
        if len(tool_seq) >= ng:
            grams = [tuple(tool_seq[i : i + ng]) for i in range(len(tool_seq) - ng + 1)]
            out[5] = _clamp01(1.0 - len(set(grams)) / len(grams))
        # 6: tool_entropy_norm
        if tools:
            counts: dict[str, int] = {}
            for name in tool_seq:
                counts[name] = counts.get(name, 0) + 1
            out[6] = _clamp01(
                _entropy_bits(list(counts.values()))
                / math.log2(norm["entropy_ref_tools"])
            )
        # 7: token_velocity_norm
        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in spans)
        out[7] = _clamp01(total_tokens / window_seconds / norm["token_rate_per_s"])
        # 8: cost_total_norm
        total_cost = sum(s.cost_usd for s in spans)
        out[8] = _clamp01(total_cost / norm["budget_usd"])
        # 9: cost_slope_norm
        if n < 2:
            out[9] = 0.5
        else:
            half = n // 2
            first = sum(s.cost_usd for s in spans[:half])
            second = sum(s.cost_usd for s in spans[half:])
            out[9] = _clamp01(((second - first) / norm["budget_usd"] + 1.0) / 2.0)
        # 10: latency_p95_norm
        out[10] = _clamp01(_p95([s.duration_ms for s in spans]) / norm["latency_ref_ms"])
        # 11: retrieval_call_ratio
        out[11] = _clamp01(len(retrievers) / n)
        # 12: retrieval_low_conf_ratio
        if retrievers:
            low = sum(
                1
                for s in retrievers
                if not s.document_scores
                or max(s.document_scores) < norm["low_confidence_threshold"]
            )
            out[12] = _clamp01(low / len(retrievers))
        # 13: depth_max_norm  /  14: fanout_max_norm
        by_id = {s.span_id: s for s in spans}
        depth_cache: dict[str, int] = {}

        def depth(s: Span) -> int:
            if s.span_id in depth_cache:
                return depth_cache[s.span_id]
            d = 1
            parent = s.parent_span_id
            seen = {s.span_id}
            while parent in by_id and parent not in seen:
                d += 1
                seen.add(parent)
                parent = by_id[parent].parent_span_id
            depth_cache[s.span_id] = d
            return d

        out[13] = _clamp01(max(depth(s) for s in spans) / norm["max_depth"])
        children: dict[str, int] = {}
        for s in spans:
            if s.parent_span_id in by_id:
                children[s.parent_span_id] = children.get(s.parent_span_id, 0) + 1
        out[14] = _clamp01((max(children.values()) if children else 0) / norm["max_fanout"])
        # 15: stagnation_norm
        duration = window_end - window_start
        if duration > 0:
            seen_keys: set[str] = set()
            last_novel_start = window_start
            for s in spans:
                key = s.kind + "\x1f" + (s.tool_name or "" if s.kind == "TOOL" else s.name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    last_novel_start = s.start_time_unix_nano
            out[15] = _clamp01((window_end - last_novel_start) / duration)

        return [float(v) for v in out]
