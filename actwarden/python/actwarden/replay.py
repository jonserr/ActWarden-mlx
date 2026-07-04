"""Replay evaluator: re-run recorded traces, collect decisions and metrics,
and verify runtime audit logs (chain integrity + decision reproducibility).

Scope note: the Python policy mirror below implements only the deterministic
rules, for offline development. The C++ engine is authoritative; the planned
`via_runtime` mode shells out to the `actwarden` CLI so replay evaluates the
real system under test rather than a reimplementation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from actwarden.features import FeaturePacker, load_spec
from actwarden.traces import Span, group_by_session, read_jsonl

# Severity order must match actwarden/runtime/include/actwarden/policy.h
ACTIONS = (
    "continue",
    "reduce_budget",
    "switch_model",
    "require_approval",
    "block_tool",
    "escalate",
)


@dataclass
class StepDecision:
    session_id: str
    span_index: int
    action: str
    reasons: list[str]
    features: list[float]
    scores: list[float] | None = None


@dataclass
class ReplayReport:
    trace_path: str
    steps: list[StepDecision] = field(default_factory=list)

    @property
    def action_counts(self) -> Counter:
        return Counter(s.action for s in self.steps)

    @property
    def flip_rate(self) -> float:
        """Fraction of consecutive steps (per session) where the action changed.
        High flip rate = an unstable governor, regardless of accuracy."""
        flips = 0
        pairs = 0
        by_session: dict[str, list[StepDecision]] = {}
        for s in self.steps:
            by_session.setdefault(s.session_id, []).append(s)
        for steps in by_session.values():
            for a, b in zip(steps, steps[1:]):
                pairs += 1
                flips += a.action != b.action
        return flips / pairs if pairs else 0.0


def rules_only_policy(features: list[float], spec: dict) -> tuple[str, list[str]]:
    """Deterministic rules R001-R004 (mirror of policy.cpp; C++ is authoritative)."""
    names = {f["name"]: f["index"] for f in spec["features"]}
    reasons: list[str] = []
    action = "continue"

    def propose(a: str, reason: str) -> None:
        nonlocal action
        reasons.append(reason)
        if ACTIONS.index(a) > ACTIONS.index(action):
            action = a

    if features[names["cost_total_norm"]] >= 1.0:
        propose("escalate", "R001:budget_exhausted")
    if features[names["depth_max_norm"]] >= 1.0:
        propose("require_approval", "R003:depth_limit")
    if features[names["error_run_norm"]] >= 0.5:
        propose("require_approval", "R004:error_streak")
    # R002 (tool pair denylist) needs raw spans, not features; runtime-only for now.
    return action, reasons


def replay_trace(
    trace_path: str | Path,
    spec_path: str | Path | None = None,
    scorer: Callable[[Sequence[float]], list[float]] | None = None,
) -> ReplayReport:
    """Stream each session span-by-span, packing and deciding at every step,
    exactly as the runtime would."""
    spec = load_spec(spec_path) if spec_path else load_spec()
    packer = FeaturePacker(spec)
    report = ReplayReport(trace_path=str(trace_path))

    for session_id, spans in group_by_session(read_jsonl(trace_path)).items():
        window: list[Span] = []
        for i, span in enumerate(spans):
            window.append(span)
            feats = packer.pack(window)
            action, reasons = rules_only_policy(feats, spec)
            scores = list(scorer(feats)) if scorer else None
            # TODO(slice, day 5): learned-threshold proposals + head metrics
            # (precision/recall vs labels) once labeled traces exist.
            report.steps.append(
                StepDecision(session_id, i, action, reasons, feats, scores)
            )
    return report


# --------------------------------------------------------------------------
# Audit-log verification (counterpart of actwarden/runtime/src/audit.{h,cpp})
# --------------------------------------------------------------------------

GENESIS_HASH = "cbf29ce484222325"  # fnv1a64(b"")


def fnv1a64(data: bytes) -> str:
    """FNV-1a 64-bit, lowercase hex. Must match fnv1a64_hex in audit.cpp
    (known vectors pinned in tests)."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


@dataclass
class AuditVerification:
    records: int
    chain_ok: bool
    decisions_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chain_ok and self.decisions_ok


def verify_audit_log(
    path: str | Path, spec_path: str | Path | None = None
) -> AuditVerification:
    """Replay an audit log without the original trace.

    Two independent checks per record:
    1. Chain integrity — "hash_prev" must equal fnv1a64 of the previous
       line's raw bytes (genesis: fnv1a64 of empty). Any byte-level edit to a
       line breaks verification of the next line.
    2. Decision reproducibility — deterministic rules are recomputed from the
       recorded feature snapshot and must match the recorded reasons/action.
       R002 (tool-pair denylist) needs raw spans, so its recorded severity is
       honored rather than recomputed. Records with learned scores only get
       the rule-subset check here (thresholds live in the C++ PolicyConfig).
    """
    spec = load_spec(spec_path) if spec_path else load_spec()
    problems: list[str] = []
    chain_ok = True
    decisions_ok = True

    lines = [
        ln for ln in Path(path).read_bytes().split(b"\n") if ln.strip()
    ]
    prev = GENESIS_HASH
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("hash_prev") != prev:
            chain_ok = False
            problems.append(f"record {i}: hash chain broken")
        prev = fnv1a64(line)

        if rec.get("spec_version") != spec["spec_version"]:
            decisions_ok = False
            problems.append(f"record {i}: spec_version mismatch")
            continue
        expected_action, expected_reasons = rules_only_policy(
            rec["features"], spec
        )
        codes = {r["code"] for r in rec["reasons"]}
        if any(c.startswith("R002:") for c in codes):
            if ACTIONS.index("block_tool") > ACTIONS.index(expected_action):
                expected_action = "block_tool"
        for code in expected_reasons:
            if code not in codes:
                decisions_ok = False
                problems.append(f"record {i}: missing rule reason {code}")
        if "scores" not in rec and rec["proposed_action"] != expected_action:
            decisions_ok = False
            problems.append(
                f"record {i}: proposed_action {rec['proposed_action']!r} != "
                f"recomputed {expected_action!r}"
            )
        if rec.get("shadow") and rec["action"] != "continue":
            decisions_ok = False
            problems.append(f"record {i}: shadow record with enforced action")

    return AuditVerification(len(lines), chain_ok, decisions_ok, problems)
