"""Tests for the audit-log chain, shadow mode, and agent instrumentation.

CLI-dependent tests activate when the runtime binary exists (same discovery
as test_feature_parity: ACTWARDEN_CLI env var, then runtime/build/actwarden,
then PATH).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from actwarden.instrument import TraceEmitter
from actwarden.replay import (
    GENESIS_HASH,
    fnv1a64,
    replay_trace,
    verify_audit_log,
)
from actwarden.traces import group_by_session, read_jsonl

ACTWARDEN_DIR = Path(__file__).resolve().parents[2]
SPEC = ACTWARDEN_DIR / "feature_spec" / "v1.json"
TRACE = ACTWARDEN_DIR / "fixtures" / "traces" / "runaway_loop.jsonl"


def find_cli() -> str | None:
    if "ACTWARDEN_CLI" in os.environ:
        return os.environ["ACTWARDEN_CLI"]
    default = ACTWARDEN_DIR / "runtime" / "build" / "actwarden"
    if default.exists():
        return str(default)
    return shutil.which("actwarden")


needs_cli = pytest.mark.skipif(
    find_cli() is None,
    reason="C++ runtime not built (set ACTWARDEN_CLI or build runtime/)",
)


# --- hash primitive: pinned cross-language vectors --------------------------


def test_fnv1a64_known_vectors():
    assert fnv1a64(b"") == "cbf29ce484222325" == GENESIS_HASH
    assert fnv1a64(b"a") == "af63dc4c8601ec8c"
    assert fnv1a64(b"foobar") == "85944171f73967e8"


# --- instrumentation ---------------------------------------------------------


def test_instrument_roundtrip(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceEmitter(str(path), session_id="s1") as em:
        with em.agent("root"):
            with em.llm(model_name="m") as h:
                h.set(prompt_tokens=10, completion_tokens=5, cost_usd=0.01)
            with pytest.raises(RuntimeError):
                with em.tool("boom"):
                    raise RuntimeError("tool failed")
            with em.retrieval() as h:
                h.set(document_scores=[0.1, 0.2])

    spans = read_jsonl(path)
    assert [s.kind for s in spans] == ["LLM", "TOOL", "RETRIEVER", "AGENT"]
    by_kind = {s.kind: s for s in spans}
    root = by_kind["AGENT"]
    assert all(
        s.parent_span_id == root.span_id for s in spans if s is not root
    )
    assert by_kind["TOOL"].is_error
    assert by_kind["LLM"].prompt_tokens == 10
    assert by_kind["RETRIEVER"].document_scores == (0.1, 0.2)
    assert len(group_by_session(spans)) == 1
    # and the whole thing replays
    assert len(replay_trace(path, SPEC).steps) == 4


# --- shadow mode + audit chain via the real CLI ------------------------------


@needs_cli
def test_shadow_mode_and_audit_chain(tmp_path):
    audit = tmp_path / "audit.jsonl"
    out = subprocess.run(
        [
            find_cli(),
            "--trace", str(TRACE),
            "--spec", str(SPEC),
            "--shadow",
            "--audit-log", str(audit),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    records = [json.loads(l) for l in out.splitlines() if l.strip()]
    assert len(records) == 15
    # shadow: effective action is always continue; the engine's view survives
    assert all(r["shadow"] for r in records)
    assert all(r["action"] == "continue" for r in records)
    assert all("proposed_action" in r for r in records)

    v = verify_audit_log(audit, SPEC)
    assert v.records == 15
    assert v.ok, v.problems


@needs_cli
def test_audit_chain_detects_tampering(tmp_path):
    audit = tmp_path / "audit.jsonl"
    subprocess.run(
        [
            find_cli(),
            "--trace", str(TRACE),
            "--spec", str(SPEC),
            "--audit-log", str(audit),
        ],
        capture_output=True,
        check=True,
    )
    assert verify_audit_log(audit, SPEC).ok

    lines = audit.read_bytes().split(b"\n")
    lines[2] = lines[2].replace(b'"span_index":2', b'"span_index":9')
    audit.write_bytes(b"\n".join(lines))
    v = verify_audit_log(audit, SPEC)
    assert not v.chain_ok
    assert any("hash chain broken" in p for p in v.problems)


@needs_cli
def test_audit_chain_resumes_across_runs(tmp_path):
    audit = tmp_path / "audit.jsonl"
    for _ in range(2):
        subprocess.run(
            [
                find_cli(),
                "--trace", str(TRACE),
                "--spec", str(SPEC),
                "--audit-log", str(audit),
            ],
            capture_output=True,
            check=True,
        )
    v = verify_audit_log(audit, SPEC)
    assert v.records == 30
    assert v.chain_ok, v.problems
