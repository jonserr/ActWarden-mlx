# ActWarden architecture overview

Status: skeleton / vertical slice in progress. This document describes intent and contracts; the
"Current state" section at the bottom says what actually works today.

## What ActWarden is

ActWarden is a learned runtime governor for AI agents and tool workflows. It ingests execution
traces (model calls, tool calls, retrieval spans, errors, latency, token and cost metadata),
packs sliding windows of spans into a fixed-length feature vector, scores execution risk with a
compact MLX model, and combines those scores with deterministic policy rules to return one of a
small set of control actions:

`continue | require_approval | reduce_budget | block_tool | switch_model | escalate`

Every decision carries structured reasons (rule IDs, threshold crossings, contributing scores) so
control actions are auditable and explainable.

## The core boundary: Python trains, C++ serves

```
   TRAINING PLANE (Python)                      RUNTIME PLANE (C++)
 ┌──────────────────────────┐              ┌─────────────────────────────┐
 │ trace synthesis / replay │              │ span ingestion (JSONL, OTel-│
 │ labeling                 │              │ flavored; OTLP receiver     │
 │ reference feature packer │   contract   │ later)                      │
 │ model training (mlx.nn)  │◄────────────►│ feature packer (must match  │
 │ offline eval / replay    │  feature_    │ reference exactly)          │
 │ export .mlxfn            │  spec/v1.json│ scorer: mx::import_function │
 └────────────┬─────────────┘              │ policy engine (rules first) │
              │ model.mlxfn                │ → Decision + reasons        │
              └───────────────────────────►└─────────────────────────────┘
```

Two artifacts cross the boundary, and nothing else:

1. `actwarden/feature_spec/v1.json` — the canonical feature specification. Both packers implement
   it; golden vectors in `actwarden/fixtures/golden/` gate parity. Train/serve skew is the classic
   failure mode of systems like this, so the spec is versioned data, not code, and parity is a CI
   test, not a convention.
2. `model.mlxfn` — an exported MLX computation graph with weights baked in as constants
   (`mx.export_function` in Python, `mx::import_function` in C++; see `examples/export/` upstream).
   No Python, no pickle, no ONNX shim at serve time.

## Data model

Ingestion accepts OpenTelemetry-flavored spans, one JSON object per line (JSONL). Fields follow
OTel naming (`trace_id`, `span_id`, `parent_span_id`, `start_time_unix_nano`, `end_time_unix_nano`,
`status_code`) plus a `session_id` grouping key. Semantic attributes use OpenInference conventions
where they exist: `openinference.span.kind` (LLM / TOOL / RETRIEVER / AGENT / CHAIN),
`llm.token_count.{prompt,completion}`, `llm.model_name`, `tool.name`,
`retrieval.document_scores`, plus the ActWarden extension `actwarden.cost_usd`.

A real OTLP/HTTP receiver is deliberately out of scope for the slice; the JSONL reader is the same
code path a receiver would feed, so the adapter is additive later.

## Feature packing

Per session, a ring buffer holds the last N spans (N=64 in v1). On each new span the window is
packed into a 16-dim float32 vector covering loopiness (n-gram repetition, tool entropy), cost and
token dynamics (totals, slope, velocity), reliability (error rate, error runs), retrieval
confidence, structure (depth, fanout), and stagnation. Exact definitions live in the spec file —
they are deliberately boring, deterministic, and cheap (no allocation-heavy stats at runtime).

## Scoring model

A compact MLP (16 → 64 → 64 → 4 sigmoid heads: `loop`, `cost`, `tool_misuse`, `dead_end`) trained
in Python with `mlx.nn`. Small on purpose: scoring must sit in the tool-call hot path, so the
budget is sub-millisecond on CPU. Sequence models (tiny GRU / attention over raw span sequences)
are a phase-2 experiment, only if the MLP measurably underperforms — and a candidate for an MLX
custom C++/Metal extension only if profiling justifies one.

Retraining produces a new `.mlxfn`; the runtime never mutates weights.

## Policy engine

Rules first, model second — the model can only add caution, never remove it:

1. Deterministic rules (hard budget exhausted, denylisted tool pairs, depth limit, error streak)
   are evaluated in order; each hit is recorded as a reason with a stable rule ID (R001…).
2. Learned scores map through per-head thresholds to proposed actions.
3. Final action = highest-severity proposal, severity order:
   `escalate > block_tool > require_approval > switch_model > reduce_budget > continue`.

Threshold hysteresis (an action must persist for K windows before downgrading) is planned to
prevent decision flapping — the most common practical failure of runtime governors.

When no model is present the engine runs rules-only. That is the permanent baseline: replay
evaluation must show the model adds lift over rules-only, or the model does not ship. The
runtime can also be *built* rules-only (`-DACTWARDEN_ENABLE_MLX=OFF`): the packer, policy
engine, CLI, and parity tests all work without MLX, and loading `--model` in such a build is a
hard error rather than a silent fallback.

## Shadow mode and the audit log

`--shadow` keeps the whole pipeline live but forces the effective action to `continue`,
reporting the engine's verdict as `proposed_action`. Observe-before-enforce is the intended
first deployment mode.

`--audit-log <path>` appends every decision — with its full feature snapshot and spec version —
to a JSONL hash chain: each record carries `hash_prev` = FNV-1a 64 of the previous line's raw
bytes (genesis: hash of empty). The chain resumes across process restarts. Any byte-level edit
breaks verification of the following record; the newest line is only protected once the next
record lands (the standard limitation of an unanchored chain — periodic external anchoring is
future work). `actwarden.replay.verify_audit_log` replays a log without the original trace:
chain integrity plus decision reproducibility (deterministic rules recomputed from the recorded
features must reproduce the recorded reasons and, for unscored records, the proposed action).

## Replay evaluation

`actwarden.replay` re-runs recorded traces through the reference packer and (optionally) a scorer,
producing per-window decisions and metrics: per-head precision/recall against labels, decision
distribution, and flip rate. A `--via-runtime` mode that shells out to the C++ CLI (making C++ the
system under test) is the end state; the Python mirror exists only to develop against.

## Deployment targets

| Target | Backend | Role | Honesty notes |
|---|---|---|---|
| macOS arm64 (M-series) | Metal | local dev, training | primary development path |
| Linux x86_64/arm64 | CPU | CI, small deployments | MLX built from source, `MLX_BUILD_METAL=OFF` |
| Linux + NVIDIA | CUDA | server / scale-out | `MLX_BUILD_CUDA=ON`; newest backend, expect rough edges; not CI-verified here yet |

One static CLI (`actwarden`) per target; container images build MLX from this tree. No cloud
vendor assumptions anywhere.

## Failure modes and open questions

- Synthetic training data can teach the model the generator, not reality. Mitigation: replay real
  instrumented agent traces as soon as possible; keep the rules-only baseline as the floor.
- Per-agent "policy drift" detection needs behavioral baselines per agent profile — phase 2.
- Score calibration (temperature/Platt) is required before thresholds are meaningful across
  retrains.
- Windowed features forget: a slow leak over 1000 spans is invisible to a 64-span window.
  Cumulative session counters may need to join the feature vector in v2.

## Vertical slice plan (< 1 week)

1. Day 1 — synthetic trace generator with scripted incident archetypes (runaway loop, cost blowup,
   error cascade, dead-end retrieval) + labels; land fixtures.
2. Day 2 — train the MLP on synthetic windows; export `model.mlxfn`; C++ smoke test that imports
   and scores a zero vector (derisks the boundary earliest).
3. Day 3 — C++ feature packer to full parity (golden test green).
4. Day 4 — policy engine thresholds + reasons finalized; CLI streams decisions end to end.
5. Day 5 — replay evaluator metrics + rules-only vs model lift report; latency microbench of
   `pack+score`.
6. Day 6–7 — Linux CPU container green in CI; docs pass; CUDA container best-effort.

## Current state

Implemented and verified: feature spec v1; Python span schema + JSONL reader; reference feature
packer with golden vectors; model definition + export path (train→export→import→score
round-trip smoke-tested); **C++ feature packer at full parity with the reference**
(`test_cpp_parity` green, including canonical window ordering); rules-first policy engine;
CLI with `--shadow`, `--audit-log` (FNV-1a hash chain, resume-safe, tamper-detected in tests),
`--emit-features`, and `--bench` (rules-only pack p50 ≈ 2µs / p99 ≈ 2.5µs on a Linux x86 CPU
sandbox — indicative only, not a claim); rules-only build option (`ACTWARDEN_ENABLE_MLX=OFF`);
agent instrumentation middleware (`actwarden.instrument`) with a runnable toy-agent example;
audit-log verification/replay from records alone; CPU container that runs the parity suite as a
build gate. C++ sources conform to the fork's `.clang-format`.

Not implemented yet: trained model artifact committed to the repo; hysteresis and score
calibration (TODOs in `policy.h`/`policy.cpp`); cumulative session counters (needs spec v2);
OTLP receiver; containers executed in CI (Dockerfiles written, not yet run); pack+score bench
with MLX enabled; fused pack+score custom extension (only if profiling justifies it).
