"""A deliberately loopy toy agent, instrumented with actwarden.instrument.

Emits a real (wall-clock) ActWarden trace: an agent that keeps re-searching
the same query, pays for each step, and hits one flaky tool error. Exists so
the governor can be demoed end to end on a trace produced by running code
rather than a hand-written fixture.

    PYTHONPATH=actwarden/python python3 actwarden/examples/toy_agent.py /tmp/toy_trace.jsonl
    ./actwarden/runtime/build/actwarden \
        --trace /tmp/toy_trace.jsonl --spec actwarden/feature_spec/v1.json \
        --shadow --audit-log /tmp/toy_audit.jsonl
"""

from __future__ import annotations

import sys
import time

from actwarden.instrument import TraceEmitter


def flaky_search(attempt: int) -> None:
    time.sleep(0.002)
    if attempt == 3:
        raise RuntimeError("search backend timeout")


def main(trace_path: str) -> None:
    with TraceEmitter(trace_path, session_id="toy-01") as em:
        with em.agent("research_agent"):
            for attempt in range(6):
                with em.llm(model_name="m-small") as h:
                    time.sleep(0.002)  # stand-in for a model call
                    h.set(
                        prompt_tokens=300 + 40 * attempt,
                        completion_tokens=90,
                        cost_usd=0.004 + 0.001 * attempt,
                    )
                try:
                    with em.tool("web_search", cost_usd=0.001):
                        flaky_search(attempt)
                except RuntimeError:
                    pass  # the agent shrugs and loops — that's the pathology
    print(f"wrote trace to {trace_path}")
    print(
        "score it:  actwarden --trace "
        f"{trace_path} --spec actwarden/feature_spec/v1.json --shadow"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/toy_agent_trace.jsonl")
