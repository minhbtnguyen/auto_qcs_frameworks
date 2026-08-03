"""Shared infrastructure checks are built on -- not an interpreter anymore.
This folder used to have each check as a checks/<check_id>.yaml, executed
by walking its steps generically here. That YAML layer is gone: a fixed
data schema can only ever express what its schema anticipated (this one
could compare two values for equality, and nothing else -- see
check_builder.py's docstring for the concrete bug that caused), while a
real check is just as easy to author -- by a human or by check_builder.py,
reviewed the same way either way -- as a short Python function calling
tools.py directly, with actual `if`/`for` available the moment a check
needs it instead of a schema to extend.

What's still worth sharing across every check, and lives here:
  - CheckContext.call: run a tool, log its observation, save its artifact
    under a name -- the bookkeeping every check needs, so a check body
    reads as the sequence of steps and nothing else.
  - CheckContext.compare: the one comparison primitive every check reuses,
    feeding the deterministic verdict rule below.
  - CheckContext.verdict / .llm_verdict: pass iff every comparison matched
    (zero LLM calls), or -- opt in, per check -- one LLM call reading the
    gathered evidence, for the real but minority case where pass/fail
    itself needs judgment (see run_disclosure_check.py, where llm_infer's
    output is free text with nothing to structurally compare).

A check file is expected to define PARAMS: list[str] and
run_check(**params) -> dict with that exact shape -- not enforced by a
schema anymore, just a convention every generated and hand-written check
follows, which is what keeps them batch-listable/introspectable despite
being free code now instead of validated data.
"""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from tools import compare_values

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


class QCReport(BaseModel):
    status: Literal["pass", "fail"]
    reasoning: str


verdict_llm = ChatAnthropic(model=MODEL, max_tokens=512).with_structured_output(QCReport)


class CheckContext:
    """One of these per run_check() call. Threads the bookkeeping (what
    got saved where, what's been compared, what to tell the solver) so the
    check body itself only has to name the sequence of steps."""

    def __init__(self) -> None:
        self.artifacts: dict[str, dict] = {}
        self.evidence: list[str] = []
        self.compares: list[dict] = []

    def call(self, tool, save_as: Optional[str] = None, **kwargs) -> dict:
        observation, artifact = tool(**kwargs, artifacts=self.artifacts)
        print(f"[run] {tool.__name__}({kwargs}) -> {observation}")
        label = save_as or tool.__name__
        self.evidence.append(f"{label}: {observation}")
        # {} rather than None when a tool has no structured artifact (e.g.
        # llm_infer) -- a later ["field"] lookup then just KeyErrors like
        # any missing dict key would, instead of TypeError-ing on None.
        artifact = artifact if artifact is not None else {}
        if save_as:
            self.artifacts[save_as] = artifact
        return artifact

    def compare(self, left, right) -> dict:
        observation, result = compare_values(
            left=str(left), right=str(right), artifacts=self.artifacts
        )
        print(f"[run] compare({left!r}, {right!r}) -> {observation}")
        self.evidence.append(observation)
        self.compares.append(result)
        return result

    def verdict(self) -> dict:
        passed = all(c["matched"] for c in self.compares)
        print("[run] orchestration LLM calls: 0")
        return {"status": "pass" if passed else "fail", "reasoning": "; ".join(self.evidence)}

    def llm_verdict(self, instructions: str) -> dict:
        context = "\n".join(self.evidence)
        report = verdict_llm.invoke([SystemMessage(instructions), HumanMessage(context)])
        print("[run] orchestration LLM calls: 1")
        return {"status": report.status, "reasoning": report.reasoning}
