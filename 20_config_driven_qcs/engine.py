"""The generic check-execution engine -- no planner, no LLM call to decide
what steps to run, for checks whose steps are fixed at authoring time
(which is most of them). Pure library, no runnable entry point of its own:
each check lives in its own checks/<check_id>.yaml, run by its own thin
run_<check_id>.py, both importing run_check from here. See yaml_builder.py
for how a check gets authored and saved in the first place.

Reuses the exact same tool functions from tools.py that a planner-based
setup would (a planner was tried first, in this folder, and dropped: for a
fixed check, a planner re-derives the identical plan on every run at LLM
cost and LLM error risk, for a decision that isn't actually being made more
than once). Steps here are the same kind of "call this tool with these
args, using an earlier step's result" a planner would produce -- the only
thing missing on purpose is the planner itself: a check's step list is data
an associate wrote once (or a yaml_builder.py session wrote once, then a
human reviewed), not a decision re-derived on every run.

Verdict is deterministic by default -- pass iff every `compare` step
matched, computed in plain code, zero LLM calls. A check only pays for an
LLM call at all if it declares a `verdict:` block, for the (real, but
minority) case where pass/fail itself needs judgment rather than aggregate
comparison -- see checks/disclosure_check.yaml, where llm_infer's output is
free text with nothing to structurally compare.
"""

import re
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from tools import ALL_TOOLS, MODEL

# Deliberately anchored to the whole string, word chars only -- a natural-
# language instructions sentence ("...must substantively describe these
# risks.") also contains a literal ".", so a naive "if '.' in value" check
# would misfire and try to treat the whole sentence as a name.field
# reference. Anchoring means only an exact, bare "name.field" token can
# ever match; a real sentence, with spaces and punctuation, never does.
_REF_PATTERN = re.compile(r"^(\w+)\.(\w+)$")


class QCReport(BaseModel):
    status: Literal["pass", "fail"]
    reasoning: str


verdict_llm = ChatAnthropic(model=MODEL, max_tokens=512).with_structured_output(
    QCReport
)


def _resolve_value(
    value: str, params: dict[str, str], artifacts: dict[str, dict]
) -> str:
    if value.startswith("$"):
        return params[value[1:]]
    match = _REF_PATTERN.match(value)
    if match:
        name, field = match.groups()
        if name in artifacts and field in artifacts[name]:
            return str(artifacts[name][field])
    return value


def run_check(check: dict, params: dict[str, str]) -> dict:
    # Fail fast on a missing param, before any tool runs -- cheaper than
    # discovering it three steps in, after real (possibly LLM-backed, i.e.
    # billed) tool calls already ran. This is also the first thing in this
    # file that actually reads a check's declared `params:` list -- until
    # now it was documentation an associate could get wrong with no
    # feedback, not something enforced.
    missing = [p for p in check.get("params", []) if p not in params]
    if missing:
        reasoning = f"missing required params: {missing}"
        print(f"[run] error: {reasoning}")
        return {"status": "error", "reasoning": reasoning}

    artifacts: dict[str, dict] = {}
    evidence: list[str] = []
    compares: list[dict] = []
    orchestration_llm_calls = 0

    for step in check["steps"]:
        # Every failure mode here (typo'd tool name, wrong argument name, a
        # step missing both 'tool' and 'compare', a $param reference that
        # doesn't match anything) surfaces as a bare KeyError or TypeError
        # deep in a dict lookup or **kwargs call -- exactly the kind of
        # thing a non-programmer authoring a check will hit constantly.
        # Catching it here means one misconfigured check returns a clear,
        # actionable status instead of crashing whatever batch of checks
        # was running it.
        try:
            if "compare" in step:
                # `compare:` is sugar for calling the compare_values tool --
                # goes through the same ALL_TOOLS entry point as every other
                # step so there's exactly one implementation of comparison
                # logic, not one in tools.py and a second duplicated here.
                left_ref, right_ref = step["compare"]
                left = _resolve_value(left_ref, params, artifacts)
                right = _resolve_value(right_ref, params, artifacts)
                observation, artifact = ALL_TOOLS["compare_values"](
                    left=left, right=right, artifacts=artifacts
                )
                compares.append({"matched": artifact["matched"]})
                print(f"[run] compare({left_ref}, {right_ref}) -> {observation}")
                evidence.append(observation)
                continue

            tool_name = step["tool"]
            tool_fn = ALL_TOOLS[tool_name]
            args = {
                key: _resolve_value(value, params, artifacts)
                for key, value in step.get("args", {}).items()
            }
            observation, artifact = tool_fn(**args, artifacts=artifacts)
        except (KeyError, TypeError) as e:
            reasoning = f"error in step {step!r}: {type(e).__name__}: {e}"
            print(f"[run] {reasoning}")
            return {"status": "error", "reasoning": reasoning}

        print(f"[run] {tool_name}({args}) -> {observation}")
        save_as = step.get("save_as", tool_name)
        evidence.append(f"{save_as}: {observation}")
        # Store {} rather than None when a tool has no structured artifact
        # (e.g. llm_infer) -- a later .field reference into it then just
        # fails the "field in artifacts[name]" check and falls through to
        # being treated as a literal, instead of raising.
        artifacts[save_as] = artifact if artifact is not None else {}

    if "verdict" in check:
        orchestration_llm_calls += 1
        context = "\n".join(evidence)
        report = verdict_llm.invoke(
            [SystemMessage(check["verdict"]["instructions"]), HumanMessage(context)]
        )
        result = {"status": report.status, "reasoning": report.reasoning}
    else:
        passed = all(c["matched"] for c in compares)
        result = {
            "status": "pass" if passed else "fail",
            "reasoning": "; ".join(evidence),
        }

    print(f"[run] orchestration LLM calls: {orchestration_llm_calls}")
    return result
