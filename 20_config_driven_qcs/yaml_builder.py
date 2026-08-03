"""Config-builder assistant -- an LLM that helps write ONE new check,
once, at authoring time, and saves it as checks/<check_id>.yaml plus a
matching run_<check_id>.py. Not a planner that runs (and re-derives the
same decision) on every check execution -- engine.py stays exactly as
planner-free as it already is; nothing here executes a check.

This is the same underlying capability the dropped runtime planner had
(read a tool catalog, translate a natural-language description into a
sequence of tool calls), repositioned in time: it runs once, when a check
is created, and a human reviews its output before ever running
run_<check_id>.py for something that matters. That's the whole difference
in risk profile -- a wrong plan here is a bad suggestion sitting in a yaml
file someone reads, not a silent wrong verdict in production on every
future run. Saving the file is not the same as trusting it: "saved" just
means "persisted somewhere reviewable," not "verified."

_validate() below checks for the same categories of mistake engine.py's
own try/except guards against at execution time (unknown tool, undeclared
param, a reference to a step that was never saved) -- structural problems,
caught automatically, before a human even starts reading. It cannot catch
semantic mismatches -- e.g. asking for "must not exceed a threshold" when
the only comparison tool available checks equality, not inequality. That
class of mistake only a human (or a richer tool library) can catch, which
is exactly why this saves a draft for review rather than executing anything
itself.
"""

import argparse
import inspect
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from tools import ALL_TOOLS, MODEL

load_dotenv()

ROOT = Path(__file__).parent
CHECKS_DIR = ROOT / "checks"


class CheckStep(BaseModel):
    reason: str = Field(
        description="One sentence: why this step exists -- for human review, not written to the yaml"
    )
    tool: Optional[str] = Field(
        default=None, description="Tool name, if this is a tool-call step"
    )
    args: Optional[dict[str, str]] = Field(
        default=None,
        description=(
            "Named args for the tool, required if tool is set. A value may be a "
            "literal, $param_name (a check parameter), or save_as_name.field_name "
            "(a named field of an earlier step's result)."
        ),
    )
    save_as: Optional[str] = Field(
        default=None,
        description="Name to save this step's result under, required if tool is set",
    )
    compare: Optional[list[str]] = Field(
        default=None,
        description="Exactly two references [left, right] to compare, if this is a comparison step instead of a tool call",
    )


class CheckDefinition(BaseModel):
    check_id: str = Field(description="A short snake_case identifier for this check")
    params: list[str] = Field(description="Every $param this check's steps reference")
    steps: list[CheckStep]
    needs_verdict: bool = Field(
        description=(
            "True only if pass/fail genuinely requires judgment (e.g. reading "
            "llm_infer's free text) rather than just aggregating compare steps"
        )
    )
    verdict_instructions: Optional[str] = Field(
        default=None, description="Required if needs_verdict is true"
    )


def _tool_catalog_text() -> str:
    lines = []
    for name, fn in ALL_TOOLS.items():
        params = [p for p in inspect.signature(fn).parameters if p != "artifacts"]
        doc = " ".join((fn.__doc__ or "").split())
        lines.append(f"- {name}({', '.join(params)}): {doc}")
    return "\n".join(lines)


BUILDER_PROMPT = (
    "You are helping an associate author ONE new quality-control check as "
    "structured data -- you are not executing anything, just proposing a "
    "check definition for a human to review. Given their description of "
    "what to check, use only the tools below -- never invent one that "
    "isn't listed.\n\n"
    f"Available tools (name(parameters): description):\n{_tool_catalog_text()}\n\n"
    "Each step is either a tool call (tool + args + save_as) or a "
    "comparison (compare: [left, right]). Set needs_verdict=true only when "
    'pass/fail cannot be reduced to "did every comparison match" -- e.g. '
    "when the last step is llm_infer and its free-text answer has to be "
    "interpreted."
)

builder_llm = ChatAnthropic(model=MODEL, max_tokens=2048).with_structured_output(
    CheckDefinition
)


def _validate(check: CheckDefinition) -> list[str]:
    problems = []
    known_names: set[str] = set()

    for i, step in enumerate(check.steps, start=1):
        if step.compare is not None and step.tool is not None:
            problems.append(
                f"step {i}: has both 'tool' and 'compare' -- must be exactly one"
            )
            continue

        if step.compare is not None:
            if len(step.compare) != 2:
                problems.append(
                    f"step {i}: compare needs exactly 2 references, got {step.compare}"
                )
            refs = step.compare
        elif step.tool is not None:
            if step.tool not in ALL_TOOLS:
                problems.append(f"step {i}: unknown tool {step.tool!r}")
            if not step.save_as:
                problems.append(f"step {i}: tool step missing save_as")
            else:
                known_names.add(step.save_as)
            refs = list((step.args or {}).values())
        else:
            problems.append(f"step {i}: has neither 'tool' nor 'compare'")
            refs = []

        for ref in refs:
            if ref.startswith("$"):
                if ref[1:] not in check.params:
                    problems.append(
                        f"step {i}: references ${ref[1:]!r} but it isn't in params {check.params}"
                    )
            elif "." in ref:
                name = ref.split(".", 1)[0]
                if name not in known_names:
                    problems.append(
                        f"step {i}: references {ref!r} but no earlier step saved as {name!r}"
                    )

    if check.needs_verdict and not check.verdict_instructions:
        problems.append("needs_verdict is true but verdict_instructions is missing")
    return problems


def _to_yaml_body(check: CheckDefinition) -> str:
    body: dict = {"params": check.params, "steps": []}
    for step in check.steps:
        if step.compare is not None:
            body["steps"].append({"compare": step.compare})
        else:
            body["steps"].append(
                {"tool": step.tool, "args": step.args or {}, "save_as": step.save_as}
            )
    if check.needs_verdict:
        body["verdict"] = {"instructions": check.verdict_instructions}
    return yaml.dump(body, sort_keys=False, default_flow_style=False)


# Plain string substitution (not .format()) on purpose -- the generated
# file is full of its own { } (f-strings, dict literals), which .format()
# would try to interpret as substitution targets too. __CHECK_ID__ is
# distinct enough it can never collide with real Python syntax.
RUNNER_TEMPLATE = '''"""Auto-generated runner for the __CHECK_ID__ check. Loads
checks/__CHECK_ID__.yaml and runs it against CLI-supplied params -- edit
the yaml to change the check itself, not this file.
"""

import argparse
from pathlib import Path

import yaml

from engine import run_check

CHECK_FILE = Path(__file__).parent / "checks" / "__CHECK_ID__.yaml"


def main() -> None:
    with open(CHECK_FILE) as f:
        check = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Run the __CHECK_ID__ check")
    for p in check.get("params", []):
        parser.add_argument(f"--{p}", required=True)
    args = parser.parse_args()

    result = run_check(check, vars(args))
    print()
    print(f"result: {result}")


if __name__ == "__main__":
    main()
'''


def render_runner(check_id: str) -> str:
    return RUNNER_TEMPLATE.replace("__CHECK_ID__", check_id)


def save_check(check: CheckDefinition) -> None:
    CHECKS_DIR.mkdir(exist_ok=True)
    yaml_path = CHECKS_DIR / f"{check.check_id}.yaml"
    yaml_path.write_text(_to_yaml_body(check))
    print(f"saved: {yaml_path.relative_to(ROOT)}")

    runner_path = ROOT / f"run_{check.check_id}.py"
    if runner_path.exists():
        print(
            f"skipped: {runner_path.relative_to(ROOT)} already exists, not overwritten"
        )
    else:
        runner_path.write_text(render_runner(check.check_id))
        print(f"saved: {runner_path.relative_to(ROOT)}")


def author_check(description: str) -> CheckDefinition:
    check = builder_llm.invoke(
        [SystemMessage(BUILDER_PROMPT), HumanMessage(description)]
    )

    print("=" * 70)
    print(f"PROPOSED CHECK: {check.check_id}")
    print("=" * 70)
    for i, step in enumerate(check.steps, start=1):
        print(f"  step {i}: {step.reason}")

    problems = _validate(check)
    print()
    if problems:
        print("VALIDATION WARNINGS (fix before trusting this):")
        for p in problems:
            print(f"  - {p}")
        print()
    else:
        print(
            "structural validation: no problems found -- still read it before trusting it\n"
        )

    print(_to_yaml_body(check))
    save_check(check)
    return check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Author a new check from a natural-language description"
    )
    parser.add_argument("description", help="What should this check verify?")
    args = parser.parse_args()
    author_check(args.description)


if __name__ == "__main__":
    main()
