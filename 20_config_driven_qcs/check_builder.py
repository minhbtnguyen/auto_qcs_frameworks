"""Check-authoring assistant -- an LLM helps write ONE new check, once, at
authoring time, and saves it as run_<check_id>.py. Not a planner that runs
(and re-derives the same decision) on every check execution -- engine.py
has no interpretation loop to re-derive anything in; nothing here executes
a check.

Same underlying capability the dropped runtime planner had (read a tool
catalog, translate a natural-language description into a sequence of tool
calls), repositioned in time: it runs once, when a check is created, and a
human reviews its output before ever running run_<check_id>.py for
something that matters. That's the whole difference in risk profile -- a
wrong plan here is a bad suggestion sitting in a file someone reads, not a
silent wrong verdict in production on every future run. Saving the file is
not the same as trusting it: "saved" just means "persisted somewhere
reviewable," not "verified."

CheckStep/CheckDefinition and _validate() are unchanged from when this
generated yaml -- the structural guarantee (only real tools, only
declared params, only references to steps that exist) is exactly as
valuable rendered as Python as it was rendered as data. What changed is
only _to_python_script(): the same validated structure now becomes a real
run_check(**params) function calling tools.py directly, not a data file
interpreted by an engine. That buys real control flow (if/for available
the moment a check needs them) at a cost this file's own docstring history
proves is real: _validate() catches structural mistakes (unknown tool,
undeclared param, a dangling reference) but never catches a semantic
mismatch -- e.g. asking for "must not exceed a threshold" when the only
comparison tool available checks equality, not inequality. That class of
mistake only a human (or a richer tool library) catches, which is exactly
why this saves a draft for review rather than trusting anything it writes.
"""

import argparse
import inspect
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from tools import ALL_TOOLS, MODEL

load_dotenv()

ROOT = Path(__file__).parent

# Deliberately anchored to the whole string, word chars only -- a free-text
# instructions sentence ("...must substantively describe these risks.")
# also contains a literal ".", so a naive "if '.' in value" check misfires
# and treats the whole sentence as a name.field reference. This is the
# exact bug that was already found and fixed once in this folder's prior
# engine.py (_resolve_value) -- reapplying the same fix here, since
# _render_value_expr below turned out to need it too.
_REF_PATTERN = re.compile(r"^(\w+)\.(\w+)$")


class CheckStep(BaseModel):
    reason: str = Field(
        description="One sentence: why this step exists -- for human review, not written to the generated file"
    )
    tool: Optional[str] = Field(default=None, description="Tool name, if this is a tool-call step")
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
        description="Valid Python identifier to save this step's result under, required if tool is set",
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
    "isn't listed. Every save_as must be a valid Python identifier (this "
    "becomes a real variable name in generated code).\n\n"
    f"Available tools (name(parameters): description):\n{_tool_catalog_text()}\n\n"
    "Each step is either a tool call (tool + args + save_as) or a "
    "comparison (compare: [left, right]). Set needs_verdict=true only when "
    'pass/fail cannot be reduced to "did every comparison match" -- e.g. '
    "when the last step is llm_infer and its free-text answer has to be "
    "interpreted."
)

builder_llm = ChatAnthropic(model=MODEL, max_tokens=2048).with_structured_output(CheckDefinition)


def _validate(check: CheckDefinition) -> list[str]:
    problems = []
    known_names: set[str] = set()

    for i, step in enumerate(check.steps, start=1):
        if step.compare is not None and step.tool is not None:
            problems.append(f"step {i}: has both 'tool' and 'compare' -- must be exactly one")
            continue

        if step.compare is not None:
            if len(step.compare) != 2:
                problems.append(f"step {i}: compare needs exactly 2 references, got {step.compare}")
            refs = step.compare
        elif step.tool is not None:
            if step.tool not in ALL_TOOLS:
                problems.append(f"step {i}: unknown tool {step.tool!r}")
            if not step.save_as:
                problems.append(f"step {i}: tool step missing save_as")
            elif not step.save_as.isidentifier():
                # Not just style -- save_as becomes a real variable name in
                # the generated file, so an invalid identifier here means
                # invalid Python, not just an ugly name.
                problems.append(f"step {i}: save_as {step.save_as!r} is not a valid Python identifier")
                known_names.add(step.save_as)
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


def _render_value_expr(value: str) -> str:
    """Render a step's raw value (a literal, $param, or name.field
    reference) as the Python expression it becomes in generated code."""
    if value.startswith("$"):
        return f"params[{value[1:]!r}]"
    match = _REF_PATTERN.match(value)
    if match:
        name, field = match.groups()
        return f"{name}[{field!r}]"
    return repr(value)


def _to_python_script(check: CheckDefinition) -> str:
    tool_names = sorted({step.tool for step in check.steps if step.tool})

    lines = [f'"""{check.check_id} -- auto-generated check."""', ""]
    lines.append("import argparse")
    lines.append("")
    lines.append("from engine import CheckContext")
    if tool_names:
        lines.append(f"from tools import {', '.join(tool_names)}")
    lines.append("")
    lines.append(f"PARAMS = {check.params!r}")
    lines.append("")
    lines.append("")
    lines.append("def run_check(**params) -> dict:")
    lines.append("    ctx = CheckContext()")
    lines.append("")

    for step in check.steps:
        lines.append(f"    # {step.reason}")
        if step.compare is not None:
            left = _render_value_expr(step.compare[0])
            right = _render_value_expr(step.compare[1])
            lines.append(f"    ctx.compare({left}, {right})")
        else:
            save_as = step.save_as if (step.save_as and step.save_as.isidentifier()) else "_missing_save_as"
            arg_parts = [f"{k}={_render_value_expr(v)}" for k, v in (step.args or {}).items()]
            call_args = ", ".join([f"save_as={save_as!r}"] + arg_parts)
            lines.append(f"    {save_as} = ctx.call({step.tool}, {call_args})")
        lines.append("")

    if check.needs_verdict:
        lines.append(f"    return ctx.llm_verdict({check.verdict_instructions!r})")
    else:
        lines.append("    return ctx.verdict()")

    lines.append("")
    lines.append("")
    lines.append("def main() -> None:")
    lines.append(f'    parser = argparse.ArgumentParser(description="Run the {check.check_id} check")')
    lines.append("    for p in PARAMS:")
    lines.append('        parser.add_argument(f"--{p}", required=True)')
    lines.append("    args = parser.parse_args()")
    lines.append("")
    lines.append("    print()")
    lines.append('    print(f"result: {run_check(**vars(args))}")')
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    main()")
    lines.append("")

    return "\n".join(lines)


def save_check(check: CheckDefinition) -> Path:
    path = ROOT / f"run_{check.check_id}.py"
    if path.exists():
        print(f"skipped: {path.relative_to(ROOT)} already exists, not overwritten")
        return path
    path.write_text(_to_python_script(check))
    print(f"saved: {path.relative_to(ROOT)}")
    return path


def author_check(description: str) -> CheckDefinition:
    check = builder_llm.invoke([SystemMessage(BUILDER_PROMPT), HumanMessage(description)])

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
        print("structural validation: no problems found -- still read it before trusting it\n")

    print(_to_python_script(check))
    save_check(check)
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description="Author a new check from a natural-language description")
    parser.add_argument("description", help="What should this check verify?")
    args = parser.parse_args()
    author_check(args.description)


if __name__ == "__main__":
    main()
