"""Multi-agent QCS — ReWOO (Reasoning WithOut Observation).

Same invoice-reconciliation domain as 18_the_agent_loop/main_langchain_complex_qcs.py,
but a different control-flow pattern. Every file in 16-18 is some flavor of
ReAct: an LLM sees the result of each tool call before deciding what happens
next, so "think" and "act" interleave, one step at a time, across however
many turns it takes.

ReWOO decouples planning from execution instead:

    planner -> worker -> solver

  - planner: ONE LLM call. Given the task, it writes out the ENTIRE plan up
    front -- every tool call it intends to make, in order -- without seeing
    any tool results. A step can reference an earlier step's result via an
    evidence variable (#E1, #E2, ...) instead of a literal value, since that
    value doesn't exist yet at planning time.
  - worker: NOT an LLM. A plain Python loop that executes the plan exactly
    as written, substituting in real evidence text for any #E_n references
    as they become available, and recording each step's observation.
  - solver: ONE LLM call. Given the original task plus the full plan and all
    gathered evidence, it synthesizes the final answer (here, a QCReport).

Trade-off versus the ReAct-style files: ReWOO makes exactly 2 LLM calls no
matter how many tool steps the plan has (planner once, solver once), instead
of one call per step -- cheaper and faster when the task decomposes cleanly
up front. The cost is inflexibility: if step 2's result reveals step 3 needs
to change, ReWOO has no way to notice until the solver reads everything at
the end, whereas a ReAct loop would adapt on the very next turn. It's the
right fit here specifically because the reconciliation steps are fully
predictable ahead of time -- the same "fixed pipeline" observation the
hierarchical file's top level was built on, pushed one step further: not
just fixed graph edges, but a fixed *plan* an LLM writes once and never
revisits.

Structurally this also means State doesn't need a `messages` list with an
add_messages reducer the way every ReAct-style file in this series does --
there's no multi-turn conversation accumulating across nodes, just a task
string in, a plan, a dict of evidence, and a report out. No turn budget or
budget_exhausted node either: the plan has a fixed, known length the moment
the planner returns, so this graph has no way to loop.

Tool logic (extraction via LLM + Pydantic, deterministic exact-match
comparison) is unchanged from main_langchain_complex_qcs.py. What's
different is *how* those tools are invoked: no @tool decorators, no
ToolNode, no Command/InjectedState -- the worker is plain code calling plain
functions, because no LLM is choosing which tool to call or reading graph
state back out mid-execution. That machinery existed in the other files
specifically to let an LLM drive tool calls turn-by-turn; ReWOO's worker
never needs an LLM to decide "what next," so it never needs it.
"""

import os
from typing import Literal

from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


# ------------------------------
# Source document and system-of-record database (unchanged from the ReAct
# versions of this domain).
# ------------------------------
RAW_DOCUMENT_TEXT = """INVOICE #INV-58291
Vendor: Acme Robotics Corp
Date: 2026-03-03

Line Items:
  1. Widget Assembly Kit x10 ................ $1,200.00
  2. Gizmo Sensor Module x4 .................. $980.00
  3. Shipping & Handling ..................... $145.50
  4. Rush Processing Fee ..................... $75.00

Subtotal: $2,400.50
Tax (8.25%): $198.04
TOTAL DUE: $2,598.54

Payment Terms: Net 30
Remit to: Acme Robotics Corp, 123 Industrial Way, Springfield
"""

_database: dict[str, dict] = {
    "INV-58291": {
        "vendor": "Acme Robotics Corp",
        "total": 2523.54,
        "date": "2026-03-03",
    },
}


# ------------------------------
# Structured records (unchanged from the ReAct versions).
# ------------------------------
class ExtractedRecord(BaseModel):
    invoice_id: str
    vendor: str
    total: float
    date: str = Field(description="ISO date, YYYY-MM-DD")
    line_items: list[str]


class Discrepancy(BaseModel):
    field: str
    expected: str
    found: str


class ReconciliationResult(BaseModel):
    invoice_id: str
    matched: bool
    discrepancies: list[Discrepancy]


class QCReport(BaseModel):
    status: Literal["pass", "fail"]
    reasoning: str


# ------------------------------
# The plan itself. `tool_input` is a plain string (usually just the invoice
# ID) that may contain an earlier step's evidence_var -- the worker
# substitutes it with that step's real observation text before calling the
# tool.
# ------------------------------
class PlanStep(BaseModel):
    plan: str = Field(description="One sentence: what this step does and why")
    evidence_var: str = Field(
        description='Evidence variable for this step\'s result, e.g. "#E1"'
    )
    tool: Literal["extract_document", "get_db_record", "compare_records"]
    tool_input: str = Field(
        description=(
            "The invoice_id to pass this tool. May reference an earlier "
            "step's evidence_var (e.g. #E1) instead of a literal value if "
            "this step genuinely needs that step's result."
        )
    )


class Plan(BaseModel):
    steps: list[PlanStep]


class State(TypedDict):
    task: str
    plan: Plan | None
    evidence: dict[str, str]
    extracted: ExtractedRecord | None
    reconciliation: ReconciliationResult | None
    qc_report: QCReport | None


# ------------------------------
# Deterministic comparison logic (unchanged) -- no LLM judgment on whether
# numbers match.
# ------------------------------
def reconcile_field(record: ExtractedRecord) -> ReconciliationResult:
    expected = _database.get(record.invoice_id)
    if expected is None:
        return ReconciliationResult(
            invoice_id=record.invoice_id,
            matched=False,
            discrepancies=[
                Discrepancy(
                    field="invoice_id",
                    expected="<a known invoice>",
                    found=record.invoice_id,
                )
            ],
        )

    discrepancies = []
    if record.vendor.strip().lower() != expected["vendor"].strip().lower():
        discrepancies.append(
            Discrepancy(
                field="vendor", expected=expected["vendor"], found=record.vendor
            )
        )
    if abs(record.total - expected["total"]) > 0.01:
        discrepancies.append(
            Discrepancy(
                field="total",
                expected=f"${expected['total']:.2f}",
                found=f"${record.total:.2f}",
            )
        )
    if record.date != expected["date"]:
        discrepancies.append(
            Discrepancy(field="date", expected=expected["date"], found=record.date)
        )

    return ReconciliationResult(
        invoice_id=record.invoice_id,
        matched=not discrepancies,
        discrepancies=discrepancies,
    )


# ------------------------------
# Tool implementations -- plain functions, not @tool. Each takes a string
# and returns a string (the observation the solver will eventually read),
# plus in two cases the structured object the next tool needs, kept in a
# local Python variable inside worker_node rather than any shared state.
# ------------------------------
EXTRACTOR_PROMPT = (
    "You are a document extraction specialist. Read the raw document text and "
    "extract the invoice ID, vendor name, total amount due, invoice date "
    "(YYYY-MM-DD), and a short description of each line item."
)

extractor_llm = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(
    ExtractedRecord
)


def _do_extract(invoice_id: str) -> ExtractedRecord:
    # Single-document demo: always reads the one seeded document.
    return extractor_llm.invoke(
        [SystemMessage(EXTRACTOR_PROMPT), HumanMessage(RAW_DOCUMENT_TEXT)]
    )


def do_extract_document(invoice_id: str) -> tuple[str, ExtractedRecord]:
    record = _do_extract(invoice_id)
    summary = (
        f"Extracted {record.invoice_id}: vendor={record.vendor!r}, "
        f"total=${record.total:.2f}, date={record.date}, "
        f"{len(record.line_items)} line item(s)."
    )
    return summary, record


def do_get_db_record(invoice_id: str) -> str:
    expected = _database.get(invoice_id.upper().strip())
    if expected is None:
        return f"error: no database record for invoice {invoice_id!r}"
    return (
        f"Database record for {invoice_id}: vendor={expected['vendor']!r}, "
        f"total=${expected['total']:.2f}, date={expected['date']}"
    )


def do_compare_records(
    invoice_id: str, extracted: ExtractedRecord | None
) -> tuple[str, ReconciliationResult]:
    """Deterministically compares against the database. `extracted` is
    whatever the worker's extract_document step already produced this run;
    if the plan never ran that step (or ran it for a different invoice), this
    re-extracts as a safety net rather than comparing against nothing."""
    key = invoice_id.upper().strip()
    record = extracted
    if record is None or record.invoice_id.upper().strip() != key:
        record = _do_extract(invoice_id)

    result = reconcile_field(record)
    if result.matched:
        summary = f"Comparison complete: invoice {key} matches the database. No discrepancies found."
    else:
        detail = "; ".join(
            f"{d.field}: expected {d.expected}, found {d.found}"
            for d in result.discrepancies
        )
        summary = (
            f"Comparison complete: invoice {key} was checked on vendor, total, "
            f"and date. This is the complete and only list of discrepancies "
            f"found ({len(result.discrepancies)} total) -- {detail}. No other "
            "fields were checked or flagged."
        )
    return summary, result


# ------------------------------
# Planner: one LLM call, writes the whole plan before any tool runs.
# ------------------------------
PLANNER_PROMPT = (
    "You are a planner for a data reconciliation quality-control process. "
    "Given the user's task, produce a complete, ordered plan BEFORE any tool "
    "is executed -- you will not see any tool results while planning, so the "
    "plan must be fully self-contained.\n\n"
    "Available tools:\n"
    "- extract_document[invoice_id]: pulls structured invoice data from the "
    "source document\n"
    "- get_db_record[invoice_id]: pulls the expected record from the "
    "system-of-record database\n"
    "- compare_records[invoice_id]: deterministically compares the two and "
    "reports every discrepancy found -- requires extract_document to have "
    "already run for this invoice_id\n\n"
    "Every step's tool_input should just be the invoice_id from the task. "
    "You may reference an earlier step's evidence_var (e.g. #E1) inside "
    "tool_input instead, if a later step genuinely needs that step's result "
    "rather than just the ID."
)

planner_llm = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(Plan)


def planner_node(state: State) -> dict:
    plan = planner_llm.invoke(
        [SystemMessage(PLANNER_PROMPT), HumanMessage(state["task"])]
    )
    print("[planner] plan:")
    for step in plan.steps:
        print(f"  {step.evidence_var} = {step.tool}[{step.tool_input}]  # {step.plan}")
    return {"plan": plan}


# ------------------------------
# Worker: not an LLM. Executes the plan exactly as written, in order,
# substituting evidence variables as they become available.
# ------------------------------
def _substitute(text: str, evidence: dict[str, str]) -> str:
    for var, value in evidence.items():
        text = text.replace(var, value)
    return text


def worker_node(state: State) -> dict:
    plan = state["plan"]
    evidence: dict[str, str] = {}
    extracted: ExtractedRecord | None = None
    reconciliation: ReconciliationResult | None = None

    for step in plan.steps:
        invoice_id = _substitute(step.tool_input, evidence).strip()

        if step.tool == "extract_document":
            observation, extracted = do_extract_document(invoice_id)
        elif step.tool == "get_db_record":
            observation = do_get_db_record(invoice_id)
        elif step.tool == "compare_records":
            observation, reconciliation = do_compare_records(invoice_id, extracted)
        else:
            observation = f"error: unknown tool {step.tool!r}"

        print(
            f"[worker] {step.evidence_var} = {step.tool}[{invoice_id}] -> {observation}"
        )
        evidence[step.evidence_var] = observation

    return {
        "evidence": evidence,
        "extracted": extracted,
        "reconciliation": reconciliation,
    }


# ------------------------------
# Solver: one LLM call, synthesizes the final verdict from the task plus the
# full plan-and-evidence trail. Same anti-fabrication discipline as the
# ReAct versions: exhaustive tool-return text, explicit instruction not to
# invent issues beyond what evidence reports.
# ------------------------------
SOLVER_PROMPT = (
    "You are rendering the final QC verdict for a data reconciliation task. "
    "You will be given the original task and the plan that was executed, "
    "with the evidence gathered for each step. Base your verdict ONLY on "
    "what the evidence explicitly reports -- do not infer, assume, or "
    "invent any additional issues. Any discrepancy reported by "
    "compare_records should result in a failed verdict."
)

solver_llm = ChatAnthropic(model=MODEL, max_tokens=512).with_structured_output(QCReport)


def _render_plan_and_evidence(state: State) -> str:
    lines = [f"Task: {state['task']}", "", "Plan and evidence:"]
    for step in state["plan"].steps:
        observation = state["evidence"].get(step.evidence_var, "<no evidence>")
        lines.append(
            f"{step.evidence_var} ({step.tool}[{step.tool_input}]): {step.plan}"
        )
        lines.append(f"  -> {observation}")
    return "\n".join(lines)


def solver_node(state: State) -> dict:
    context = _render_plan_and_evidence(state)
    report = solver_llm.invoke([SystemMessage(SOLVER_PROMPT), HumanMessage(context)])
    print(f"[solver] -> {report.status} ({report.reasoning})")
    return {"qc_report": report}


# ------------------------------
# Graph: a straight line, no conditional edges. The plan already fixed how
# many tool steps happen (inside worker_node's own for-loop); the graph
# level only ever needs to go planner -> worker -> solver -> END, once.
# ------------------------------
graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("worker", worker_node)
graph.add_node("solver", solver_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "worker")
graph.add_edge("worker", "solver")
graph.add_edge("solver", END)

app = graph.compile()


def run_agent(user_message: str) -> dict:
    initial: State = {
        "task": user_message,
        "plan": None,
        "evidence": {},
        "extracted": None,
        "reconciliation": None,
        "qc_report": None,
    }
    return app.invoke(initial)


def main() -> None:
    print("=" * 70)
    print("MULTI-AGENT QCS — ReWOO (Plan -> Work -> Solve)")
    print("=" * 70)

    print("\ngraph structure (Mermaid):")
    print(app.get_graph().draw_mermaid())
    print()

    result = run_agent(
        "Please reconcile invoice INV-58291. Steps: first extract the invoice "
        "data, then get the expected record from the database, then reconcile "
        "them and report any discrepancies."
    )

    print()
    print(f"extracted:      {result['extracted']}")
    print(f"reconciliation: {result['reconciliation']}")
    print(f"qc_report:      {result['qc_report']}")


if __name__ == "__main__":
    main()
