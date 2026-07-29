"""Multi-agent QCS v3 — hierarchical: two teams, deterministic top-level flow.

Concrete answer to "is there a scalable way with more subagents and tools":
not a flatter supervisor with more options, but a hierarchy with bounded
fan-out at every level.

    data_team (its own supervisor + 2 specialists)
      |-- extractor   (bound only to extract_document)
      `-- database    (bound only to get_db_record)
    audit_team (its own supervisor + 2 specialists)
      |-- comparator   (bound only to compare_records)
      `-- risk_assessor (bound only to assess_risk)

    data_team -> audit_team -> verdict

The team level genuinely needs LLM-driven routing: within data_team,
extractor and database are independent, so which one goes first isn't
knowable ahead of time -- that's a real decision, and it was reliable across
every run. The top level does not need that: data must always be gathered
before it's audited, and there's never a reason to run either team twice.
That's the same "fixed pipeline" lesson main_langchain_complex_qcs.py's
original design was built on (don't spend an LLM call on a decision that was
never actually in question) -- it just applies one level up here.

An earlier version of this file used an LLM-routed top-level supervisor
(mirroring the flat trading-file's pattern), deciding between
data_team/audit_team/end on every step. It worked most of the time, but on
some runs it re-dispatched a team that had already finished, or looped until
the turn budget ran out -- because it was asking an LLM to reliably enforce
an ordering constraint that plain graph edges enforce for free, every time,
with zero tokens spent. This version replaces that routing supervisor with a
straight sequence and keeps the LLM only for the one genuinely
judgment-requiring step at the top: rendering the final pass/fail QCReport
from what the teams reported.

Each team's compiled graph is used as an ordinary node in the parent graph
(`graph.add_node("data_team", data_team_app)`); LangGraph treats a compiled
StateGraph as a Runnable, so a whole team's internal loop (its own supervisor
<-> specialists <-> tools cycle) collapses into one atomic step from the
parent's point of view.

State is intentionally shared end-to-end (one TypedDict, reused by the parent
and both team subgraphs) rather than isolated per team -- this keeps the demo
close to the flat version's simplicity. `team_next`/`team_turns` are owned by
whichever team is currently running; a small `reset_for_audit` node clears
them between teams so audit_team doesn't silently inherit data_team's
leftover routing state or spent turn budget.

Everything else -- Command-based tool state updates, ID-only handoffs between
tools, deterministic comparison, exhaustive tool-return text -- is unchanged
from main_langchain_complex_qcs.py. This file is additive on top of that
design, not a different philosophy.
"""

import os
import sys
from typing import Annotated, Literal

from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
MAX_TEAM_TURNS = 6


# ------------------------------
# Source document and system-of-record database (unchanged from v2).
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
# Structured records
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


class RiskAssessment(BaseModel):
    invoice_id: str
    level: Literal["low", "medium", "high"]
    detail: str


class QCReport(BaseModel):
    status: Literal["pass", "fail"]
    reasoning: str


# ------------------------------
# State -- shared by the parent graph and both team subgraphs. Data fields
# (extracted, reconciliation, risk, qc_report) flow freely across the whole
# hierarchy; team_next/team_turns belong to whichever team is currently
# running (the top level no longer has its own routing fields -- it isn't an
# LLM-routed decision anymore, see module docstring).
# ------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    team_turns: int
    team_next: str
    extracted: ExtractedRecord | None
    reconciliation: ReconciliationResult | None
    risk: RiskAssessment | None
    qc_report: QCReport | None


# ------------------------------
# Deterministic logic (no LLM judgment on exact-match comparison or on the
# risk-scoring arithmetic) -- reused by tools below.
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


def _parse_dollar(value: str) -> float:
    return float(value.replace("$", "").replace(",", ""))


def score_risk(result: ReconciliationResult) -> RiskAssessment:
    if result.matched:
        return RiskAssessment(
            invoice_id=result.invoice_id, level="low", detail="No discrepancies found."
        )

    total_disc = next((d for d in result.discrepancies if d.field == "total"), None)
    if total_disc:
        expected = _parse_dollar(total_disc.expected)
        found = _parse_dollar(total_disc.found)
        pct = abs(found - expected) / expected * 100 if expected else 100.0
    else:
        # A non-total mismatch (vendor/date) has no natural percentage --
        # treat identity/date discrepancies as high risk outright.
        pct = 100.0

    level: Literal["low", "medium", "high"] = (
        "high" if pct >= 5 else "medium" if pct >= 1 else "low"
    )
    return RiskAssessment(
        invoice_id=result.invoice_id,
        level=level,
        detail=f"Total variance is {pct:.2f}% of the expected amount.",
    )


# ------------------------------
# Extraction is LLM-backed (unstructured text needs judgment); everything
# else in this section is unchanged in spirit from the flat v2 file.
# ------------------------------
EXTRACTOR_PROMPT = (
    "You are a document extraction specialist. Read the raw document text and "
    "extract the invoice ID, vendor name, total amount due, invoice date "
    "(YYYY-MM-DD), and a short description of each line item."
)

extractor_llm_structured = ChatAnthropic(
    model=MODEL, max_tokens=1024
).with_structured_output(ExtractedRecord)


def _do_extract(invoice_id: str) -> ExtractedRecord:
    # Single-document demo: always reads the one seeded document.
    return extractor_llm_structured.invoke(
        [SystemMessage(EXTRACTOR_PROMPT), HumanMessage(RAW_DOCUMENT_TEXT)]
    )


@tool
def extract_document(
    invoice_id: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Extract structured invoice data (vendor, total, date, line items) from
    the raw source document for the given invoice ID."""
    record = _do_extract(invoice_id)
    summary = (
        f"Extracted {record.invoice_id}: vendor={record.vendor!r}, "
        f"total=${record.total:.2f}, date={record.date}, "
        f"{len(record.line_items)} line item(s)."
    )
    return Command(
        update={
            "extracted": record,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
def get_db_record(invoice_id: str) -> str:
    """Look up the system-of-record database entry for an invoice ID."""
    expected = _database.get(invoice_id.upper().strip())
    if expected is None:
        return f"error: no database record for invoice {invoice_id!r}"
    return (
        f"Database record for {invoice_id}: vendor={expected['vendor']!r}, "
        f"total=${expected['total']:.2f}, date={expected['date']}"
    )


@tool
def compare_records(
    invoice_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
) -> Command:
    """Deterministically compare the extracted invoice data against the
    database record for this invoice ID and report every discrepancy found.
    Reads the extracted record from graph state; re-extracts only if it
    isn't there yet."""
    key = invoice_id.upper().strip()
    update: dict = {}

    record = state.get("extracted")
    if record is None or record.invoice_id.upper().strip() != key:
        record = _do_extract(invoice_id)
        update["extracted"] = record

    result = reconcile_field(record)
    update["reconciliation"] = result

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

    update["messages"] = [ToolMessage(content=summary, tool_call_id=tool_call_id)]
    return Command(update=update)


@tool
def assess_risk(
    invoice_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[State, InjectedState],
) -> Command:
    """Assess the materiality/risk level of any discrepancies reconciliation
    found for this invoice, based on the size of the total variance relative
    to the expected amount. Requires compare_records to have run first."""
    key = invoice_id.upper().strip()
    result = state.get("reconciliation")

    if result is None or result.invoice_id.upper().strip() != key:
        summary = (
            f"error: no reconciliation result for invoice {key} yet -- "
            "run compare_records first."
        )
        return Command(
            update={
                "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]
            }
        )

    assessment = score_risk(result)
    summary = (
        f"Risk assessment for {key}: {assessment.level} risk -- {assessment.detail}"
    )
    return Command(
        update={
            "risk": assessment,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


def make_specialist_node(llm):
    """Factory: every specialist below is this same one-line shape, just
    bound to a different single-tool LLM -- narrow binding per specialist,
    same rule as the flat file, just applied to 4 specialists instead of 1."""

    def node(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    return node


def after_specialist(state: State) -> str:
    """Shared by every specialist in both teams -- each team's own subgraph
    has its own "tools"/"supervisor" nodes, so this one routing function is
    safe to reuse across both without name collisions."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "supervisor"


def budget_exhausted_node(state: State) -> dict:
    return {"messages": [AIMessage("budget exhausted")]}


# ------------------------------
# Data team: extractor + database specialists
# ------------------------------
extractor_node = make_specialist_node(
    ChatAnthropic(model=MODEL, max_tokens=512).bind_tools([extract_document])
)
database_node = make_specialist_node(
    ChatAnthropic(model=MODEL, max_tokens=512).bind_tools([get_db_record])
)


class DataTeamRoute(BaseModel):
    next: Literal["extractor", "database", "end"]
    reason: str = Field(description="One sentence on why this specialist goes next")


DATA_TEAM_PROMPT = (
    "You are the data-gathering team supervisor. Dispatch 'extractor' to pull "
    "invoice data from the source document, and 'database' to pull the "
    "expected record from the system of record. Once both have reported, "
    "route to 'end'."
)

data_team_supervisor_llm = ChatAnthropic(
    model=MODEL, max_tokens=256
).with_structured_output(DataTeamRoute)


def data_team_supervisor_node(state: State) -> dict:
    route = data_team_supervisor_llm.invoke(
        [SystemMessage(DATA_TEAM_PROMPT)] + state["messages"]
    )
    print(f"    [data-team supervisor] -> {route.next} ({route.reason})")
    return {"team_next": route.next, "team_turns": state["team_turns"] + 1}


def route_data_team_supervisor(state: State) -> str:
    if state["team_turns"] >= MAX_TEAM_TURNS:
        return "budget_exhausted"
    return state["team_next"]


data_team_tool_node = ToolNode(tools=[extract_document, get_db_record])

data_team_graph = StateGraph(State)
data_team_graph.add_node("supervisor", data_team_supervisor_node)
data_team_graph.add_node("extractor", extractor_node)
data_team_graph.add_node("database", database_node)
data_team_graph.add_node("tools", data_team_tool_node)
data_team_graph.add_node("budget_exhausted", budget_exhausted_node)

data_team_graph.set_entry_point("supervisor")
data_team_graph.add_conditional_edges(
    "supervisor",
    route_data_team_supervisor,
    {
        "extractor": "extractor",
        "database": "database",
        "end": END,
        "budget_exhausted": "budget_exhausted",
    },
)
data_team_graph.add_conditional_edges(
    "extractor", after_specialist, {"tools": "tools", "supervisor": "supervisor"}
)
data_team_graph.add_conditional_edges(
    "database", after_specialist, {"tools": "tools", "supervisor": "supervisor"}
)
data_team_graph.add_edge("tools", "supervisor")
data_team_graph.add_edge("budget_exhausted", END)

data_team_app = data_team_graph.compile()


# ------------------------------
# Audit team: comparator -> risk_assessor, a fixed 2-step pipeline, not a
# supervisor-routed choice. Unlike data_team's extractor/database (which are
# genuinely independent -- either order works), risk_assessor has a hard data
# dependency on comparator's output (assess_risk reads state["reconciliation"]
# and errors without it), and skipping it is never valid: both steps are
# mandatory, in a fixed order. An earlier version routed this with an LLM
# supervisor (mirroring data_team's pattern); on some runs it decided
# comparator alone was "enough" and ended without ever calling risk_assessor
# -- leaving state["risk"] as None while the top-level verdict LLM, still
# only reading conversation text, went ahead and fabricated a risk
# characterization anyway. Same root cause as the v2 investigator
# hallucination bug (an LLM asked to track completeness from prose instead
# of state), just relocated. Since the order and requiredness are both fixed
# ahead of time, this is now plain graph edges plus a state-derived check
# (`state.get("risk") is not None`) -- no LLM can skip a step code doesn't
# offer it a path around.
# ------------------------------
comparator_llm = ChatAnthropic(model=MODEL, max_tokens=512).bind_tools(
    [compare_records]
)
risk_llm = ChatAnthropic(model=MODEL, max_tokens=512).bind_tools([assess_risk])


def comparator_node(state: State) -> dict:
    print("    [audit-team] -> comparator")
    response = comparator_llm.invoke(state["messages"])
    return {"messages": [response], "team_turns": state["team_turns"] + 1}


def risk_node(state: State) -> dict:
    print("    [audit-team] -> risk_assessor")
    response = risk_llm.invoke(state["messages"])
    return {"messages": [response], "team_turns": state["team_turns"] + 1}


def after_comparator(state: State) -> str:
    if state["team_turns"] >= MAX_TEAM_TURNS:
        return "budget_exhausted"
    last = state["messages"][-1]
    # compare_records is mandatory -- if the LLM replied without calling it,
    # dispatch comparator again rather than moving on.
    return "tools" if getattr(last, "tool_calls", None) else "comparator"


def after_audit_tools(state: State) -> str:
    # Deterministic branch on graph state, not on LLM judgment: once risk is
    # populated the mandatory pipeline is complete.
    return "end" if state.get("risk") is not None else "risk_assessor"


def after_risk_assessor(state: State) -> str:
    if state["team_turns"] >= MAX_TEAM_TURNS:
        return "budget_exhausted"
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "risk_assessor"


audit_team_tool_node = ToolNode(tools=[compare_records, assess_risk])

audit_team_graph = StateGraph(State)
audit_team_graph.add_node("comparator", comparator_node)
audit_team_graph.add_node("risk_assessor", risk_node)
audit_team_graph.add_node("tools", audit_team_tool_node)
audit_team_graph.add_node("budget_exhausted", budget_exhausted_node)

audit_team_graph.set_entry_point("comparator")
audit_team_graph.add_conditional_edges(
    "comparator",
    after_comparator,
    {
        "tools": "tools",
        "comparator": "comparator",
        "budget_exhausted": "budget_exhausted",
    },
)
audit_team_graph.add_conditional_edges(
    "tools", after_audit_tools, {"risk_assessor": "risk_assessor", "end": END}
)
audit_team_graph.add_conditional_edges(
    "risk_assessor",
    after_risk_assessor,
    {
        "tools": "tools",
        "risk_assessor": "risk_assessor",
        "budget_exhausted": "budget_exhausted",
    },
)
audit_team_graph.add_edge("budget_exhausted", END)

audit_team_app = audit_team_graph.compile()


# ------------------------------
# Top level: data_team -> audit_team -> verdict. Plain edges, not an LLM
# routing decision -- see module docstring for why. The only LLM call up
# here renders the final QCReport from what the teams already reported.
# ------------------------------
def _reset_team_routing(state: State) -> dict:
    """Runs between data_team and audit_team so audit_team's supervisor
    starts with a clean team_turns budget and no leftover team_next from
    data_team's last internal step."""
    return {"team_turns": 0, "team_next": ""}


VERDICT_PROMPT = (
    "You are rendering the final QC verdict for a data reconciliation "
    "process. The data-gathering and audit teams have already finished; "
    "read the conversation to see what they found. Base your verdict ONLY "
    "on what reconciliation and risk assessment explicitly reported -- do "
    "not infer, assume, or invent any additional issues. A high or medium "
    "risk level should generally result in a failed verdict."
)

verdict_llm = ChatAnthropic(model=MODEL, max_tokens=512).with_structured_output(
    QCReport
)


def top_verdict_node(state: State) -> dict:
    report = verdict_llm.invoke([SystemMessage(VERDICT_PROMPT)] + state["messages"])
    print(f"[verdict] -> {report.status} ({report.reasoning})")
    verdict_text = (
        f"QC Verdict: {report.status.upper()}\n\nReasoning: {report.reasoning}"
    )
    return {"qc_report": report, "messages": [AIMessage(verdict_text)]}


graph = StateGraph(State)
graph.add_node("data_team", data_team_app)
graph.add_node("reset_for_audit", _reset_team_routing)
graph.add_node("audit_team", audit_team_app)
graph.add_node("verdict", top_verdict_node)

graph.set_entry_point("data_team")
graph.add_edge("data_team", "reset_for_audit")
graph.add_edge("reset_for_audit", "audit_team")
graph.add_edge("audit_team", "verdict")
graph.add_edge("verdict", END)

app = graph.compile()


# ------------------------------
# Observation formatter (unchanged shape)
# ------------------------------
def _ai_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = [
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return " ".join(p for p in parts if p)


def pretty_trace(messages: list[AnyMessage]) -> None:
    observations = {
        m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)
    }
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            print(f"[{i:02d}    user] {message.content}")
        elif isinstance(message, AIMessage):
            text = _ai_text(message)
            if message.tool_calls:
                if text:
                    print(f"[{i:02d} thought] {text}")
                for call in message.tool_calls:
                    observation = observations.get(call["id"], "?")
                    print(
                        f"[{i:02d}  action] {call['name']}({call['args']}) -> {observation}"
                    )
            else:
                print(f"[{i:02d}   final] {text}")


def run_agent(user_message: str) -> dict:
    initial = {
        "messages": [HumanMessage(user_message)],
        "team_turns": 0,
        "team_next": "",
        "extracted": None,
        "reconciliation": None,
        "risk": None,
        "qc_report": None,
    }
    return app.invoke(initial)


def main() -> None:
    print("=" * 70)
    print("MULTI-AGENT QCS v3 — Hierarchical (top supervisor + 2 teams)")
    print("=" * 70)

    print("\ntop-level graph structure (Mermaid):")
    print(app.get_graph().draw_mermaid())

    print("\ndata_team subgraph structure (Mermaid):")
    print(data_team_app.get_graph().draw_mermaid())

    print("\naudit_team subgraph structure (Mermaid):")
    print(audit_team_app.get_graph().draw_mermaid())

    result = run_agent(
        "Please reconcile invoice INV-58291. Steps: first extract the invoice "
        "data, then get the expected record from the database, then reconcile "
        "them, assess the risk, and report any discrepancies."
    )
    messages = result["messages"]

    print()
    pretty_trace(messages)
    print()
    print(f"extracted:      {result['extracted']}")
    print(f"reconciliation: {result['reconciliation']}")
    print(f"risk:           {result['risk']}")
    print(f"qc_report:      {result['qc_report']}")


if __name__ == "__main__":
    main()
