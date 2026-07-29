"""Multi-agent Quality Control System (QCS) v2 — supervisor + reconciliation subagent.

Redesign of the fixed-pipeline version: instead of separate top-level nodes
for each stage, this reuses the exact supervisor/specialist skeleton from
main_langchain_complex.py (the trading file) -- one supervisor loop, but a
single subagent this time (`reconciliation`) bound to three tools instead of
two single-tool specialists:

  - extract_document -> pulls structured invoice data from the source doc
  - get_db_record     -> pulls the expected record from the database
  - compare_records    -> deterministic exact-match comparison (no LLM
                          judgment on whether numbers match)

The supervisor reads the user's stated steps, dispatches to `reconciliation`,
and once the subagent reports back, renders the final structured verdict
itself (`QCReport`) -- same "who has the full picture" reasoning as the
trading file's `final_message`.

Two design choices worth calling out:

1. All three tools key off `invoice_id` alone and cache/re-derive their own
   data, rather than having the subagent copy the extracted total/vendor/date
   as literal arguments into compare_records. Copying a short ID between tool
   calls is low-risk; copying a dollar figure and a vendor name by hand across
   three separate calls is exactly the transcription surface that caused the
   hallucination in the previous version of this file.

2. Structure protects code-to-code handoffs and graph routing -- it does NOT
   automatically protect LLM-to-LLM handoffs, because an LLM can only ever
   read text, never a Python object directly. compare_records' return string
   is deterministic and *exhaustive* ("this is the complete and only list"),
   and both the reconciliation and supervisor prompts explicitly forbid
   inventing discrepancies beyond what was reported -- that combination, not
   the Pydantic models alone, is what prevents the fabrication bug seen last
   time.

3. extract_document and compare_records return `Command(update={...})`
   instead of a plain string. A tool isn't limited to only appending a
   ToolMessage -- returning a Command lets it write directly to other State
   fields (`extracted`, `reconciliation`) in the same step. The model still
   only ever gets the text summary (that part is unavoidable -- it can only
   read tokens), but graph State now holds the real Pydantic objects too,
   visible in --trace and inspectable by code, instead of hiding in a
   module-level cache dict nothing else could see.
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
MAX_TURNS = 10


# ------------------------------
# Source document and system-of-record database. Deliberately seeded with one
# discrepancy -- the DB total is stale by exactly the rush fee -- so the
# pipeline has something real to catch, not a rubber-stamp pass.
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


class QCReport(BaseModel):
    status: Literal["pass", "fail"]
    reasoning: str


# ------------------------------
# State -- defined before the tools below since compare_records takes an
# Annotated[State, InjectedState] parameter and needs the name to exist yet.
# ------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: int
    next: str
    extracted: ExtractedRecord | None
    reconciliation: ReconciliationResult | None
    qc_report: QCReport | None


# ------------------------------
# Deterministic comparison logic (unchanged from v1) -- reused by the
# compare_records tool below.
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
# Extraction is LLM-backed (reading unstructured text requires judgment) but
# exposed to the reconciliation subagent as a plain tool.
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
    # Single-document demo: always reads the one seeded document. A real
    # system would look up the raw text by invoice_id from storage here.
    return extractor_llm.invoke(
        [SystemMessage(EXTRACTOR_PROMPT), HumanMessage(RAW_DOCUMENT_TEXT)]
    )


@tool
def extract_document(invoice_id: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
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
    Reads the extracted record directly from graph state (populated by
    extract_document); re-extracts only if it isn't there yet, so it's safe
    to call even if extract_document wasn't called first."""
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


RECONCILIATION_TOOLS = [extract_document, get_db_record, compare_records]


# ------------------------------
# Reconciliation subagent: single-shot per dispatch, bound to all three
# tools, hands control back to the supervisor once it's done.
# ------------------------------
RECONCILIATION_PROMPT = (
    "You are a reconciliation subagent. Given a task describing which invoice "
    "to reconcile, follow these steps in order: (1) call extract_document to "
    "pull the invoice's data from the source document, (2) call get_db_record "
    "to pull the expected data from the database, (3) call compare_records to "
    "deterministically check them against each other. Do not attempt the "
    "comparison yourself -- always use compare_records for it, and report "
    "only what it returns, with no additional discrepancies of your own. "
    "Once compare_records has run, summarize its result in one message and "
    "stop calling tools."
)

reconciliation_llm = ChatAnthropic(model=MODEL, max_tokens=768).bind_tools(
    RECONCILIATION_TOOLS
)


def reconciliation_node(state: State) -> dict:
    response = reconciliation_llm.invoke(
        [SystemMessage(RECONCILIATION_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def after_reconciliation(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "supervisor"


# ------------------------------
# Supervisor: reads the user's stated steps, dispatches to reconciliation,
# then renders the final structured verdict once the subagent reports back.
# ------------------------------
class Route(BaseModel):
    next: Literal["reconciliation", "end"]
    reason: str = Field(description="One sentence on why this is the next step")
    qc_report: QCReport | None = Field(
        default=None,
        description="Required when next='end': the final pass/fail verdict with reasoning.",
    )


SUPERVISOR_PROMPT = (
    "You are a supervisor for a data reconciliation quality-control process. "
    "The user describes which invoice to reconcile and the steps to follow. "
    "Route to 'reconciliation' to have the subagent extract the data, pull "
    "the database record, and compare them. Once the subagent reports its "
    "comparison result, route to 'end' and render the final QC verdict "
    "yourself: pass only if no discrepancies were reported; fail otherwise. "
    "Base your verdict ONLY on discrepancies the subagent explicitly "
    "reported -- do not infer, assume, or invent any additional issues."
)

supervisor_llm = ChatAnthropic(model=MODEL, max_tokens=512).with_structured_output(
    Route
)


def supervisor_node(state: State) -> dict:
    route = supervisor_llm.invoke(
        [SystemMessage(SUPERVISOR_PROMPT)] + state["messages"]
    )
    print(f"  [supervisor] -> {route.next} ({route.reason})")

    update: dict = {"next": route.next, "turns": state["turns"] + 1}
    if route.next == "end" and route.qc_report:
        update["qc_report"] = route.qc_report
        update["messages"] = [
            AIMessage(
                f"QC Verdict: {route.qc_report.status.upper()}\n\n"
                f"Reasoning: {route.qc_report.reasoning}"
            )
        ]
    return update


def route_supervisor(state: State) -> str:
    if state["turns"] >= MAX_TURNS:
        return "budget_exhausted"
    return state["next"]


def budget_exhausted_node(state: State) -> dict:
    return {"messages": [AIMessage("budget exhausted")]}


# ------------------------------
# Graph: same skeleton as main_langchain_complex.py's trading supervisor --
# one supervisor loop, now with a single subagent bound to three tools.
# ------------------------------
tool_node = ToolNode(tools=RECONCILIATION_TOOLS)

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("reconciliation", reconciliation_node)
graph.add_node("tools", tool_node)
graph.add_node("budget_exhausted", budget_exhausted_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "reconciliation": "reconciliation",
        "end": END,
        "budget_exhausted": "budget_exhausted",
    },
)
graph.add_conditional_edges(
    "reconciliation",
    after_reconciliation,
    {"tools": "tools", "supervisor": "supervisor"},
)
graph.add_edge("tools", "reconciliation")
graph.add_edge("budget_exhausted", END)

app = graph.compile()


# ------------------------------
# Observation formatter
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


def run_agent(user_message: str, trace: bool = False) -> dict:
    """Run the graph. With trace=True, stream node-by-node and print each
    node's full State input and the partial update it returned."""
    initial = {
        "messages": [HumanMessage(user_message)],
        "turns": 0,
        "next": "",
        "extracted": None,
        "reconciliation": None,
        "qc_report": None,
    }

    if not trace:
        return app.invoke(initial)

    state = dict(initial)
    state["messages"] = list(initial["messages"])
    for update in app.stream(initial, stream_mode="updates"):
        for node_name, partial in update.items():
            # Normally one dict per node. But when a ToolNode executes
            # multiple tool calls and at least one returns a Command, this
            # becomes a list of update dicts instead -- normalize to a list
            # either way so the merge/print logic below doesn't care which.
            partial_updates = partial if isinstance(partial, list) else [partial]

            print("=" * 70)
            print(f"NODE: {node_name}")
            print("-" * 70)
            print("INPUT  (accumulated state this node received):")
            for key, value in state.items():
                if key != "messages":
                    print(f"  {key} = {value}")
            print(f"  messages ({len(state['messages'])}):")
            for i, msg in enumerate(state["messages"]):
                print(f"    [{i}] {msg.type}: {str(msg.content)[:90]!r}")

            print("\nOUTPUT (partial update this node returned):")
            for p in partial_updates:
                for key, value in p.items():
                    if key == "messages":
                        for msg in value:
                            print(f"  messages += {msg.type}: {str(msg.content)[:120]!r}")
                    else:
                        print(f"  {key} = {value}")
            print()

            for p in partial_updates:
                if "messages" in p:
                    state["messages"] = state["messages"] + p["messages"]
                for key, value in p.items():
                    if key != "messages":
                        state[key] = value

    print("=" * 70)
    print("FINAL RETAINED STATE:")
    for key, value in state.items():
        if key != "messages":
            print(f"  {key} = {value}")
    print(f"  messages = {len(state['messages'])} total")

    return state


def main() -> None:
    print("=" * 70)
    print("MULTI-AGENT QCS v2 — Supervisor + Reconciliation Subagent")
    print("=" * 70)

    print("\ngraph structure (Mermaid):")
    print(app.get_graph().draw_mermaid())

    trace = "--trace" in sys.argv
    if trace:
        print("\n--trace enabled: showing per-node input/output\n")

    result = run_agent(
        "Please reconcile invoice INV-58291. Steps: first extract the invoice "
        "data, then get the expected record from the database, then reconcile "
        "them and report any discrepancies.",
        trace=trace,
    )
    messages = result["messages"]

    print()
    pretty_trace(messages)
    print()
    print(f"extracted:      {result['extracted']}")
    print(f"reconciliation: {result['reconciliation']}")
    print(f"qc_report:      {result['qc_report']}")
    print(f"turns used:     {result['turns']}")


if __name__ == "__main__":
    main()
