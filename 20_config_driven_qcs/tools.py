"""Tool registry for the config-driven QCS engine -- one function per data
source (PDF, database, spreadsheet, API, email) plus one generic comparison
tool, all sharing engine.py's uniform (str, artifacts) -> (str, dict |
None) signature.

Every source tool returns a plain dict, not a bespoke Pydantic type per
source, specifically so compare_values can reach into any two of them the
same way regardless of where they came from. Each docstring enumerates the
exact fields that tool returns -- whoever writes (or whoever's
yaml_builder.py generates) a checks/<check_id>.yaml entry needs to know
"vendor" is a real field name before writing pdf.vendor into a compare
step, since nothing checks that reference against the tool's actual output
until the check runs.

Each check under checks/ references only the tools it actually needs --
proof that adding a tool here never requires touching engine.py, and that a
given check only pays for (and only needs docs on) the tools its own steps
call for.

Fixture/mock data lives in mock_data.py, not here -- see that file's
docstring for why the split matters at 300+ checks. query_database,
read_spreadsheet, and call_api are also collapsed into one
_make_lookup_tool factory below: all three were the identical shape (strip
a key, look it up in a dict, format an error or a summary), differing only
in which store they read and what to call it in messages. A 4th or 5th
simple lookup source is one factory call now, not a copied function --
read_pdf and read_email stay explicit, since they're LLM-backed extraction,
a genuinely different shape, and there are only two of them so far.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mock_data import (
    API_RESPONSES,
    DATABASE,
    EMAILS,
    PDF_DOCS,
    PDF_SECTIONS,
    SPREADSHEET,
)

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


# ------------------------------
# PDF -- unstructured text, needs LLM judgment to extract.
# ------------------------------
class _InvoiceFields(BaseModel):
    invoice_id: str
    vendor: str
    total: float
    date: str = Field(description="ISO date, YYYY-MM-DD")


_invoice_extractor = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(
    _InvoiceFields
)


def read_pdf(doc_id: str, artifacts: dict[str, Any]) -> tuple[str, Optional[dict]]:
    """Extract structured fields from a PDF invoice by its document ID.
    Input: the invoice ID, e.g. "INV-58291". Returns fields: invoice_id,
    vendor, total, date."""
    doc_id = doc_id.strip()
    text = PDF_DOCS.get(doc_id)
    if text is None:
        return f"error: no PDF found for {doc_id!r}", None
    fields = _invoice_extractor.invoke(
        [
            SystemMessage(
                "Extract the invoice ID, vendor, total amount due, and date "
                "(YYYY-MM-DD) from this document."
            ),
            HumanMessage(text),
        ]
    )
    record = fields.model_dump()
    summary = (
        f"Extracted from PDF {doc_id}: vendor={record['vendor']!r}, "
        f"total=${record['total']:.2f}, date={record['date']}"
    )
    return summary, record


def get_pdf_section(
    doc_id: str, section: str, artifacts: dict[str, Any]
) -> tuple[str, Optional[dict]]:
    """Retrieve one section's text from a pre-parsed, multi-section PDF
    (already chunked and indexed by section name -- this does not parse a
    raw file itself). Input: doc_id (e.g. "10-K-2025") and section (e.g.
    "Risk Factors", "MD&A"). Returns fields: doc_id, section, text."""
    key = (doc_id.strip(), section.strip())
    text = PDF_SECTIONS.get(key)
    if text is None:
        return f"error: no section {section!r} found in document {doc_id!r}", None
    record = {"doc_id": doc_id, "section": section, "text": text}
    preview = text if len(text) <= 150 else text[:150] + "..."
    summary = f"Section {section!r} of {doc_id}: {preview}"
    return summary, record


# ------------------------------
# Simple exact-key lookups -- query_database, read_spreadsheet, and call_api
# are the same shape (strip a key, look it up in a dict, format an error or
# a summary), differing only in which store they read and what to call it
# in messages. One factory, three one-line instantiations below; a 4th or
# 5th simple lookup source is one more factory call, not a copied function.
# Each still gets its own real docstring -- that per-tool field-name
# documentation is load-bearing (yaml_builder.py's catalog is generated
# from exactly this text), so the factory takes it as a parameter rather
# than trying to generate something generic.
# ------------------------------
def _make_lookup_tool(source: dict[str, dict], label: str, doc: str):
    def lookup(key: str, artifacts: dict[str, Any]) -> tuple[str, Optional[dict]]:
        key = key.strip()
        record = source.get(key)
        if record is None:
            return f"error: no {label} found for {key!r}", None
        # label[0].upper() + label[1:], not .capitalize() -- .capitalize()
        # force-lowercases the rest of the string too, which turns "API
        # response" into "Api response" and mangles the acronym.
        summary = f"{label[0].upper() + label[1:]} {key}: " + ", ".join(
            f"{k}={v!r}" for k, v in record.items()
        )
        return summary, record

    lookup.__doc__ = doc
    return lookup


# Database -- different record IDs return different shapes (this mock
# serves three unrelated use cases), which is exactly why each shape is
# spelled out below -- whoever writes a check against this tool can't infer
# a shape they've never seen.
query_database = _make_lookup_tool(
    DATABASE,
    "database record",
    "Look up a record in the system-of-record database by ID. Input: key. "
    "Returns different fields depending which ID you look up: an invoice ID "
    'like "INV-58291" returns {vendor, total, date}; a policy ID like '
    '"POLICY-TRAVEL" returns {category, required_title, threshold}; a '
    'vendor ID like "V-100" returns {vendor_id, legal_name, status}.',
)

# Spreadsheet -- stands in for a parsed XLSX row lookup.
read_spreadsheet = _make_lookup_tool(
    SPREADSHEET,
    "spreadsheet row",
    "Look up a row from the expense ledger spreadsheet by expense ID. "
    'Input: key, e.g. "EXP-4471". Returns fields: expense_id, submitter, '
    "amount, category.",
)

# API -- deterministic (mocked) external service call.
call_api = _make_lookup_tool(
    API_RESPONSES,
    "API response",
    "Call the external vendor-registry API for a vendor's current legal "
    'name and status. Input: key, e.g. "V-100". Returns fields: vendor_id, '
    "legal_name, status.",
)


# ------------------------------
# Email -- unstructured text, needs LLM judgment to extract.
# ------------------------------
class _ApprovalFields(BaseModel):
    expense_id: str
    approver: str
    approver_title: str
    approved_amount: float


_email_extractor = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(
    _ApprovalFields
)


def read_email(thread_id: str, artifacts: dict[str, Any]) -> tuple[str, Optional[dict]]:
    """Extract the approver, their title, and the approved amount from an
    approval email thread. Input: the thread ID, e.g. "THREAD-882". Returns
    fields: expense_id, approver, approver_title, approved_amount."""
    thread_id = thread_id.strip()
    text = EMAILS.get(thread_id)
    if text is None:
        return f"error: no email thread found for {thread_id!r}", None
    fields = _email_extractor.invoke(
        [
            SystemMessage(
                "Extract the expense_id being discussed, the approver's "
                "name, their job title, and the dollar amount they approved "
                "from this email."
            ),
            HumanMessage(text),
        ]
    )
    record = fields.model_dump()
    summary = (
        f"Extracted from email {thread_id}: approver={record['approver']!r}, "
        f"title={record['approver_title']!r}, "
        f"approved_amount=${record['approved_amount']:.2f}"
    )
    return summary, record


# ------------------------------
# Generic inference -- unlike every tool above, this one has no fixed
# output schema at all: what it produces depends entirely on the
# instructions given in a check's step, not something this file can know
# in advance. That's why it returns free text and no artifact (None)
# -- there's no dict shape to hand downstream steps like compare. It can
# still be the last step of a check (a verdict step reads its text like any
# other evidence); it just can't feed a structured comparison. A more
# capable version could build a Pydantic model at runtime from a
# caller-specified field list to recover a comparable artifact -- a real
# option, just a bigger feature than this template needs for a genuinely
# open-ended "read this and tell me X" step.
# ------------------------------
_inference_llm = ChatAnthropic(model=MODEL, max_tokens=1024)


def llm_infer(
    text: str, instructions: str, artifacts: dict[str, Any]
) -> tuple[str, Optional[dict]]:
    """Run a one-shot inference over arbitrary text given free-form
    instructions, e.g. instructions='Does this mention supply chain risk? '
    'Answer yes or no and quote the relevant sentence.' Input: text (the
    raw text to read, often a prior step's save_as name plus .text, e.g.
    section_data.text) and instructions (what to do with it). Returns
    free-form text only -- no structured fields, since the answer's shape
    depends entirely on the instructions given."""
    response = _inference_llm.invoke(
        [
            SystemMessage(f"Follow these instructions exactly: {instructions}"),
            HumanMessage(text),
        ]
    )
    return response.content, None


# ------------------------------
# Generic comparison -- the one tool every check reuses, regardless of
# which two sources it's comparing or what fields they call things. Takes
# two already-resolved values, not raw references -- engine.py's
# _resolve_value turns a "name.field" reference into the real value before
# this (or any) tool ever sees it, uniformly for every tool, so this one
# doesn't do its own artifact lookup.
# ------------------------------
def _values_match(a: str, b: str) -> bool:
    try:
        a_num, b_num = float(a), float(b)
        return abs(a_num - b_num) <= max(abs(a_num), abs(b_num), 1.0) * 0.001
    except (TypeError, ValueError):
        return a.strip().lower() == b.strip().lower()


def compare_values(
    left: str, right: str, artifacts: dict[str, Any]
) -> tuple[str, Optional[dict]]:
    """Deterministically compare two values and report whether they match
    -- numeric values are compared with a small tolerance, everything else
    with a case-insensitive exact match. Input: left and right, typically
    each a "name.field" reference (resolved to the real value before this
    tool runs), but a literal value works too."""
    matched = _values_match(left, right)
    result = {"left": left, "right": right, "matched": matched}
    if matched:
        summary = f"Match: {left!r} == {right!r}"
    else:
        summary = f"Discrepancy: {left!r} != {right!r} -- these do not match."
    return summary, result


ALL_TOOLS = {
    "read_pdf": read_pdf,
    "get_pdf_section": get_pdf_section,
    "query_database": query_database,
    "read_spreadsheet": read_spreadsheet,
    "call_api": call_api,
    "read_email": read_email,
    "llm_infer": llm_infer,
    "compare_values": compare_values,
}
