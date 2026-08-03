"""Fake in-memory backends standing in for a real PDF parser, SQL
connection, XLSX reader, HTTP client, and mail API.

Kept separate from tools.py on purpose: at 300+ checks, this is the file
that keeps growing (every new invoice, vendor, expense you want to test
against needs a fixture here) while tools.py -- the actual tool logic --
should almost never need to change once a source type exists. Swapping a
mock for a real backend later means changing what this file exports (or
replacing an import in tools.py with a real DB client / HTTP client), not
touching a single tool function's logic.

Data is deliberately seeded with exactly one real discrepancy per source
pairing (never a rubber-stamp pass, per the same lesson as every earlier
file in this series).
"""

# Example: invoice reconciliation. DB total is stale by exactly the rush
# fee ($75, ~3%) -- vendor matches, total doesn't.
PDF_DOCS = {
    "INV-58291": """INVOICE #INV-58291
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
""",
}

# Example: expense approval. Ledger claims $750, but the approver only
# signed off on $700 -- approver's title matches policy, amount doesn't.
SPREADSHEET = {
    "EXP-4471": {
        "expense_id": "EXP-4471",
        "submitter": "J. Lee",
        "amount": 750.00,
        "category": "Travel",
    },
}

EMAILS = {
    "THREAD-882": """From: mpatel@company.com
To: finance@company.com
Subject: Re: Expense approval EXP-4471

Approved. I'm signing off on this as Director for a travel expense of
$700.00 submitted by J. Lee.

-- M. Patel, Director
""",
}

# Example: vendor freshness. legal_name matches, but our internal record is
# stale on status -- the registry shows this vendor was suspended.
DATABASE = {
    "INV-58291": {
        "vendor": "Acme Robotics Corp",
        "total": 2523.54,
        "date": "2026-03-03",
    },
    "POLICY-TRAVEL": {
        "category": "Travel",
        "required_title": "Director",
        "threshold": 500.00,
    },
    "V-100": {
        "vendor_id": "V-100",
        "legal_name": "Acme Robotics Corp",
        "status": "active",
    },
}

API_RESPONSES = {
    "V-100": {
        "vendor_id": "V-100",
        "legal_name": "Acme Robotics Corp",
        "status": "suspended",
    },
}

# Example: disclosure check. The Risk Factors section describes supply-
# chain risk in substance (single-source suppliers, shipping delays, no
# redundant sourcing) without ever using that exact phrase -- a genuine
# test of judgment, not a keyword match a deterministic tool could do.
PDF_SECTIONS = {
    ("10-K-2025", "Risk Factors"): (
        "Our business depends heavily on a small number of single-source "
        "component suppliers located in one region. Any disruption to "
        "these suppliers -- due to natural disaster, geopolitical "
        "instability, or shipping delays -- could materially impact our "
        "ability to manufacture and deliver products on time. We do not "
        "currently maintain redundant sourcing arrangements."
    ),
    ("10-K-2025", "MD&A"): (
        "Revenue grew 12% year over year, driven primarily by increased "
        "unit sales in our core product line."
    ),
}
