"""invoice_reconciliation -- pure-compare shape: pass iff both comparisons
match, computed in plain code, zero LLM calls for orchestration (read_pdf
still makes one LLM call internally, but that's extraction needing
judgment, not a routing decision).
"""

import argparse

from engine import CheckContext
from tools import query_database, read_pdf

PARAMS = ["invoice_id"]


def run_check(**params) -> dict:
    ctx = CheckContext()

    pdf = ctx.call(read_pdf, save_as="pdf", doc_id=params["invoice_id"])
    db = ctx.call(query_database, save_as="db", key=params["invoice_id"])

    ctx.compare(pdf["vendor"], db["vendor"])
    ctx.compare(pdf["total"], db["total"])

    return ctx.verdict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the invoice_reconciliation check")
    for p in PARAMS:
        parser.add_argument(f"--{p}", required=True)
    args = parser.parse_args()

    print()
    print(f"result: {run_check(**vars(args))}")


if __name__ == "__main__":
    main()
