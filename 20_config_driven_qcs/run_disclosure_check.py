"""disclosure_check -- llm_infer's output is free text, not a value to
compare, so there's nothing to aggregate -- pass/fail itself needs
judgment. One LLM call for the verdict, paid only because this check
declares one, not by every check.
"""

import argparse

from engine import CheckContext
from tools import get_pdf_section, llm_infer

PARAMS = ["doc_id", "section"]

VERDICT_INSTRUCTIONS = (
    "Based on the evidence gathered, render a pass/fail verdict on whether "
    "the disclosure is adequate, with reasoning."
)


def run_check(**params) -> dict:
    ctx = CheckContext()

    section = ctx.call(
        get_pdf_section, save_as="section_data", doc_id=params["doc_id"], section=params["section"]
    )
    ctx.call(
        llm_infer,
        save_as="analysis",
        text=section["text"],
        instructions=(
            "Does this adequately disclose supply chain risk -- specifically "
            "dependency on single-source suppliers and risk of sourcing "
            "disruption? It does not need to use the exact phrase 'supply "
            "chain', but must substantively describe these risks."
        ),
    )

    return ctx.llm_verdict(VERDICT_INSTRUCTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the disclosure_check check")
    for p in PARAMS:
        parser.add_argument(f"--{p}", required=True)
    args = parser.parse_args()

    print()
    print(f"result: {run_check(**vars(args))}")


if __name__ == "__main__":
    main()
