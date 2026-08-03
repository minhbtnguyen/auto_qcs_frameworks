"""vendor_registry_match -- drift detection: has our internal record
fallen out of sync with the live source of truth.
"""

import argparse

from engine import CheckContext
from tools import call_api, query_database

PARAMS = ["vendor_id"]


def run_check(**params) -> dict:
    ctx = CheckContext()

    internal = ctx.call(query_database, save_as="internal_vendor", key=params["vendor_id"])
    external = ctx.call(call_api, save_as="external_vendor", key=params["vendor_id"])

    ctx.compare(internal["legal_name"], external["legal_name"])
    ctx.compare(internal["status"], external["status"])

    return ctx.verdict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vendor_registry_match check")
    for p in PARAMS:
        parser.add_argument(f"--{p}", required=True)
    args = parser.parse_args()

    print()
    print(f"result: {run_check(**vars(args))}")


if __name__ == "__main__":
    main()
