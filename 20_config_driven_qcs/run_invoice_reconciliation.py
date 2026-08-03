"""Auto-generated runner for the invoice_reconciliation check. Loads
checks/invoice_reconciliation.yaml and runs it against CLI-supplied params -- edit
the yaml to change the check itself, not this file.
"""

import argparse
from pathlib import Path

import yaml

from engine import run_check

CHECK_FILE = Path(__file__).parent / "checks" / "invoice_reconciliation.yaml"


def main() -> None:
    with open(CHECK_FILE) as f:
        check = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Run the invoice_reconciliation check")
    for p in check.get("params", []):
        parser.add_argument(f"--{p}", required=True)
    args = parser.parse_args()

    result = run_check(check, vars(args))
    print()
    print(f"result: {result}")


if __name__ == "__main__":
    main()
