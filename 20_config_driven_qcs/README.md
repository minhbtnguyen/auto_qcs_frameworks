# Config-Driven QCS

A quality-control check engine where the orchestration is data (YAML), not
an LLM decision. Each check declares a fixed sequence of tool calls; a
shared engine executes that sequence deterministically, with an LLM
involved only where a step or a verdict genuinely needs judgment.

This folder is the third and final stop in a progression explored across
this repo: [`18_the_agent_loop`](../18_the_agent_loop) (ReAct — an LLM
decides every step, live), [`19_rewoo`](../19_rewoo) (ReWOO — an LLM
decides the whole plan once per run, then a deterministic worker executes
it), and this folder (the plan is decided once, ever, at authoring time,
and reused for every future run). The [Discussion](#discussion-react-vs-rewoo-vs-config-driven)
section below covers why that progression happened and when each pattern
actually fits.

## Architecture

```
description (natural language)
       |
       v
 yaml_builder.py --- one LLM call, structurally validated, --->  checks/<check_id>.yaml
                      human reviews before trusting it                   |
                                                                          v
                                                              run_<check_id>.py (CLI)
                                                                          |
                                                                          v
                                                                     engine.py
                                                              (run_check: dispatch,
                                                               resolve references,
                                                               deterministic verdict
                                                               unless one is declared)
                                                                          |
                                                                          v
                                                                     tools.py
                                                          (ALL_TOOLS registry; each tool
                                                           reads mock_data.py)
```

| File | Role |
|---|---|
| `engine.py` | Pure library. `run_check(check, params)` executes a check's `steps`, resolves `$param` / `name.field` references, computes a deterministic pass/fail unless the check declares a `verdict` block. No runnable entry point of its own. |
| `tools.py` | The tool registry (`ALL_TOOLS`). One function per data source (`read_pdf`, `get_pdf_section`, `query_database`, `read_spreadsheet`, `call_api`, `read_email`) plus one generic comparison tool (`compare_values`) and one generic inference tool (`llm_infer`). Every tool shares the signature `(**named_args, artifacts) -> (observation_text, structured_dict_or_None)`. |
| `mock_data.py` | Fixture data standing in for a real PDF store, database, spreadsheet, API, and mailbox. Deliberately separate from `tools.py` — this is the file that grows with every new test case; the tool logic shouldn't have to. |
| `yaml_builder.py` | `python yaml_builder.py "<description>"` — one LLM call proposes a check (Pydantic-structured), `_validate()` catches structural problems (unknown tool, undeclared param, dangling reference), then saves `checks/<check_id>.yaml` and generates `run_<check_id>.py` from a template — unless a runner with that name already exists, so it never clobbers a hand-edited one. |
| `checks/*.yaml` | One file per check. Each declares `params` (what the caller must supply), `steps` (tool calls and `compare` shorthands), and an optional `verdict` block. |
| `run_<check_id>.py` | Auto-generated, ~25 lines, identical shape for every check: load this check's yaml, build `--flag` CLI args from its declared `params`, call `run_check`, print the result. All the check-specific content lives in the yaml, not here. |

### Mechanics worth knowing

- **`$param` vs `name.field`.** A step's `args` values can be a literal, `$param_name` (filled from the CLI at run time), or `save_as_name.field_name` (one named field of an earlier step's structured result, resolved by `engine.py` before the tool ever sees it — uniformly, for every tool, so no tool does its own artifact lookup).
- **`compare:` is sugar**, not a separate code path — it calls the real `compare_values` tool through the same `ALL_TOOLS` entry point every other step uses, so there is exactly one implementation of comparison logic.
- **Verdict is deterministic by default**: pass iff every `compare` step matched, computed in plain code, zero LLM calls. A check pays for an LLM call only if it declares `verdict:` — for the real but minority case (see `checks/disclosure_check.yaml`) where pass/fail requires reading free text (`llm_infer`'s output) rather than aggregating comparisons.
- **A tool's docstring is load-bearing, not decoration.** `yaml_builder.py`'s tool catalog and the LLM's field-name guesses are generated straight from each tool's real Python signature (`inspect.signature`) and docstring — document a tool once, in one place, and the authoring assistant's knowledge of it updates automatically.

## Usage

Run an existing check:
```bash
python run_invoice_reconciliation.py --invoice_id=INV-58291
python run_disclosure_check.py --doc_id=10-K-2025 --section="Risk Factors"
python run_vendor_registry_match.py --vendor_id=V-100
```

Author a new one:
```bash
python yaml_builder.py "Check that expense EXP-4471's claimed amount matches what was approved in email thread THREAD-882."
```
This saves a draft `checks/<check_id>.yaml` and `run_<check_id>.py`. **Saved is not the same as trusted** — read the printed proposal (and the yaml) before running it against anything that matters. `_validate()` only catches structural mistakes (unknown tool, dangling reference); it cannot catch a semantic mismatch, such as asking for "must not exceed a threshold" when the only comparison tool available checks equality, not inequality. That specific bug happened for real while building this folder — see the Discussion below.

Add a new tool: write one function in `tools.py` with a docstring that states its real parameter names and every field it returns, add its fixture data to `mock_data.py`, add one line to `ALL_TOOLS`. Nothing else needs to change — no check references a tool it doesn't use, so adding one never touches `engine.py` or any existing check.

## Known limitations

- **No loops or conditionals in the schema.** A check needing an unknown-at-authoring-time number of steps ("reconcile however many invoices are in this folder") can't be expressed as a fixed `steps` list. A *known* count ("these 3 invoices") is fine — that's just 3 params or 3 rows, decided by whoever authors the check, not discovered at run time.
- **`compare_values` is equality-only** — no `<=`/`>=`. This is a real, open gap (see Discussion).
- **The trigger layer is still a demo.** `run_<check_id>.py` takes CLI args; nothing here yet decides *when* a check should run (a schedule, an event, a batch of IDs). That's a real design question (cron? a queue consumer? a CLI wrapper that loops over a CSV of IDs?) deliberately left open rather than guessed at.

## Discussion: ReAct vs ReWOO vs Config-Driven

The throughline across all three patterns in this repo is **where the decision "what happens next" gets made, and how many times it gets re-made.**

```
ReAct           [LLM] -> [tool] -> [LLM] -> [tool] -> [LLM] -> [tool] -> [LLM: done?]
                 one decision per step, fresh every run, needs a turn budget

ReWOO           [LLM: plan] -> [tool] -> [tool] -> [tool] -> [LLM: solve]
                 one decision per run, made before execution, fixed length

Config-Driven   [tool] -> [tool] -> [tool] -> ([LLM: verdict], only if declared)
                 zero decisions per run -- decided once, at authoring time, ever
```

| | ReAct (`18_the_agent_loop`) | ReWOO (`19_rewoo`) | Config-Driven (here) |
|---|---|---|---|
| Where "what to do" is decided | Every step, at runtime | Once per run, at runtime, before any tool executes | Once, at authoring time — reused across every future run |
| LLM calls for orchestration | 1 per turn, unbounded | Exactly 2 (planner + solver), regardless of step count | 0, plus 1 optional (`verdict`) only when judgment is genuinely needed |
| Needs a turn/step budget | Yes — real risk of looping forever | No — plan has a fixed, known length the moment it's written | No — same reason, one level further |
| Can adapt mid-task | Yes — next turn sees the last observation | No — committed before any tool runs | No — steps existed before this run's inputs did |
| Failure mode actually observed in this repo | Mis-routing, skipped mandatory steps, hallucinated dialogue (the hierarchical QCS file's original top-level supervisor) | Reference-resolution bugs (`#E1.text` not substituting, a `.` in free text colliding with `#E1.field` syntax) — mechanism bugs, not routing or judgment mistakes, because the worker has no LLM in it | Crashes on malformed config (fixed with `try/except` in `engine.py`); a semantic bug — equality used where an inequality was needed — caught only because a human read the draft before trusting it |
| Best fit | The step sequence genuinely can't be known in advance — it depends on what's discovered along the way (e.g. following a cross-reference in a 200-page filing) | The sequence is knowable from the task text alone, but the task varies enough (different phrasings, different combinations of steps) that hand-writing every variant isn't practical, and 2 LLM calls per run is an acceptable cost | The sequence is fixed and known, run repeatedly — possibly at real volume — and near-zero per-run cost matters |

### The pattern is a progression, not three unrelated choices

Every transition in this repo happened for the same reason: a decision that *looked* like it needed live judgment turned out to be constant, and paying an LLM to re-derive a constant, on every run, was pure waste plus unnecessary risk.

- The hierarchical ReAct file's top-level supervisor was an LLM choosing between `data_team` and `audit_team` on every turn — but that order was *never actually in question*. Replacing it with plain graph edges didn't lose anything; the LLM's "decision" had never been a real decision.
- ReWOO's planner looked necessary until the evidence said otherwise: across every verification run of the same check, it produced an *identical* plan structure, every time. A step that reliably reproduces the same output isn't planning, it's re-deriving a constant at LLM cost, LLM latency, and (concretely, twice in this repo) LLM error risk.
- Config-driven is what's left once that constant is written down once instead of re-derived: the same plan, minus the LLM call that kept confirming it hadn't changed.

**A practical way to use all three together**: use ReAct (or a ReWOO planner) to *discover* what a new check's steps should be, especially when the shape isn't obvious yet. Once the planner keeps reproducing the same structure run after run — which is easy to notice, exactly as it was noticed here — that repetition is the signal the task graduated from "needs judgment about what to do" to "the *what* is already known, only the input values change." At that point, moving it into a `checks/<check_id>.yaml` (by hand, or by running `yaml_builder.py` once and reviewing the result) removes the planning LLM call from the critical path entirely — not because capability was sacrificed, but because that call was never buying anything once the plan stopped varying.

### Where config-driven's own ceiling is

None of this makes config-driven strictly "the end state" — it has the narrowest expressive range of the three by design. It cannot express genuine branching or an unknown-in-advance loop count without either folding that logic into a tool's own code (fine, if the loop body is uniform and judgment-free) or dropping back to ReAct/ReWOO for that specific check. And the equality-only `compare_values` gap is a real, live example: a check asking for "must not exceed a threshold" silently got mapped onto the only comparison primitive available, because the schema had no slot for an inequality — the same mistake a human would never make writing `if amount > threshold` in three seconds of plain Python, precisely because Python has no such ceiling. That's the honest trade config-driven makes: it buys determinism and near-zero orchestration cost by giving up the ability to express anything the schema wasn't built to say.

The other open question is who is actually authoring the checks. Config-driven's real advantage — beyond removing the runtime LLM call — is that a check becomes readable and writable by someone who has never written Python. If that person exists in practice, this is the right layer. If the checks are all being written by engineers anyway, plain Python functions calling `tools.py` directly, through the same kind of shared harness `engine.py` already provides, are likely simpler *and* more powerful, with real IDE tooling, real stack traces, and no structural ceiling — and the yaml layer becomes indirection that isn't paying for itself.
