# Config-Driven QCS

A quality-control check engine where the *sequence of steps* is never an
LLM's runtime decision. Each check is a short, real Python function calling
a shared tool registry; an LLM is involved only where a step or a verdict
genuinely needs judgment — and, optionally, once at authoring time to draft
a new check for a human to review.

This folder is the third stop in a progression explored across this repo:
[`18_the_agent_loop`](../18_the_agent_loop) (ReAct — an LLM decides every
step, live), [`19_rewoo`](../19_rewoo) (ReWOO — an LLM decides the whole
plan once per run, then a deterministic worker executes it), and this
folder (the plan is decided once, ever, at authoring time, and reused for
every future run). See [Discussion](#discussion-react-vs-rewoo-vs-config-driven)
for why that progression happened and when each pattern fits.

**This folder itself went through a second, smaller evolution worth naming
up front**: it originally expressed checks as YAML data, interpreted by a
generic engine. That's gone. Checks are real Python now — see
[How this folder changed](#how-this-folder-changed) for exactly what that
traded away and what it bought back.

## Why not ReAct, ReWOO, or the original Config-Driven (YAML)?

Three earlier architectures were built and actually run in this repo before
landing here — each rejected for a specific, observed reason, not on
general principle. See [Discussion](#discussion-react-vs-rewoo-vs-config-driven)
below for the full comparison; this is the short version.

**ReAct** (`18_the_agent_loop`) — an LLM decides every step, live, one at a
time. Rejected because the step sequence for a given QC check is always the
same; a live per-step decision pays LLM cost, LLM latency, and needs a turn
budget to guard against looping forever, for a choice that was never
actually being made fresh each run. It also produced the worst failure
modes actually observed anywhere in this repo: mis-routing, skipped
mandatory steps, hallucinated dialogue that was never really sent.

**ReWOO** (`19_rewoo`) — an LLM plans the whole sequence once per run
(exactly 2 LLM calls, planner + solver), then a deterministic worker
executes it, no turn budget needed. A real improvement over ReAct, but
still rejected: across every verification run of the same check, the
planner reproduced an *identical* plan structure every time. That's not
planning, it's re-deriving a constant at LLM cost and LLM error risk — this
repo hit two separate reference-resolution bugs from exactly that
mechanism — for a decision that was never varying run to run.

**Config-Driven, YAML edition** (this folder's own earlier state) — removed
the redundant re-planning entirely: the sequence is decided once, at
authoring time, and reused for free on every future run. This was the
correct fix for ReWOO's actual problem, and it's still the shape of what's
here now. But being *data* meant being bounded by whatever the schema
anticipated — `compare_values` could only ever check equality, and a check
asking for "must not exceed a threshold" got silently, wrongly mapped onto
it, because the format had no way to say anything else. For a team of
developers — not the non-programmer associates the YAML format existed to
serve — that ceiling was a real, paid cost buying a benefit nobody was
actually using.

What's here now keeps YAML's actual win (zero orchestration LLM calls,
decided once at authoring time, never re-derived) and removes YAML's
ceiling (real Python, real `if`/`for` available the moment a check needs
them) — at the cost of a narrower, more honestly-scoped LLM-authoring
benefit than YAML had, covered in
[Is `check_builder.py` actually worth it](#is-check_builderpy-actually-worth-it-given-developers-maintain-this)
further down.

## Architecture

```
description (natural language)
       |
       v
 check_builder.py --- one LLM call, extracts a validated  --->  run_<check_id>.py
                       CheckDefinition (same schema as             (real Python:
                       before), _validate()'s the same              PARAMS + a
                       structural mistakes it always did,            run_check()
                       renders it as Python, human reviews            function)
                       before trusting it                                |
                                                                          v
                                                                     engine.py
                                                                  (CheckContext:
                                                                   call/compare/
                                                                   verdict helpers)
                                                                          |
                                                                          v
                                                                     tools.py
                                                          (ALL_TOOLS registry; each tool
                                                           reads mock_data.py)
```

| File | Role |
|---|---|
| `engine.py` | `CheckContext` — the only shared infrastructure a check needs. `.call(tool, save_as=..., **kwargs)` runs a tool, logs it, saves its result under a name. `.compare(left, right)` runs the one shared comparison primitive and feeds the verdict rule. `.verdict()` is the deterministic pass-iff-everything-matched rule (0 LLM calls); `.llm_verdict(instructions)` is the opt-in judgment call for checks that need it (1 LLM call). No interpretation loop, no runnable entry point. |
| `tools.py` | The tool registry (`ALL_TOOLS`). One function per data source (`read_pdf`, `get_pdf_section`, `query_database`, `read_spreadsheet`, `call_api`, `read_email`) plus `compare_values` and `llm_infer`. Every tool shares the signature `(**named_args, artifacts) -> (observation_text, structured_dict_or_None)`. |
| `mock_data.py` | Fixture data standing in for a real PDF store, database, spreadsheet, API, and mailbox — kept separate from `tools.py` since this is the file that grows with every new test case. |
| `check_builder.py` | `python check_builder.py "<description>"` — one LLM call extracts a `CheckDefinition` (Pydantic-structured, unchanged from the YAML era), `_validate()` catches structural problems, then `_to_python_script()` renders it as a real `run_<check_id>.py` file — unless one already exists, so it never clobbers a hand-edited check. |
| `run_<check_id>.py` | A complete, standalone check: `PARAMS: list[str]`, a `run_check(**params) -> dict` function built from `CheckContext`, and a standard argparse CLI wrapper. Nothing else needs to exist for a check to run — no separate data file. |

### Mechanics worth knowing

- **A check body is genuinely readable Python**, not data being interpreted:
  ```python
  def run_check(**params) -> dict:
      ctx = CheckContext()
      pdf = ctx.call(read_pdf, save_as="pdf", doc_id=params["invoice_id"])
      db = ctx.call(query_database, save_as="db", key=params["invoice_id"])
      ctx.compare(pdf["vendor"], db["vendor"])
      ctx.compare(pdf["total"], db["total"])
      return ctx.verdict()
  ```
  `read_pdf`/`query_database` are real imported functions — jump-to-definition, real stack traces, real type hints all work, none of which a YAML string `tool: read_pdf` could ever give you.
- **Verdict is deterministic by default**: `ctx.verdict()` passes iff every `ctx.compare()` call matched, zero LLM calls. A check only pays for an LLM call by explicitly calling `ctx.llm_verdict(instructions)` instead — for the real but minority case (see `run_disclosure_check.py`) where pass/fail requires reading free text (`llm_infer`'s output) rather than aggregating comparisons.
- **A tool's docstring is still load-bearing.** `check_builder.py`'s tool catalog and the LLM's field-name guesses are generated straight from each tool's real Python signature (`inspect.signature`) and docstring — document a tool once, and the authoring assistant's knowledge of it updates automatically. Unchanged from the YAML era.
- **The `PARAMS` + `run_check(**params)` convention is what keeps checks batch-introspectable** despite being free code now instead of validated data — a scheduler can `import run_invoice_reconciliation; run_invoice_reconciliation.PARAMS` without executing anything past that. It's enforced by convention now, not by a schema, which is the trade this migration made — see below.

## Usage

Run an existing check:
```bash
python run_invoice_reconciliation.py --invoice_id=INV-58291
python run_disclosure_check.py --doc_id=10-K-2025 --section="Risk Factors"
python run_vendor_registry_match.py --vendor_id=V-100
```

Author a new one:
```bash
python check_builder.py "Check that expense EXP-4471's claimed amount matches what was approved in email thread THREAD-882."
```
This saves `run_<check_id>.py` directly — no separate data file. **Saved is
not trusted** — read the printed proposal (and the generated file itself)
before running it against anything that matters. `_validate()` only catches
structural mistakes (unknown tool, undeclared param, dangling reference);
it cannot catch a semantic mismatch, such as asking for "must not exceed a
threshold" when the only comparison primitive available checks equality,
not inequality. That specific bug happened for real while building this
folder, in the YAML era — see Discussion below. **The difference now**: if
you hit that bug today, you fix it by editing the one check's Python
directly (`if float(pdf["total"]) > threshold: ...`), not by extending a
schema everything else depends on.

Add a new tool: write one function in `tools.py` with a docstring stating
its real parameter names and every field it returns, add fixture data to
`mock_data.py`, add one line to `ALL_TOOLS`. Nothing else needs to change.

## How this folder changed

The original version had each check as a `checks/<check_id>.yaml`, executed
by `engine.py` walking its steps generically — a `compare` step, a `$param`
reference, a `name.field` reference were all resolved at *run time* by an
interpreter. That's gone. What forced the change: a check asking for "must
not exceed a threshold" got silently mapped onto the only comparison
primitive the schema had (`compare_values`, equality-only), because the
data format had no way to say anything else — a mistake real Python would
never make, since `if amount > threshold` costs nothing extra and needs no
schema to permit it.

**What was preserved, deliberately:** `check_builder.py` still extracts a
validated `CheckDefinition` — the exact same Pydantic schema, the exact
same `_validate()` structural checks — before ever writing a file. This
matters because it's a real, load-bearing decision, not an accident: an
LLM asked to produce free-form code with no structural constraint is a
fundamentally bigger trust boundary than one asked to fill in a typed,
mechanically-validated form. Going all the way to free-form generation
would have removed the *last* real limitation (no conditionals, no loops)
at the cost of `_validate()`'s guarantees entirely — worth knowing as a
deliberate line that was drawn, not a limitation nobody considered. If a
future check genuinely needs the LLM itself to reason in open-ended code,
that's the next fork, and it's a bigger one than this migration was.

**A bug worth remembering, because it repeated:** the reference-resolution
logic (turning `#E1.field`-style, then `name.field`-style, references into
real values) has now broken the *identical* way twice in this repo's
history — once in `engine.py`'s YAML-era `_resolve_value`, and again in
`check_builder.py`'s `_render_value_expr` while building this very
migration, because the fix wasn't carried over to the new function. Both
times: a naive `"." in value` check misfired on free text that happened to
contain a period (an `llm_infer` instructions sentence), treating a whole
sentence as a `name.field` reference. Both times the fix was the same —
anchor the match to the whole string (`^(\w+)\.(\w+)$`) so a sentence with
spaces and punctuation can never match. Worth internalizing as a general
lesson about this kind of code: the fix belongs to the *pattern*, not to
the one function it was first found in.

## Known limitations

- **`check_builder.py`'s output is still bounded by `CheckDefinition`'s schema** — a freshly generated check still can't express a comparison operator beyond equality, or a loop over an unknown-in-advance count. What changed is that this ceiling is now escapable *per check*, by hand-editing the generated Python directly, without touching shared infrastructure. It's still real for anything nobody has hand-edited yet.
- **`compare_values` itself is still equality-only** — no `<=`/`>=` primitive exists yet. Any check needing one has to inline the comparison itself rather than call `ctx.compare()`.
- **The trigger layer is still a demo.** `run_<check_id>.py` takes CLI args; nothing here decides *when* a check should run (a schedule, an event, a batch of IDs). Left open deliberately rather than guessed at.
- **Every check duplicates its own CLI boilerplate.** In the YAML era, `run_<check_id>.py` was a generated, uniform template — changing CLI behavior meant editing one file. Now each check's `argparse` block is literal, repeated text in every file: 300 checks means 300 near-identical copies, and a CLI-behavior change means editing all of them, or writing a codemod. This is a real scaling regression the migration introduced, not a hypothetical one. The fix: pull the CLI entry point back into `engine.py` as a shared runner that imports `PARAMS`/`run_check` from whichever check module it's pointed at, instead of every check regenerating that logic.
- **`_validate()` runs exactly once, at generation time, and never again.** The moment a developer hand-edits a check — which will happen constantly, since escaping the comparison-operator ceiling is the whole point of being in Python — there is no structural guardrail left. A developer who forgets to call `ctx.compare()` for a field they meant to check gets no error; the check silently checks less than intended and keeps passing, because `ctx.verdict()` only ever sees whatever comparisons actually ran. YAML's declarative `steps:` list made "did I check everything I meant to" verifiable at a glance; free-form Python is harder to audit that way by construction — the direct cost of the expressiveness this migration bought.
- **No tests exist yet against any `run_check()` function**, despite being trivially unit-testable. The capability is real; the discipline of using it hasn't been exercised in this repo.

## Is `check_builder.py` actually worth it, given developers maintain this?

Yes, but the honest margin is much narrower than "use an LLM to write your
checks" makes it sound, and it's worth being precise about exactly what the
narrow part is, since a vague "yes" here would be a worse answer than the
specific one.

Split "faster/scalable/reliable/maintainable" apart rather than answering
in the aggregate:

- **Faster** — only for producing the first draft of a *new* check.
  `check_builder.py` never touches a file that already exists, so it does
  nothing for review (which needs the same scrutiny as reviewing any code,
  LLM-written or not) and nothing for maintenance.
- **Scalable** — real for the shared parts (`CheckContext`, `tools.py`), but
  see the CLI-boilerplate-duplication limitation above: that specific
  regression came directly *from* moving to free-form Python and hasn't
  been fixed yet.
- **Reliable** — strong once a check is written correctly; weaker than it
  looks during the maintenance window, since `_validate()`'s guarantees
  don't survive a single hand-edit (see above).
- **Maintainable** — genuinely true for developers: real diffs, real git
  blame, real IDE refactoring, real pytest-ability. This is the one
  dimension where the answer isn't qualified.

**What's actually different between "draft with `check_builder.py`, then
maintain by hand" and "write the check by hand from the start"?** Less than
it first appears. Some of what `_validate()` catches — an unknown tool
name, a reference to an undefined variable — a developer writing Python
directly in a real IDE gets for free, in real time, faster than running any
validator; that part of `_validate()` only earns its keep because LLM
generation can introduce that specific class of mistake in a way a human
typing with IDE autocomplete naturally doesn't. But one check `_validate()`
does is *not* redundant with IDE tooling either way: whether a `$param`
used inside the function body was actually declared in `PARAMS`. That's
plain string-keyed dict access (`params["invoice_id"]`) — no type checker
catches a typo there, hand-written or generated, until it's a runtime
`KeyError`. That specific check is a general-purpose lint that happens to
currently live only inside `check_builder.py`, gated to the moment of
generation, which means it provides zero ongoing value to a check that's
since been edited — or to any check that was hand-written from the start.
Pulling just that check out into a standalone script that runs against any
`run_*.py` file, generated or not, ideally in CI, would close this gap
entirely. At that point the honest, remaining difference between the two
workflows is exactly one thing: how fast you want the first draft to
appear. Not authoring, not review, not correctness, not reliability — draft
speed, and nothing else.

## Discussion: ReAct vs ReWOO vs Config-Driven

The throughline across all three patterns in this repo is **where the
decision "what happens next" gets made, and how many times it gets
re-made.**

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
| LLM calls for orchestration | 1 per turn, unbounded | Exactly 2 (planner + solver), regardless of step count | 0, plus 1 optional (`.llm_verdict()`) only when judgment is genuinely needed |
| Needs a turn/step budget | Yes — real risk of looping forever | No — plan has a fixed, known length the moment it's written | No — same reason, one level further |
| Can adapt mid-task | Yes — next turn sees the last observation | No — committed before any tool runs | No — steps existed before this run's inputs did |
| Failure mode actually observed in this repo | Mis-routing, skipped mandatory steps, hallucinated dialogue (the hierarchical QCS file's original top-level supervisor) | Reference-resolution bugs (`#E1.text` not substituting, a `.` in free text colliding with `#E1.field` syntax) — mechanism bugs, not routing or judgment mistakes | The same reference-resolution bug class recurring in `check_builder.py`'s Python renderer (see above); a semantic bug (equality used where an inequality was needed), caught only because a human read the draft before trusting it |
| Best fit | The step sequence genuinely can't be known in advance — it depends on what's discovered along the way | The sequence is knowable from the task text, but the task varies enough that hand-writing every variant isn't practical, and 2 LLM calls per run is acceptable | The sequence is fixed and known, run repeatedly — and the authors are comfortable reading and debugging Python |

### The pattern is a progression, not three unrelated choices

Every transition in this repo happened for the same reason: a decision
that *looked* like it needed live judgment turned out to be constant, and
paying an LLM to re-derive a constant on every run was pure waste plus
unnecessary risk. The hierarchical ReAct file's supervisor was choosing
between two teams on every turn, for an order that was never actually in
question. ReWOO's planner reliably reproduced the *identical* plan
structure across every verification run of the same check — not planning,
re-deriving a constant at LLM cost and LLM error risk. Config-driven is
what's left once that constant is written down once instead of re-derived.

**A practical way to use all three together**: use ReAct (or a ReWOO
planner) to *discover* what a new check's steps should be. Once the same
structure keeps reproducing run after run, that repetition is the signal
the task graduated from "needs judgment about what to do" to "the *what* is
already known." At that point, `check_builder.py` (or hand-writing the
function directly) removes the planning LLM call from the critical path
entirely — not because capability was sacrificed, but because that call was
never buying anything once the plan stopped varying.

### Data vs. code turned out to be the wrong axis; LLM vs. human turned out to be a smaller question than it looked

Two things were worth working out explicitly, because they weren't obvious
going in and the answer reshaped this folder:

**"YAML vs. Python" is really "schema-validated structure vs. free-form
code," and that axis is independent of who or what writes the check.** A
schema gives you a guarantee — no unknown tool, no dangling reference —
that holds regardless of whether a human or an LLM produced the values.
Free-form code has no such guarantee either way; reviewing LLM-written
Python takes exactly the scrutiny reviewing hand-written Python would. This
folder deliberately stayed on the schema-validated side of that axis (see
[How this folder changed](#how-this-folder-changed)) while moving the
*rendering* from YAML to Python — which is why `_validate()` still means
something here, and why the equality-vs-threshold bug is a *known,
documented* ceiling rather than a silent one.

**Once code is genuinely free-form, "LLM-authored vs. hand-written" stops
being an architectural question and becomes a personal-workflow one** —
did you dictate the function or type it. At that point a dedicated builder
tool's value shrinks to convenience (a consistent prompt template,
automatic file placement); the thing that made it more than "a prompt" —
the schema — is gone by construction. That's *not* the state this folder
ended up in, deliberately: `check_builder.py` kept the schema, which is
exactly why it's still worth having as a distinct tool rather than just
asking a general coding assistant to write the function.

**The other open, load-bearing question is who actually authors these
checks.** Config-driven's real advantage beyond removing the runtime LLM
call is that a check stays *reviewable* by someone who reads Python
carefully but wouldn't write it fluently from a blank file — the generated
first draft plus `_validate()`'s structural guarantee lowers the bar for
that reviewer specifically. If every check is written and read exclusively
by fluent engineers, the value of `check_builder.py` narrows to "saves
typing," which is real but far smaller than the case it was built for.
