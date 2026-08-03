"""ReWOO, stripped to the bare pattern -- no invoice domain, no anti-
hallucination prompt engineering, just planner -> worker -> solver.

    "What is the combined population of the capital of France and the
    capital of Japan?"

None of the three tools below can answer that alone. The plan has to chain
them: find each capital, then look up each capital's population, then add
the two numbers -- and the calculator step genuinely can't be written until
the lookups have run, since nobody knows the populations in advance. That's
the one thing main_langchain_complex_qcs_rewoo.py's tools don't really
exercise (they all just take an invoice_id, so evidence substitution is
never load-bearing there). Here it is: the planner writes
`calculator[#E3 + #E4]` without knowing what #E3 or #E4 will turn out to be,
and the worker substitutes the real numbers in only once they exist.

  - planner: 1 LLM call. Writes the whole plan (which tool, in what order,
    with what input) before any tool runs.
  - worker: 0 LLM calls. Plain code -- for each step, substitute any #E_n
    references in tool_input with real evidence text, call the named tool,
    record the result.
  - solver: 1 LLM call. Reads the original question plus every step's
    evidence and answers -- told explicitly to use only that evidence, not
    its own world knowledge, so the answer is traceable to what was
    actually looked up.

For the fuller, real-domain version of this same pattern (Pydantic records,
deterministic reconciliation, exhaustive tool-return text to prevent
fabrication), see main_langchain_complex_qcs_rewoo.py in this folder.

Questions
- What if complex task? Worker as executioner?
- How to productionize planner?
- The steps make sense but isolated? What if the output of step 2 depending on step 1? Calculator is equate from LLMs not from the tool's funcition
- Is this a good production code template structure? 
- 
"""

import os
from typing import Literal

from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


# ------------------------------
# Fake world data + tools. Plain functions, all the same (str) -> str
# shape -- no @tool decorator, no ToolNode: nothing here ever needs an LLM
# to decide which one to call, since the planner already decided that.
# ------------------------------
_CAPITALS = {"France": "Paris", "Japan": "Tokyo"}
_POPULATIONS = {"Paris": 2_148_000, "Tokyo": 13_960_000}


def lookup_capital(country: str) -> str:
    capital = _CAPITALS.get(country.strip())
    return capital if capital else f"error: no capital known for {country!r}"


def lookup_population(city: str) -> str:
    population = _POPULATIONS.get(city.strip())
    return str(population) if population else f"error: no population known for {city!r}"


def calculator(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed):
        return "error: illegal character in expr"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


TOOLS = {
    "lookup_capital": lookup_capital,
    "lookup_population": lookup_population,
    "calculator": calculator,
}


# ------------------------------
# Plan: the planner's entire output. tool_input may textually contain an
# earlier step's evidence_var -- the worker resolves that before calling.
# ------------------------------
class PlanStep(BaseModel):
    plan: str = Field(description="One sentence: what this step does and why")
    evidence_var: str = Field(description='Evidence variable for this result, e.g. "#E1"')
    tool: Literal["lookup_capital", "lookup_population", "calculator"]
    tool_input: str = Field(
        description=(
            "The input to pass this tool. May reference an earlier step's "
            "evidence_var (e.g. #E1) instead of a literal value, if this "
            "step genuinely needs that step's result."
        )
    )


class Plan(BaseModel):
    steps: list[PlanStep]


class State(TypedDict):
    task: str
    plan: Plan | None
    evidence: dict[str, str]
    answer: str | None


# ------------------------------
# Planner -- the only place any reasoning about STRATEGY happens.
# ------------------------------
PLANNER_PROMPT = (
    "You are a planner. Given a question, write a complete, ordered plan "
    "BEFORE any tool runs -- you will not see any tool results while "
    "planning, so the plan must stand on its own.\n\n"
    "Available tools:\n"
    "- lookup_capital[country]: returns the capital city of a country\n"
    "- lookup_population[city]: returns a city's population\n"
    "- calculator[expr]: evaluates an arithmetic expression\n\n"
    "If a step needs an earlier step's result, reference its evidence_var "
    "(e.g. #E1) in tool_input instead of guessing the value yourself."
)

planner_llm = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(Plan)


def planner_node(state: State) -> dict:
    plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(state["task"])])
    print("[planner] plan:")
    for step in plan.steps:
        print(f"  {step.evidence_var} = {step.tool}[{step.tool_input}]  # {step.plan}")
    return {"plan": plan}


# ------------------------------
# Worker -- the only place execution happens. No reasoning, no choices.
# ------------------------------
def _substitute(text: str, evidence: dict[str, str]) -> str:
    for var, value in evidence.items():
        text = text.replace(var, value)
    return text


def worker_node(state: State) -> dict:
    evidence: dict[str, str] = {}
    for step in state["plan"].steps:
        tool_input = _substitute(step.tool_input, evidence)
        observation = TOOLS[step.tool](tool_input)
        print(f"[worker] {step.evidence_var} = {step.tool}[{tool_input}] -> {observation}")
        evidence[step.evidence_var] = observation
    return {"evidence": evidence}


# ------------------------------
# Solver -- the only place the gathered evidence gets interpreted.
# ------------------------------
SOLVER_PROMPT = (
    "Answer the original question using ONLY the evidence below -- do not "
    "use any outside knowledge, even if you already know the answer."
)


def _render_plan_and_evidence(state: State) -> str:
    lines = [f"Question: {state['task']}", "", "Plan and evidence:"]
    for step in state["plan"].steps:
        observation = state["evidence"][step.evidence_var]
        lines.append(f"{step.evidence_var} = {step.tool}[{step.tool_input}] -> {observation}")
    return "\n".join(lines)


solver_llm = ChatAnthropic(model=MODEL, max_tokens=256)


def solver_node(state: State) -> dict:
    context = _render_plan_and_evidence(state)
    response = solver_llm.invoke([SystemMessage(SOLVER_PROMPT), HumanMessage(context)])
    print(f"[solver] -> {response.content}")
    return {"answer": response.content}


# ------------------------------
# Graph: planner -> worker -> solver -> END. Straight line, no conditional
# edges -- the plan's length is fixed the moment the planner returns.
# ------------------------------
graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("worker", worker_node)
graph.add_node("solver", solver_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "worker")
graph.add_edge("worker", "solver")
graph.add_edge("solver", END)

app = graph.compile()


def run_agent(question: str) -> dict:
    initial: State = {"task": question, "plan": None, "evidence": {}, "answer": None}
    return app.invoke(initial)


def main() -> None:
    print("=" * 70)
    print("SIMPLEST REWOO EXAMPLE — Plan -> Work -> Solve")
    print("=" * 70)

    print("\ngraph structure (Mermaid):")
    print(app.get_graph().draw_mermaid())
    print()

    result = run_agent(
        "What is the combined population of the capital of France and the "
        "capital of Japan?"
    )

    print()
    print(f"answer: {result['answer']}")


if __name__ == "__main__":
    main()
