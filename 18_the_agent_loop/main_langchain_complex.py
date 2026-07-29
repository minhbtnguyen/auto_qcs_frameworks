"""Multi-agent ReAct system — LangChain/LangGraph + real Claude.

A supervisor routes to narrow-scoped specialist subagents instead of one agent
holding every tool:
  - trader:   bound only to `trade`      -> executes portfolio buy/sell orders
  - analyst:  bound only to `calculator` -> does arithmetic (profit, valuation)

Two different questions, two different answers on raw text vs. structure:
  - Specialists reading the conversation so far: raw messages are fine -- same
    reasoning as the single-agent case, it's still "one LLM re-reading a shared
    history."
  - The supervisor's routing decision: needs structure. `next` drives an actual
    conditional edge (`route_supervisor`), so it has to be a reliable value the
    graph can dispatch on, not prose another LLM call would have to interpret.
    The supervisor's structured output also carries the final synthesized
    answer (`final_message`) for the same reason: nobody else in this graph is
    responsible for writing a holistic summary, since each specialist only ever
    sees its own narrow task.

Specialists are single-shot per dispatch: they act (or don't), then always hand
control back to the supervisor. All looping happens at the supervisor -- no
nested per-specialist loops -- which keeps the graph flat and easy to trace.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()

MAX_TURNS = 10
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")


# ------------------------------
# Tools, split by specialty
# ------------------------------
@tool
def calculator(expr: str) -> str:
    """Evaluate a basic arithmetic expression using +, -, *, /, and parentheses."""
    allowed = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed):
        return "error: illegal character in expr"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


_portfolio: dict[str, int] = {"AAPL": 10}


@tool
def trade(ticker: str, shares: int) -> str:
    """Buy or sell shares of a ticker in the portfolio. Pass a positive number of
    shares to buy, a negative number to sell, or 0 to just check the current
    holding. Returns the new share count for that ticker. Buying a ticker not
    yet held opens a new position; selling more shares than are held errors."""
    ticker = ticker.upper().strip()
    current = _portfolio.get(ticker, 0)
    new_count = current + shares
    if new_count < 0:
        return f"error: cannot sell {-shares} shares of {ticker}; only {current} held"
    _portfolio[ticker] = new_count
    return f"{ticker}: {new_count} shares held"


TOOLS = [calculator, trade]


# ------------------------------
# State
# ------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: int
    next: str


# ------------------------------
# Supervisor: structured routing decision + (when done) the final answer
# ------------------------------
class Route(BaseModel):
    next: Literal["trader", "analyst", "end"]
    reason: str = Field(description="One sentence on why this specialist goes next")
    final_message: str | None = Field(
        default=None,
        description=(
            "Required when next='end': a complete answer to the user's original "
            "question, synthesizing everything the specialists found. Null otherwise."
        ),
    )


SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating two specialists:\n"
    "- trader: can buy or sell shares in the portfolio (owns the `trade` tool)\n"
    "- analyst: can do arithmetic, e.g. computing profit or valuation (owns the "
    "`calculator` tool)\n\n"
    "Given the conversation so far, decide which specialist should act next. "
    "Route to 'end' only once the user's original question has been fully "
    "answered -- and when you do, write the complete final answer yourself in "
    "final_message, since neither specialist sees the full picture."
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
    if route.next == "end" and route.final_message:
        update["messages"] = [AIMessage(route.final_message)]
    return update


def route_supervisor(state: State) -> str:
    if state["turns"] >= MAX_TURNS:
        return "budget_exhausted"
    return state["next"]


# ------------------------------
# Specialists: single-shot, bound to only their own tool
# ------------------------------
trader_llm = ChatAnthropic(model=MODEL, max_tokens=512).bind_tools([trade])
analyst_llm = ChatAnthropic(model=MODEL, max_tokens=512).bind_tools([calculator])


def trader_node(state: State) -> dict:
    return {"messages": [trader_llm.invoke(state["messages"])]}


def analyst_node(state: State) -> dict:
    return {"messages": [analyst_llm.invoke(state["messages"])]}


def after_specialist(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "supervisor"


def budget_exhausted_node(state: State) -> dict:
    return {"messages": [AIMessage("budget exhausted")]}


tool_node = ToolNode(tools=TOOLS)

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("trader", trader_node)
graph.add_node("analyst", analyst_node)
graph.add_node("tools", tool_node)
graph.add_node("budget_exhausted", budget_exhausted_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "trader": "trader",
        "analyst": "analyst",
        "end": END,
        "budget_exhausted": "budget_exhausted",
    },
)
graph.add_conditional_edges(
    "trader", after_specialist, {"tools": "tools", "supervisor": "supervisor"}
)
graph.add_conditional_edges(
    "analyst", after_specialist, {"tools": "tools", "supervisor": "supervisor"}
)
graph.add_edge("tools", "supervisor")
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
    """Render the trace using main.py's own vocabulary (user/thought/action/final).
    Doesn't distinguish which specialist produced a given AI message -- every
    AIMessage prints the same way regardless of whether trader, analyst, or the
    supervisor's final_message wrote it. The `[supervisor] -> ...` routing lines
    printed live during the run (not part of this replay) are what show you who
    acted at each step.
    """
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
    node's input (accumulated state it received) and output (the partial
    update it returned) instead of just invoking straight through."""
    initial = {"messages": [HumanMessage(user_message)], "turns": 0, "next": ""}

    if not trace:
        return app.invoke(initial)

    state = {"messages": list(initial["messages"]), "turns": 0, "next": ""}
    for update in app.stream(initial, stream_mode="updates"):
        for node_name, partial in update.items():
            print("=" * 70)
            print(f"NODE: {node_name}")
            print("-" * 70)
            print("INPUT  (accumulated state this node received):")
            print(f"  turns = {state['turns']}, next = {state['next']!r}")
            print(f"  messages ({len(state['messages'])}):")
            for i, msg in enumerate(state["messages"]):
                print(f"    [{i}] {msg.type}: {str(msg.content)[:90]!r}")

            print("\nOUTPUT (partial update this node returned):")
            for key, value in partial.items():
                if key == "messages":
                    for msg in value:
                        print(f"  messages += {msg.type}: {str(msg.content)[:120]!r}")
                else:
                    print(f"  {key} = {value}")
            print()

            if "messages" in partial:
                state["messages"] = state["messages"] + partial["messages"]
            if "turns" in partial:
                state["turns"] = partial["turns"]
            if "next" in partial:
                state["next"] = partial["next"]

    print("=" * 70)
    print("FINAL RETAINED STATE:")
    print(f"  turns = {state['turns']}, next = {state['next']!r}")
    print(f"  messages = {len(state['messages'])} total")

    return state


def main() -> None:
    print("=" * 70)
    print("MULTI-AGENT REACT — Supervisor + Trader + Analyst")
    print("=" * 70)

    print("\ngraph structure (Mermaid):")
    print(app.get_graph().draw_mermaid())

    trace = "--trace" in sys.argv
    if trace:
        print("\n--trace enabled: showing per-node input/output\n")

    result = run_agent(
        "I already hold 10 shares of AAPL. Buy 15 more shares at "
        "$180 each, then sell 5 shares at $195 each. What's the "
        "profit on the shares I sold, and how many AAPL shares do "
        "I hold now?",
        trace=trace,
    )
    messages = result["messages"]
    final = messages[-1].content

    print()
    pretty_trace(messages)
    print()
    print(f"final answer: {final}")
    print(f"turns used:   {result['turns']}")
    print(f"tools used:   {sorted(t.name for t in TOOLS)}")


if __name__ == "__main__":
    main()
