"""ReAct agent loop — LangChain/LangGraph + real Claude.

Same five ingredients as main.py, mapped onto framework primitives instead of
hand-rolled ones:
  1. message buffer      -> State.messages (LangGraph's add_messages reducer)
  2. tool registry        -> @tool-decorated functions + ToolNode
  3. stop condition       -> conditional edge: no tool_calls -> END
  4. turn budget          -> State.turns, checked in the conditional edge
  5. observation formatter -> pretty_trace() over real BaseMessage objects

Unlike ToyLLM's scripted playback, `llm` here is a real ChatAnthropic model
that actually decides which tools to call and when to stop.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

MAX_TURNS = 10


# ------------------------------
# Tools: calculator + a stock portfolio (dict-backed buy/sell ledger)
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
# State, model, and graph
# ------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: int


llm = ChatAnthropic(
    model=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5"), max_tokens=1024
).bind_tools(TOOLS)


def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "turns": state["turns"] + 1}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    if state["turns"] >= MAX_TURNS:
        return "budget_exhausted"
    return "tools"


def budget_exhausted_node(state: State) -> dict:
    return {"messages": [AIMessage("budget exhausted")]}


tool_node = ToolNode(tools=TOOLS)

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("budget_exhausted", budget_exhausted_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "budget_exhausted": "budget_exhausted", END: END},
)
graph.add_edge("tools", "agent")
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
    """Render the trace using main.py's own vocabulary (user/thought/action/final)
    instead of raw LangChain message types, so the two lessons read the same way.
    Each action is merged with its observation onto one line, exactly like
    main.py's `{call.name}({call.args}) -> {observation}`.
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
    initial = {"messages": [HumanMessage(user_message)], "turns": 0}

    if not trace:
        return app.invoke(initial)

    state = {"messages": list(initial["messages"]), "turns": initial["turns"]}
    for update in app.stream(initial, stream_mode="updates"):
        for node_name, partial in update.items():
            print("=" * 70)
            print(f"NODE: {node_name}")
            print("-" * 70)
            print("INPUT  (accumulated state this node received):")
            print(f"  turns = {state['turns']}")
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

    print("=" * 70)
    print("FINAL RETAINED STATE:")
    print(f"  turns = {state['turns']}")
    print(f"  messages = {len(state['messages'])} total")

    return state


def main() -> None:
    print("=" * 70)
    print("REACT LOOP — LangChain/LangGraph + Claude")
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
