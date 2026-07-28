"""
Agent State Machine: Thought -> Action -> Observation
"""

# ------------------------------
# 1. State, Tools, and Model
# ------------------------------
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


MOCK_SEARCH_RESULTS = {
    "anthropic headquarters": (
        "Anthropic's headquarters is at 548 Market St, San Francisco, CA 94104."
    ),
}

MOCK_FILES = {
    "README.md": "# Agent State Machine Demo\nBuilt with LangGraph + Claude.",
}


@tool
def search_web(query: str) -> str:
    """Search the web for information and return a short summary."""
    key = query.lower().strip()
    for db_key, answer in MOCK_SEARCH_RESULTS.items():
        if db_key in key or key in db_key:
            return answer
    return f"No results found for '{query}'."


@tool
def read_file(path: str) -> str:
    """Read the contents of a file by path."""
    if path not in MOCK_FILES:
        return f"File '{path}' not found. Available files: {list(MOCK_FILES.keys())}"
    return MOCK_FILES[path]


@tool
def delete_database(name: str) -> str:
    """Delete a database by name. DESTRUCTIVE -- always requires human approval before use."""
    return f"Database '{name}' has been deleted. (mocked -- no real deletion occurs)"


TOOLS = [search_web, read_file, delete_database]

llm = ChatAnthropic(
    model=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5"), max_tokens=1024
).bind_tools(TOOLS)


def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


tool_node = ToolNode(tools=TOOLS)

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")


# ------------------------------
# 2. Run with a thread
# ------------------------------
def run_basic_demo():
    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "user-42"}}

    print("=" * 60)
    print("  Basic run -- streamed state updates")
    print("=" * 60)
    for event in app.stream(
        {"messages": [HumanMessage("find the Anthropic headquarters address")]},
        config,
        stream_mode="updates",
    ):
        print(event)


# ------------------------------
# 3. Add Human In The Loop Interruption
# ------------------------------
def run_human_in_the_loop_demo():
    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],  # pause before every tool call
    )
    config = {"configurable": {"thread_id": "user-hitl"}}

    print("\n" + "=" * 60)
    print("  Human-in-the-loop -- pause before every tool call")
    print("=" * 60)

    state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    "Delete the scratch database named 'temp-test-042' -- "
                    "it's leftover from a finished test run and no longer needed."
                )
            ]
        },
        config,
    )

    pending_calls = getattr(state["messages"][-1], "tool_calls", None)
    if not pending_calls:
        print("  No tool call was proposed -- nothing to approve.")
        return app, config

    print(f"  Paused before tool call(s): {pending_calls}")

    approved = True  # in a real app, ask a human reviewer here instead
    if approved:
        print("  Reviewer approved -- resuming.")
        state = app.invoke(Command(resume=True), config)
    else:
        print("  Reviewer denied -- blocking and recording the rejection.")
        app.update_state(
            config, {"messages": [AIMessage("Blocked by human reviewer.")]}
        )
        state = app.get_state(config).values

    print(f"  Final message: {state['messages'][-1].content}")
    return app, config


# ------------------------------
# 4. Time-Travel for Debugging
# ------------------------------
def run_time_travel_demo(app, config):
    print("\n" + "=" * 60)
    print("  Time-travel -- checkpoint history + fork")
    print("=" * 60)

    history = list(app.get_state_history(config))
    for snapshot in history:
        messages = snapshot.values.get("messages")
        preview = messages[-1].content[:80] if messages else "(no messages yet)"
        print(f"  {preview!r} {snapshot.config}")

    if len(history) > 3:
        target = history[3].config  # three steps back
        print("\n  Replaying from 3 steps back:")
        for event in app.stream(None, target, stream_mode="values"):
            messages = event.get("messages")
            preview = messages[-1].content if messages else event
            print(f"    {preview}")
    else:
        print(
            f"\n  Only {len(history)} checkpoint(s) recorded -- skipping fork demo (need > 3)."
        )


class _Tee:
    """Writes to both the original stream and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    log_path = Path(__file__).parent / "logs.txt"
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            run_basic_demo()
            hitl_app, hitl_config = run_human_in_the_loop_demo()
            run_time_travel_demo(hitl_app, hitl_config)
        finally:
            sys.stdout = sys.__stdout__
