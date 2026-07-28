"""
Function Calling & Tool Use
"""

# ------------------------------------
# 1. Define the tool registry
# ------------------------------------

import json
import math
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

TOOL_REGISTRY = {}


def register_tool(name, description, input_schema, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        },
        "function": function,
    }


# ------------------------------------
# 2. Implement 5 Tools
# ------------------------------------
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {
            "error": True,
            "message": f"Invalid characters in expression: {expression}",
        }
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {
        "temp_c": 16,
        "condition": "foggy",
        "humidity": 80,
        "wind_kph": 18,
    },
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {
            "title": "OpenAI Function Calling Guide",
            "url": "https://platform.openai.com/docs/guides/function-calling",
            "snippet": "Learn how to connect LLMs to external tools.",
        },
        {
            "title": "Anthropic Tool Use",
            "url": "https://docs.anthropic.com/en/docs/tool-use",
            "snippet": "Claude can interact with external tools and APIs.",
        },
    ],
    "MCP protocol": [
        {
            "title": "Model Context Protocol",
            "url": "https://modelcontextprotocol.io",
            "snippet": "An open standard for connecting AI models to data sources.",
        },
    ],
    "weather API": [
        {
            "title": "OpenWeatherMap API",
            "url": "https://openweathermap.org/api",
            "snippet": "Free weather API with current, forecast, and historical data.",
        },
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key.lower() in key or key in db_key.lower():
            return {
                "query": query,
                "results": results[:max_results],
                "total": len(results),
            }
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {
            "error": True,
            "message": "Path traversal not allowed.",
            "code": "FORBIDDEN",
        }
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {
            "error": True,
            "message": f"File '{path}' not found.",
            "available_files": available,
            "code": "NOT_FOUND",
        }
    content = FILE_SYSTEM[path]
    return {
        "path": path,
        "content": content,
        "size_bytes": len(content),
        "lines": content.count("\n") + 1,
    }


def run_code(code, language="python"):
    if language != "python":
        return {
            "error": True,
            "message": f"Language '{language}' not supported. Only 'python' is available.",
        }
    forbidden = [
        "import os",
        "import sys",
        "import subprocess",
        "exec(",
        "eval(",
        "__import__",
        "open(",
    ]
    for pattern in forbidden:
        if pattern in code:
            return {
                "error": True,
                "message": f"Forbidden operation: {pattern}",
                "code": "SECURITY_VIOLATION",
            }
    try:
        local_vars = {}
        exec(
            code,
            {
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "math": math,
                }
            },
            local_vars,
        )
        result = local_vars.get("result", None)
        return {
            "success": True,
            "result": result,
            "variables": {
                k: str(v) for k, v in local_vars.items() if not k.startswith("_")
            },
        }
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}


# ------------------------------------
# 3. Register All Tools
# ------------------------------------
def register_all_tools():
    register_tool(
        "calculator",
        "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '(10 + 5) * 3'",
                },
                "precision": {
                    "type": "integer",
                    "description": "Decimal places in result",
                    "default": 2,
                },
            },
            "required": ["expression"],
        },
        calculator,
    )
    register_tool(
        "get_weather",
        "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo' or 'San Francisco'",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units, defaults to celsius",
                },
            },
            "required": ["city"],
        },
        get_weather,
    )
    register_tool(
        "web_search",
        "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        web_search,
    )
    register_tool(
        "read_file",
        "Read the contents of a file. Returns the file content, size, and line count.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path, e.g. 'data/config.json'",
                }
            },
            "required": ["path"],
        },
        read_file,
    )
    register_tool(
        "run_code",
        "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "language": {
                    "type": "string",
                    "enum": ["python"],
                    "description": "Programming language",
                },
            },
            "required": ["code"],
        },
        run_code,
    )


# ------------------------------------
# 4. Build Function Calling Loop
# Let Claude decide which tool(s) to call, execute them, and feed results back
# ------------------------------------
def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {
            "error": True,
            "message": f"Unknown tool: {name}",
            "code": "UNKNOWN_TOOL",
        }

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(client, model, user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    iterations = 0
    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tool_definitions,
            messages=conversation,
        )
        iterations += 1
        conversation.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results_content = []
        for block in tool_use_blocks:
            result = execute_tool_call({"name": block.name, "arguments": block.input})
            all_tool_results.append(result)
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result["result"]),
                }
            )

        conversation.append({"role": "user", "content": tool_results_content})

        if response.stop_reason != "tool_use":
            break

    return {
        "conversation": conversation,
        "tool_results": all_tool_results,
        "iterations": iterations,
    }


# ------------------------------------
# 5. Argument Validation
# Check tool arguments against the JSON Schema before execution
# ------------------------------------
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["input_schema"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(
                    f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}"
                )

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(
                f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}"
            )

    return errors


# ------------------------------------
# 6. Run Demo
# ------------------------------------
def run_demo():
    load_dotenv()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")

    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["description"][:60]
        params = list(tool["definition"]["input_schema"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(client, model, query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(client, model, multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(
            f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}"
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
            run_demo()
        finally:
            sys.stdout = sys.__stdout__
