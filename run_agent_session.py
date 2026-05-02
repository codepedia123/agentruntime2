#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json_file(path: str | None, default: Any) -> Any:
    if not path:
        return default
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help                 Show commands\n"
        "  /state                Print current state\n"
        "  /context              Print current full context JSON\n"
        "  /vars                 Print current variables JSON\n"
        "  /save <file>          Save current context JSON to file\n"
        "  /load <file>          Load context JSON from file\n"
        "  /reset                Reset context to initial loaded context\n"
        "  /exit                 Exit session\n"
    )


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Persistent multi-turn local runner for PartsWale agents")
    parser.add_argument("--role", required=True, choices=["mechanic", "dealer"])
    parser.add_argument("--context-file", help="Initial context JSON file")
    parser.add_argument("--conversation-file", help="Optional legacy conversation history file")
    parser.add_argument("--variables-file", help="Optional variables JSON file")
    parser.add_argument("--api-key", help="LLM API key override")
    args = parser.parse_args()

    port = "8001" if args.role == "mechanic" else "8002"
    os.environ["PORT"] = port

    runtime = importlib.import_module("main")

    context = load_json_file(args.context_file, {})
    conversation_history = load_json_file(args.conversation_file, [])
    variables = load_json_file(args.variables_file, {})
    initial_context = json.loads(json.dumps(context))

    print(f"[session] role={args.role} port={port}")
    print(f"[session] thread_id={context.get('thread_id', '')}")
    print_help()

    while True:
        try:
            raw = input(f"{args.role}> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[session] interrupted")
            break

        if not raw:
            continue

        if raw == "/help":
            print_help()
            continue
        if raw == "/exit":
            break
        if raw == "/state":
            print(context.get("state") or (context.get("variables") or {}).get("current_state") or "")
            continue
        if raw == "/context":
            print(dump_json(context))
            continue
        if raw == "/vars":
            print(dump_json((context or {}).get("variables", {})))
            continue
        if raw == "/reset":
            context = json.loads(json.dumps(initial_context))
            print("[session] context reset")
            continue
        if raw.startswith("/save "):
            path = raw.split(" ", 1)[1].strip()
            Path(path).write_text(dump_json(context) + "\n", encoding="utf-8")
            print(f"[session] saved context -> {path}")
            continue
        if raw.startswith("/load "):
            path = raw.split(" ", 1)[1].strip()
            context = load_json_file(path, {})
            print(f"[session] loaded context <- {path}")
            continue

        result = runtime.run_agent(
            context,
            conversation_history,
            raw,
            variables,
            args.api_key,
        )

        if isinstance(result, dict) and isinstance(result.get("context"), dict):
            context = result["context"]

        print(dump_json(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
