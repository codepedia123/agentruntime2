#!/usr/bin/env python3
"""
Direct local runner for PartsWale agents.

This bypasses FastAPI/HTTP and calls main.run_agent(...) directly.

Examples:
  python3 run_agent_local.py \
    --role mechanic \
    --message "Request a Part" \
    --context-file /tmp/mechanic_context.json

  python3 run_agent_local.py \
    --role dealer \
    --message "Send Quote" \
    --context-file /tmp/dealer_context.json \
    --api-key "$GROQ_API_KEY"
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROLE_TO_PORT = {
    "mechanic": 8001,
    "dealer": 8002,
}


def load_json_file(path: str | None, default: Any) -> Any:
    if not path:
        return default
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=sorted(ROLE_TO_PORT))
    parser.add_argument("--message", required=True)
    parser.add_argument("--context-file")
    parser.add_argument("--conversation-file")
    parser.add_argument("--variables-file")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    os.environ["PORT"] = str(ROLE_TO_PORT[args.role])

    runtime = importlib.import_module("main")

    context = load_json_file(args.context_file, {})
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")

    conversation = load_json_file(args.conversation_file, [])
    if conversation is None:
        conversation = []
    if not isinstance(conversation, list):
        raise ValueError("conversation must be a JSON array")

    variables = load_json_file(args.variables_file, {})
    if variables is None:
        variables = {}
    if not isinstance(variables, dict):
        raise ValueError("variables must be a JSON object")

    api_key = args.api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

    result = runtime.run_agent(
        context,
        conversation,
        args.message,
        variables,
        api_key,
    )

    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
