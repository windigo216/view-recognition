#!/usr/bin/env python3
"""
Run the view-recognition evaluation using VisionAgent.

VisionAgent runs a multi-turn vision and reasoning loop via OpenRouter,
presenting each question as a single PAT-format composite image.
Prompt caching is used for Claude to reduce per-turn image token cost.

Supported models: gpt, opus, gemini

API key required in ../.env:
  OPENROUTER_API_KEY

Usage:
    python run_custom.py <model> [--max-turns N]

Examples:
    python run_custom.py opus
    python run_custom.py opus --max-turns 8
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import prepare_questions

load_dotenv(Path(__file__).parent.parent / ".env")

MODELS = {
    "gpt":    "openai/gpt-5.4",
    "opus":   "anthropic/claude-opus-4.6",
    "gemini": "google/gemini-3.1-pro-preview",
}

AGENT_IMPORT = "vision_agent:VisionAgent"
TASK_PATH    = str(Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Cap the agent's reasoning turns (default: unlimited)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env or environment")
        sys.exit(1)

    prepare_questions()

    model = f"openrouter/{MODELS[args.model]}"

    print(f"Model:     {model}")
    print(f"Task:      {TASK_PATH}")
    print(f"Agent:     {AGENT_IMPORT}")
    print(f"max_turns: {args.max_turns if args.max_turns is not None else 'unlimited'}\n")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = TASK_PATH + (":" + existing if existing else "")

    cmd = [
        "harbor", "run",
        "-p", TASK_PATH,
        "--agent-import-path", AGENT_IMPORT,
        "-m", model,
        "--force-build",
        "--ae", f"OPENROUTER_API_KEY={api_key}",
    ]
    if args.max_turns is not None:
        cmd += ["--ak", f"max_turns={args.max_turns}"]

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
