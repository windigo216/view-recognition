#!/usr/bin/env python3
"""
Run all three frontier models against the view-recognition-custom-agent task
k times each in parallel, then compute per-question pass@k statistics.

Uses VisionAgent

Usage:
    python run_analytics.py --runs 5
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from config import prepare_questions

HERE          = Path(__file__).parent
JOBS_DIR      = HERE / "jobs"
ANALYTICS_DIR = HERE / "analytics"

load_dotenv(HERE.parent / ".env")

MODELS = {
    "gemini": "google/gemini-3.1-pro-preview",
    "opus":   "anthropic/claude-opus-4.6",
    "gpt":    "openai/gpt-5.4",
}

AGENT_IMPORT = "vision_agent:VisionAgent"
TASK_PATH    = str(HERE)

JOB_START_COOLDOWN = 5.0  # seconds between any two job starts
_launch_lock       = threading.Lock()
_last_launch_time  = 0.0


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(model_key: str, model_id: str, api_key: str, run_idx: int,
              max_turns=None) -> dict:
    """Launch one harbor trial and block until it completes."""
    global _last_launch_time
    with _launch_lock:
        wait = JOB_START_COOLDOWN - (time.monotonic() - _last_launch_time)
        if wait > 0:
            time.sleep(wait)
        _last_launch_time = time.monotonic()
    print(f"  [{model_key} #{run_idx + 1}] starting...")
    t0 = time.monotonic()

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = TASK_PATH + (":" + existing if existing else "")

    cmd = [
        "harbor", "run",
        "-p", TASK_PATH,
        "--agent-import-path", AGENT_IMPORT,
        "-m", model_id,
        "--force-build",
        "--ae", f"OPENROUTER_API_KEY={api_key}",
    ]
    if max_turns is not None:
        cmd += ["--ak", f"max_turns={max_turns}"]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - t0
    status = "done" if proc.returncode == 0 else f"exit={proc.returncode}"
    print(f"  [{model_key} #{run_idx + 1}] {status} ({elapsed:.1f}s)")
    if proc.returncode != 0 and elapsed < 10:
        if proc.stdout.strip():
            print(f"  STDOUT:\n{proc.stdout}")
        if proc.stderr.strip():
            print(f"  STDERR:\n{proc.stderr}")
    return {
        "model_key":   model_key,
        "model_id":    model_id,
        "run_idx":     run_idx,
        "returncode":  proc.returncode,
        "elapsed":     elapsed,
        "finish_time": time.time(),
    }


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def find_new_job_dirs(known_jobs: set) -> list:
    return [d for d in sorted(JOBS_DIR.iterdir())
            if d.is_dir() and d.name not in known_jobs]


def parse_job(job_dir: Path):
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None

    config     = json.loads(config_path.read_text())
    # Custom agent jobs store model_name in agents list or agent dict
    try:
        model_name = config["agents"][0]["model_name"]
    except (KeyError, IndexError):
        try:
            model_name = config["agent"]["model_name"]
        except KeyError:
            return None

    stdout_paths = list(job_dir.glob("*/verifier/test-stdout.txt"))
    if not stdout_paths:
        exc_paths = list(job_dir.glob("*/exception.txt"))
        reason = exc_paths[0].read_text().strip().splitlines()[-1] if exc_paths else "no exception.txt"
        print(f"    parse_job: test-stdout.txt not found in {job_dir.name} — {reason}")
        return None

    per_question = {}
    for line in stdout_paths[0].read_text().splitlines():
        m = re.match(r"\s*(q\d+):\s*(OK|WRONG)", line)
        if m:
            per_question[m.group(1)] = (m.group(2) == "OK")

    return {"model_name": model_name, "per_question": per_question, "job_dir": str(job_dir)}

def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if c == n:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio


def compute_analytics(results_by_model: dict, n_runs: int, q_ids: list) -> dict:
    analytics = {}
    for model_key, runs in results_by_model.items():
        n = len(runs)
        per_question = {}
        for q_id in q_ids:
            c = sum(1 for run in runs if run.get(q_id, False))
            entry = {
                "n": n, "correct": c,
                "pass@1": pass_at_k(n, c, 1),
                "pass@2": pass_at_k(n, c, 2),
            }
            if n_runs > 2:
                entry[f"pass@{n_runs}"] = pass_at_k(n, c, n_runs)
            per_question[q_id] = entry

        valid = [v for v in per_question.values() if not math.isnan(v["pass@2"])]
        model_data = {
            "n_runs":      n,
            "per_question": per_question,
            "mean_pass@1": sum(v["pass@1"] for v in valid) / len(valid) if valid else float("nan"),
            "mean_pass@2": sum(v["pass@2"] for v in valid) / len(valid) if valid else float("nan"),
        }
        if n_runs > 2:
            pk_key      = f"pass@{n_runs}"
            mean_pk_key = f"mean_pass@{n_runs}"
            valid_pk = [v for v in per_question.values()
                        if not math.isnan(v.get(pk_key, float("nan")))]
            model_data[mean_pk_key] = (
                sum(v[pk_key] for v in valid_pk) / len(valid_pk) if valid_pk else float("nan")
            )
        analytics[model_key] = model_data
    return analytics

def write_summary(analytics: dict, path: Path, q_ids: list):
    lines = ["pass@k results — view-recognition-custom-agent", "=" * 60]
    first   = next(iter(analytics.values())) if analytics else {}
    n_runs  = first.get("n_runs", 2)
    show_pk = n_runs > 2

    for model_key, data in sorted(analytics.items()):
        n            = data["n_runs"]
        header_extra = f" {'pass@' + str(n_runs):>10}" if show_pk else ""
        sep_extra    = f" {'-'*10}"                    if show_pk else ""
        lines += [
            f"\nModel: {model_key}  ({n} runs)",
            f"  mean pass@1 = {data['mean_pass@1']:.3f}",
            f"  mean pass@2 = {data['mean_pass@2']:.3f}",
        ]
        if show_pk:
            lines.append(f"  mean pass@{n_runs} = {data.get(f'mean_pass@{n_runs}', float('nan')):.3f}")
        lines += [
            "",
            f"  {'Question':<10} {'Correct':>9} {'pass@1':>8} {'pass@2':>8}{header_extra}",
            f"  {'-'*10} {'-'*9} {'-'*8} {'-'*8}{sep_extra}",
        ]
        for q_id in q_ids:
            q   = data["per_question"][q_id]
            c   = q["correct"]
            p1  = f"{q['pass@1']:.3f}" if not math.isnan(q["pass@1"]) else "  n/a"
            p2  = f"{q['pass@2']:.3f}" if not math.isnan(q["pass@2"]) else "  n/a"
            row = f"  {q_id:<10} {c:>4}/{n:<5} {p1:>8} {p2:>8}"
            if show_pk:
                pk_val = q.get(f"pass@{n_runs}", float("nan"))
                row   += f" {f'{pk_val:.3f}' if not math.isnan(pk_val) else '  n/a':>10}"
            lines.append(row)

    summary = "\n".join(lines) + "\n"
    path.write_text(summary)
    print(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, required=True,
                        help="Number of trials per model (>= 2 for pass@2)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Cap the agent's reasoning turns (default: unlimited)")
    args = parser.parse_args()

    if args.runs < 2:
        print("Error: --runs must be >= 2 to compute pass@2")
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env or environment")
        sys.exit(1)

    q_ids = prepare_questions()
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    known_jobs = {d.name for d in JOBS_DIR.iterdir() if d.is_dir()}

    tasks = [
        (mk, f"openrouter/{mid}", api_key, ri)
        for mk, mid in MODELS.items()
        for ri in range(args.runs)
    ]
    print(f"Launching {len(tasks)} trials ({len(MODELS)} models × {args.runs} runs)...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for mk, mid, ak, ri in tasks:
            futures[executor.submit(run_trial, mk, mid, ak, ri, args.max_turns)] = (mk, ri)

        for future in as_completed(futures):
            mk, ri = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  [{mk} #{ri + 1}] ERROR: {exc}")

    print("\nParsing results...")
    new_job_dirs = find_new_job_dirs(known_jobs)
    results_by_model = {mk: [] for mk in MODELS}

    raw_results = []
    for job_dir in new_job_dirs:
        parsed = parse_job(job_dir)
        if not parsed:
            print(f"  Warning: could not parse {job_dir.name}")
            continue
        raw_results.append(parsed)
        mk = next((k for k, v in MODELS.items()
                   if f"openrouter/{v}" == parsed["model_name"]), None)
        if mk:
            results_by_model[mk].append(parsed["per_question"])
            correct = sum(parsed["per_question"].values())
            print(f"  [{mk}] {job_dir.name}  score={correct}/{len(parsed['per_question'])}")
        else:
            print(f"  Warning: unrecognised model '{parsed['model_name']}' in {job_dir.name}")

    (ANALYTICS_DIR / "raw_results.json").write_text(json.dumps(raw_results, indent=2))
    analytics = compute_analytics(results_by_model, args.runs, q_ids)
    (ANALYTICS_DIR / "pass_at_k.json").write_text(json.dumps(analytics, indent=2))
    write_summary(analytics, ANALYTICS_DIR / "summary.txt", q_ids)
    print(f"Results saved to {ANALYTICS_DIR}/")


if __name__ == "__main__":
    main()
