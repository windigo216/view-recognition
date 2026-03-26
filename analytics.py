#!/usr/bin/env python3
"""
Reanalyze all jobs in jobs/ and compute pass@k statistics.

Parses every job directory regardless of when it was created, so results
from multiple batched runs are combined into a single report.

Usage:
    python analytics.py
    python analytics.py --k 5
"""

import argparse
import json
import math
import re
from pathlib import Path

HERE          = Path(__file__).parent
JOBS_DIR      = HERE / "jobs"
ANALYTICS_DIR = HERE / "analytics"

MODELS = {
    "gpt":    "openai/gpt-5.4",
    "opus":   "anthropic/claude-opus-4.6",
    "gemini": "google/gemini-3.1-pro-preview",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_job(job_dir: Path):
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None

    config = json.loads(config_path.read_text())
    try:
        model_name = config["agents"][0]["model_name"]
    except (KeyError, IndexError):
        try:
            model_name = config["agent"]["model_name"]
        except KeyError:
            return None

    stdout_paths = list(job_dir.glob("*/verifier/test-stdout.txt"))
    if not stdout_paths:
        return None

    per_question = {}
    for line in stdout_paths[0].read_text().splitlines():
        m = re.match(r"\s*(q\d+):\s*(OK|WRONG)", line)
        if m:
            per_question[m.group(1)] = (m.group(2) == "OK")

    if not per_question:
        return None

    return {"model_name": model_name, "per_question": per_question, "job_dir": str(job_dir)}


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if c == n:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio


def compute_analytics(results_by_model: dict, k: int) -> dict:
    analytics = {}
    for model_key, runs in results_by_model.items():
        if not runs:
            continue
        n = len(runs)
        q_ids = sorted({q for run in runs for q in run})
        per_question = {}
        for q_id in q_ids:
            c = sum(1 for run in runs if run.get(q_id, False))
            entry = {"n": n, "correct": c, "pass@1": pass_at_k(n, c, 1),
                     "pass@2": pass_at_k(n, c, 2)}
            if k > 2:
                entry[f"pass@{k}"] = pass_at_k(n, c, k)
            per_question[q_id] = entry

        def _mean(key):
            vals = [v[key] for v in per_question.values() if not math.isnan(v.get(key, float("nan")))]
            return sum(vals) / len(vals) if vals else float("nan")

        model_data = {
            "n_runs":       n,
            "per_question": per_question,
            "mean_pass@1":  _mean("pass@1"),
            "mean_pass@2":  _mean("pass@2"),
        }
        if k > 2:
            model_data[f"mean_pass@{k}"] = _mean(f"pass@{k}")

        analytics[model_key] = model_data
    return analytics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary(analytics: dict, path: Path, k: int):
    lines = ["pass@k results — view-recognition-custom-agent", "=" * 60]

    for model_key, data in sorted(analytics.items()):
        n       = data["n_runs"]
        q_ids   = sorted(data["per_question"])
        show_pk = k > 2

        lines += [
            f"\nModel: {model_key}  ({n} runs)",
            f"  mean pass@1 = {data['mean_pass@1']:.3f}",
            f"  mean pass@2 = {data['mean_pass@2']:.3f}",
        ]
        if show_pk:
            lines.append(f"  mean pass@{k} = {data.get(f'mean_pass@{k}', float('nan')):.3f}")

        header_extra = f" {'pass@' + str(k):>10}" if show_pk else ""
        sep_extra    = f" {'-'*10}"               if show_pk else ""
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
                pk_val = q.get(f"pass@{k}", float("nan"))
                row   += f" {f'{pk_val:.3f}' if not math.isnan(pk_val) else '  n/a':>10}"
            lines.append(row)

    summary = "\n".join(lines) + "\n"
    path.write_text(summary)
    print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reanalyze all jobs in jobs/ and compute pass@k statistics.")
    parser.add_argument("--k", type=int, default=2,
                        help="Maximum k for pass@k (default: 2)")
    args = parser.parse_args()

    if not JOBS_DIR.exists():
        print(f"No jobs directory found at {JOBS_DIR}")
        return

    job_dirs = sorted(d for d in JOBS_DIR.iterdir() if d.is_dir())
    print(f"Found {len(job_dirs)} job(s) in {JOBS_DIR}\n")

    results_by_model = {mk: [] for mk in MODELS}
    raw_results = []
    skipped = 0

    for job_dir in job_dirs:
        parsed = parse_job(job_dir)
        if not parsed:
            skipped += 1
            continue
        raw_results.append(parsed)
        mk = next((k for k, v in MODELS.items()
                   if f"openrouter/{v}" == parsed["model_name"]), None)
        if mk:
            results_by_model[mk].append(parsed["per_question"])
            correct = sum(parsed["per_question"].values())
            total   = len(parsed["per_question"])
            print(f"  [{mk}] {job_dir.name}  score={correct}/{total}")
        else:
            print(f"  [?] {job_dir.name}  unrecognised model '{parsed['model_name']}'")

    if skipped:
        print(f"\n  ({skipped} job(s) skipped — incomplete or missing verifier output)")

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYTICS_DIR / "raw_results.json").write_text(json.dumps(raw_results, indent=2))

    analytics = compute_analytics(results_by_model, args.k)
    (ANALYTICS_DIR / "pass_at_k.json").write_text(json.dumps(analytics, indent=2))
    write_summary(analytics, ANALYTICS_DIR / "summary.txt", args.k)
    print(f"Results saved to {ANALYTICS_DIR}/")


if __name__ == "__main__":
    main()
