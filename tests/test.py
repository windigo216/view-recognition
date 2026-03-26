"""
Verifier for view-recognition eval.

Reads /home/user/sol.txt (one letter per line, q01 first) and scores it
against the ground-truth answers in /tests/solutions.json.
Writes a fractional reward to /logs/verifier/reward.txt.
"""

import json
import sys
from pathlib import Path

SOLUTIONS_FILE = Path("/tests/solutions.json")
SOL_FILE       = Path("/home/user/sol.txt")
REWARD_FILE    = Path("/logs/verifier/reward.txt")


def main():
    solutions = json.loads(SOLUTIONS_FILE.read_text())
    q_ids = sorted(solutions)

    try:
        lines = SOL_FILE.read_text().strip().splitlines()
    except FileNotFoundError:
        print("ERROR: /home/user/sol.txt not found")
        REWARD_FILE.write_text("0.0")
        sys.exit(1)

    agent = {}
    for i, q_id in enumerate(q_ids):
        raw = lines[i].strip().upper() if i < len(lines) else ""
        agent[q_id] = next((c for c in raw if c in "ABCD"), "?")

    correct = 0
    for q_id in q_ids:
        expected = solutions[q_id]
        got = agent.get(q_id, "?")
        ok = got == expected
        correct += ok
        status = "OK" if ok else f"WRONG (got '{got}', expected '{expected}')"
        print(f"  {q_id}: {status}")

    total  = len(solutions)
    reward = correct / total
    print(f"\nScore: {correct}/{total} = {reward:.2f}")

    REWARD_FILE.write_text(f"{reward:.4f}")
    sys.exit(0 if correct == total else 1)


if __name__ == "__main__":
    main()
