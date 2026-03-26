#!/bin/bash
# Oracle solution for view-recognition.
# Reads the bundled solutions.json and writes correct answers to sol.txt
# in the order questions appear under /questions/.

python3 - <<'EOF'
import json
from pathlib import Path

solutions = json.loads((Path("/solution/solutions.json")).read_text())
q_ids = sorted(d.name for d in Path("/questions").iterdir() if d.is_dir())

answers = []
for q_id in q_ids:
    answer = solutions.get(q_id)
    if answer is None:
        raise KeyError(f"No solution found for {q_id} — update solution/solutions.json")
    answers.append(answer)

Path("/home/user/sol.txt").write_text("\n".join(answers) + "\n")
print(f"Wrote {len(answers)} answer(s): {answers}")
EOF
