"""
Eval configuration for view-recognition.

Edit OFFICIAL_QUESTIONS to choose which questions to include.
Set to None to include all 15. Order determines the sequential naming
(q01, q02, ...) inside the Docker environment.
"""

import json
import shutil
from pathlib import Path

HERE    = Path(__file__).parent
ENV_DIR = HERE / "environment"

# Any subset of 1–15 in any order. Set to None to include all.
OFFICIAL_QUESTIONS = None


def prepare_questions() -> list:
    """
    Build environment/selected_questions/ from environment/official_questions/.
    Copies each selected qXX.png → selected_questions/qXX/composite.png,
    renaming sequentially (q01, q02, ...) regardless of original numbers.
    Writes solutions to tests/solutions.json and solution/solutions.json.
    Returns the list of new q_ids.
    """
    src_dir = ENV_DIR / "official_questions"
    dst_dir = ENV_DIR / "selected_questions"

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir()

    all_solutions = json.loads((src_dir / "solutions.json").read_text())

    selected = OFFICIAL_QUESTIONS if OFFICIAL_QUESTIONS is not None else sorted(
        int(q_id[1:]) for q_id in all_solutions
    )

    new_solutions = {}
    for new_idx, orig_num in enumerate(selected, start=1):
        orig_id = f"q{orig_num:02d}"
        new_id  = f"q{new_idx:02d}"
        src_png = src_dir / f"{orig_id}.png"
        if not src_png.exists():
            raise FileNotFoundError(f"{src_png} not found in official_questions/")
        dst_q = dst_dir / new_id
        dst_q.mkdir()
        shutil.copy2(src_png, dst_q / "composite.png")
        new_solutions[new_id] = all_solutions[orig_id]

    data = json.dumps(new_solutions, indent=2)
    (HERE / "tests"    / "solutions.json").write_text(data)
    (HERE / "solution" / "solutions.json").write_text(data)

    q_ids = [f"q{i:02d}" for i in range(1, len(selected) + 1)]
    print(f"Questions: {[f'q{orig_num:02d}' for orig_num in selected]} → {q_ids}")
    return q_ids
