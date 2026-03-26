# view-recognition

A benchmark for evaluating the perceptual reasoning abilities of frontier model agents. Agents are presented with orthographic projection problems drawn from the Perceptual Ability Test (PAT) and must identify the correct third of a 3D solid given two out of its top, end, and front views.

Each question is rendered as a single composite image in standard PAT format. No code execution, terminal access, or internet access. The agent reasons from vision and perceptual ability alone.

---

## Overview

Questions are drawn from a fixed set of 15 PAT problems located in `environment/official_questions/`.

Three frontier models are evaluated:

| Key      | Model                              |
|----------|------------------------------------|
| `gpt`    | `openai/gpt-5.4`                   |
| `opus`   | `anthropic/claude-opus-4.6`        |
| `gemini` | `google/gemini-3.1-pro-preview`    |

All models are routed through [OpenRouter](https://openrouter.ai).

---

## Requirements

- Python 3.9 or later
- [Harbor](https://github.com/brainlid/harbor) installed and available on `PATH`
- Docker

Install Python dependencies:

```
pip install -r requirements.txt
```

Provide your OpenRouter API key in a `.env` file one directory above the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

---

## Directory Structure

```
view-recognition/
├── vision_agent.py            # Custom Harbor agent (multi-turn vision + reasoning loop)
├── config.py                  # Question selection and Docker environment preparation
├── run_custom.py              # Single-model trial runner
├── run_analytics.py           # Multi-model parallel runner with pass@k statistics
├── analytics.py               # Reanalyze all accumulated jobs in jobs/
├── task.toml                  # Harbor task configuration
├── instruction.md             # Task description passed to the agent
├── requirements.txt
├── environment/
│   ├── Dockerfile
│   ├── official_questions/    # Hand-authored PAT questions (not in Docker image)
│   └── selected_questions/    # Prepared subset copied into Docker image
├── solution/
│   ├── solve.sh               # Oracle solution script
│   └── solutions.json         # Answer key synced from tests/ at prep time
└── tests/
    ├── test.py                # Verifier: scores sol.txt against solutions.json
    └── solutions.json         # Ground-truth answers (not in Docker image)
```

---

## Selecting Questions

To control which questions are included in a run, edit `config.py`:

```python
# Any subset of 1–15 in any order, or None for all
OFFICIAL_QUESTIONS = None
```

---

## Running a Single Trial

```
python run_custom.py <model> [--max-turns N]
```

**Arguments:**

| Argument | Description |
|---|---|
| `model` | One of `gpt`, `opus`, `gemini` |
| `--max-turns N` | Cap the agent's reasoning turns (default: unlimited) |

**Examples:**

```
python run_custom.py opus
python run_custom.py gemini
python run_custom.py opus --max-turns 10
```

Results are written to `jobs/<timestamp>/`.

---

## Running Analytics

Runs all three models in parallel for `k` trials each and computes pass@k statistics per question.

```
python run_analytics.py --runs <k> [--max-turns N]
```

`--runs` must be at least 2 to compute pass@2.

**Examples:**

```
python run_analytics.py --runs 5
python run_analytics.py --runs 7 --max-turns 10
```

Results are written to `analytics/`:

- `raw_results.json` — per-job scores
- `pass_at_k.json` — pass@k values per model per question
- `summary.txt` — formatted table

### pass@k

pass@k is the probability that at least one of k randomly sampled runs is correct, estimated using the unbiased estimator from Chen et al. (2021):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where `n` is the total number of runs and `c` is the number of correct runs.

---

## Agent Behaviour

`VisionAgent` operates in a multi-turn loop with no access to a terminal or code execution environment. Each turn:

1. The model receives the composite question images and is asked to respond with a structured JSON block containing `analysis`, `plan`, and `answers` fields.
2. The loop continues until the model sets `answers` to a complete list of letters, or `max_turns` is reached.
3. Answers are written to `/home/user/sol.txt` inside the Docker container, which the verifier scores.

A trajectory in ATIF-v1.6 format is written to the job log directory after each run.

---

## Reanalyzing Results

To reanalyze all accumulated jobs in `jobs/` without running new trials:

```
python analytics.py [--k N]
```

**Arguments:**

| Argument | Description |
|---|---|
| `--k N` | Maximum k for pass@k statistics (default: 5) |

This reads every job directory regardless of when it was created, allowing results to be accumulated across multiple batched runs and reanalyzed at any time. Output is written to `analytics/`.

---

## Agent Timeout

The default agent timeout is 2700 seconds (45 minutes), set in `task.toml`. Adjust as needed:

```toml
[agent]
timeout_sec = 2700.0
```

---

## Oracle

To verify the task is solvable and the verifier is correct, run the oracle agent:

```
harbor run -p . -a oracle --force-build
```

The oracle reads `solution/solutions.json` (kept in sync with `tests/solutions.json` by `config.py`) and writes the correct answers to `sol.txt`. A reward of 1.000 confirms the verifier is working correctly.
