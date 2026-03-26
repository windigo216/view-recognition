"""
Custom Harbor agent for the view-recognition PAT benchmark.

Multi-turn vision and reasoning loop; no terminal access. Writes trajectory.json
in ATIF-v1.6 format. max_turns accepted as an __init__ kwarg (--ak max_turns=N).
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import litellm

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

THINKING_BUDGET        = 10_000  # tokens; used for Claude (budget_tokens)
GEMINI_THINKING_EFFORT = "medium"   # effort level for Gemini 3.x: "none", "low", "medium", "high"

SYSTEM_PROMPT = (
    "You are a careful and methodical reasoner.\n"
    "Each turn, you MUST respond with ONLY a JSON block in this exact format:\n"
    "```json\n"
    "{\n"
    '  "analysis": "<your observations and reasoning this turn>",\n'
    '  "plan": "<what you intend to examine or verify next>",\n'
    '  "answers": null\n'
    "}\n"
    "```\n"
    "When you have verified all answers and are fully confident, set `answers` to a list of "
    "strings (one letter per question, in question order), for example:\n"
    "```json\n"
    "{\n"
    '  "analysis": "<final verification notes>",\n'
    '  "plan": "done",\n'
    '  "answers": ["B", "A", "C"]\n'
    "}\n"
    "```\n"
    "Do NOT set `answers` until you have genuinely cross-checked every question. "
    "Take as many turns as you need."
)

INITIAL_PROMPT = (
    "There are {n} question(s) in total. "
    "The composite question image(s) follow."
)

CONTINUE_PROMPT = (
    "Continue your analysis. Respond with a JSON block containing your `analysis`, `plan`, "
    "and `answers` (null if not yet ready). There are {total} question(s) total."
)

FINISH_PROMPT = (
    "You have been working through {total} question(s). "
    "Please finalize your analysis and output your JSON block now. "
    "If you are confident in all answers, set `answers` to a list of letters."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_claude(model_name: str) -> bool:
    mn = model_name.lower()
    return "claude" in mn or "anthropic" in mn


def _is_gemini(model_name: str) -> bool:
    mn = model_name.lower()
    return "gemini" in mn or "google" in mn


def _parse_json_response(text: str) -> dict | None:
    """Extract the first JSON object from a fenced or bare block."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _parse_answers_from_json(text: str) -> list[str]:
    """Return the `answers` list from a structured JSON response, or [] if not ready."""
    data = _parse_json_response(text)
    if not data:
        return []
    answers = data.get("answers")
    if not answers or not isinstance(answers, list):
        return []
    result = []
    for a in answers:
        letter = str(a).strip().upper()
        if letter in ("A", "B", "C", "D"):
            result.append(letter)
    return result


def _user_message_text(content: list[dict] | str) -> str:
    """Render user message content as a string for the trajectory; images are summarised."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "image_url":
            parts.append("[composite.png — base64 image omitted]")
    return "\n".join(parts)


class VisionAgent(BaseAgent):

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = 1000000,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            *args,
            **kwargs,
        )
        if max_turns is not None:
            self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "vision-pat"

    def version(self) -> str | None:
        return "1.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        session_id = str(uuid.uuid4())
        steps: list[dict] = []
        step_id = 0

        def _next_id() -> int:
            nonlocal step_id
            step_id += 1
            return step_id

        # ── 1. Discover questions ─────────────────────────────────────────────
        result = await environment.exec("ls /questions/ | sort")
        q_ids = [q for q in result.stdout.strip().splitlines() if q.strip()]
        if not q_ids:
            self.logger.error("No question directories found under /questions/")
            return
        self.logger.info(f"Found questions: {q_ids}")

        # ── 2. Base64-encode composite images ─────────────────────────────────
        composites: dict[str, str | None] = {}
        for q_id in q_ids:
            path = f"/questions/{q_id}/composite.png"
            r = await environment.exec(f"base64 -w 0 {path}")
            if r.return_code != 0 or not r.stdout.strip():
                self.logger.warning(f"Failed to encode {path}: {r.stderr}")
                composites[q_id] = None
            else:
                composites[q_id] = r.stdout.strip()

        # ── 3. Build first user message ────────────────────────────────────────
        use_cache = _is_claude(self.model_name)

        initial_content: list[dict] = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": INITIAL_PROMPT.format(n=len(q_ids))},
        ]
        for q_id in q_ids:
            initial_content.append({"type": "text", "text": f"\n=== {q_id} ==="})
            b64 = composites.get(q_id)
            if b64:
                initial_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            else:
                initial_content.append({"type": "text", "text": "[image unavailable]"})

        if use_cache:
            initial_content[-1]["cache_control"] = {"type": "ephemeral"}

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": initial_content},
        ]

        steps.append({
            "step_id":   _next_id(),
            "timestamp": _now(),
            "source":    "user",
            "message":   f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n"
                         + _user_message_text(initial_content),
        })

        # ── 4. Multi-turn reasoning loop ──────────────────────────────────────
        api_key   = os.environ.get("OPENROUTER_API_KEY")
        answers: list[str] = []

        context.n_input_tokens  = 0
        context.n_output_tokens = 0
        total_cached_tokens     = 0
        total_cost_usd          = 0.0

        for turn in range(1, self._max_turns + 1):
            self.logger.info(f"Turn {turn}/{self._max_turns} — calling {self.model_name}")

            call_kwargs: dict = dict(
                model=self.model_name,
                messages=messages,
                api_key=api_key,
            )

            if use_cache and not self.model_name.startswith("openrouter/"):
                call_kwargs["extra_headers"] = {
                    "anthropic-beta": "prompt-caching-2024-07-31"
                }

            if _is_claude(self.model_name):
                thinking_param = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
                if self.model_name.startswith("openrouter/"):
                    call_kwargs["extra_body"] = {"thinking": thinking_param}
                else:
                    call_kwargs["thinking"] = thinking_param
            elif _is_gemini(self.model_name):
                call_kwargs["extra_body"] = {
                    "reasoning": {"effort": GEMINI_THINKING_EFFORT}
                }

            try:
                response = await litellm.acompletion(**call_kwargs)
            except Exception as e:
                self.logger.error(f"LLM call failed on turn {turn}: {e}")
                break

            raw = response.choices[0].message.content or ""
            if not isinstance(raw, str):
                raw = "\n".join(
                    b["text"]
                    for b in raw
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                )
            raw = raw.strip()

            prompt_tokens      = 0
            completion_tokens  = 0
            cached_tokens      = 0
            step_cost          = None
            if response.usage:
                prompt_tokens     = response.usage.prompt_tokens    or 0
                completion_tokens = response.usage.completion_tokens or 0
                # LiteLLM may surface cached tokens in prompt_tokens_details
                details = getattr(response.usage, "prompt_tokens_details", None)
                if details:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                context.n_input_tokens  += prompt_tokens
                context.n_output_tokens += completion_tokens
                total_cached_tokens     += cached_tokens

            try:
                step_cost = litellm.completion_cost(completion_response=response)
                total_cost_usd += step_cost
            except Exception:
                pass

            step: dict = {
                "step_id":    _next_id(),
                "timestamp":  _now(),
                "source":     "agent",
                "model_name": self.model_name,
                "message":    raw,
                "tool_calls": [],
                "metrics": {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }
            if cached_tokens:
                step["metrics"]["cached_tokens"] = cached_tokens
            if step_cost is not None:
                step["metrics"]["cost_usd"] = step_cost
            steps.append(step)

            found = _parse_answers_from_json(raw)

            if len(found) >= len(q_ids):
                answers = found[: len(q_ids)]
                self.logger.info(f"Got all {len(q_ids)} answers on turn {turn}")
                break

            messages.append({"role": "assistant", "content": raw})

            follow_up = (
                FINISH_PROMPT.format(total=len(q_ids))
                if turn == self._max_turns
                else CONTINUE_PROMPT.format(total=len(q_ids))
            )
            messages.append({"role": "user", "content": follow_up})

            steps.append({
                "step_id":   _next_id(),
                "timestamp": _now(),
                "source":    "user",
                "message":   follow_up,
            })
            self.logger.info(
                f"Turn {turn}: answers not yet committed — continuing"
            )
        else:
            self.logger.warning(
                f"Reached max_turns={self._max_turns} without a complete answer set"
            )

        while len(answers) < len(q_ids):
            self.logger.warning("Padding missing answer with '?'")
            answers.append("?")
        answers = answers[: len(q_ids)]
        self.logger.info(f"Final answers: {dict(zip(q_ids, answers))}")

        # ── 5. Write sol.txt ──────────────────────────────────────────────────
        sol_content = "\n".join(answers)
        write_cmd = (
            f"python3 -c \""
            f"open('/home/user/sol.txt','w').write({repr(sol_content + chr(10))})"
            f"\""
        )
        r = await environment.exec(write_cmd)
        if r.return_code != 0:
            self.logger.error(f"Failed to write sol.txt: {r.stderr}")
        else:
            self.logger.info("sol.txt written successfully")

        # ── 6. Write trajectory.json ──────────────────────────────────────────
        trajectory = {
            "schema_version": "ATIF-v1.6",
            "session_id":     session_id,
            "agent": {
                "name":       self.name(),
                "version":    self.version(),
                "model_name": self.model_name,
                "extra": {
                    "max_turns":       self._max_turns,
                    "thinking_budget": THINKING_BUDGET if _is_claude(self.model_name) else None,
                    "gemini_thinking_effort": GEMINI_THINKING_EFFORT if _is_gemini(self.model_name) else None,
                    "prompt_caching":  use_cache,
                },
            },
            "steps": steps,
            "final_metrics": {
                "total_prompt_tokens":     context.n_input_tokens  or 0,
                "total_completion_tokens": context.n_output_tokens or 0,
                "total_cached_tokens":     total_cached_tokens,
                "total_cost_usd":          total_cost_usd if total_cost_usd else None,
            },
        }

        traj_path = self.logs_dir / "trajectory.json"
        try:
            traj_path.write_text(json.dumps(trajectory, indent=2))
            self.logger.info(f"trajectory.json written to {traj_path}")
        except Exception as e:
            self.logger.error(f"Failed to write trajectory.json: {e}")
