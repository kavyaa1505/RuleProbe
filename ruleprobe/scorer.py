"""Scoring: deterministic checks + LLM judge."""

from __future__ import annotations

import re
from typing import Any

from .models import BehavioralFingerprint, Run, Score


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

class CheckRegistry:
    """Registry of named deterministic checks applied to agent outputs."""

    def run_check(self, check: dict[str, Any], run: Run) -> tuple[str, bool, str]:
        """Return (check_name, passed, detail)."""
        check_type = check.get("type", "contains")
        name = check.get("name", check_type)
        output = run.output

        if check_type == "contains":
            pattern = check.get("value", "")
            passed = pattern.lower() in output.lower()
            return name, passed, f"Expected '{pattern}' in output"

        elif check_type == "not_contains":
            pattern = check.get("value", "")
            passed = pattern.lower() not in output.lower()
            return name, passed, f"Expected '{pattern}' absent from output"

        elif check_type == "regex":
            pattern = check.get("pattern", "")
            passed = bool(re.search(pattern, output, re.IGNORECASE))
            return name, passed, f"Regex '{pattern}'"

        elif check_type == "max_length":
            limit = int(check.get("value", 500))
            passed = len(output.split()) <= limit
            return name, passed, f"Output length {len(output.split())} <= {limit} words"

        elif check_type == "min_length":
            limit = int(check.get("value", 1))
            passed = len(output.split()) >= limit
            return name, passed, f"Output length {len(output.split())} >= {limit} words"

        elif check_type == "tool_call":
            expected_tool = check.get("value", "")
            tool_names = [tc.name for tc in run.tool_calls]
            passed = expected_tool in tool_names
            return name, passed, f"Expected tool call '{expected_tool}'"

        elif check_type == "refusal":
            refusal_phrases = ["i cannot", "i can't", "i'm unable", "i must decline", "i won't"]
            passed = any(p in output.lower() for p in refusal_phrases)
            return name, passed, "Expected refusal"

        elif check_type == "no_refusal":
            refusal_phrases = ["i cannot", "i can't", "i'm unable", "i must decline", "i won't"]
            passed = not any(p in output.lower() for p in refusal_phrases)
            return name, passed, "Expected no refusal"

        elif check_type == "json_valid":
            import json
            try:
                json.loads(output)
                passed = True
            except Exception:
                passed = False
            return name, passed, "Expected valid JSON output"

        return name, True, "unknown check type (skipped)"


CHECK_REGISTRY = CheckRegistry()


def score_run(run: Run, checks: list[dict[str, Any]]) -> Score:
    """Score a single run against a list of checks."""
    check_scores: dict[str, bool] = {}

    for check in checks:
        name, passed, detail = CHECK_REGISTRY.run_check(check, run)
        check_scores[name] = passed

    # Aggregate: fraction of checks passed
    if check_scores:
        aggregate = sum(check_scores.values()) / len(check_scores)
    else:
        aggregate = 1.0  # no checks → assume pass

    # Build fingerprint
    fingerprint = BehavioralFingerprint(
        tool_call_sequence=[tc.name for tc in run.tool_calls],
        step_count=len(run.tool_calls),
        tokens_used=run.tokens_in + run.tokens_out,
        refusal=any(
            p in run.output.lower()
            for p in ["i cannot", "i can't", "i'm unable", "i won't"]
        ),
        latency_ms=run.latency_ms,
    )

    return Score(
        run_id=run.id,
        check_scores=check_scores,
        aggregate_score=aggregate * 100,  # 0-100
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict evaluator. Given an agent's response and a rubric, 
score the response from 0 to 100.

Rubric: {rubric}

Agent response:
{output}

Return ONLY a JSON object: {{"score": <0-100>, "rationale": "<one sentence>", "confidence": <0.0-1.0>}}
"""


class LLMJudge:
    """LLM-based rubric judge. Versioned and model-pinned."""

    VERSION = "v1"

    def __init__(self, adapter, rubric: str = "Is the response helpful, accurate, and policy-compliant?"):
        self.adapter = adapter
        self.rubric = rubric

    async def judge(self, run: Run) -> tuple[float, str, float]:
        """Return (score 0-100, rationale, confidence)."""
        import json

        prompt = JUDGE_PROMPT.format(rubric=self.rubric, output=run.output[:2000])
        result = await self.adapter.run(
            instructions="You are a strict AI evaluator. Return only valid JSON.",
            scenario_input=prompt,
            seed=42,
        )

        try:
            data = json.loads(result.output.strip())
            score = float(data.get("score", 50))
            rationale = str(data.get("rationale", ""))
            confidence = float(data.get("confidence", 0.8))
            return score, rationale, confidence
        except Exception:
            # Fallback: try to extract a number
            nums = re.findall(r"\b(\d{1,3})\b", result.output)
            score = float(nums[0]) if nums else 50.0
            return score, result.output[:100], 0.3
