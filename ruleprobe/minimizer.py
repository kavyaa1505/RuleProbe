"""Delta-debugging minimizer: finds smallest passing rule subset."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Sequence

from .models import Run, Rule, RuleVerdict, Verdict

logger = logging.getLogger(__name__)


class PromptMinimizer:
    """
    Implements ddmin (delta debugging) to find the minimal subset of rules
    that preserves the same behavioral performance as the full prompt.

    Uses cached runs wherever possible.
    """

    def __init__(
        self,
        runner_fn: Callable[[str, list[Rule]], Awaitable[float]],
        pass_threshold: float = 90.0,
        max_iterations: int = 50,
    ):
        """
        runner_fn: async callable(prompt_text, rules) → average_score (0-100)
        pass_threshold: score must stay >= this fraction of baseline score
        """
        self.runner_fn = runner_fn
        self.pass_threshold = pass_threshold
        self.max_iterations = max_iterations

    async def minimize(
        self,
        original_prompt: str,
        rules: list[Rule],
        verdicts: list[RuleVerdict],
    ) -> tuple[list[Rule], float]:
        """
        Returns (minimal_rules, baseline_score).
        Pinned rules are always included.
        """
        # Never remove pinned rules
        pinned = [r for r in rules if r.pinned]
        candidates = [r for r in rules if not r.pinned]

        # Pre-filter: only test rules not already marked load_bearing
        # (Load-bearing rules very likely needed; keep them in by default)
        lb_ids = {v.rule_id for v in verdicts if v.verdict == Verdict.load_bearing}
        load_bearing = [r for r in candidates if r.id in lb_ids]
        removable_candidates = [r for r in candidates if r.id not in lb_ids]

        # Baseline score
        baseline_score = await self.runner_fn(original_prompt, rules)
        threshold = baseline_score * (self.pass_threshold / 100.0)
        logger.info(f"Minimizer: baseline={baseline_score:.1f}, threshold={threshold:.1f}")

        # Start with just load-bearing + pinned
        minimal = list(pinned) + list(load_bearing)

        # ddmin: try removing chunks of removable_candidates
        kept = list(removable_candidates)
        n = 2
        iterations = 0

        while len(kept) > 0 and iterations < self.max_iterations:
            iterations += 1
            chunk_size = max(1, len(kept) // n)
            chunks = [kept[i: i + chunk_size] for i in range(0, len(kept), chunk_size)]

            any_removed = False
            for chunk in chunks:
                trial = [r for r in kept if r not in chunk]
                trial_rules = minimal + trial
                trial_prompt = self._rebuild_prompt(trial_rules)
                score = await self.runner_fn(trial_prompt, trial_rules)

                if score >= threshold:
                    # Removing this chunk is safe
                    kept = trial
                    any_removed = True
                    logger.info(f"  Removed {len(chunk)} rules (score={score:.1f})")
                    break

            if not any_removed:
                if n >= len(kept):
                    break
                n = min(n * 2, len(kept))

        minimal_rules = minimal + kept
        return minimal_rules, baseline_score

    def _rebuild_prompt(self, rules: list[Rule]) -> str:
        lines = []
        section = None
        for r in sorted(rules, key=lambda x: x.position):
            if r.section != section:
                if r.section and r.section != "root":
                    lines.append(f"\n## {r.section}\n")
                section = r.section
            lines.append(f"- {r.text}")
        return "\n".join(lines)


# Type alias for async callable
from typing import Awaitable
