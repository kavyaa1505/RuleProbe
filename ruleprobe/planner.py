"""Planner: builds experiment matrix from rules × operators × scenarios × repeats."""

from __future__ import annotations

import uuid
from typing import Sequence

from .models import (
    ExperimentConfig,
    ExperimentPlan,
    Mutant,
    Operator,
    Rule,
    Scenario,
)
from .operators import (
    DuplicateOperator,
    MoveOperator,
    NegateOperator,
    PairRemoveOperator,
    ParaphraseOperator,
    RemoveOperator,
)

# Average token counts for cost estimation (conservative)
AVG_TOKENS_IN = 500
AVG_TOKENS_OUT = 150

# Rough pricing per 1M tokens (use GPT-4o-mini as default baseline)
PRICE_PER_1M_IN = 0.15
PRICE_PER_1M_OUT = 0.60


def _get_operators(config: ExperimentConfig):
    op_map = {
        Operator.remove: RemoveOperator(),
        Operator.negate: NegateOperator(),
        Operator.move: MoveOperator("start"),
        Operator.paraphrase: ParaphraseOperator(),
        Operator.duplicate: DuplicateOperator(),
    }
    return [op_map[op] for op in config.operators if op in op_map]


class ExperimentPlanner:
    """
    Builds the full experiment matrix:
    1. Create a baseline mutant (original prompt)
    2. For each rule × operator → create a mutant prompt
    3. Pair each mutant with every relevant scenario
    4. Respect budget cap using screening-then-refine ordering
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def plan(
        self,
        original_prompt: str,
        rules: list[Rule],
        scenarios: list[Scenario],
    ) -> ExperimentPlan:
        config = self.config
        operators = _get_operators(config)
        model = config.model
        N = config.runs_per_mutant

        mutants: list[Mutant] = []
        plan_runs: list[tuple[str, str, str, int]] = []

        # --- Baseline mutant (original prompt, no changes)
        baseline_id = f"M_baseline"
        baseline_mutant = Mutant(
            id=baseline_id,
            operator=Operator.remove,  # placeholder
            rule_ids=[],
            prompt_text=original_prompt,
        )
        mutants.append(baseline_mutant)
        for scenario in scenarios:
            for rep in range(N):
                plan_runs.append((baseline_id, scenario.id, model, rep))

        # --- Per-rule mutants (screening pass: remove + negate first)
        for rule in rules:
            if rule.pinned:
                continue
            for op in operators:
                try:
                    mutant = op.apply(original_prompt, rule, rules)
                    mutants.append(mutant)
                    # Map scenarios: prefer target_rule_ids, fall back to all
                    relevant = [
                        s for s in scenarios
                        if not s.target_rule_ids or rule.id in s.target_rule_ids
                    ] or scenarios
                    for scenario in relevant:
                        for rep in range(N):
                            plan_runs.append((mutant.id, scenario.id, model, rep))
                except Exception:
                    continue

        # --- Pair-remove for detecting interactions (expensive — do last)
        if Operator.pair_remove in config.operators and len(rules) >= 2:
            pair_op = PairRemoveOperator()
            # Only pair rules with shared scenario coverage
            for i, r1 in enumerate(rules[:8]):  # cap at 8 for cost
                for r2 in rules[i + 1: i + 4]:
                    if r1.pinned or r2.pinned:
                        continue
                    mutant = pair_op.apply(original_prompt, r1, r2)
                    mutants.append(mutant)
                    for scenario in scenarios[:3]:  # limited scenarios for pairs
                        for rep in range(min(N, 3)):
                            plan_runs.append((mutant.id, scenario.id, model, rep))

        # --- Cost estimate
        total_tokens_in = len(plan_runs) * AVG_TOKENS_IN
        total_tokens_out = len(plan_runs) * AVG_TOKENS_OUT
        estimated_cost = (
            total_tokens_in * PRICE_PER_1M_IN
            + total_tokens_out * PRICE_PER_1M_OUT
        ) / 1_000_000

        # --- Budget cap: trim runs to fit budget
        if estimated_cost > config.budget_usd:
            ratio = config.budget_usd / estimated_cost
            keep = max(1, int(len(plan_runs) * ratio))
            plan_runs = plan_runs[:keep]
            # Recalculate
            estimated_cost = config.budget_usd

        return ExperimentPlan(
            experiment_id=str(uuid.uuid4()),
            runs=plan_runs,
            estimated_cost=round(estimated_cost, 4),
            estimated_tokens=len(plan_runs) * (AVG_TOKENS_IN + AVG_TOKENS_OUT),
            mutants=mutants,
            scenarios=scenarios,
        )
