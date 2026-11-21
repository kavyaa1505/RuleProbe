"""Main pipeline: orchestrates parse → plan → run → score → analyze → report."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .adapters import AgentAdapter, get_adapter
from .analyzer import RuleAnalyzer, detect_conflicts
from .models import (
    ExperimentConfig,
    Experiment,
    Run,
    RuleVerdict,
    Score,
    Scenario,
)
from .parser import InstructionParser, parse_file
from .planner import ExperimentPlanner
from .reporter import format_json, format_markdown_pr_comment, format_verdict_table
from .runner import ExperimentRunner, RunCache
from .scenarios import generate_scenarios_for_rules, load_scenarios
from .scorer import score_run
from .storage import Storage

logger = logging.getLogger(__name__)


class RuleProbe:
    """
    High-level orchestrator. Runs the full pipeline and returns results.

    Usage:
        rp = RuleProbe(config)
        result = await rp.analyze()
        print(result["report"])
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    async def analyze(self) -> dict[str, Any]:
        cfg = self.config
        t_start = time.perf_counter()
        exp_id = str(uuid.uuid4())

        # --- 1. PARSE
        logger.info(f"Parsing {cfg.instruction_file}")
        original_text = Path(cfg.instruction_file).read_text(encoding="utf-8")
        parser = InstructionParser(pinned_ids=cfg.pinned_rules)
        rules = parser.parse(original_text)
        logger.info(f"  Parsed {len(rules)} rules")

        # --- 2. SCENARIOS
        if cfg.scenarios_file:
            scenarios = load_scenarios(cfg.scenarios_file)
            logger.info(f"  Loaded {len(scenarios)} scenarios from {cfg.scenarios_file}")
        else:
            scenarios = generate_scenarios_for_rules(rules, n_per_rule=1)
            logger.info(f"  Generated {len(scenarios)} heuristic scenarios")

        if not scenarios:
            raise ValueError("No scenarios available. Provide a scenarios file or use --generate-scenarios.")

        # --- 3. PLAN
        planner = ExperimentPlanner(cfg)
        plan = planner.plan(original_text, rules, scenarios)
        logger.info(
            f"  Plan: {len(plan.runs)} runs, {len(plan.mutants)} mutants, "
            f"estimated cost ${plan.estimated_cost:.4f}"
        )

        if plan.estimated_cost > cfg.budget_usd:
            logger.warning(f"  Budget cap applied: ${cfg.budget_usd:.2f}")

        # --- 4. EXECUTE
        adapter = get_adapter(cfg.adapter, model=cfg.model)
        cache = RunCache(cfg.cache_db)
        runner = ExperimentRunner(
            adapter=adapter,
            concurrency=cfg.concurrency,
            cache=cache,
            budget_usd=cfg.budget_usd,
        )

        logger.info("Running experiment...")
        runs = await runner.run_experiment(plan)
        spent = runner.spent
        logger.info(f"  Completed {len(runs)} runs, spent ${spent:.4f}")

        # --- 5. SCORE
        # Build lookup: scenario_id → checks
        scenario_checks = {s.id: s.checks for s in scenarios}
        # Build run scores
        run_scores: dict[str, Score] = {}
        for run in runs:
            if run.error:
                continue
            checks = scenario_checks.get(run.scenario_id, [])
            score = score_run(run, checks)
            run_scores[run.id] = score

        # --- 6. ANALYZE
        # Separate baseline mutant
        baseline_mutant_id = plan.mutants[0].id  # first is always baseline

        # Build score maps
        # baseline: scenario_id → [scores]
        baseline_scores: dict[str, list[float]] = {}
        # mutant: mutant_id → scenario_id → [scores]
        mutant_score_map: dict[str, dict[str, list[float]]] = {}

        for run in runs:
            if run.error or run.id not in run_scores:
                continue
            s = run_scores[run.id].aggregate_score
            if run.mutant_id == baseline_mutant_id:
                baseline_scores.setdefault(run.scenario_id, []).append(s)
            else:
                mutant_score_map.setdefault(run.mutant_id, {}).setdefault(run.scenario_id, []).append(s)

        # Build mutant → (rule_id, operator) mapping
        mutant_map = {m.id: m for m in plan.mutants}
        mutant_to_rule: dict[str, tuple[str, Any]] = {}
        for mutant in plan.mutants:
            if mutant.rule_ids:
                mutant_to_rule[mutant.id] = (mutant.rule_ids[0], mutant.operator)

        analyzer = RuleAnalyzer(
            materiality_threshold=cfg.materiality_threshold,
        )
        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores=baseline_scores,
            mutant_scores=mutant_score_map,
            mutant_to_rule=mutant_to_rule,
        )

        conflicts = detect_conflicts(verdicts)

        # --- 7. REPORT
        elapsed = time.perf_counter() - t_start

        if cfg.output_format == "json":
            report = format_json(rules, verdicts, spent_usd=spent, elapsed_s=elapsed)
        elif cfg.output_format == "markdown":
            report = format_markdown_pr_comment(rules, verdicts, conflicts=conflicts, spent_usd=spent)
        else:
            report = format_verdict_table(rules, verdicts, conflicts=conflicts, spent_usd=spent)

        return {
            "rules": rules,
            "verdicts": verdicts,
            "conflicts": conflicts,
            "runs": runs,
            "scores": run_scores,
            "spent_usd": spent,
            "elapsed_s": elapsed,
            "plan": plan,
            "report": report,
            "experiment_id": exp_id,
        }
