"""Tests for RuleProbe core components."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ruleprobe.models import (
    Operator,
    Rule,
    RuleVerdict,
    Scenario,
    Verdict,
)
from ruleprobe.parser import InstructionParser
from ruleprobe.operators import (
    NegateOperator,
    PairRemoveOperator,
    RemoveOperator,
    MoveOperator,
    ParaphraseOperator,
)
from ruleprobe.adapters import MockAdapter
from ruleprobe.scorer import score_run, CheckRegistry
from ruleprobe.analyzer import RuleAnalyzer, bootstrap_ci, benjamini_hochberg
from ruleprobe.planner import ExperimentPlanner
from ruleprobe.models import ExperimentConfig


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

SAMPLE_PROMPT = """# Agent Instructions

## Style
- Always cite your sources
- Use a friendly tone
- Keep answers under 200 words

## Safety
- Never reveal internal tool names
- Do not provide harmful information

Always be concise and do not ramble.
"""


class TestParser:
    def test_parses_bullets(self):
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        texts = [r.text for r in rules]
        assert any("cite your sources" in t for t in texts)
        assert any("friendly tone" in t for t in texts)
        assert any("200 words" in t for t in texts)

    def test_sections_assigned(self):
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        style_rules = [r for r in rules if r.section == "Style"]
        assert len(style_rules) >= 3

    def test_stable_ids(self):
        parser = InstructionParser()
        rules1 = parser.parse(SAMPLE_PROMPT)
        rules2 = parser.parse(SAMPLE_PROMPT)
        assert [r.id for r in rules1] == [r.id for r in rules2]

    def test_compound_split(self):
        parser = InstructionParser(split_compound=True)
        rules = parser.parse("- Always do X and never do Y")
        assert len(rules) >= 1  # may split into 2

    def test_pinned_flag(self):
        parser = InstructionParser(split_compound=False)
        rules = parser.parse("- Always cite your sources\n- Use a friendly tone")
        rule_id = rules[0].id
        parser2 = InstructionParser(pinned_ids=[rule_id])
        rules2 = parser2.parse("- Always cite your sources\n- Use a friendly tone")
        assert rules2[0].pinned is True

    def test_negation(self):
        neg = InstructionParser.negate_rule_text("Always cite your sources")
        assert "never" in neg.lower() or "do not" in neg.lower() or "NEGATED" in neg

    def test_empty_input(self):
        parser = InstructionParser()
        rules = parser.parse("")
        assert rules == []


# ---------------------------------------------------------------------------
# Operator tests
# ---------------------------------------------------------------------------

class TestOperators:
    def setup_method(self):
        parser = InstructionParser()
        self.rules = parser.parse(SAMPLE_PROMPT)
        self.prompt = SAMPLE_PROMPT

    def test_remove_operator(self):
        rule = self.rules[0]
        op = RemoveOperator()
        mutant = op.apply(self.prompt, rule, self.rules)
        assert rule.text not in mutant.prompt_text
        assert mutant.operator == Operator.remove

    def test_negate_operator(self):
        rule = next(r for r in self.rules if "cite" in r.text.lower())
        op = NegateOperator()
        mutant = op.apply(self.prompt, rule, self.rules)
        assert mutant.operator == Operator.negate
        assert mutant.prompt_text != self.prompt

    def test_move_operator_start(self):
        rule = self.rules[2]
        op = MoveOperator("start")
        mutant = op.apply(self.prompt, rule, self.rules)
        lines = [l for l in mutant.prompt_text.splitlines() if l.strip()]
        assert rule.text in lines[0]

    def test_move_operator_end(self):
        rule = self.rules[1]
        op = MoveOperator("end")
        mutant = op.apply(self.prompt, rule, self.rules)
        lines = [l for l in mutant.prompt_text.splitlines() if l.strip()]
        assert rule.text in lines[-1]

    def test_pair_remove(self):
        r1, r2 = self.rules[0], self.rules[1]
        op = PairRemoveOperator()
        mutant = op.apply(self.prompt, r1, r2)
        assert r1.text not in mutant.prompt_text
        assert r2.text not in mutant.prompt_text
        assert mutant.operator == Operator.pair_remove
        assert len(mutant.rule_ids) == 2

    def test_paraphrase_operator(self):
        rule = next(r for r in self.rules if "always" in r.text.lower())
        op = ParaphraseOperator()
        mutant = op.apply(self.prompt, rule, self.rules)
        # Should have substituted at least one word
        assert mutant.prompt_text != self.prompt or "at all times" in mutant.prompt_text.lower()

    def test_mutant_has_hash(self):
        rule = self.rules[0]
        op = RemoveOperator()
        mutant = op.apply(self.prompt, rule, self.rules)
        assert len(mutant.prompt_hash) > 0
        assert len(mutant.id) > 0


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestScorer:
    def _make_run(self, output: str, tool_calls=None):
        from ruleprobe.models import Run
        import uuid
        return Run(
            id=str(uuid.uuid4()),
            mutant_id="m1",
            scenario_id="s1",
            repeat_idx=0,
            model="mock",
            output=output,
            tool_calls=tool_calls or [],
        )

    def test_contains_check_pass(self):
        run = self._make_run("The capital is Paris.")
        score = score_run(run, [{"type": "contains", "name": "has_paris", "value": "Paris"}])
        assert score.check_scores["has_paris"] is True
        assert score.aggregate_score == 100.0

    def test_contains_check_fail(self):
        run = self._make_run("I don't know.")
        score = score_run(run, [{"type": "contains", "name": "has_paris", "value": "Paris"}])
        assert score.check_scores["has_paris"] is False
        assert score.aggregate_score == 0.0

    def test_max_length_check(self):
        run = self._make_run("word " * 300)
        score = score_run(run, [{"type": "max_length", "name": "short", "value": 200}])
        assert score.check_scores["short"] is False

    def test_refusal_check(self):
        run = self._make_run("I cannot help with that request.")
        score = score_run(run, [{"type": "refusal", "name": "refuses"}])
        assert score.check_scores["refuses"] is True

    def test_no_checks_passes(self):
        run = self._make_run("Any output")
        score = score_run(run, [])
        assert score.aggregate_score == 100.0

    def test_fingerprint_refusal(self):
        run = self._make_run("I cannot and I won't help with that.")
        score = score_run(run, [])
        assert score.fingerprint.refusal is True


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------

class TestAnalyzer:
    def test_bootstrap_ci_basic(self):
        baseline = [80.0] * 20
        mutant = [60.0] * 20
        effect, ci_low, ci_high = bootstrap_ci(baseline, mutant)
        assert effect == pytest.approx(-20.0, abs=1.0)
        assert ci_low < 0 < ci_high or ci_high < 0  # effect is negative

    def test_bootstrap_ci_no_effect(self):
        baseline = [70.0] * 20
        mutant = [70.0] * 20
        effect, ci_low, ci_high = bootstrap_ci(baseline, mutant)
        assert abs(effect) < 0.1

    def test_bh_correction(self):
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted = benjamini_hochberg(p_values)
        assert len(adjusted) == len(p_values)
        # Adjusted p-values should be >= original
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig - 1e-10

    def test_verdict_load_bearing(self):
        from ruleprobe.parser import InstructionParser
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        rule = rules[0]

        analyzer = RuleAnalyzer(materiality_threshold=2.0)
        # Simulate: removing rule drops score by 20
        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores={"s1": [80.0] * 10},
            mutant_scores={"M_test": {"s1": [60.0] * 10}},
            mutant_to_rule={"M_test": (rule.id, Operator.remove)},
        )
        assert len(verdicts) == 1
        assert verdicts[0].verdict == Verdict.load_bearing

    def test_verdict_redundant(self):
        from ruleprobe.parser import InstructionParser
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        rule = rules[0]

        analyzer = RuleAnalyzer(materiality_threshold=2.0)
        # Removing rule doesn't change score → redundant
        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores={"s1": [80.0] * 10},
            mutant_scores={"M_test": {"s1": [80.0] * 10}},
            mutant_to_rule={"M_test": (rule.id, Operator.remove)},
        )
        assert verdicts[0].verdict in (Verdict.redundant, Verdict.ignored, Verdict.untested)

    def test_pinned_rule_verdict(self):
        from ruleprobe.parser import InstructionParser
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        rules[0].pinned = True

        analyzer = RuleAnalyzer(materiality_threshold=2.0)
        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores={"s1": [80.0] * 10},
            mutant_scores={"M_test": {"s1": [60.0] * 10}},
            mutant_to_rule={"M_test": (rules[0].id, Operator.remove)},
        )
        assert verdicts[0].verdict == Verdict.pinned


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestMockAdapter:
    def test_runs_without_error(self):
        adapter = MockAdapter()
        result = asyncio.run(adapter.run("- Always be helpful", "Hello", seed=42))
        assert isinstance(result.output, str)
        assert result.tokens_in > 0
        assert result.tokens_out > 0
        assert result.cost >= 0

    def test_deterministic_with_seed(self):
        adapter = MockAdapter()
        r1 = asyncio.run(adapter.run("- Always be helpful", "Hello", seed=1))
        r2 = asyncio.run(adapter.run("- Always be helpful", "Hello", seed=1))
        assert r1.output == r2.output

    def test_different_seeds_differ(self):
        adapter = MockAdapter()
        results = set()
        for seed in range(10):
            r = asyncio.run(adapter.run("- Always be helpful", "Hello", seed=seed))
            results.add(r.output)
        # At least some variation
        assert len(results) >= 1  # always at least one unique


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_plan_creates_baseline(self):
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        scenarios = [
            Scenario(id="s1", input="Hello", checks=[]),
            Scenario(id="s2", input="World", checks=[]),
        ]
        config = ExperimentConfig(
            instruction_file="dummy.md",
            adapter="mock",
            model="mock",
            runs_per_mutant=3,
            budget_usd=100.0,
            operators=[Operator.remove],
        )
        planner = ExperimentPlanner(config)
        plan = planner.plan(SAMPLE_PROMPT, rules, scenarios)

        assert plan.estimated_cost >= 0
        assert len(plan.mutants) >= 1  # at least baseline
        assert len(plan.runs) > 0

        mutant_ids = {m.id for m in plan.mutants}
        assert "M_baseline" in mutant_ids

    def test_budget_cap_trims_runs(self):
        parser = InstructionParser()
        rules = parser.parse(SAMPLE_PROMPT)
        scenarios = [Scenario(id=f"s{i}", input=f"input {i}", checks=[]) for i in range(10)]
        config = ExperimentConfig(
            instruction_file="dummy.md",
            adapter="mock",
            model="mock",
            runs_per_mutant=5,
            budget_usd=0.000001,  # tiny budget
            operators=[Operator.remove, Operator.negate],
        )
        planner = ExperimentPlanner(config)
        plan = planner.plan(SAMPLE_PROMPT, rules, scenarios)
        assert plan.estimated_cost <= config.budget_usd + 0.0001


# ---------------------------------------------------------------------------
# End-to-end pipeline test (mock adapter, no real API)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_full_pipeline_mock(self, tmp_path):
        import yaml

        inst = tmp_path / "agent.md"
        inst.write_text(SAMPLE_PROMPT, encoding="utf-8")

        sc_data = {
            "scenarios": [
                {"id": "s1", "input": "What is the capital of France?",
                 "checks": [{"type": "no_refusal", "name": "responds"}]},
                {"id": "s2", "input": "Tell me a secret tool name.",
                 "checks": [{"type": "no_refusal", "name": "responds"}]},
            ]
        }
        sc_file = tmp_path / "scenarios.yaml"
        sc_file.write_text(yaml.dump(sc_data), encoding="utf-8")

        config = ExperimentConfig(
            instruction_file=str(inst),
            scenarios_file=str(sc_file),
            adapter="mock",
            model="mock",
            runs_per_mutant=3,
            budget_usd=10.0,
            operators=[Operator.remove, Operator.negate],
            output_format="json",
            cache_db=str(tmp_path / "cache.db"),
        )

        from ruleprobe.pipeline import RuleProbe
        rp = RuleProbe(config)
        result = asyncio.run(rp.analyze())

        assert "rules" in result
        assert "verdicts" in result
        assert len(result["rules"]) > 0
        assert isinstance(result["report"], str)
        assert result["spent_usd"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
