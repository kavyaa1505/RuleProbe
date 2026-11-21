"""Statistical analysis: paired comparisons, bootstrap CIs, FDR correction, verdicts."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

from .models import Operator, Rule, RuleVerdict, Run, Score, Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    baseline_scores: Sequence[float],
    mutant_scores: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for mean(mutant) - mean(baseline).
    Returns (effect_size, ci_low, ci_high).
    """
    if not baseline_scores or not mutant_scores:
        return 0.0, 0.0, 0.0

    b = np.array(baseline_scores, dtype=float)
    m = np.array(mutant_scores, dtype=float)
    effect = float(np.mean(m) - np.mean(b))

    rng = np.random.default_rng(42)
    b_sample = rng.choice(b, size=(n_boot, len(b)), replace=True)
    m_sample = rng.choice(m, size=(n_boot, len(m)), replace=True)
    boot_diffs = np.mean(m_sample, axis=1) - np.mean(b_sample, axis=1)

    ci_low = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return effect, ci_low, ci_high


def permutation_p_value(
    baseline_scores: Sequence[float],
    mutant_scores: Sequence[float],
    n_perms: int = 1000,
) -> float:
    """Two-sided permutation test p-value."""
    if not baseline_scores or not mutant_scores:
        return 1.0

    b = np.array(baseline_scores, dtype=float)
    m = np.array(mutant_scores, dtype=float)
    observed = abs(np.mean(m) - np.mean(b))
    if observed < 1e-9:
        return 1.0

    all_scores = np.concatenate([b, m])
    n_b = len(b)
    rng = np.random.default_rng(42)
    perms = np.array([rng.permutation(all_scores) for _ in range(n_perms)])
    diffs = np.abs(np.mean(perms[:, :n_b], axis=1) - np.mean(perms[:, n_b:], axis=1))
    count = int(np.sum(diffs >= observed - 1e-9))
    return (count + 1) / (n_perms + 1)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed), 1):
        adj = min(prev, p * n / (n - rank + 1))
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


# ---------------------------------------------------------------------------
# Noise floor baseline
# ---------------------------------------------------------------------------

def compute_noise_floor(baseline_runs: list[tuple[Run, Score]]) -> float:
    """
    Estimate natural score variance from repeated baseline runs.
    Returns std dev of baseline aggregate scores.
    """
    scores = [s.aggregate_score for _, s in baseline_runs]
    if len(scores) < 2:
        return 0.0
    return float(np.std(scores))


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class RuleAnalyzer:
    """
    Compares mutant runs against baseline, computes per-rule verdicts.
    """

    def __init__(
        self,
        materiality_threshold: float = 2.0,
        alpha: float = 0.05,
        n_boot: int = 1000,
    ):
        self.materiality_threshold = materiality_threshold
        self.alpha = alpha
        self.n_boot = n_boot

    def analyze(
        self,
        rules: list[Rule],
        baseline_scores: dict[str, list[float]],  # scenario_id → scores
        mutant_scores: dict[str, dict[str, list[float]]],  # mutant_id → scenario_id → scores
        mutant_to_rule: dict[str, tuple[str, Operator]],  # mutant_id → (rule_id, operator)
        compliance_rates: dict[str, float] | None = None,  # rule_id → compliance rate
    ) -> list[RuleVerdict]:
        """
        Produce one RuleVerdict per (rule, operator) pair.
        """
        # Flatten baseline scores to a list
        all_baseline = []
        for scores in baseline_scores.values():
            all_baseline.extend(scores)

        noise_floor = float(np.std(all_baseline)) if len(all_baseline) >= 2 else 0.0

        verdicts: list[RuleVerdict] = []
        rule_map = {r.id: r for r in rules}

        # Collect per-mutant p-values for FDR correction
        mutant_effects: list[tuple[str, str, Operator, float, float, float, float, list[str]]] = []

        for mutant_id, scenario_score_map in mutant_scores.items():
            if mutant_id not in mutant_to_rule:
                continue
            rule_id, operator = mutant_to_rule[mutant_id]

            # Gather paired scores
            paired_baseline, paired_mutant, supporting = [], [], []
            for scenario_id, m_scores in scenario_score_map.items():
                b_scores = baseline_scores.get(scenario_id, [])
                if b_scores and m_scores:
                    paired_baseline.extend(b_scores)
                    paired_mutant.extend(m_scores)
                    supporting.append(scenario_id)

            effect, ci_low, ci_high = bootstrap_ci(paired_baseline, paired_mutant, self.n_boot)
            p_val = permutation_p_value(paired_baseline, paired_mutant)
            mutant_effects.append((mutant_id, rule_id, operator, effect, ci_low, ci_high, p_val, supporting))

        # FDR correction
        p_values = [e[6] for e in mutant_effects]
        p_adj_list = benjamini_hochberg(p_values)

        for i, (mutant_id, rule_id, operator, effect, ci_low, ci_high, p_val, supporting) in enumerate(mutant_effects):
            p_adj = p_adj_list[i]
            rule = rule_map.get(rule_id)

            verdict = self._assign_verdict(
                rule=rule,
                operator=operator,
                effect=effect,
                ci_low=ci_low,
                ci_high=ci_high,
                p_adj=p_adj,
                noise_floor=noise_floor,
                compliance_rate=compliance_rates.get(rule_id) if compliance_rates else None,
                has_scenarios=len(supporting) > 0,
            )

            verdicts.append(RuleVerdict(
                rule_id=rule_id,
                operator=operator,
                effect_size=round(effect, 2),
                ci_low=round(ci_low, 2),
                ci_high=round(ci_high, 2),
                p_value=round(p_val, 4),
                p_adj=round(p_adj, 4),
                verdict=verdict,
                supporting_scenarios=supporting,
                compliance_rate=compliance_rates.get(rule_id) if compliance_rates else None,
            ))

        return verdicts

    def _assign_verdict(
        self,
        rule: Rule | None,
        operator: Operator,
        effect: float,
        ci_low: float,
        ci_high: float,
        p_adj: float,
        noise_floor: float,
        compliance_rate: float | None,
        has_scenarios: bool = True,
    ) -> Verdict:
        # Pinned rules are never recommended for removal
        if rule and rule.pinned:
            return Verdict.pinned

        # No scenarios tested this rule
        if not has_scenarios:
            return Verdict.untested

        significant = p_adj < self.alpha
        material = abs(effect) > max(self.materiality_threshold, noise_floor)

        if not significant or not material:
            # Effect is small / not significant → dead weight
            # Split: ignored vs redundant using compliance rate
            if compliance_rate is not None:
                if compliance_rate < 0.5:
                    return Verdict.ignored
                else:
                    return Verdict.redundant
            return Verdict.redundant

        if operator == Operator.remove:
            if effect < 0:
                return Verdict.load_bearing  # removing hurt performance
            else:
                return Verdict.harmful  # removing helped → rule hurts

        if operator == Operator.move:
            return Verdict.order_sensitive

        if operator == Operator.negate:
            if effect < 0:
                return Verdict.load_bearing
            else:
                return Verdict.ignored

        return Verdict.load_bearing


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------

def detect_conflicts(verdicts: list[RuleVerdict]) -> list[tuple[str, str, str]]:
    """
    Detect pairs of rules where one's removal improves another's compliance.
    Returns list of (rule_id_a, rule_id_b, description).
    """
    # Simplified: flag pairs where both have harmful verdict
    harmful = [v for v in verdicts if v.verdict == Verdict.harmful]
    conflicts = []
    for i, a in enumerate(harmful):
        for b in harmful[i + 1:]:
            conflicts.append((a.rule_id, b.rule_id, f"Both rules have harmful verdict — may conflict"))
    return conflicts
