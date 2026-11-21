"""Statistical analysis: scenario-level paired comparisons, hierarchical bootstrap CIs,
sign-flip permutation tests, FDR correction, and materiality-gated verdicts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

from .models import Operator, Rule, RuleVerdict, Run, Score, Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario-level hierarchical bootstrap CI
# ---------------------------------------------------------------------------

def scenario_paired_bootstrap_ci(
    baseline_scores: dict[str, list[float]],
    mutant_scores: dict[str, list[float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Two-level hierarchical paired bootstrap for scenario-level mean differences:
        d_s = mean(mutant runs in s) - mean(baseline runs in s)
        effect = mean_s(d_s)

    Level 1: Resamples runs with replacement within each scenario.
    Level 2: Resamples scenarios with replacement.

    Returns (effect_size, ci_low, ci_high).
    """
    common_scenarios = [
        s for s in baseline_scores
        if s in mutant_scores and len(baseline_scores[s]) > 0 and len(mutant_scores[s]) > 0
    ]
    if not common_scenarios:
        return 0.0, 0.0, 0.0

    S = len(common_scenarios)
    b_runs_list = [np.array(baseline_scores[s], dtype=float) for s in common_scenarios]
    m_runs_list = [np.array(mutant_scores[s], dtype=float) for s in common_scenarios]

    d_s = [float(np.mean(m_runs_list[i]) - np.mean(b_runs_list[i])) for i in range(S)]
    effect = float(np.mean(d_s))

    # Single scenario edge-case: resample runs directly
    if S == 1:
        return bootstrap_ci(b_runs_list[0], m_runs_list[0], n_boot=n_boot, alpha=alpha)

    rng = np.random.default_rng(seed)

    # Level 1: Resample runs within each scenario to get bootstrap scenario diffs (S, n_boot)
    d_boot_by_scen = np.zeros((S, n_boot), dtype=float)
    for i in range(S):
        b_r = b_runs_list[i]
        m_r = m_runs_list[i]
        b_idx = rng.integers(0, len(b_r), size=(n_boot, len(b_r)))
        m_idx = rng.integers(0, len(m_r), size=(n_boot, len(m_r)))
        b_means = np.mean(b_r[b_idx], axis=1)
        m_means = np.mean(m_r[m_idx], axis=1)
        d_boot_by_scen[i, :] = m_means - b_means

    # Level 2: Resample scenarios with replacement
    scen_idx = rng.integers(0, S, size=(n_boot, S))
    # Shape: (n_boot, S) -> average over scenarios for each bootstrap sample
    boot_means = np.mean(np.take_along_axis(d_boot_by_scen.T, scen_idx, axis=1), axis=1)

    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return effect, ci_low, ci_high


def scenario_sign_flip_p_value(
    d_s: list[float],
    n_perms: int = 2000,
    seed: int = 42,
) -> float:
    """
    Two-sided sign-flip permutation test on paired scenario differences d_s.
    Under H0, each scenario difference is symmetric around 0 (equal probability of +/- sign).
    """
    d = np.array(d_s, dtype=float)
    S = len(d)
    if S == 0:
        return 1.0

    t_obs = abs(float(np.mean(d)))
    if t_obs < 1e-9:
        return 1.0

    # If S is small (<= 12), enumerate all 2^S sign flips exactly
    if S <= 12:
        n_comb = 1 << S
        comb = np.arange(n_comb, dtype=np.int32)[:, None]
        bits = (comb >> np.arange(S)) & 1
        signs = 2.0 * bits - 1.0  # shape (2^S, S)
        stats = np.abs(np.mean(signs * d, axis=1))
        count = int(np.sum(stats >= t_obs - 1e-9))
        return count / n_comb
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perms, S))
        stats = np.abs(np.mean(signs * d, axis=1))
        count = int(np.sum(stats >= t_obs - 1e-9))
        return (count + 1) / (n_perms + 1)


# ---------------------------------------------------------------------------
# Fallback / 1D run-level helpers (retained for backward compatibility)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    baseline_scores: Sequence[float],
    mutant_scores: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """1D paired bootstrap CI for run-level scores."""
    if len(baseline_scores) == 0 or len(mutant_scores) == 0:
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
    seed: int = 42,
) -> float:
    """1D two-sided permutation test p-value."""
    if len(baseline_scores) == 0 or len(mutant_scores) == 0:
        return 1.0

    b = np.array(baseline_scores, dtype=float)
    m = np.array(mutant_scores, dtype=float)
    observed = abs(np.mean(m) - np.mean(b))
    if observed < 1e-9:
        return 1.0

    all_scores = np.concatenate([b, m])
    n_b = len(b)
    rng = np.random.default_rng(seed)
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
# Empirical Noise Floor Diagnostic
# ---------------------------------------------------------------------------

def compute_empirical_noise_floor(
    baseline_scores: dict[str, list[float]],
    n_splits: int = 100,
    seed: int = 42,
) -> float:
    """
    Estimate empirical noise floor via split-half differences of baseline runs across scenarios.
    Returns the half-width (95th percentile of absolute differences) of the A/A scenario difference distribution.
    Used purely as a diagnostic to check if the materiality margin m is reasonable.
    """
    scenarios = [s for s, sc in baseline_scores.items() if len(sc) >= 2]
    if not scenarios:
        return 0.0

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_splits):
        scen_diffs = []
        for s in scenarios:
            scores = list(baseline_scores[s])
            rng.shuffle(scores)
            mid = len(scores) // 2
            half_a = scores[:mid]
            half_b = scores[mid:]
            scen_diffs.append(np.mean(half_b) - np.mean(half_a))
        diffs.append(abs(float(np.mean(scen_diffs))))

    if not diffs:
        return 0.0
    return float(np.percentile(diffs, 95))


def compute_noise_floor(baseline_runs: list[tuple[Run, Score]]) -> float:
    """Legacy helper returning std dev of baseline aggregate scores."""
    scores = [s.aggregate_score for _, s in baseline_runs]
    if len(scores) < 2:
        return 0.0
    return float(np.std(scores))


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

class RuleAnalyzer:
    """
    Compares mutant runs against baseline using scenario-level inference:
    - Unit of analysis: SCENARIO. d_s = mean(mutant_s) - mean(baseline_s).
    - Hierarchical paired bootstrap for confidence intervals.
    - Sign-flip permutation test on d_s for p-values.
    - Benjamini-Hochberg FDR correction across all tests.
    - Materiality margin m:
        * load_bearing / harmful: p_adj < alpha AND |mean d| > m
        * redundant / ignored: 95% CI entirely inside [-m, +m]
        * inconclusive: CI extends outside [-m, +m] without significant effect
    """

    def __init__(
        self,
        materiality_threshold: float = 5.0,  # default 5 points
        alpha: float = 0.05,
        n_boot: int = 1000,
        n_perms: int = 2000,
        seed: int = 42,
    ):
        self.materiality_threshold = materiality_threshold
        self.alpha = alpha
        self.n_boot = n_boot
        self.n_perms = n_perms
        self.seed = seed
        self.last_empirical_noise_floor: float = 0.0

    def analyze(
        self,
        rules: list[Rule],
        baseline_scores: dict[str, list[float]],  # scenario_id -> scores
        mutant_scores: dict[str, dict[str, list[float]]],  # mutant_id -> scenario_id -> scores
        mutant_to_rule: dict[str, tuple[str, Operator]],  # mutant_id -> (rule_id, operator)
        compliance_rates: dict[str, float] | None = None,  # rule_id -> compliance rate
    ) -> list[RuleVerdict]:
        """Produce one RuleVerdict per (rule, operator) pair."""
        # Compute empirical noise floor as a diagnostic check
        self.last_empirical_noise_floor = compute_empirical_noise_floor(baseline_scores, seed=self.seed)
        if self.materiality_threshold < self.last_empirical_noise_floor:
            logger.warning(
                f"Configured materiality margin ({self.materiality_threshold} pts) is below "
                f"empirical noise floor ({self.last_empirical_noise_floor:.2f} pts). "
                f"Effects smaller than noise floor may be indistinguishable from split-half variance."
            )

        verdicts: list[RuleVerdict] = []
        rule_map = {r.id: r for r in rules}

        mutant_effects: list[tuple[str, str, Operator, float, float, float, float, list[str]]] = []

        for mutant_id, scenario_score_map in mutant_scores.items():
            if mutant_id not in mutant_to_rule:
                continue
            rule_id, operator = mutant_to_rule[mutant_id]

            common_scenarios = [
                s for s in scenario_score_map
                if s in baseline_scores and len(baseline_scores[s]) > 0 and len(scenario_score_map[s]) > 0
            ]

            if not common_scenarios:
                mutant_effects.append((mutant_id, rule_id, operator, 0.0, 0.0, 0.0, 1.0, []))
                continue

            # Scenario differences d_s
            d_s = [
                float(np.mean(scenario_score_map[s]) - np.mean(baseline_scores[s]))
                for s in common_scenarios
            ]

            # Hierarchical paired bootstrap for CI
            effect, ci_low, ci_high = scenario_paired_bootstrap_ci(
                baseline_scores,
                scenario_score_map,
                n_boot=self.n_boot,
                alpha=self.alpha,
                seed=self.seed,
            )

            # Sign-flip permutation test for p-value
            if len(common_scenarios) == 1:
                # Single-scenario fallback to run-level permutation test
                p_val = permutation_p_value(
                    baseline_scores[common_scenarios[0]],
                    scenario_score_map[common_scenarios[0]],
                    n_perms=self.n_perms,
                    seed=self.seed,
                )
            else:
                p_val = scenario_sign_flip_p_value(d_s, n_perms=self.n_perms, seed=self.seed)

            mutant_effects.append((mutant_id, rule_id, operator, effect, ci_low, ci_high, p_val, common_scenarios))

        # FDR correction across ALL (rule x operator) tests together
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
        compliance_rate: float | None = None,
        has_scenarios: bool = True,
    ) -> Verdict:
        # Pinned rules are never recommended for removal
        if rule and rule.pinned:
            return Verdict.pinned

        # No scenarios tested this rule
        if not has_scenarios:
            return Verdict.untested

        m = self.materiality_threshold

        # 1. Statistically significant AND practically material
        if p_adj < self.alpha and abs(effect) > m:
            if operator == Operator.remove:
                return Verdict.load_bearing if effect < -m else Verdict.harmful
            elif operator == Operator.move:
                return Verdict.order_sensitive
            elif operator == Operator.negate:
                return Verdict.load_bearing if effect < -m else Verdict.ignored
            else:
                return Verdict.load_bearing if effect < -m else Verdict.harmful

        # 2. Equivalence: 95% CI lies entirely inside [-m, +m]
        # Rule's effect is proven immaterial within margin m
        if ci_low >= -m and ci_high <= m:
            if compliance_rate is not None and compliance_rate < 0.5:
                return Verdict.ignored
            return Verdict.redundant

        # 3. Inconclusive: CI extends outside [-m, +m] while not meeting significance
        # Never label such a rule redundant!
        return Verdict.inconclusive


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------

def detect_conflicts(verdicts: list[RuleVerdict]) -> list[tuple[str, str, str]]:
    """
    Detect pairs of rules where one's removal improves another's compliance.
    Returns list of (rule_id_a, rule_id_b, description).
    """
    harmful = [v for v in verdicts if v.verdict == Verdict.harmful]
    conflicts = []
    for i, a in enumerate(harmful):
        for b in harmful[i + 1:]:
            conflicts.append((a.rule_id, b.rule_id, "Both rules have harmful verdict — may conflict"))
    return conflicts
