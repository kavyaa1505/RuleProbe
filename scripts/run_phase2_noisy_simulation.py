"""Phase 2: Noisy simulated benchmark with StochasticBenchmarkAdapter.

Sweeps runs-per-mutant N in {3, 5, 10, 20} and effect sizes {small, medium, large}
across >= 200 simulated experiments per setting.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ruleprobe.analyzer import RuleAnalyzer
from ruleprobe.models import Operator, Rule, Verdict


# ---------------------------------------------------------------------------
# Ground Truth Definitions
# ---------------------------------------------------------------------------

PLANTED_RULES = [
    Rule(id="R_lb", text="Always cite factual sources", position=0),
    Rule(id="R_red", text="Use a standard professional tone", position=1),
    Rule(id="R_ign", text="Always include Latin taxonomic names", position=2),
    Rule(id="R_harm", text="Always elaborate extensively without stopping", position=3),
    Rule(id="R_ord", text="Ask clarifying questions first", position=4),
    Rule(id="R_dec", text="Prefer blue markdown headings", position=5),
]

EXPECTED_VERDICTS = {
    "R_lb": Verdict.load_bearing,
    "R_red": Verdict.redundant,
    "R_ign": Verdict.ignored,
    "R_harm": Verdict.harmful,
    "R_ord": Verdict.order_sensitive,
    "R_dec": Verdict.untested,
}

# Probabilities P(compliant | present) vs P(compliant | absent)
EFFECT_PROBS = {
    "large": {
        "R_lb": (0.95, 0.15),   # Delta = -80%
        "R_red": (0.90, 0.90),  # Delta = 0%, high compliance
        "R_ign": (0.20, 0.20),  # Delta = 0%, low compliance
        "R_harm": (0.20, 0.90), # Delta = +70% (removal fixes conflict)
        "R_ord": (0.95, 0.20),  # Sensitive to move operator
    },
    "medium": {
        "R_lb": (0.85, 0.45),   # Delta = -40%
        "R_red": (0.85, 0.85),
        "R_ign": (0.25, 0.25),
        "R_harm": (0.35, 0.75), # Delta = +40%
        "R_ord": (0.85, 0.45),
    },
    "small": {
        "R_lb": (0.80, 0.60),   # Delta = -20%
        "R_red": (0.80, 0.80),
        "R_ign": (0.30, 0.30),
        "R_harm": (0.50, 0.70), # Delta = +20%
        "R_ord": (0.80, 0.60),
    },
}


# ---------------------------------------------------------------------------
# Stochastic Simulator
# ---------------------------------------------------------------------------

class StochasticSimulator:
    """Simulates agent evaluation under realistic non-deterministic output noise."""

    def __init__(self, effect_size: str = "large", random_fail_rate: float = 0.05, rng: np.random.Generator | None = None):
        self.probs = EFFECT_PROBS[effect_size]
        self.random_fail_rate = random_fail_rate
        self.rng = rng or np.random.default_rng(42)

    def sample_scores(self, p_compliant: float, n_runs: int) -> list[float]:
        """Sample scores with Bernoulli compliance + occasional failure + measurement noise."""
        scores = []
        for _ in range(n_runs):
            # Occasional random failure (e.g. network/rate limit/transient glitch)
            if self.rng.random() < self.random_fail_rate:
                scores.append(0.0)
                continue
            compliant = self.rng.random() < p_compliant
            base = 100.0 if compliant else 0.0
            # Continuous score jitter
            jitter = self.rng.normal(0, 5.0)
            score = max(0.0, min(100.0, base + jitter))
            scores.append(score)
        return scores


def run_single_simulation_experiment(
    simulator: StochasticSimulator,
    n_runs: int,
) -> dict[str, Any]:
    """Runs one simulated experiment for the 6 planted rules."""
    baseline_scores: dict[str, list[float]] = {}
    mutant_scores: dict[str, dict[str, list[float]]] = {}
    mutant_to_rule: dict[str, tuple[str, Operator]] = {}
    compliance_rates: dict[str, float] = {}

    # Scenarios targeting specific rules (R_dec has no scenario)
    active_rules = ["R_lb", "R_red", "R_ign", "R_harm", "R_ord"]

    for rid in active_rules:
        sid = f"sc_{rid}"
        p_pres, p_abs = simulator.probs[rid]

        # Baseline: rule is present
        b_scores = simulator.sample_scores(p_pres, n_runs)
        baseline_scores[sid] = b_scores
        compliance_rates[rid] = float(np.mean(b_scores) / 100.0)

        # Mutant:
        # For R_ord, operator is move; for others, operator is remove
        op = Operator.move if rid == "R_ord" else Operator.remove
        mid = f"M_{rid}"
        mutant_to_rule[mid] = (rid, op)
        m_scores = simulator.sample_scores(p_abs, n_runs)
        mutant_scores[mid] = {sid: m_scores}

    # Analyze
    analyzer = RuleAnalyzer(materiality_threshold=2.0)
    verdicts = analyzer.analyze(
        rules=PLANTED_RULES,
        baseline_scores=baseline_scores,
        mutant_scores=mutant_scores,
        mutant_to_rule=mutant_to_rule,
        compliance_rates=compliance_rates,
    )

    verdict_map = {v.rule_id: v.verdict for v in verdicts}
    return verdict_map


# ---------------------------------------------------------------------------
# Wilson score interval calculation
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_parameter_sweep(
    n_values: list[int] = [3, 5, 10, 20],
    effect_sizes: list[str] = ["small", "medium", "large"],
    n_experiments_per_setting: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    print(f"Starting parameter sweep: N in {n_values}, effects in {effect_sizes}, {n_experiments_per_setting} exp/setting (seed={seed})")
    rng = np.random.default_rng(seed)

    sweep_results: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()

    for effect in effect_sizes:
        sweep_results[effect] = {}
        for n_runs in n_values:
            simulator = StochasticSimulator(effect_size=effect, random_fail_rate=0.05, rng=rng)

            # Confusion counts per ground truth class: TP, FP, FN
            classes = [Verdict.load_bearing, Verdict.redundant, Verdict.ignored, Verdict.harmful, Verdict.order_sensitive, Verdict.untested]
            tp = {c: 0 for c in classes}
            fp = {c: 0 for c in classes}
            fn = {c: 0 for c in classes}

            for _ in range(n_experiments_per_setting):
                v_map = run_single_simulation_experiment(simulator, n_runs=n_runs)

                for rule in PLANTED_RULES:
                    expected = EXPECTED_VERDICTS[rule.id]
                    actual = v_map.get(rule.id, Verdict.untested)

                    if actual == expected:
                        tp[expected] += 1
                    else:
                        fn[expected] += 1
                        fp[actual] += 1

            # Compute precision, recall, F1 and Wilson 95% CIs per class
            metrics_per_class = {}
            for c in classes:
                t_pos = tp[c]
                f_pos = fp[c]
                f_neg = fn[c]

                prec = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0.0
                rec = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                rec_ci = wilson_ci(t_pos, t_pos + f_neg)
                prec_ci = wilson_ci(t_pos, t_pos + f_pos)

                metrics_per_class[c.value] = {
                    "tp": t_pos,
                    "fp": f_pos,
                    "fn": f_neg,
                    "precision": round(prec, 4),
                    "precision_95ci": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
                    "recall": round(rec, 4),
                    "recall_95ci": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
                    "f1": round(f1, 4),
                }

            # Macro averages across classes
            macro_prec = float(np.mean([m["precision"] for m in metrics_per_class.values()]))
            macro_rec = float(np.mean([m["recall"] for m in metrics_per_class.values()]))
            macro_f1 = float(np.mean([m["f1"] for m in metrics_per_class.values()]))

            sweep_results[effect][f"N_{n_runs}"] = {
                "n_runs": n_runs,
                "effect_size": effect,
                "n_experiments": n_experiments_per_setting,
                "macro_precision": round(macro_prec, 4),
                "macro_recall": round(macro_rec, 4),
                "macro_f1": round(macro_f1, 4),
                "classes": metrics_per_class,
            }

            lb_rec = metrics_per_class[Verdict.load_bearing.value]["recall"]
            print(f"  [{effect.upper():<6} | N={n_runs:>2}] Macro F1={macro_f1:.1%}, LoadBearing Recall={lb_rec:.1%}")

    elapsed = time.perf_counter() - t0
    print(f"Sweep completed in {elapsed:.2f}s")

    # Determine minimum N for recall >= 0.90 per effect size
    min_n_recommendations = {}
    for effect in effect_sizes:
        rec_found = None
        for n_runs in n_values:
            lb_rec = sweep_results[effect][f"N_{n_runs}"]["classes"][Verdict.load_bearing.value]["recall"]
            if lb_rec >= 0.90:
                rec_found = n_runs
                break
        min_n_recommendations[effect] = rec_found

    return {
        "sweep": sweep_results,
        "min_n_for_recall_90": min_n_recommendations,
        "elapsed_s": round(elapsed, 2),
        "seed": seed,
    }


def main():
    results = run_parameter_sweep(
        n_values=[3, 5, 10, 20],
        effect_sizes=["small", "medium", "large"],
        n_experiments_per_setting=200,
        seed=42,
    )

    out_file = Path("results/phase2_noisy_simulation.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved full simulation sweep data to {out_file}")


if __name__ == "__main__":
    main()
