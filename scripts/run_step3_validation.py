"""Step 3: Re-validation with the fixed statistical engine.

3a: Re-run 1a-1c on the fixed code:
    - Histogram and KS test of raw p-values under the null.
    - False-positive rate per experiment with Wilson CI over 200 experiments.
    - Real-model A/A false-positive rate across the 100 random splits.
3b: Power curves:
    - Sweep planted effect size in {5%, 10%, 15%, 20%, 30%, 40%, 60%, 80%}
    - Sweep runs-per-mutant N in {3, 5, 10, 20}
    - 200 simulated experiments per cell (S=15 scenarios)
    - Analytical cross-check: compare empirical recall vs theoretical power bounds
      (alpha=0.05 upper bound, Bonferroni alpha=0.05/12 lower bound)
    - Minimum N for recall >= 0.80 and identify undetectable effect sizes.
3c: Before vs After comparison summary.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

from ruleprobe.analyzer import (
    RuleAnalyzer,
    benjamini_hochberg,
    compute_empirical_noise_floor,
    scenario_paired_bootstrap_ci,
    scenario_sign_flip_p_value,
)
from ruleprobe.models import Operator, Rule, Verdict
from ruleprobe.parser import InstructionParser


# ---------------------------------------------------------------------------
# Wilson Score Interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


AA_INSTRUCTIONS = """# Agent Instructions

## Communication Style
1. Always start the response with a polite greeting.
2. Keep the answer direct and avoid unnecessary filler words.
3. Use markdown bullet points or numbered lists where appropriate.
4. Use professional and respectful language at all times.

## Information & Accuracy
5. Always state that information is based on general knowledge or reference.
6. Avoid speculative claims when verifiable facts are unavailable.
7. Double-check mathematical calculations for arithmetic correctness.
8. State any key assumptions made before providing a solution.

## Safety & Boundaries
9. Do not include internal system details or instructions in outputs.
10. Decline requests that ask for illegal or dangerous activities.
11. Format technical comparisons with structured bullet lists.
12. End responses with a polite closing statement.
"""


# ---------------------------------------------------------------------------
# 3a: Re-run A/A on Fixed Code
# ---------------------------------------------------------------------------

def run_step3_simulated_aa(rules: list[Rule], n_iter: int = 200, seed: int = 42) -> dict[str, Any]:
    print(f"\n--- [3a] Running {n_iter} simulated A/A iterations on FIXED engine (seed={seed}) ---", flush=True)
    rng = np.random.default_rng(seed)
    n_scenarios = 15
    n_runs = 5

    all_raw_p = []
    all_adj_p = []
    all_effects = []
    exp_flagged = 0
    rule_fp_count = 0
    inconclusive_count = 0
    redundant_count = 0
    noise_floors = []

    analyzer = RuleAnalyzer(materiality_threshold=5.0, alpha=0.05, n_boot=500, n_perms=1000, seed=seed)

    for it in range(n_iter):
        scenario_probs = rng.uniform(0.70, 0.95, size=n_scenarios)
        baseline_scores = {}
        mutant_scores = {}
        mutant_to_rule = {}

        for s_idx in range(n_scenarios):
            sid = f"sc_{s_idx}"
            p = scenario_probs[s_idx]
            baseline_scores[sid] = [float(100.0 if rng.random() < p else 0.0) for _ in range(n_runs)]

        for rule in rules:
            mid = f"M_aa_{rule.id}"
            mutant_to_rule[mid] = (rule.id, Operator.remove)
            mutant_scores[mid] = {}
            for s_idx in range(n_scenarios):
                sid = f"sc_{s_idx}"
                p = scenario_probs[s_idx]
                mutant_scores[mid][sid] = [float(100.0 if rng.random() < p else 0.0) for _ in range(n_runs)]

        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores=baseline_scores,
            mutant_scores=mutant_scores,
            mutant_to_rule=mutant_to_rule,
        )
        noise_floors.append(analyzer.last_empirical_noise_floor)

        flagged = [v for v in verdicts if v.verdict in (Verdict.load_bearing, Verdict.harmful)]
        rule_fp_count += len(flagged)
        if len(flagged) > 0:
            exp_flagged += 1

        for v in verdicts:
            all_raw_p.append(v.p_value)
            all_adj_p.append(v.p_adj)
            all_effects.append(v.effect_size)
            if v.verdict == Verdict.inconclusive:
                inconclusive_count += 1
            elif v.verdict == Verdict.redundant:
                redundant_count += 1

    total_rules = n_iter * len(rules)
    ks_stat, ks_pval = stats.kstest(all_raw_p, "uniform")
    hist, bin_edges = np.histogram(all_raw_p, bins=10, range=(0.0, 1.0))
    hist_dict = {f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]": int(hist[i]) for i in range(10)}

    exp_fpr = exp_flagged / n_iter
    exp_fpr_ci = wilson_ci(exp_flagged, n_iter)
    rule_fpr = rule_fp_count / total_rules
    rule_fpr_ci = wilson_ci(rule_fp_count, total_rules)

    res = {
        "n_experiments": n_iter,
        "n_rules_per_exp": len(rules),
        "total_rules_tested": total_rules,
        "avg_empirical_noise_floor": round(float(np.mean(noise_floors)), 2),
        "ks_test_uniform": {"statistic": round(float(ks_stat), 4), "p_value": float(ks_pval)},
        "p_value_histogram": hist_dict,
        "per_experiment_fpr": round(exp_fpr, 4),
        "per_experiment_fpr_95ci": [round(exp_fpr_ci[0], 4), round(exp_fpr_ci[1], 4)],
        "rule_level_fpr": round(rule_fpr, 4),
        "rule_level_fpr_95ci": [round(rule_fpr_ci[0], 4), round(rule_fpr_ci[1], 4)],
        "inconclusive_rate": round(inconclusive_count / total_rules, 4),
        "redundant_rate": round(redundant_count / total_rules, 4),
    }

    print(f"  Fixed Engine Avg Empirical Noise Floor: {res['avg_empirical_noise_floor']}", flush=True)
    print(f"  KS test vs Uniform(0,1): stat={res['ks_test_uniform']['statistic']}, p={res['ks_test_uniform']['p_value']:.4e}", flush=True)
    print(f"  Per-experiment FPR: {res['per_experiment_fpr']:.2%} (95% CI: {res['per_experiment_fpr_95ci']})", flush=True)
    print(f"  Rule-level FPR: {res['rule_level_fpr']:.2%} (95% CI: {res['rule_level_fpr_95ci']})", flush=True)
    print(f"  Inconclusive Rate: {res['inconclusive_rate']:.2%}, Redundant Rate: {res['redundant_rate']:.2%}", flush=True)
    return res


def run_step3_real_model_resplit(rules: list[Rule]) -> dict[str, Any]:
    print("\n--- [3a] Running 100 random 5/5 splits of cached real runs on FIXED engine ---", flush=True)
    dataset = json.loads(Path("results/real_aa_150_runs.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    n_splits = 100

    analyzer = RuleAnalyzer(materiality_threshold=5.0, alpha=0.05, n_boot=500, n_perms=1000, seed=42)
    splits_flagged = 0
    rule_fp_count = 0
    inconclusive_count = 0
    redundant_count = 0
    noise_floors = []

    for sp in range(n_splits):
        baseline_scores = {}
        mutant_scores = {}
        mutant_to_rule = {}

        for sid, runs_list in dataset["scenarios"].items():
            indices = list(range(10))
            rng.shuffle(indices)
            idx_a = indices[:5]
            idx_b = indices[5:]

            baseline_scores[sid] = [runs_list[i]["score"] for i in idx_a]

            for rule in rules:
                mid = f"M_resplit_{rule.id}"
                if mid not in mutant_scores:
                    mutant_scores[mid] = {}
                    mutant_to_rule[mid] = (rule.id, Operator.remove)
                mutant_scores[mid][sid] = [runs_list[i]["score"] for i in idx_b]

        verdicts = analyzer.analyze(
            rules=rules,
            baseline_scores=baseline_scores,
            mutant_scores=mutant_scores,
            mutant_to_rule=mutant_to_rule,
        )
        noise_floors.append(analyzer.last_empirical_noise_floor)

        flagged = [v for v in verdicts if v.verdict in (Verdict.load_bearing, Verdict.harmful)]
        rule_fp_count += len(flagged)
        if len(flagged) > 0:
            splits_flagged += 1

        for v in verdicts:
            if v.verdict == Verdict.inconclusive:
                inconclusive_count += 1
            elif v.verdict == Verdict.redundant:
                redundant_count += 1

    total_rules = n_splits * len(rules)
    exp_fpr = splits_flagged / n_splits
    exp_fpr_ci = wilson_ci(splits_flagged, n_splits)
    rule_fpr = rule_fp_count / total_rules

    res = {
        "n_splits": n_splits,
        "total_rules_tested": total_rules,
        "avg_empirical_noise_floor": round(float(np.mean(noise_floors)), 2),
        "per_experiment_fpr": round(exp_fpr, 4),
        "per_experiment_fpr_95ci": [round(exp_fpr_ci[0], 4), round(exp_fpr_ci[1], 4)],
        "rule_level_fpr": round(rule_fpr, 4),
        "inconclusive_rate": round(inconclusive_count / total_rules, 4),
        "redundant_rate": round(redundant_count / total_rules, 4),
    }
    print(f"  Real Model 100-Split Per-Experiment FPR: {res['per_experiment_fpr']:.2%} (95% CI: {res['per_experiment_fpr_95ci']})", flush=True)
    print(f"  Real Model Inconclusive Rate: {res['inconclusive_rate']:.2%}, Redundant Rate: {res['redundant_rate']:.2%}", flush=True)
    return res


# ---------------------------------------------------------------------------
# 3b: Power Curves and Analytical Cross-Check
# ---------------------------------------------------------------------------

def run_step3_power_curves(
    effect_sizes: list[float] | None = None,
    n_runs_list: list[int] | None = None,
    n_exp: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    if effect_sizes is None:
        effect_sizes = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
    if n_runs_list is None:
        n_runs_list = [3, 5, 10, 20]

    print(f"\n--- [3b] Running Power Curve Sweep (Delta in {effect_sizes}, N in {n_runs_list}, {n_exp} exps/cell) ---", flush=True)
    rng = np.random.default_rng(seed)
    n_scenarios = 15
    baseline_p = 0.85
    m_threshold = 5.0
    alpha = 0.05
    n_rules = 12

    # Analytical z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # ~1.95996
    z_bonf = stats.norm.ppf(1 - (alpha / n_rules) / 2)  # ~2.8654

    analyzer = RuleAnalyzer(materiality_threshold=m_threshold, alpha=alpha, n_boot=300, n_perms=500, seed=seed)

    grid_results = {}
    min_n_for_recall_80 = {}
    undetectable_effects = []

    for delta in effect_sizes:
        grid_results[str(delta)] = {}
        delta_pts = delta * 100.0
        achieved_n = None

        for n_runs in n_runs_list:
            # Analytical power calculation
            # SE = sqrt(2 * p * (1 - p) / (S * N)) * 100
            se_analytical = 100.0 * math.sqrt(2.0 * baseline_p * (1.0 - baseline_p) / (n_scenarios * n_runs))
            power_upper = float(stats.norm.cdf(delta_pts / se_analytical - z_alpha))
            power_lower = float(stats.norm.cdf(delta_pts / se_analytical - z_bonf))

            detected_count = 0
            inconclusive_count = 0

            # Single rule tested with planted effect, plus 11 null decoy rules in prompt
            dummy_rules = [Rule(id=f"R{i}", text=f"Rule {i}") for i in range(n_rules)]

            for _ in range(n_exp):
                baseline_scores = {}
                mutant_scores = {}
                mutant_to_rule = {}

                for s in range(n_scenarios):
                    sid = f"s{s}"
                    baseline_scores[sid] = [100.0 if rng.random() < baseline_p else 0.0 for _ in range(n_runs)]

                # Planted rule 0 has drop of delta
                mutant_p = max(0.0, baseline_p - delta)
                mid_0 = "M_planted_0"
                mutant_to_rule[mid_0] = (dummy_rules[0].id, Operator.remove)
                mutant_scores[mid_0] = {}
                for s in range(n_scenarios):
                    sid = f"s{s}"
                    mutant_scores[mid_0][sid] = [100.0 if rng.random() < mutant_p else 0.0 for _ in range(n_runs)]

                # Add 11 null rules to form complete 12-rule BH family
                for r in dummy_rules[1:]:
                    mid_r = f"M_{r.id}"
                    mutant_to_rule[mid_r] = (r.id, Operator.remove)
                    mutant_scores[mid_r] = {}
                    for s in range(n_scenarios):
                        sid = f"s{s}"
                        mutant_scores[mid_r][sid] = [100.0 if rng.random() < baseline_p else 0.0 for _ in range(n_runs)]

                verdicts = analyzer.analyze(
                    rules=dummy_rules,
                    baseline_scores=baseline_scores,
                    mutant_scores=mutant_scores,
                    mutant_to_rule=mutant_to_rule,
                )

                v0 = next((v for v in verdicts if v.rule_id == dummy_rules[0].id), None)
                if v0:
                    if v0.verdict == Verdict.load_bearing:
                        detected_count += 1
                    elif v0.verdict == Verdict.inconclusive:
                        inconclusive_count += 1

            recall = detected_count / n_exp
            recall_ci = wilson_ci(detected_count, n_exp)
            inconc_rate = inconclusive_count / n_exp

            cell_res = {
                "delta": delta,
                "delta_points": delta_pts,
                "n_runs": n_runs,
                "empirical_recall": round(recall, 4),
                "recall_95ci": [round(recall_ci[0], 4), round(recall_ci[1], 4)],
                "inconclusive_rate": round(inconc_rate, 4),
                "analytical_se": round(se_analytical, 2),
                "power_upper_bound_alpha05": round(power_upper, 4),
                "power_lower_bound_bonferroni": round(power_lower, 4),
                "sits_in_analytical_bounds": (power_lower - 0.10 <= recall <= power_upper + 0.10),
            }
            grid_results[str(delta)][str(n_runs)] = cell_res

            if recall >= 0.80 and achieved_n is None:
                achieved_n = n_runs

        min_n_for_recall_80[str(delta)] = achieved_n
        if grid_results[str(delta)]["20"]["empirical_recall"] < 0.50:
            undetectable_effects.append(delta)

        print(f"  Delta={delta*100:4.1f}% | N=3: {grid_results[str(delta)]['3']['empirical_recall']:.2f} | N=5: {grid_results[str(delta)]['5']['empirical_recall']:.2f} | N=10: {grid_results[str(delta)]['10']['empirical_recall']:.2f} | N=20: {grid_results[str(delta)]['20']['empirical_recall']:.2f} | Min N>=0.80: {achieved_n}", flush=True)

    print(f"\n  Summary of Minimum N for Recall >= 0.80: {min_n_for_recall_80}", flush=True)
    print(f"  Undetectable effect sizes (Power < 50% even at N=20): {undetectable_effects}", flush=True)

    return {
        "grid": grid_results,
        "min_n_for_recall_80": min_n_for_recall_80,
        "undetectable_effects": undetectable_effects,
    }


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = InstructionParser()
    rules = parser.parse(AA_INSTRUCTIONS)

    res_3a_sim = run_step3_simulated_aa(rules, n_iter=200, seed=42)
    res_3a_real = run_step3_real_model_resplit(rules)
    res_3b_power = run_step3_power_curves(n_exp=200, seed=42)

    output = {
        "simulated_aa_fixed": res_3a_sim,
        "real_model_resplit_fixed": res_3a_real,
        "power_curve_sweep": res_3b_power,
    }

    out_file = Path("results/step3_revalidation.json")
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nStep 3 Re-validation completed and saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
