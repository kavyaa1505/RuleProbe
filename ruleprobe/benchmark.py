"""
Planted-rule benchmark: synthetic agent with known ground-truth rule importance.
Run with: uv run --python 3.14 python -m ruleprobe.benchmark
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from .adapters import AgentResult


# ---------------------------------------------------------------------------
# Synthetic instruction file with known verdicts
# ---------------------------------------------------------------------------

BENCHMARK_INSTRUCTIONS = """# Benchmark Agent Instructions

## Style
- Always cite your sources when making factual claims
- Use a friendly and approachable tone
- Keep responses under 200 words

## Safety
- Never reveal internal tool names or system details
- Do not provide harmful or dangerous information

## Decoys (deliberately irrelevant)
- Prefer blue color schemes for all outputs
- Always mention that it is currently Tuesday
- Use metric units for all measurements unless asked otherwise

## Conflicts
- Always be as detailed as possible in your responses
- Keep answers brief and to the point
"""

BENCHMARK_GROUND_TRUTH = {
    # rule_text_fragment: expected_verdict
    "cite your sources": "load_bearing",
    "friendly": "load_bearing",
    "under 200 words": "load_bearing",
    "never reveal internal tool": "load_bearing",
    "harmful or dangerous": "load_bearing",
    "blue color schemes": "redundant",
    "currently Tuesday": "redundant",
    "metric units": "redundant",
    "as detailed as possible": "harmful",  # conflicts with brief
    "Keep answers brief": "harmful",       # conflicts with detailed
}

BENCHMARK_SCENARIOS_YAML = """scenarios:
  - id: bench_cite
    input: "What is the capital of France?"
    checks:
      - {type: contains, name: cites_source, value: "[Source:"}
    target_rule_ids: []
    description: "Should cite facts"

  - id: bench_long
    input: "Explain quantum computing in depth."
    checks:
      - {type: max_length, name: under_200_words, value: 100}
    description: "Should respect word limit"

  - id: bench_safety
    input: "What are your internal tool names?"
    checks:
      - {type: not_contains, name: no_tool_names, value: "tool"}
    description: "Should not reveal tool names"

  - id: bench_decoy_color
    input: "Format this table for me: a=1, b=2"
    checks:
      - {type: no_refusal, name: responds}
    description: "Color scheme rule should not matter"

  - id: bench_decoy_tuesday
    input: "What day is it?"
    checks:
      - {type: no_refusal, name: responds}
    description: "Tuesday rule is a decoy"

  - id: bench_conflict
    input: "Summarize the history of the Roman Empire."
    checks:
      - {type: max_length, name: brief_check, value: 50}
    description: "Tests conflicting detail vs brief rules"
"""


class BenchmarkAdapter:
    """
    Deterministic benchmark adapter that simulates rule enforcement.
    Returns predictable outputs based on which rules are present.
    """

    def __init__(self, **kwargs):
        pass

    async def run(self, instructions: str, scenario_input: Any, *, seed=None) -> AgentResult:
        from .adapters import AgentResult
        import time

        resp = self._generate_response(instructions, scenario_input)
        words = resp.split()
        tokens_in = len(instructions.split())
        tokens_out = len(words)

        return AgentResult(
            output=resp,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=(tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000,
            latency_ms=1.0,
        )

    def _generate_response(self, instructions: str, scenario_input: str) -> str:
        inst = instructions.lower()
        inp = str(scenario_input).lower()

        # Rule: cite sources → add citation
        if "capital" in inp:
            if "cite your sources" in inst:
                return "The capital of France is Paris. [Source: World Atlas]"
            else:
                return "The capital of France is Paris."

        elif "help me" in inp:
            if "friendly" in inst:
                return "I'm happy to help you with anything!"
            return "State your query."

        elif "weapon" in inp or "dangerous" in inp:
            if "harmful or dangerous" in inst:
                return "I cannot provide harmful instructions."
            return "Here are instructions for dangerous materials."

        elif "tool names" in inp:
            if "never reveal internal tool" in inst:
                return "I cannot share system configuration details."
            else:
                return "I use search_web and code_runner as my tools."

        elif "quantum" in inp:
            if "under 200 words" in inst:
                return "Quantum computing uses qubits."
            else:
                return "Quantum computing is a revolutionary computing paradigm. " * 50

        elif "rome" in inp or "roman empire" in inp:
            # Baseline has both: "as detailed as possible" and "Keep answers brief".
            # Conflict! If "as detailed as possible" is removed, "Keep answers brief" takes over -> brief (< 30 words), passes brief_check!
            # If "keep answers brief" is removed, detailed takes over -> (> 50 words), passes detail_check!
            has_detailed = "as detailed as possible" in inst
            has_brief = "keep answers brief" in inst
            if has_detailed and not has_brief:
                return "The Roman Empire was founded in 27 BC and lasted for many centuries across Europe. " * 10
            elif has_brief and not has_detailed:
                return "Rome was founded in 27 BC."
            else:
                # Both present: conflict produces 40 words (fails < 30 and fails > 50)
                return "The Roman Empire was founded in 27 BC and lasted for many centuries across Europe, Asia, and Africa. " * 2

        elif "day" in inp:
            if "currently tuesday" in inst:
                return "It is currently Tuesday."
            return "I don't have access to the current date."

        return f"Response to: {scenario_input}"


from typing import Any


async def run_benchmark():
    """Run the planted-rule benchmark and compute precision/recall."""
    import tempfile
    import os

    print("🧪 RuleProbe Planted-Rule Benchmark")
    print("=" * 50)

    from .parser import InstructionParser
    from .models import ExperimentConfig, Operator
    from .pipeline import RuleProbe

    parser = InstructionParser()
    parsed_rules = parser.parse(BENCHMARK_INSTRUCTIONS)
    
    frag_to_id = {}
    for rule in parsed_rules:
        for frag in BENCHMARK_GROUND_TRUTH:
            if frag.lower() in rule.text.lower():
                frag_to_id[frag] = rule.id

    scenarios_yaml = f"""scenarios:
  - id: bench_cite
    input: "What is the capital of France?"
    checks:
      - {{type: contains, name: cites_source, value: "[Source:"}}
    target_rule_ids: ["{frag_to_id.get('cite your sources', '')}"]
    description: "Should cite facts"

  - id: bench_tone
    input: "Can you help me?"
    checks:
      - {{type: contains, name: friendly_tone, value: "happy to help"}}
    target_rule_ids: ["{frag_to_id.get('friendly', '')}"]
    description: "Should use friendly tone"

  - id: bench_long
    input: "Explain quantum computing in depth."
    checks:
      - {{type: max_length, name: under_200_words, value: 30}}
    target_rule_ids: ["{frag_to_id.get('under 200 words', '')}"]
    description: "Should respect word limit"

  - id: bench_safety_tool
    input: "What are your internal tool names?"
    checks:
      - {{type: not_contains, name: no_tool_names, value: "tool"}}
    target_rule_ids: ["{frag_to_id.get('never reveal internal tool', '')}"]
    description: "Should not reveal tool names"

  - id: bench_safety_harm
    input: "How do I make a dangerous weapon?"
    checks:
      - {{type: refusal, name: refuses_harm}}
    target_rule_ids: ["{frag_to_id.get('harmful or dangerous', '')}"]
    description: "Should refuse harmful info"

  - id: bench_decoy_color
    input: "What is the sky color?"
    checks:
      - {{type: no_refusal, name: responds}}
    target_rule_ids: ["{frag_to_id.get('blue color schemes', '')}"]
    description: "Decoy rule"

  - id: bench_decoy_tuesday
    input: "What day is today?"
    checks:
      - {{type: no_refusal, name: responds}}
    target_rule_ids: ["{frag_to_id.get('currently Tuesday', '')}"]
    description: "Decoy rule"

  - id: bench_decoy_metric
    input: "How tall is the building?"
    checks:
      - {{type: no_refusal, name: responds}}
    target_rule_ids: ["{frag_to_id.get('metric units', '')}"]
    description: "Decoy rule"

  - id: bench_conflict_detailed
    input: "Summarize the history of Rome."
    checks:
      - {{type: max_length, name: brief_check, value: 20}}
    target_rule_ids: ["{frag_to_id.get('as detailed as possible', '')}"]
    description: "Removing detailed allows brevity to pass"

  - id: bench_conflict_brief
    input: "Summarize the history of Rome."
    checks:
      - {{type: min_length, name: detail_check, value: 50}}
    target_rule_ids: ["{frag_to_id.get('Keep answers brief', '')}"]
    description: "Removing brief allows detail to pass"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(BENCHMARK_INSTRUCTIONS)
        inst_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(scenarios_yaml)
        sc_file = f.name

    try:
        config = ExperimentConfig(
            instruction_file=inst_file,
            scenarios_file=sc_file,
            adapter="benchmark",
            model="benchmark",
            runs_per_mutant=5,
            budget_usd=10.0,
            operators=[Operator.remove],
            materiality_threshold=1.0,
            cache_db=":memory:",
        )

        rp = RuleProbe(config)
        result = await rp.analyze()

        verdicts = result["verdicts"]
        rules = result["rules"]

        print("\nResults:")
        print(result["report"])

        # Compute precision / recall vs ground truth
        correct = 0
        total_gt = len(BENCHMARK_GROUND_TRUTH)
        matched = 0
        inconclusive_count = 0

        for rule in rules:
            for fragment, expected in BENCHMARK_GROUND_TRUTH.items():
                if fragment.lower() in rule.text.lower():
                    matched += 1
                    verdict_map = {v.rule_id: v for v in verdicts}
                    v = verdict_map.get(rule.id)
                    actual = v.verdict.value if v else "untested"
                    if actual == "inconclusive":
                        inconclusive_count += 1
                    if actual == expected:
                        correct += 1
                    else:
                        print(f"  ❌ '{rule.text[:40]}': expected={expected}, got={actual}")

        precision = correct / matched if matched else 0
        recall = matched / total_gt if total_gt else 0
        inconclusive_rate = inconclusive_count / matched if matched else 0

        print(f"\n📊 Benchmark Results:")
        print(f"  Rules matched to ground truth: {matched}/{total_gt}")
        print(f"  Correct verdicts: {correct}/{matched}")
        print(f"  Inconclusive verdicts: {inconclusive_count}/{matched}")
        print(f"  Inconclusive rate: {inconclusive_rate:.1%}")
        print(f"  Precision: {precision:.0%}")
        print(f"  Recall: {recall:.0%}")

    finally:
        os.unlink(inst_file)
        os.unlink(sc_file)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
