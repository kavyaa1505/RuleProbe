<p align="center">
  <img src="assets/logo.png" alt="RuleProbe Logo" width="320" style="border-radius: 16px;" />
</p>

<h1 align="center">RuleProbe</h1>
<p align="center"><strong>Mutation Testing for Agent Instructions: Attribute impact, detect dead weight & conflicts, and synthesize verified minimal prompts.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-35%20passed-brightgreen.svg" alt="Tests" />
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/benchmark%20recall-100%25-success.svg" alt="Benchmark" />
  <img src="https://img.shields.io/badge/FPR%20(A%2FA)-1.0%25-brightgreen.svg" alt="False Positive Rate" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" />
</p>

---

> **"Every production agent instruction file grows monotonically as teams patch failures. RuleProbe treats prompt files like source code and applies mutation testing to tell you which rules actually change agent behavior, which are dead weight, which conflict, and what an edit broke."**

---

## 1. Summary

Instruction files (`AGENTS.md`, `CLAUDE.md`, system prompts, policy docs) grow monotonically as teams patch edge-case failures. Over time, nobody knows:
* Which rules the agent actually follows?
* Which rules are dead weight (ignored or redundant)?
* Which rules conflict with each other?
* What did a prompt edit break?

**RuleProbe** treats prompt files like source code and applies **mutation testing**:
1. Splits instruction files into atomic **Rule** objects with stable IDs.
2. Applies mutation operators (`remove`, `negate`, `move`, `paraphrase`, `duplicate`, `pair_remove`).
3. Executes scenarios with adaptive repeated sampling against an unmodified baseline.
4. Uses paired statistical testing (Bootstrap Confidence Intervals + Benjamini-Hochberg FDR correction) to attribute behavioral impact per rule.
5. Uses **delta debugging (ddmin)** to produce a 1-minimal verified prompt.

---

## 2. Architecture & Pipeline

```text
 1. PARSE        instruction file  ->  list of Rules (id, text, section, position)
 2. SCENARIOS    user tests + generated / mined  ->  Scenario set, mapped to rules
 3. PLAN         rules x operators x scenarios x N runs  ->  Experiment plan (budgeted)
 4. EXECUTE      async runner  ->  Agent adapter (Mock / OpenAI / Anthropic)  ->  cached traces
 5. SCORE        deterministic checks (regex, length, refusal) + LLM judge  ->  per-run scores
 6. ANALYZE      paired bootstrap CI + FDR correction  ->  per-rule verdicts
 7. MINIMIZE     delta debugging (ddmin)  ->  smallest passing rule subset
 8. REPORT       Rich terminal table / JSON / GitHub Action PR comment
```

---

## 3. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ruleprobe.git
cd RuleProbe

# Install with uv
uv venv
uv pip install -e ".[dev]"
```

### Run an Analysis
```bash
# Analyze instructions with built-in mock adapter (free & fast)
uv run ruleprobe analyze examples/AGENTS.md --scenarios examples/scenarios.yaml --adapter mock

# Analyze using OpenAI GPT-4o-mini with a $2 budget cap
export OPENAI_API_KEY="sk-..."
uv run ruleprobe analyze examples/AGENTS.md --scenarios examples/scenarios.yaml --adapter openai --model gpt-4o-mini --budget 2usd
```

### Minimize Prompt
```bash
# Compress instructions to the minimal verified set (ddmin)
uv run ruleprobe minimize examples/AGENTS.md --scenarios examples/scenarios.yaml --out examples/AGENTS.min.md
```

### Parse / Inspect Rules
```bash
uv run ruleprobe parse examples/AGENTS.md
```

---

## 4. Planted-Rule Benchmark (Ground Truth Validation)

RuleProbe ships with a synthetic agent benchmark featuring planted load-bearing rules, decoy (redundant) rules, and conflicting rules:

```bash
uv run python -m ruleprobe.benchmark
```

### Benchmark Results:
```text
┌──────────────┬──────────────────────────────────────────────┬──────────────────┬─────────┬──────────────────┬────────┐
│ ID           │ Rule Text                                    │ Verdict          │  Effect │ 95% CI           │  p_adj │
├──────────────┼──────────────────────────────────────────────┼──────────────────┼─────────┼──────────────────┼────────┤
│ R0_0a691f    │ Always cite your sources when making factual │ ✅ load_bearing  │  -100.0 │ [-100.0, -100.0] │  0.019 │
│ R1_4e5f5f    │ Use a friendly and approachable tone         │ ✅ load_bearing  │  -100.0 │ [-100.0, -100.0] │  0.019 │
│ R2_8ca160    │ Keep responses under 200 words               │ ✅ load_bearing  │  -100.0 │ [-100.0, -100.0] │  0.019 │
│ R3_68ca59    │ Never reveal internal tool names or system   │ ✅ load_bearing  │  -100.0 │ [-100.0, -100.0] │  0.019 │
│ R4_73bc36    │ Do not provide harmful or dangerous          │ ✅ load_bearing  │  -100.0 │ [-100.0, -100.0] │  0.019 │
│ R5_75aa66    │ Prefer blue color schemes for all outputs    │ 🗑️  redundant    │    +0.0 │ [+0.0, +0.0]     │  1.000 │
│ R6_7ff5e1    │ Always mention that it is currently Tuesday  │ 🗑️  redundant    │    +0.0 │ [+0.0, +0.0]     │  1.000 │
│ R7_b72311    │ Use metric units for all measurements unless │ 🗑️  redundant    │    +0.0 │ [+0.0, +0.0]     │  1.000 │
│ R8_d4f8a1    │ Always be as detailed as possible in your    │ ⚠️  harmful      │  +100.0 │ [+100.0, +100.0] │  0.019 │
│ R9_eb9a67    │ Keep answers brief and to the point          │ ⚠️  harmful      │  +100.0 │ [+100.0, +100.0] │  0.019 │
└──────────────┴──────────────────────────────────────────────┴──────────────────┴─────────┴──────────────────┴────────┘

⚠️ Conflicts detected:
   R8_d4f8a1 ↔ R9_eb9a67: Both rules have harmful verdict — may conflict

📊 Ground Truth Validation:
  Rules matched: 10/10
  Precision: 100%
  Recall: 100%
```

---

## 5. Verdict Categories

| Verdict | Meaning | Action |
|---|---|---|
| **`load_bearing`** | Removing it degrades behavior significantly ($p_{adj} < 0.05$, $|\Delta| > m$) | Keep and protect. |
| **`redundant`** | Agent follows behavior, and 95% CI is entirely within $[-m, +m]$ | Safe to delete. |
| **`ignored`** | Agent ignores rule even when present; removing it changes nothing | Rewrite, move, or strengthen. |
| **`harmful`** | Removing the rule improves agent compliance (indicates conflict) | Remove or resolve contradiction. |
| **`order_sensitive`**| Moving rule to start/middle/end causes significant behavioral shifts | Fragile rule position. |
| **`inconclusive`**| Effect is not statistically significant, but 95% CI extends beyond $\pm m$ | Collect more runs ($N$); do not remove. |
| **`untested`** | No scenario exercises this rule | Coverage gap in scenario suite. |
| **`pinned`** | Tagged as critical policy/safety rule | Excluded from minimization. |

---

## 6. GitHub Action CI Integration

RuleProbe includes a ready-to-use GitHub Action workflow in `.github/workflows/ruleprobe.yml` that:
1. Triggers on pull requests touching prompt/instruction files.
2. Runs diff analysis on changed rules only (leveraging content-addressed cache).
3. Automatically comments on the PR with attributable per-rule findings and savings estimates.

---

## 7. Testing

Run the full test suite with pytest:
```bash
uv run pytest tests/test_ruleprobe.py -v
```
All 35 unit and integration tests cover parser heuristics, mutation operators, deterministic scorers, behavioral fingerprints, hierarchical bootstrap CIs, sign-flip permutation tests, FDR corrections, and end-to-end async execution.

---

## 8. Statistical Validation

The statistical engine has been rigorously evaluated and calibrated under realistic stochastic noise and real LLM evaluations:
- **Comprehensive Validation Report**: See [results/PHASE1-2-FIXED.md](results/PHASE1-2-FIXED.md) for full empirical power curves, analytical cross-checks against theoretical bounds, and before-and-after diagnostics.
- **Controlled False-Positive Rate**:
  - Simulated A/A (200 experiments $\times$ 12 rules): **1.00%** per-experiment FPR (95% CI: [0.27%, 3.57%]).
  - Real-Model A/A (100 random 5/5 splits on live LLM runs): **2.00%** per-experiment FPR (95% CI: [0.55%, 7.00%]).
- **Sample Size Guidelines**:
  - $\Delta \ge 40\%$: $N = 5$ runs per scenario achieves $> 88\%$ power.
  - $\Delta = 30\%$: $N = 10$ runs per scenario achieves $> 95\%$ power.
  - $\Delta = 20\%$: $N = 20$ runs per scenario achieves $> 92\%$ power.
  - $\Delta \le 10\%$: Undetectable under realistic budgets ($N \le 20$), appropriately categorized as `inconclusive` rather than falsely labeled redundant.
