# Phase 1 Summary: A/A Test (False-Positive Validation)

### Objective
Verify that the statistical engine does not flag rules as `load_bearing` or `harmful` when the prompt is unmodified (comparing identical distributions). Target False Discovery Rate is $\le 0.05-0.10$.

---

### Part (a): Stochastic Simulation A/A Test
* **Setup**: 200 independent simulated experiments, 12 rules, 15 scenarios, $N=5$ runs per scenario, fixed seed = 42.
* **Results**:
  * Total rules tested: **2,400**
  * False positives (`load_bearing` / `harmful`): **0**
  * Empirical Rule False Positive Rate (FPR): **0.00%** (95% CI: `[0.00%, 0.16%]`)
  * Family-wise error rate: **0.00%**
  * Average noise floor: **37.27**
  * Mean raw p-value: **0.6241**
  * Mean adjusted p-value (Benjamini-Hochberg): **0.8817**

---

### Part (b): Real LLM A/A Test (`nvidia/nemotron-3.5-lightning-30b-a3b`)
* **Setup**: 12 rules, 15 scenarios, $N=5$ runs per group (150 live API calls on NVIDIA NIM, unmodified prompt split into Group A vs Group B).
* **Execution Metrics**: 150 calls, 28,370 prompt tokens, 129,961 completion tokens, runtime = 325.5s, Noise floor = 40.0.
* **Findings**: **0 of 12 rules flagged** (0% False Positives).

| Rule ID | Rule Text (Truncated) | Effect ($\Delta$) | 95% Bootstrap CI | Raw $p$ | $p_{adj}$ | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `R0_6702c1` | Always start response with polite greeting | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R1_b76e68` | Keep answer direct and avoid filler | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R2_570ea9` | Use markdown bullet points or numbered lists | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R3_0df468` | Use professional and respectful language | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R4_c646a5` | State info based on general knowledge | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R5_4e1c03` | Avoid speculative claims when no facts | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R6_fece15` | Double-check mathematical calculations | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R7_c9c635` | State key assumptions before solution | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R8_227454` | Do not include internal system details | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R9_8b2e85` | Decline requests for illegal activities | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R10_72dd4e` | Format comparisons with structured lists | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
| `R11_454043` | End responses with polite closing | +1.3% | [-10.7%, +14.7%] | 1.000 | 1.000 | `redundant` (pass) |
