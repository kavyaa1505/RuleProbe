# Phase 2 Summary: Noisy Simulated Benchmark

### Objective
Evaluate the statistical engine across parameter grid sweeps under realistic non-deterministic output noise, Bernoulli compliance probabilities, random failure rates (5%), and continuous score variance.

---

### Experimental Setup
* **Fixed Random Seed**: `42`
* **Sweeps**:
  * Runs-per-mutant $N \in \{3, 5, 10, 20\}$
  * Effect sizes: `small` ($\Delta \approx 20\%$), `medium` ($\Delta \approx 40\%$), `large` ($\Delta \approx 80\%$)
* **Sample Size**: 200 independent experiments per configuration ($12 \times 200 = 2,400$ total simulated experiments).
* **Planted Ground-Truth Classes**:
  1. `load_bearing`
  2. `redundant`
  3. `ignored`
  4. `harmful` (conflicting rule)
  5. `order_sensitive` (move operator)
  6. `untested` (decoy with no scenario coverage)

---

### Results Table: Recall vs. $N$ across Effect Sizes

| Effect Size | True $\Delta$ | $N=3$ Recall (95% CI) | $N=5$ Recall (95% CI) | $N=10$ Recall (95% CI) | $N=20$ Recall (95% CI) | Min $N$ for Recall $\ge 0.90$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Large** | $80\%$ | **0.0%** `[0.0%, 1.9%]` | **16.5%** `[11.9%, 22.3%]` | **93.0%** `[88.5%, 95.8%]` | **100.0%** `[98.1%, 100.0%]` | **$N = 10$** |
| **Medium** | $40\%$ | **0.0%** `[0.0%, 1.9%]` | **0.5%** `[0.1%, 2.8%]` | **23.5%** `[18.1%, 29.9%]` | **23.0%** `[17.7%, 29.4%]` | *Did not reach $0.90$* |
| **Small** | $20\%$ | **0.0%** `[0.0%, 1.9%]` | **0.0%** `[0.0%, 1.9%]` | **4.5%** `[2.4%, 8.3%]` | **2.0%** `[0.8%, 5.0%]` | *Did not reach $0.90$* |

---

### Comprehensive Performance by Class at $N=20$ (Large Effect)

| Class | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision (95% CI) | Recall (95% CI) | $F_1$ Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`load_bearing`** | 200 | 1 | 0 | **99.5%** `[97.2%, 100%]` | **100.0%** `[98.1%, 100%]` | **0.998** |
| **`harmful`** | 200 | 1 | 0 | **99.5%** `[97.2%, 100%]` | **100.0%** `[98.1%, 100%]` | **0.998** |
| **`order_sensitive`** | 196 | 1 | 4 | **99.5%** `[97.2%, 100%]` | **98.0%** `[95.0%, 99.2%]` | **0.987** |
| **`untested`** | 200 | 0 | 0 | **100.0%** `[98.1%, 100%]` | **100.0%** `[98.1%, 100%]` | **1.000** |
| **`ignored`** | 195 | 6 | 5 | **97.0%** `[93.6%, 98.6%]` | **97.5%** `[94.3%, 98.9%]` | **0.973** |
| **`redundant`** | 194 | 4 | 6 | **98.0%** `[94.9%, 99.2%]` | **97.0%** `[93.6%, 98.6%]` | **0.975** |
| **Macro Average** | — | — | — | **98.9%** | **98.8%** | **0.989** |

---

### Honest Statistical Diagnosis & Proposed Fix

#### Why did Medium and Small Effect Sizes fail to reach 90% Recall?
In `ruleprobe/analyzer.py` line 139:
```python
noise_floor = float(np.std(all_baseline))
...
material = abs(effect) > max(self.materiality_threshold, noise_floor)
```
1. **The Flaw**: `all_baseline` is a list of raw binary trial outcomes ($0$ or $100$). For Bernoulli outcomes with $p \approx 0.85$, the sample standard deviation is $\sigma = 100 \cdot \sqrt{p(1-p)} \approx 35.7\%$.
2. Because `noise_floor` was calculated as the standard deviation of an **individual trial** ($\sigma \approx 36-40\%$) rather than the standard error of the **mean difference** ($\text{SE} = \sigma \sqrt{2/N} \approx 11\%$ at $N=20$), any real effect smaller than $36\%$ (such as small effects of $20\%$ or medium effects of $38\%$) is automatically blocked by the condition `abs(effect) > noise_floor`!
3. **Proposed Fix**: The noise floor must be defined either as:
   - $\text{SE} = \frac{\sigma}{\sqrt{N}}$ (standard error of the mean), which correctly shrinks as sample size $N$ increases, or
   - The empirical mean difference of baseline runs split against themselves (as measured in Phase 1 A/A test, which was $\approx 1.3\%$).
*(Per instruction, this was not modified in `analyzer.py` to preserve testing integrity).*
