# Quantization experiment: pilot analysis

- Results file: `data/raw/pilot_results.jsonl`
- Items analyzed: 50 (complete cases across 2 levels)
- Generated: 2026-11-03 09:27
- Command: `python analyze.py data/raw/pilot_results.jsonl --min-effect 0.05`

> Pilot run: hypothesis tests, the figure and the pairwise CSV are skipped. The pilot is for checking the pipeline and estimating the discordance rate only.

## 1. Accuracy

| Level | Correct | Accuracy | 95% Wilson CI | Invalid outputs |
|---|---|---|---|---|
| q4_K_M | 36/50 | 72.0% | [58.3, 82.5] | 0 |
| q8_0 | 35/50 | 70.0% | [56.2, 80.9] | 0 |

## 5. Sensitivity analysis: excluding invalid outputs

No invalid outputs at any level; scoring invalid outputs as incorrect had no effect.

## 6. Power analysis (two-sided McNemar, Connor 1987)

```text
  alpha = 0.05/3 = 0.0167  ->  z(1-alpha/2) = 2.3940
  power = 0.8                ->  z(1-beta)    = 0.8416
  minimum effect of interest delta = 0.05
  n = [z(1-alpha/2)*sqrt(psi) + z(1-beta)*sqrt(psi - delta^2)]^2 / delta^2

  q4_K_M vs q8_0:
    discordant pairs: 0 (q8_0 only) + 1 (q4_K_M only) = 1 of 50
    psi_hat = 1/50 = 0.020, 95% Wilson CI [0.004, 0.105]
    planning psi = 0.105 (upper bound: conservative)
    n = [2.3940*sqrt(0.1050) + 0.8416*sqrt(0.1050 - 0.0025)]^2 / 0.0025
      = [0.7756 + 0.2694]^2 / 0.0025 = 1.0919 / 0.0025 = 436.8  ->  n = 437

  The pilot is used only for psi. delta is the minimum effect you chose, not the pilot's observed
  difference, which is too noisy to plan with.
```
