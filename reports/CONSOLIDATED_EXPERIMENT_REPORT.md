# Causal-GraIL v5 Consolidated Experiment Report

Updated: 2026-06-04 16:45 CST

## Scope

This report consolidates the previous timestamped work reports into a compact project-level summary. Detailed per-run rows are retained in `RESULTS_LEDGER.md`; running context and latest conclusions are retained in `PROJECT_CONTEXT.md`.

## Current Code State

Branch:

```text
v5-alpha-beta-mask
```

Latest relevant commit:

```text
ef31ec5 Add WN18RR v1 mask-only auxiliary diagnostic
```

Important default-off additions now available:

```text
--selection_metric {auc,auc_pr}
--log_all_score_modes_validation
--mask_logit_l2_weight
--causal_mask_entropy_floor_weight
--causal_mask_logit_l2_weight
--effect_loss_warmup_epochs
--effect_loss_ramp_epochs
--effect_gradient_mode
--effect_score_clamp
--masked_aux_gradient_mode {full,mask_only}
```

Baseline path remains default-preserving: causal training is only active with `--use_causal_training`, and new parameters are default-off or default-compatible.

## Main Configuration

The current strongest v5 mainline is:

```bash
--use_causal_training
--causal_loss_weight 0.5
--effect_loss_weight 0.0
--mask_gamma 0.5
--mask_budget_weight 0.05
--mask_overlap_weight 0.01
--mask_logit_l2_weight 0.001
--causal_mask_entropy_floor_weight 0.1
--causal_mask_logit_l2_weight 0.002
--mask_entropy_floor 0.2
--causal_mask_target 0.45
--shortcut_mask_target 0.45
--score_mode original
--selection_metric auc_pr
--log_all_score_modes_validation
```

Effect loss is not part of the current mainline. Multiple diagnostics showed that effect-loss variants destabilize masks or scorer magnitudes, especially on WN18RR.

## Formal Positive Results

| Dataset | Experiment | AUC | AUC-PR | MRR | Hits@1 | Hits@5 | Hits@10 | Conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `nell_v1` | `diag_v5_nell_v1_original_score_w05_allmodes_10ep` | 0.8943 | 0.9230 | 0.7247 | 0.6469 | 0.8212 | 0.8531 | Beats same-env baseline and paper AUC-PR/Hits@10 |
| `fb237_v1` | `diag_v5_fb237_v1_original_score_w05_allmodes_10ep_lc` | 0.8396 | 0.8546 | 0.5575 | 0.4492 | 0.6829 | 0.7622 | Beats same-env baseline and paper AUC-PR/Hits@10 |

Same-environment baselines:

```text
NELL_v1:  AUC 0.8691, AUC-PR 0.7967, MRR 0.5246, Hits@1 0.4750, Hits@10 0.5700
FB237_v1: AUC 0.7938, AUC-PR 0.8349, MRR 0.4862, Hits@1 0.4024, Hits@10 0.6537
```

Paper targets:

```text
NELL-995 v1: AUC-PR 0.8605, Hits@10 0.5950
FB15k-237 v1: AUC-PR 0.8469, Hits@10 0.6415
```

## Stability Evidence

| Dataset | Repeat | Best Validation AUC | Best Validation AUC-PR | Test Run? | Conclusion |
|---|---|---:|---:|---|---|
| `fb237_v1` | `repeat_v5_fb237_v1_original_score_w05_allmodes_10ep_lc` | 0.8504 | 0.8683 | No | Positive validation repeat; supports formal test result |
| `nell_v1` | `repeat_v5_nell_v1_original_score_w05_allmodes_10ep` | 0.9009 | 0.9175 | No | Positive validation repeat; weaker than formal run but still far above targets |

No additional tests were run for these repeats to avoid tuning from test feedback.

## WN18RR Status

WN18RR remains negative despite improved mask health.

| Dataset | Best Current Diagnostic | Best Validation AUC-PR | Same-Env Baseline AUC-PR | Status |
|---|---|---:|---:|---|
| `WN18RR_v1` | original-score/all-mode, causal weight 0.5 | 0.9358 validation, but formal test AUC-PR 0.9221 | 0.9350 | Negative formal test |
| `WN18RR_v2` | original-score/all-mode, 5ep | 0.9272 | 0.9479 | Negative validation |
| `WN18RR_v4` | original-score/all-mode, 5ep | 0.9143 | 0.9354 | Negative validation |

Recent WN18RR_v1 structural diagnostics:

```text
shortcut_penalty_weight=0.1:
  best validation AUC-PR 0.9123
  failed; shortcut raw/overlap rose

masked_aux_gradient_mode=mask_only:
  best validation AUC-PR 0.9166
  masks healthy, but score still below baseline
```

Conclusion: WN18RR failure is no longer primarily v4-style mask collapse. The current objective produces healthy masks but does not improve the scorer enough on WN18RR.

## Mask Health Summary

v4 failure pattern:

```text
causal_mask_mean -> 1.0
shortcut_mask_mean -> 0.0
mask_entropy -> 0
```

v5 current mainline:

```text
NELL_v1 formal: selected raw=0.7385/0.4106, entropy=0.3464/0.6759
FB237_v1 formal: final raw=0.5376/0.4259, entropy=0.4054/0.6809
WN18RR_v1 mask-only diagnostic: final raw=0.6464/0.4176, entropy=0.5617/0.6789
```

Overall, v5 materially mitigates v4 mask collapse across tested datasets. The remaining problem is dataset-dependent score quality.

## Negative Paths

Do not prioritize these without a new design reason:

```text
effect_loss_weight > 0 as mainline
effect ramp alone
detach_shortcut effect gradients
effect score clamp alone
lower LR plus effect loss
shortcut_penalty_weight=0.1
masked_aux_gradient_mode=mask_only as a standalone WN fix
more WN18RR scalar sweeps on the same objective
```

## Current Judgment

```text
Causal-GraIL v5 is currently positive on NELL_v1 and FB237_v1, with formal tests and validation repeats.
It is negative on WN18RR_v1/v2/v4, despite much healthier masks.
The project has credible evidence that v5 mitigates mask collapse, but gains are dataset-dependent.
```

## Next Optimization Direction

The next useful optimization should focus on WN18RR without using test feedback:

1. Analyze relation-level validation behavior and mask budgets.
2. Try relation-aware mask targets if they can be justified from training/validation diagnostics, not test results.
3. Rework effect objective only if it avoids prior instability: no scorer explosion, no shortcut entropy collapse, no high-overlap budget domination.
4. Keep NELL/FB237 test results fixed; use validation-only repeats or ablations if needed.

## Files Kept After Consolidation

```text
reports/CONSOLIDATED_EXPERIMENT_REPORT.md
reports/PROJECT_CONTEXT.md
reports/RESULTS_LEDGER.md
```
