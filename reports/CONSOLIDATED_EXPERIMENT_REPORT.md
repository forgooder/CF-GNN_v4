# Causal-GraIL v5 Consolidated Experiment Report

Updated: 2026-06-04 20:35 CST

## Scope

This report consolidates the previous timestamped work reports into a compact project-level summary. Detailed per-run rows are retained in `RESULTS_LEDGER.md`; running context and latest conclusions are retained in `PROJECT_CONTEXT.md`.

## Current Code State

Branch:

```text
v5-alpha-beta-mask
```

Latest relevant commit:

```text
daac33a Add causal loss warmup ramp
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
--causal_loss_warmup_epochs
--causal_loss_ramp_epochs
--effect_gradient_mode
--effect_score_clamp
--masked_aux_gradient_mode {full,mask_only}
--log_relation_metrics_validation
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

## Latest Optimization Addendum

Added default-off validation diagnostics:

```text
--log_relation_metrics_validation
```

This logs validation AUC/AUC-PR grouped by target relation for the selected score mode. It does not change metric computation, checkpoint selection, negative sampling, subgraph extraction, data, or baseline behavior.

Smoke:

```text
Experiment: smoke_v5_relation_metrics_validation
Dataset: WN18RR_v1
Code state: 2a8db3e + local relation-metrics patch
Best smoke validation original AUC/AUC-PR: 0.8439/0.8913
Mask: raw=0.4526/0.4312, entropy=0.5520/0.6824
```

Initial relation-level observation from WN18RR_v1 smoke:

```text
_hypernym is the dominant weak relation:
support=336, positives=168, negatives=168, AUC-PR about 0.696-0.699 in early smoke validation.

Stronger relations in the same smoke include:
_derivationally_related_form support=732, AUC-PR about 0.946-0.973
_also_see support=52, AUC-PR about 0.885-0.969
_verb_group support=58, AUC-PR about 0.935-0.938
```

Next WN18RR step:

```text
Run a validation-only WN18RR_v1 mainline diagnostic with relation metrics enabled. Use relation-level validation only to design future relation-aware budgets/objectives; do not run WN18RR test unless validation clearly qualifies.
```

Completed relation-metrics diagnostic:

```text
Experiment: diag_v5_wn18rr_v1_relation_metrics_original_w05_8ep
Code: 63bb624
Best validation original AUC/AUC-PR: 0.9180/0.9206
No test run.
Final mask: raw=0.5989/0.4215, entropy=0.5778/0.6802, budget=0.0620, overlap=0.2510
```

Best-point relation metrics:

```text
_hypernym: support=336, AUC-PR=0.7069
_has_part: support=64, AUC-PR=0.6877
_synset_domain_topic_of: support=14, AUC-PR=0.7857
_also_see: support=52, AUC-PR=0.8846
_verb_group: support=58, AUC-PR=0.9535
_derivationally_related_form: support=732, AUC-PR=0.9566
```

Interpretation:

```text
WN18RR_v1 aggregate failure is concentrated in relation families where GraIL-style enclosing subgraphs are likely ambiguous for hierarchy/part-whole reasoning, especially _hypernym and _has_part. Current v5 mask regularization keeps masks healthy but does not repair these relation-specific ranking failures.
```

Relation-aware budget diagnostic:

```text
Config: configs/wn18rr_v1_weak_relation_budget.json
Experiment: diag_v5_wn18rr_v1_relation_budget_weak065035_original_w05_8ep
Budget: _hypernym and _has_part causal target 0.65, shortcut target 0.35; other relations default 0.45/0.45
Best validation original AUC/AUC-PR: 0.9117/0.9169
No test run.
Final mask: raw=0.5649/0.4066, entropy=0.6263/0.6742, budget=0.0283, overlap=0.2250
```

Conclusion:

```text
This relation-aware budget is negative. It does not improve aggregate WN18RR_v1 validation and does not reliably repair _hypernym or _has_part. The next relation-aware attempt should not simply force weak relations to larger causal masks; it needs either per-relation diagnostics of mask means or a different relation-specific objective.
```

## Code Review Update: Causal Loss Warmup/Ramp

Review finding:

```text
WN18RR diagnostics now have healthy masks but weak original validation scores. One likely training issue is that the masked auxiliary causal loss competes with the original GraIL scorer from epoch 1, before the original scorer has stabilized. This can depress the score mode that currently works best on NELL_v1 and FB237_v1: original.
```

Code change:

```text
Added default-off --causal_loss_warmup_epochs and --causal_loss_ramp_epochs.
The training loop now logs causal_loss_weight and causal_loss_ramp_factor.
Default behavior is preserved: warmup=0 and ramp=0 keep immediate activation at --causal_loss_weight.
Also hardened relation-level validation logging for multi-negative batches by repeating relation labels when score_neg has expanded length.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py managers/evaluator.py
python train.py --help | rg "causal_loss_warmup|causal_loss_ramp|causal_loss_weight"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_causal_loss_warmup \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --causal_loss_warmup_epochs 1 --causal_loss_ramp_epochs 3 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr --log_all_score_modes_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8988/0.9067
Best validation causal AUC/AUC-PR: 0.9223/0.9244
Epoch1 causal_loss_weight=0.0 and causal_loss_ramp_factor=0.0 as expected.
Epoch1 mask raw=0.4545/0.4325, entropy=0.6890/0.6839, budget=0.0004, overlap=0.1966.
```

Conclusion:

```text
Smoke passed. This is not a performance result. The change is default-off and keeps baseline behavior unchanged. Next step is a validation-only WN18RR_v1 run with causal_loss_warmup=3 and causal_loss_ramp=3; run test only if validation AUC-PR and mask health qualify.
```

## Files Kept After Consolidation

```text
reports/CONSOLIDATED_EXPERIMENT_REPORT.md
reports/PROJECT_CONTEXT.md
reports/RESULTS_LEDGER.md
```
