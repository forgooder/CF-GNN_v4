# Causal-GraIL v5 Consolidated Experiment Report

Updated: 2026-06-04 22:18 CST

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
--log_relation_mask_validation
--relation_overlap_penalty_path
--relation_overlap_penalty_weight
--relation_shortcut_floor_path
--relation_shortcut_floor_weight
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

## WN18RR_v1 Causal Loss Warmup/Ramp Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_causal_warmup3_ramp3_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --causal_loss_warmup_epochs 3 --causal_loss_ramp_epochs 3 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_all_score_modes_validation --log_relation_metrics_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9103/0.9165
No test run.
```

Mask:

```text
epoch1 causal_loss_weight=0.0 raw=0.4554/0.4318 entropy=0.6889/0.6836
epoch3 causal_loss_weight=0.0 raw=0.4546/0.4323 entropy=0.6890/0.6840
epoch6 causal_loss_weight=0.5 raw=0.6082/0.4208 entropy=0.4427/0.6794 budget=0.1206 overlap=0.2511
epoch8 causal_loss_weight=0.5 raw=0.6473/0.4184 entropy=0.5435/0.6792 budget=0.0843 overlap=0.2750
```

Relation observations:

```text
Best-point _hypernym AUC-PR about 0.7013
Best-point _has_part AUC-PR about 0.6678
Strong relation _derivationally_related_form remains high around 0.9562
```

Conclusion:

```text
Negative validation result. Causal loss warmup/ramp works mechanically and keeps masks non-collapsed, but does not close the WN18RR_v1 baseline gap. Once causal loss ramps on, validation drops and causal mask raw mean/overlap rise. Do not test this configuration. WN18RR still needs a relation-aware objective or feature/scorer change targeted at _hypernym and _has_part, not another global auxiliary-loss schedule.
```

## Code Review Update: Relation-Level Mask Diagnostics

Code change:

```text
Added default-off --log_relation_mask_validation.
When enabled, validation logs edge_count, causal_raw_mean, shortcut_raw_mean, causal_entropy, shortcut_entropy, and overlap grouped by target relation.
This is diagnostic-only and does not affect training loss, checkpoint selection, data, negative sampling, subgraph extraction, or metric computation.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py managers/evaluator.py
python train.py --help | rg "log_relation_mask_validation|log_relation_metrics_validation"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_relation_mask_validation \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8987/0.9163
No test run.
Final epoch raw=0.3502/0.4386 entropy=0.4681/0.6843
```

Relation mask observations from smoke:

```text
Early validation point:
_hypernym causal_raw_mean=0.0916, causal_entropy=0.2654
_has_part causal_raw_mean=0.1347, causal_entropy=0.3346
_derivationally_related_form causal_raw_mean=0.0356, causal_entropy=0.1432

Later validation point:
_hypernym causal_raw_mean=0.4204, causal_entropy=0.1535
_has_part causal_raw_mean=0.6957, overlap=0.2902
_derivationally_related_form causal_raw_mean=0.3246, causal_entropy=0.5982
```

Conclusion:

```text
Smoke passed and shows that WN weak relations do not share one mask failure mode: _hypernym becomes low-entropy, while _has_part becomes high-causal/high-overlap. The next diagnostic should run the relation-mask logger on an 8 epoch validation-only WN18RR_v1 mainline configuration before designing relation-specific regularization.
```

## WN18RR_v1 Relation-Mask Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_relation_mask_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9090/0.9168
Best all-mode observed: causal AUC/AUC-PR 0.9175/0.9191
No test run.
```

Global mask:

```text
epoch1 raw=0.4064/0.4341 entropy=0.4329/0.6825
epoch4 raw=0.5742/0.4235 entropy=0.6451/0.6812
epoch8 raw=0.6603/0.4174 entropy=0.5437/0.6789 overlap=0.2725
```

Best-point relation mask observations:

```text
_hypernym score AUC-PR=0.6987, causal_raw=0.7013, shortcut_raw=0.4134, causal_entropy=0.5337, overlap=0.2877
_has_part score AUC-PR=0.6522, causal_raw=0.7026, shortcut_raw=0.4157, causal_entropy=0.5827, overlap=0.2913
_derivationally_related_form score AUC-PR=0.9582, causal_raw=0.5613, shortcut_raw=0.4260, causal_entropy=0.6550, overlap=0.2382
```

Conclusion:

```text
Negative validation result. Relation-mask logging confirms v5 no longer has global v4-style collapse, but WN weak relation families still fail. _hypernym and _has_part often show higher causal raw and overlap than strong relations, so forcing larger causal budgets is not the right direction. The next WN attempt should either penalize weak-relation overlap/high causal saturation, or introduce relation-family-specific scoring/objective changes; do not run test for this configuration.
```

## Code Review Update: Relation-Specific Overlap Penalty

Code change:

```text
Added default-off --relation_overlap_penalty_path and --relation_overlap_penalty_weight.
The JSON maps relation ids or names to overlap multipliers. The loss is a weighted average of raw causal*shortcut overlap on selected target relations, then scaled by relation_overlap_penalty_weight.
Added configs/wn18rr_v1_weak_relation_overlap.json for _hypernym and _has_part.
Default behavior is unchanged because the weight is 0.0 and the path is empty by default.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py
python train.py --help | rg "relation_overlap_penalty"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_relation_overlap_penalty \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --relation_overlap_penalty_path configs/wn18rr_v1_weak_relation_overlap.json \
  --relation_overlap_penalty_weight 0.05 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.9217/0.9217
No test run.
Final epoch relation_overlap_loss=0.0460
Final global raw=0.4910/0.3590 entropy=0.4934/0.6142
```

Initial observation:

```text
The penalty strongly reduced selected weak-relation overlap in smoke.
_has_part AUC-PR reached 0.8743 at a later validation point, much higher than prior relation diagnostics.
_hypernym remained weak around AUC-PR 0.699, and aggregate AUC-PR fell at the later validation point.
```

Conclusion:

```text
Smoke passed. Relation-specific overlap penalty is a plausible local tool for _has_part but does not solve _hypernym in smoke. It needs validation-only follow-up before any test run.
```

## WN18RR_v1 Relation-Overlap Penalty Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_relation_overlap_w005_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --relation_overlap_penalty_path configs/wn18rr_v1_weak_relation_overlap.json \
  --relation_overlap_penalty_weight 0.05 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9150/0.9187
Best all-mode observed causal_plus_effect AUC/AUC-PR 0.9179/0.9201
No test run.
```

Mask diagnostics:

```text
epoch1 raw=0.4913/0.3690 entropy=0.5503/0.6388 relation_overlap_loss=0.0545
epoch5 raw=0.6511/0.2848 entropy=0.5403/0.5620 relation_overlap_loss=0.0858
epoch6 _hypernym shortcut_raw=0.0763 shortcut_entropy=0.2371
epoch6 _has_part shortcut_raw=0.0253 shortcut_entropy=0.1139
epoch8 raw=0.5065/0.3433 entropy=0.4768/0.5954 relation_overlap_loss=0.0516
```

Conclusion:

```text
Negative validation result. The relation-overlap penalty can reduce overlap for selected weak relations, but at weight 0.05 it often does so by suppressing shortcut masks and lowering shortcut entropy, which is a new mask-health failure. It does not improve aggregate WN18RR_v1 validation beyond the prior mainline and remains below same-env AUC-PR 0.9350. Do not test. Future WN work should not use this exact penalty; if revisited, it needs a floor-preserving formulation that penalizes high overlap without pushing shortcut raw toward zero.
```

## Code Review Update: Relation Shortcut Floor

Code change:

```text
Added default-off --relation_shortcut_floor_path and --relation_shortcut_floor_weight.
The JSON maps relation ids or names to shortcut raw-mask floors. The loss is a weighted average of relu(floor - shortcut_raw)^2 on selected target relations.
Added configs/wn18rr_v1_weak_relation_shortcut_floor.json with floor 0.35 for _hypernym and _has_part.
Default behavior is unchanged because the path is empty and weight is 0.0 by default.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py
python train.py --help | rg "relation_shortcut_floor|relation_overlap_penalty"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_relation_overlap_floor \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --relation_overlap_penalty_path configs/wn18rr_v1_weak_relation_overlap.json \
  --relation_overlap_penalty_weight 0.03 \
  --relation_shortcut_floor_path configs/wn18rr_v1_weak_relation_shortcut_floor.json \
  --relation_shortcut_floor_weight 0.1 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.9228/0.9256
No test run.
Final relation_overlap_loss=0.0992, relation_shortcut_floor_loss=0.0010
Final global raw=0.4582/0.3950 entropy=0.5778/0.6663
```

Relation mask observations:

```text
At best validation:
_hypernym shortcut_raw=0.3741, shortcut_entropy=0.6593, overlap=0.1325
_has_part shortcut_raw=0.3965, shortcut_entropy=0.6710, overlap=0.1520
```

Conclusion:

```text
Smoke passed. The shortcut floor prevents the relation-overlap penalty from collapsing weak-relation shortcut masks and reaches a stronger WN18RR_v1 smoke AUC-PR than the previous overlap-only diagnostic. It is still below same-env baseline AUC-PR 0.9350, so it requires validation-only follow-up before any test run.
```

## WN18RR_v1 Relation Overlap + Shortcut Floor Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_relation_overlap_floor_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --relation_overlap_penalty_path configs/wn18rr_v1_weak_relation_overlap.json \
  --relation_overlap_penalty_weight 0.03 \
  --relation_shortcut_floor_path configs/wn18rr_v1_weak_relation_shortcut_floor.json \
  --relation_shortcut_floor_weight 0.1 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9158/0.9197
No test run.
```

Best-point mask:

```text
global raw=0.6148/0.3604 entropy=0.5666/0.6462
_hypernym AUC-PR=0.7078, causal_raw=0.3759, shortcut_raw=0.3545, entropy=0.6435/0.6491, overlap=0.1339
_has_part AUC-PR=0.6877, causal_raw=0.4576, shortcut_raw=0.3925, entropy=0.6698/0.6693, overlap=0.1784
```

Conclusion:

```text
Negative validation result. The shortcut floor fixes the mask-health failure introduced by relation-overlap penalty, but it does not fix WN18RR weak-relation ranking. The best validation AUC-PR remains below same-env baseline 0.9350 and below the no-floor/no-overlap relation-metrics diagnostic. Do not test. This narrows the WN issue further: mask health can be made acceptable, but _hypernym and _has_part remain structurally hard for the current scorer/objective.
```

## Code Review Update: Optional Nonlinear Scorer

Code change:

```text
Added default-off --score_hidden_dim and --score_dropout.
score_hidden_dim=0 preserves the original linear GraIL scorer.
When score_hidden_dim > 0, GraphClassifier uses Linear -> ReLU -> Dropout -> Linear on the existing graph/head/tail/relation representation.
No data, negative sampling, subgraph extraction, metric, or baseline default behavior changed.
```

Verification:

```bash
python -m py_compile train.py model/dgl/graph_classifier.py managers/trainer.py managers/evaluator.py utils/score_utils.py
python train.py --help | rg "score_hidden_dim|score_dropout"
```

## WN18RR_v1 Nonlinear Scorer Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_wn18rr_v1_mlp_scorer64 \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --score_hidden_dim 64 --score_dropout 0.1 \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.9270/0.9367
No test run.
Epoch1 final raw=0.3759/0.4362, entropy=0.5604/0.6841, overlap=0.1620
```

Relation observations:

```text
First validation point: _hypernym AUC-PR=0.8773, much higher than prior WN relation diagnostics near 0.70.
Second validation point: _hypernym AUC-PR=0.8511, _has_part AUC-PR=0.7059.
Masks were healthy in the smoke; shortcut entropy stayed high and no raw mask collapsed.
```

Conclusion:

```text
The smoke suggested that a nonlinear scorer can temporarily improve WN18RR_v1 relation ranking, especially _hypernym, without immediate mask collapse. This was not treated as a performance result because it used batch_size=4 and only one epoch. It justified a validation-only batch_size=16 diagnostic, not a test run.
```

## WN18RR_v1 Nonlinear Scorer Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_mlp_scorer64_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --score_hidden_dim 64 --score_dropout 0.1 \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9168/0.9207
Best all-mode observed causal AUC/AUC-PR 0.9169/0.9211
No test run.
```

Diagnostics:

```text
Same-env WN18RR_v1 baseline AUC-PR: 0.9350
Epoch3 relation masks showed localized alpha saturation: _also_see causal_raw=0.9931, _has_part causal_raw=0.9316.
Later validation fell as low as original AUC/AUC-PR 0.8419/0.8888.
Weight norm rose from 141.8 at epoch1 to 209.4 at epoch8.
Final global raw=0.3523/0.4381, entropy=0.4027/0.6841.
```

Conclusion:

```text
Negative validation result. The nonlinear scorer smoke was a false positive under the standard batch_size=16 diagnostic. The MLP scorer increased capacity but did not repair stable WN18RR_v1 relation ranking; _hypernym returned to about 0.70 AUC-PR and validation stayed below same-env baseline. It also introduced score/mask co-adaptation: scorer weight norm and score magnitudes rose while relation-level causal masks oscillated or saturated. Do not test this configuration. If this direction is revisited, it needs explicit scorer regularization or lower-capacity/stronger-dropout validation, not test-set feedback.
```

## Files Kept After Consolidation

```text
reports/CONSOLIDATED_EXPERIMENT_REPORT.md
reports/PROJECT_CONTEXT.md
reports/RESULTS_LEDGER.md
```
