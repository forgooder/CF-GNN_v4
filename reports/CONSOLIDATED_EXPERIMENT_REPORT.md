# Causal-GraIL v5 Consolidated Experiment Report

Updated: 2026-06-05 12:00 CST

## Scope

This report consolidates the previous timestamped work reports into a compact project-level summary. Detailed per-run rows are retained in `RESULTS_LEDGER.md`; running context and latest conclusions are retained in `PROJECT_CONTEXT.md`.

## Current Code State

Branch:

```text
v5-alpha-beta-mask
```

Latest relevant commit:

```text
ec14f4e Add graph maxpool diagnostics
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

## 12-Task Target Clarification

The intended target set is now:

```text
NELL_v1-v4
FB237_v1-v4
WN18RR_v1-v4
```

To exceed 80% over baseline, at least 10 of these 12 tasks need credible AUC-PR/Hits@10 gains in the same environment.

Current completed formal-positive count from the ledger:

| Group | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| NELL | Positive formal test | Same-env baseline established; v5 validation negative | Not yet evaluated | Not yet evaluated |
| FB237 | Positive formal test | Not yet evaluated | Not yet evaluated | Not yet evaluated |
| WN18RR | Negative formal test / negative diagnostics | Negative validation/test diagnostics | Not yet evaluated | Negative validation diagnostics |

Current score:

```text
2 / 12 formal-positive tasks.
Need at least 10 / 12.
```

This means the immediate project state is substantially short of the expanded target. The next scheduling priority should shift from only repairing WN18RR_v1 to broad validation coverage of the untested NELL/FB237 variants, while continuing WN-specific work without test-set tuning.

## NELL_v2 Coverage Start

Same-environment baseline:

```text
Experiment: baseline_nell_v2_sameenv_10ep
Validation-only, 10 epochs, batch_size=16, original score, selection_metric=auc_pr.
Best validation AUC/AUC-PR: 0.9613/0.9624 at epoch8.
No test run yet.
```

Failed infrastructure attempt:

```text
Experiment: diag_v5_nell_v2_original_score_w05_allmodes_10ep
Failure: lmdb.MapResizedError / MDB_MAP_RESIZED during concurrent cache growth.
Cause: v5 and baseline were both started on first-time nell_v2 LMDB construction.
Conclusion: do not run first-time cache construction concurrently for the same dataset. This is not a model result.
```

v5 validation diagnostics:

```text
Experiment: diag_v5_nell_v2_original_score_w05_allmodes_10ep_rerun
Stopped after epoch1 because runtime was high and validation was far below baseline.
Validation original AUC/AUC-PR: 0.8846/0.8968.
Best all-mode AUC-PR: causal_plus_effect 0.8985.
Mask state: raw causal/shortcut 0.6434/0.4182, entropy 0.5239/0.6789, overlap 0.2682.

Experiment: diag_v5_nell_v2_original_w05_warm3_ramp2_5ep
Stopped after epoch1 because warmup did not preserve scoring quality.
Validation original AUC/AUC-PR: 0.8520/0.8711.
Mask state: raw causal/shortcut 0.4545/0.4324, entropy 0.6890/0.6839, overlap 0.1965.
```

Conclusion:

```text
NELL_v2 is currently negative for v5. The same-env baseline is very strong at validation AUC-PR 0.9624, while both v5 mainline and warmup/ramp diagnostics are far below it.
Mask collapse is not the immediate problem on NELL_v2; the current causal-training path weakens the scorer.
Do not run NELL_v2 test for these v5 configs.
Next broad-coverage work should establish FB237_v2 and NELL_v3/FB237_v3 baselines sequentially, then run v5 validation only after each cache is built.
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

## Graph Summary Addendum

Added a default-off graph maxpool feature to the scorer:

```text
--add_graph_maxpool_features
```

WN18RR_v1 results:

```text
smoke_v5_graph_maxpool_validation:
  best validation AUC/AUC-PR = 0.8987/0.9117
  final raw = 0.3789/0.4352
  _hypernym margin_tie_rate = 0.5059523809523809

diag_v5_wn18rr_v1_graph_maxpool_original_w05_8ep:
  best validation AUC/AUC-PR = 0.9161/0.9191
  final raw = 0.7694/0.7131
  _hypernym AUC-PR = 0.7023
```

Interpretation:

```text
Graph maxpool is a healthy code path and slightly changes relation-wise score dynamics, but it does not close the WN18RR_v1 gap to same-env baseline AUC-PR 0.9350. The same-size _hypernym tie pattern remains visible.
```

Failed graph-stat branch:

```text
smoke_v5_graph_stat_validation_v1-v4:
  attempted max/min/std graph summary features
  failed first on DGL compatibility (min_nodes unavailable)
  then on validation-time NaN/inf scores even after compatibility fixes
```

Interpretation:

```text
The richer graph-stat path is not a useful next step in this environment as implemented. It is lower priority than relation-specific discrimination on the existing healthy graph-maxpool path.
```

## WN18RR_v1 MLP Aggregator Diagnostic

Existing built-in option tested:

```text
--gnn_agg_type mlp
```

Smoke:

```text
Experiment: smoke_v5_wn18rr_v1_mlp_agg_original_w05
Best validation AUC/AUC-PR: 0.9143/0.9309
No test run.
Signal: first validation reduced _hypernym margin_tie_rate from about 0.506 to 0.095 and raised _hypernym AUC-PR to 0.8785.
```

Standard validation-only follow-up:

```text
Experiment: diag_v5_wn18rr_v1_mlp_agg_original_w05_8ep
Best validation AUC/AUC-PR: 0.9051/0.9124
No test run.
```

Conclusion:

```text
MLP aggregation does not replicate the smoke signal under standard batch_size=16 validation. _hypernym returns to the same paired tie pattern and aggregate AUC-PR remains below same-env WN18RR_v1 baseline 0.9350. Do not test this configuration.
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

## WN18RR_v1 Low-Capacity Nonlinear Scorer Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_mlp_scorer16_drop03_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --score_hidden_dim 16 --score_dropout 0.3 \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9136/0.9186
Best all-mode observed shortcut AUC/AUC-PR 0.9137/0.9189
No test run.
```

Diagnostics:

```text
Same-env WN18RR_v1 baseline AUC-PR: 0.9350
First validation point showed alpha closed on many relations: _hypernym causal_raw=0.0226, _has_part causal_raw=0.0101.
Later masks recovered but weak-relation ranking did not: best-point _hypernym AUC-PR=0.7015, _has_part AUC-PR=0.6877.
Validation later fell to original AUC/AUC-PR 0.9159/0.9125.
Score magnitude still drifted: original_score_mean reached 78.28 at epoch6, and weight_norm reached 208.0 at epoch8.
```

Conclusion:

```text
Negative validation result. Reducing nonlinear scorer capacity and increasing dropout did not beat the linear scorer mainline or same-env baseline. It also did not reliably control scorer magnitude. WN weak relations stayed near prior failure levels, so low-capacity MLP is not a WN fix and should not be tested.
```

## Code Review Update: Score Magnitude Regularizer

Code change:

```text
Added default-off --score_l2_weight.
When causal training is enabled and score_l2_weight > 0, the trainer adds an L2 penalty on original/causal/shortcut positive and negative score magnitudes.
The loss logs score_l2_loss and score_l2_reg_loss.
Default behavior is unchanged because score_l2_weight=0.0.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py model/dgl/graph_classifier.py
python train.py --help | rg "score_l2_weight|score_hidden_dim|score_dropout"
```

## WN18RR_v1 Score-L2 Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_wn18rr_v1_mlp16_score_l2_001 \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --score_hidden_dim 16 --score_dropout 0.3 --score_l2_weight 0.001 \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8872/0.9104
No test run; no 8-epoch follow-up.
Epoch1 score_l2_loss=282.1560, score_l2_reg_loss=0.2822.
Epoch1 score means were lower than the unregularized 8-epoch late drift: original=13.79, causal=11.54, shortcut=8.90.
Final raw=0.4717/0.3359, entropy=0.5740/0.5852.
```

Conclusion:

```text
Negative smoke. Score L2 with weight 0.001 controls score magnitude, but validation AUC-PR is weak and shortcut masks are pushed down. The second validation point showed _has_part causal_raw=0.8132 and overlap=0.3076. Do not run a full diagnostic at this weight. If scorer magnitude regularization is revisited, it needs a much weaker coefficient or a centered/calibrated score penalty; this direct L2 formulation is not ready for WN testing.
```

## Code Review Update: Relation-Weighted Ranking Loss

Code change:

```text
Added default-off --relation_loss_weight_path.
The JSON maps relation ids or names to ranking-loss weights used only during causal training.
When no path is provided, ranking loss uses the original MarginRankingLoss reduction and baseline behavior is unchanged.
Added configs/wn18rr_v1_weak_relation_loss_weight2.json with 2x loss weights for _hypernym and _has_part.
The trainer logs relation_loss_weight_mean for causal-training batches.
```

Verification:

```bash
python -m py_compile train.py managers/trainer.py model/dgl/graph_classifier.py
python train.py --help | rg "relation_loss_weight|score_l2_weight"
```

## WN18RR_v1 Relation-Loss Weight Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_wn18rr_v1_relation_loss_w2 \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --relation_loss_weight_path configs/wn18rr_v1_weak_relation_loss_weight2.json \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.9155/0.9327
No test run.
Epoch1 relation_loss_weight_mean=1.3398.
Final raw=0.4358/0.4325, entropy=0.5986/0.6831.
```

Relation observations:

```text
First validation point was weak: AUC/AUC-PR 0.8757/0.8951.
Second validation point improved weak relations: _hypernym AUC-PR=0.8633 and _has_part AUC-PR=0.9218.
The second validation point still remained below same-env WN18RR_v1 baseline AUC-PR 0.9350.
```

Conclusion:

```text
Smoke showed a plausible but narrow weak-relation signal, so it justified a validation-only batch_size=16 diagnostic. It was not eligible for test evaluation.
```

## WN18RR_v1 Relation-Loss Weight Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_relation_loss_w2_original_w05_8ep \
  --gpu 0 --use_causal_training --num_epochs 8 --batch_size 16 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --relation_loss_weight_path configs/wn18rr_v1_weak_relation_loss_weight2.json \
  --score_mode original --selection_metric auc_pr \
  --log_all_score_modes_validation --log_relation_metrics_validation --log_relation_mask_validation
```

Best validation:

```text
original AUC/AUC-PR 0.9153/0.9194
Best all-mode observed original AUC/AUC-PR 0.9153/0.9194
No test run.
```

Diagnostics:

```text
Same-env WN18RR_v1 baseline AUC-PR: 0.9350
relation_loss_weight_mean stayed about 1.34, confirming the weak-relation weights were active.
Best-point weak relation metrics stayed poor: _hypernym AUC-PR=0.7048, _has_part AUC-PR=0.6700.
Later validation dropped to original AUC/AUC-PR 0.8345/0.8861.
Relation masks were unstable: early _hypernym causal_raw=0.8571 with overlap=0.3406; later _has_part causal_raw=0.0467 with entropy=0.1401.
```

Conclusion:

```text
Negative validation result. Relation-weighted ranking loss produced a short smoke improvement but did not reproduce under the standard batch_size=16 diagnostic. It increases pressure on weak relations but does not fix their ranking and can destabilize relation-level masks in both directions. Do not test this configuration. Further WN work should avoid simply upweighting weak relations and should instead inspect whether _hypernym/_has_part require relation-specific features or a different subgraph scorer, while preserving the unchanged data/evaluation path.
```

## Code Review Update: Head/Tail Interaction Scorer Features

Code change:

```text
Added default-off --add_ht_interaction_features.
When enabled with add_ht_emb=True, GraphClassifier appends head * tail and abs(head - tail) features to the existing graph/head/tail/relation scorer input.
Default behavior is unchanged because the flag is off by default.
No data, negative sampling, subgraph extraction, metric, or baseline path changed.
```

Verification:

```bash
python -m py_compile train.py model/dgl/graph_classifier.py managers/trainer.py
python train.py --help | rg "add_ht_interaction|score_hidden_dim"
```

## WN18RR_v1 Head/Tail Interaction Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_wn18rr_v1_ht_interactions \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --add_ht_interaction_features \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8301/0.8842
No test run; no 8-epoch follow-up.
Epoch1 final raw=0.4950/0.4275, entropy=0.5815/0.6815.
```

Relation observations:

```text
_has_part improved at the second validation point to AUC-PR=0.9158 and _also_see to 0.9692.
_hypernym stayed weak around AUC-PR=0.6909.
Aggregate validation remained far below same-env WN18RR_v1 baseline AUC-PR 0.9350.
Masks were globally healthy, so the failure is score quality rather than collapse.
```

Conclusion:

```text
Negative smoke. Simple head/tail product and distance features do not fix WN18RR_v1 and substantially hurt aggregate validation. Do not run a full diagnostic or test this configuration. The result further narrows WN failure: local pairwise scorer features can help some relation families but do not repair _hypernym, the dominant weak relation.
```

## Relation Score Stats Diagnostic

Code change:

```text
Added default-off --log_relation_score_stats_validation.
When enabled, validation logs relation-level positive/negative score means/stds, mean score gap, paired margin percentiles, and paired positive-greater-than-negative rate when positive/negative counts match.
Default behavior is unchanged; checkpoint selection, data, negative sampling, subgraph extraction, and metric formulas are unchanged.
```

Verification:

```bash
conda run -n cignn_dgl python -m py_compile managers/evaluator.py managers/trainer.py train.py
conda run -n cignn_dgl python train.py --help | rg "log_relation_score_stats_validation|log_relation_metrics_validation|log_relation_mask_validation"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n cignn_dgl python -u train.py -d WN18RR_v1 -e smoke_v5_relation_score_stats_validation \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_score_stats_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.9038/0.9116
No test run; no 8-epoch follow-up.
Epoch1 final raw=0.4463/0.4311, entropy=0.5012/0.6821, overlap=0.1887.
```

Relation score observations:

```text
At the best validation point, _hypernym had AUC-PR=0.6851, score_gap=8.3145, margin_p10=0.0, margin_p50=0.0, and pos_gt_neg_rate=0.4583.
_has_part was much healthier in the same validation point: AUC-PR=0.9190, score_gap=9.5612, margin_p50=3.8275, pos_gt_neg_rate=0.9375.
_derivationally_related_form remained strong: AUC-PR=0.9570, score_gap=40.1381, margin_p50=43.7551, pos_gt_neg_rate=0.9180.
```

Conclusion:

```text
Code smoke passed and gives a sharper WN18RR_v1 diagnosis. _hypernym is not merely suffering from a low mean positive score; many paired positive/negative examples are tied or non-separable by the current scorer, with median margin exactly 0 and pos_gt_neg_rate below 0.5. This supports next diagnostics around relation-specific margin/separation or score calibration on validation, not more mask-budget or overlap sweeps. Do not test this smoke.
```

## Relation Score Tie-Rate Diagnostic

Code change:

```text
Extended --log_relation_score_stats_validation with paired margin_le0_rate and margin_tie_rate.
Default behavior remains unchanged because the logger is still off by default.
```

Verification:

```bash
conda run -n cignn_dgl python -m py_compile managers/evaluator.py managers/trainer.py train.py
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n cignn_dgl python -u train.py -d WN18RR_v1 -e smoke_v5_relation_score_tie_stats_validation \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_score_stats_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8552/0.9032
No test run; no 8-epoch follow-up.
Epoch1 final raw=0.4807/0.4288, entropy=0.5584/0.6819, overlap=0.2027.
```

Tie-rate observations:

```text
At both validation points, _hypernym had margin_p50=0.0, pos_gt_neg_rate=0.4583, margin_le0_rate=0.5417, and margin_tie_rate=0.5060.
_derivationally_related_form stayed well separated at the best point: margin_p50=58.4347, margin_le0_rate=0.0328, margin_tie_rate=0.0273.
_has_part was unstable in aggregate, but tie_rate dropped from 0.4375 at the first validation to 0.0313 at the best point; its AUC-PR remained only 0.7215 in this smoke.
```

Conclusion:

```text
Negative smoke but useful diagnostic. The repeated _hypernym tie_rate around 50% confirms that the WN18RR_v1 dominant weak relation has a paired-score degeneracy under the current graph/scorer representation. This makes further mask-budget/overlap tuning unlikely to solve WN18RR_v1 by itself. The next defensible experiment should target relation-specific margin separation or inspect whether tied _hypernym pairs are structurally indistinguishable in extracted subgraphs, without changing extraction or sampling.
```

## Relation Pair Structure Diagnostic

Code change:

```text
Added default-off --log_relation_pair_stats_validation.
When enabled, validation logs paired positive/negative margin stats plus same-relation, same-node-count, same-edge-count, same-size, and tie-subset same-size rates by relation.
Default behavior is unchanged; training, checkpoint selection, data, negative sampling, subgraph extraction, and metric formulas are unchanged.
```

Verification:

```bash
conda run -n cignn_dgl python -m py_compile managers/evaluator.py managers/trainer.py train.py
conda run -n cignn_dgl python train.py --help | rg "log_relation_pair_stats_validation|log_relation_score_stats_validation"
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n cignn_dgl python -u train.py -d WN18RR_v1 -e smoke_v5_relation_pair_stats_validation \
  --gpu 0 --use_causal_training --num_epochs 1 --batch_size 4 \
  --causal_loss_weight 0.5 --effect_loss_weight 0.0 \
  --mask_gamma 0.5 --mask_budget_weight 0.05 --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 --causal_mask_target 0.45 --shortcut_mask_target 0.45 \
  --score_mode original --selection_metric auc_pr \
  --log_relation_metrics_validation --log_relation_score_stats_validation \
  --log_relation_pair_stats_validation --log_relation_mask_validation
```

Smoke result:

```text
Best validation original AUC/AUC-PR: 0.8498/0.8974
No test run; no 8-epoch follow-up.
Epoch1 final raw=0.4230/0.4332, entropy=0.5140/0.6829, overlap=0.1819.
```

Pair-structure observations:

```text
At the first validation point, _hypernym pair stats showed same_size_rate=0.5060, margin_tie_rate=0.5060, tie_same_size_rate=1.0, margin_p50=0.0.
At the second validation point, _hypernym repeated same_size_rate=0.5060, margin_tie_rate=0.5060, tie_same_size_rate=1.0, margin_p50=0.0.
The relation-level AUC-PR stayed weak for _hypernym: 0.7050 then 0.6997.
_derivationally_related_form had much lower same_size_rate/tie_rate at the best point, both 0.0792, while retaining AUC-PR=0.9760.
```

Conclusion:

```text
Negative smoke but important structural diagnosis. _hypernym paired score ties align exactly with positive/negative pairs that have identical node and edge counts in the extracted subgraphs. This does not prove full graph isomorphism, but it shows the dominant WN18RR_v1 weakness is strongly associated with same-size paired subgraphs under the existing extraction, not global mask collapse. Because data, negative sampling, and extraction are fixed by constraint, the next code-side experiment should add default-off diagnostics or model capacity that can distinguish same-size _hypernym pairs using existing node/edge features, and should qualify on validation before any test.
```

## Files Kept After Consolidation

```text
reports/CONSOLIDATED_EXPERIMENT_REPORT.md
reports/PROJECT_CONTEXT.md
reports/RESULTS_LEDGER.md
```
