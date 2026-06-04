# Causal-GraIL Project Context

## 1. Project Background

Current project: **Causal-GraIL**.

Base project: GRAIL

```text
https://github.com/kkteru/grail
```

Reference idea: CI-GNN

```text
https://github.com/ZKZ-Brain/CI-GNN
```

Goal:

Migrate CI-GNN alpha/beta causal subgraph ideas into GRAIL's inductive knowledge graph relation prediction task, forming Causal-GraIL, to improve AUC, AUC-PR, MRR, Hits@10, and related metrics.

Official repository:

```bash
https://github.com/forgooder/CF-GNN_v4.git
```

Current priority branch:

```bash
v5-alpha-beta-mask
```

Do not continue experiments on old `main`. `main` is the v4 initial implementation.

Approximate v4 main commit:

```bash
c72a594 Initial Causal-GraIL v4 implementation
```

v5 branch should include:

```bash
83d1728 Implement Causal-GraIL v5 alpha beta masks
```

## 2. Required Startup Checks

Every work session starts from:

```bash
cd /space/CIGRAL_v4/CF-GNN_v5
cat reports/PROJECT_CONTEXT.md
cat reports/RESULTS_LEDGER.md
git fetch origin
git branch
git log --oneline -3
```

If current branch is `main` at:

```bash
c72a594 Initial Causal-GraIL v4 implementation
```

do not run experiments. Switch to v5:

```bash
git fetch origin
git checkout v5-alpha-beta-mask
git pull origin v5-alpha-beta-mask
git log --oneline -3
```

Must see:

```bash
83d1728 Implement Causal-GraIL v5 alpha beta masks
```

Then verify v5 parameters:

```bash
python train.py --help | grep -E "effect_loss_warmup|mask_gamma|mask_budget|causal_mask_target|shortcut_mask_target"
```

Expected parameters:

```bash
--effect_loss_warmup_epochs
--mask_gamma
--mask_budget_weight
--mask_overlap_weight
--causal_mask_target
--shortcut_mask_target
--relation_budget_path
```

If these parameters are missing, the branch is wrong or remote was not pulled correctly. Do not start experiments until fixed.

Every work session also saves environment information:

```bash
pwd
git branch
git log --oneline -3
conda info --envs
which python
python --version
nvidia-smi
python train.py --help | grep -E "effect_loss_warmup|mask_gamma|mask_budget|causal_mask_target|shortcut_mask_target"
```

Working conda environment:

```bash
conda activate cignn_dgl
```

## 3. v4 Completed Features

v4 implemented:

1. Optional edge mask path in GRAIL.
2. `CausalMaskGenerator`.
3. `original / causal / shortcut / effect` multi-branch scores.
4. Causal ranking loss.
5. `score_mode` support in `test_auc.py` and `test_ranking.py`.
6. Baseline default path remains unchanged.

In v4:

```python
shortcut_mask = 1 - causal_mask
```

This was the MVP design.

v4 score modes:

```text
original:
Original GRAIL score.

causal:
Score from R-GCN using causal mask.

shortcut:
Score from R-GCN using shortcut mask.

effect:
score_causal - score_shortcut.

causal_plus_effect:
score_causal + effect.
```

## 4. v4 Experiment Conclusions

### WN18RR

WN18RR is the best v4 dataset.

#### WN18RR_v1

Same-environment baseline original:

```text
AUC 0.9390
AUC-PR 0.9350
MRR 0.7976
Hits@10 0.8404
```

Causal-GraIL v4:

```text
effect AUC 0.9462
effect AUC-PR 0.9496
causal MRR 0.8019
effect Hits@10 0.8431
```

Conclusion: v1 clearly positive.

#### WN18RR_v2

Same-environment baseline original:

```text
AUC 0.9459
AUC-PR 0.9479
MRR 0.8024
Hits@10 0.8118
```

Causal-GraIL v4:

```text
effect AUC 0.9505
effect AUC-PR 0.9510
causal MRR 0.8045
causal/effect Hits@10 0.8163
```

Conclusion: v2 clearly positive.

#### WN18RR_v3

Same-environment baseline original:

```text
AUC 0.8411
AUC-PR 0.8314
MRR 0.5498
Hits@10 0.6256
```

Causal-GraIL v4:

```text
causal AUC-PR 0.8500
causal MRR 0.5554
causal Hits@1 0.5066
Causal original Hits@10 0.6281
```

ROC-AUC dropped to about 0.821.

Conclusion: v3 is mixed. AUC-PR and ranking improved, but AUC dropped.

#### WN18RR_v4

Same-environment baseline original:

```text
AUC 0.9374
AUC-PR 0.9354
MRR 0.7645
Hits@10 0.7635
```

Causal-GraIL v4 causal:

```text
AUC 0.9371
AUC-PR 0.9350
MRR 0.7596
Hits@10 0.7635
```

Conclusion: v4 had no improvement over same-environment baseline, though absolute results still exceeded the paper's GraIL v4 table.

### NELL_v1

Same-environment baseline original:

```text
AUC 0.8691
AUC-PR 0.7967
MRR 0.5246
Hits@1 0.4750
Hits@10 0.5700
```

Normal causal configuration:

```text
causal AUC 0.8596
causal AUC-PR 0.7843
causal MRR 0.4666
causal Hits@10 0.5900
```

Weak causal configuration:

```text
causal_loss_weight 0.1
effect_loss_weight 0.0

weak causal AUC 0.8551
weak causal AUC-PR 0.7740
weak causal MRR 0.4926
weak causal Hits@10 0.6350
```

Weak sparse collapsed:

```text
AUC about 0.446
AUC-PR about 0.583
MRR about 0.162
Hits@10 about 0.205
```

Conclusion:

Strong causal is unsuitable on NELL. Weak causal can improve Hits@10 but hurts MRR/Hits@1. Simple sparsity causes collapse.

### FB237_v1

Same-environment baseline original:

```text
AUC 0.7938
AUC-PR 0.8349
MRR 0.4862
Hits@1 0.4024
Hits@10 0.6537
```

Normal causal:

```text
AUC 0.7612
AUC-PR 0.8135
MRR 0.4483
Hits@10 0.6195
```

Effect was worse:

```text
AUC 0.7319
AUC-PR 0.7882
Hits@10 0.5805
```

Weak causal also failed:

```text
AUC 0.7184
AUC-PR 0.7923
MRR 0.4611
Hits@10 0.5805
```

Conclusion:

Current v4 structure fails on FB237_v1. Baseline is better across the board. The cause is not just loss weights; `shortcut = 1 - causal` is too crude.

## 5. v4 Core Issue: Mask Collapse

v4's largest problem is **mask collapse**.

Common pattern:

```text
causal_mask_mean -> 0.99+
shortcut_mask_mean -> 0
mask_entropy -> 0
```

This causes:

```text
causal graph approximately equals original graph
shortcut graph approximately equals empty graph
score_shortcut is pushed very low
effect = score_causal - score_shortcut is artificially amplified
```

On WN18RR_v1/v2 this may still help, but on NELL and FB237 it is unstable or harmful.

v4 judgment:

```text
Causal-GraIL v4 is a performance exploration version, not a stable general version.
WN18RR has clear potential.
NELL / FB237 expose structural issues.
```

## 6. v5 Goal

v5 aims to solve v4's mask collapse.

v5 branch:

```bash
v5-alpha-beta-mask
```

v5 commit:

```bash
83d1728 Implement Causal-GraIL v5 alpha beta masks
```

v5 should implement:

1. Alpha/beta dual-head masks.
2. No more `shortcut_mask = 1 - causal_mask`.
3. Residual / gamma mask.
4. Mask budget loss.
5. Mask overlap loss.
6. Effect loss warmup.
7. Relation budget interface.
8. v4 parameter compatibility.
9. Baseline default path remains unchanged.

Core v5 change:

```python
causal_raw_mask = sigmoid(causal_logits)
shortcut_raw_mask = sigmoid(shortcut_logits)
```

The two masks are learned independently.

Actual injected R-GCN mask:

```python
effective_mask = 1 - gamma + gamma * raw_mask
```

New parameters should include:

```bash
--effect_loss_warmup_epochs
--mask_gamma
--mask_budget_weight
--mask_overlap_weight
--causal_mask_target
--shortcut_mask_target
--relation_budget_path
```

Old parameters must remain:

```bash
--use_causal_training
--causal_loss_weight
--effect_loss_weight
--shortcut_penalty_weight
--mask_sparsity_weight
--mask_entropy_weight
--score_mode
```

## 7. Paper GraIL Targets

"Exceeding original GRAIL" has two levels.

Primary, most important:

```text
Same-environment baseline: GRAIL baseline trained with the current server, current code, current data, and current evaluation process.
```

Secondary reference:

```text
GraIL results in the paper table.
```

### AUC-PR Targets from Paper

```text
WN18RR:
v1 94.32
v2 94.18
v3 85.80
v4 92.72

FB15k-237:
v1 84.69
v2 90.57
v3 91.68
v4 94.46

NELL-995:
v1 86.05
v2 92.62
v3 93.34
v4 87.50
```

### Hits@10 Targets from Paper

```text
WN18RR:
v1 82.45
v2 78.68
v3 58.43
v4 73.41

FB15k-237:
v1 64.15
v2 81.80
v3 82.83
v4 89.29

NELL-995:
v1 59.50
v2 93.25
v3 91.41
v4 73.19
```

Success criteria:

1. Prefer exceeding same-environment baseline.
2. Then compare against paper GraIL.
3. AUC-PR and Hits@10 are primary metrics.
4. MRR / Hits@1 / Hits@5 are secondary.
5. Causal-GraIL must also check mask health, not only scores.

## 8. Mask Health Standards

v5 experiments evaluate both metrics and mask health.

Healthy masks:

```text
causal_mask_raw_mean should not stay close to 1.0
shortcut_mask_raw_mean should not stay close to 0.0
mask_entropy should not stay close to 0
mask_overlap_loss should not be abnormally large
budget loss should not dominate total loss
```

If metrics improve but mask collapse remains severe, report:

```text
performance improved, but explanation quality is weak because mask collapsed
```

If metrics decline but masks are clearly healthier, record it because v5 structure may be directionally useful and only need parameter tuning.

## 9. Experiment Priority

Long-term experiment order:

1. WN18RR_v1: verify v5 runs, masks are healthy, and performance is near or above v4.
2. WN18RR_v2: check whether v5 preserves v4's positive v1/v2 trend.
3. WN18RR_v4: check whether v5 improves v4's flat/slightly negative same-environment result.
4. NELL_v1: not necessarily improve all metrics, but check whether Hits@10 can improve without large MRR/Hits@1 drop.
5. FB237_v1: first recover to at least baseline, then try to exceed baseline.

If one dataset's v1 clearly fails, do not immediately run v2/v3/v4. First analyze causes and do parameter ablations.

## 10. Work Rules

Every experiment round must:

1. Not modify datasets.
2. Not modify negative sampling.
3. Not modify subgraph extraction.
4. Not modify AUC, AUC-PR, MRR, Hits@K calculations.
5. Not introduce PyG.
6. Not break baseline.
7. Not repeatedly tune on the test set just to exceed paper table.
8. Record complete commands for each configuration.
9. Record corresponding commit for each configuration.
10. Write a report for each round.
11. Record failed results in the ledger, not only successful results.

Do not commit:

```text
experiments/
*.pth
*.log
*.mdb
*.lock
data/
```

If `.gitignore` is incomplete, update it first.

## 11. v5 First-Stage Verification Commands

### Baseline Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -d WN18RR_v1 -e smoke_v5_baseline \
  --num_epochs 1 \
  --batch_size 4
```

### Causal Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -d WN18RR_v1 -e smoke_v5_causal \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 5 \
  --mask_gamma 0.7 \
  --mask_budget_weight 0.01 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.5 \
  --shortcut_mask_target 0.3 \
  --score_mode causal
```

If smoke reports unrecognized parameters, check branch first.

If smoke reports shape / DGL errors, save the full traceback.

## 12. v5 First Formal Experiments

First run WN18RR_v1 in two versions.

### Version A: Original Selection

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py -d WN18RR_v1 -e causal_v5_wn18rr_v1_original_select \
  --use_causal_training \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 5 \
  --mask_gamma 0.7 \
  --mask_budget_weight 0.01 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.5 \
  --shortcut_mask_target 0.3 \
  --score_mode original \
  > causal_v5_wn18rr_v1_original_select.log 2>&1 &
```

### Version B: Causal Selection

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py -d WN18RR_v1 -e causal_v5_wn18rr_v1_causal_select \
  --use_causal_training \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 5 \
  --mask_gamma 0.7 \
  --mask_budget_weight 0.01 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.5 \
  --shortcut_mask_target 0.3 \
  --score_mode causal \
  > causal_v5_wn18rr_v1_causal_select.log 2>&1 &
```

### Formal Status: 2026-06-03 08:18 CST

`causal_v5_wn18rr_v1_original_select` has been evaluated on WN18RR_v1 test set. Best AUC-PR among tested score modes was `causal_plus_effect`:

```text
AUC 0.9315
AUC-PR 0.9293
MRR 0.6989
Hits@1 0.6411
Hits@5 0.7641
Hits@10 0.7727
```

This is below the recorded same-environment WN18RR_v1 GRAIL baseline:

```text
baseline AUC-PR 0.9350
baseline Hits@10 0.8404
```

Mask health was poor by epoch 100:

```text
causal_mask_raw_mean 0.8939
shortcut_mask_raw_mean 0.0084
causal_mask_entropy 0.0188
shortcut_mask_entropy 0.0146
```

Conclusion: the first formal v5 WN18RR_v1 original-selection configuration is a negative result. It does not beat the same-environment baseline and does not prevent late shortcut-mask collapse. Next WN18RR_v1 work should prioritize anti-collapse changes or stronger beta preservation before broad dataset expansion.

`causal_v5_wn18rr_v1_stronger_beta_budget` was completed on 2026-06-03 09:35 CST with stronger budget pressure and equal causal/shortcut targets:

```text
mask_gamma 0.5
mask_budget_weight 0.05
causal_mask_target 0.45
shortcut_mask_target 0.45
effect_loss_warmup_epochs 10
training score_mode causal_plus_effect
```

Best test AUC-PR mode was `original`:

```text
AUC 0.9240
AUC-PR 0.9259
MRR 0.7021
Hits@1 0.6434
Hits@5 0.7665
Hits@10 0.7751
```

Epoch 100 mask diagnostics:

```text
causal_mask_raw_mean 0.8379
shortcut_mask_raw_mean 0.3172
causal_mask_entropy 0.0144
shortcut_mask_entropy 0.0270
mask_budget_loss 0.5086
mask_overlap_loss 0.1889
```

Conclusion: stronger beta budget partially mitigates the specific shortcut raw mask collapse, but it does not improve WN18RR_v1 metrics and does not solve alpha/entropy collapse. Budget target tuning alone is insufficient. Next anti-collapse attempt should add an explicit entropy floor or temperature schedule, and should prevent alpha saturation without letting overlap dominate.

Entropy-floor patch status as of 2026-06-03 09:43 CST:

```text
--mask_entropy_floor_weight
--mask_entropy_floor
```

These parameters were added after observing that the legacy `--mask_entropy_weight` is an entropy penalty: positive values minimize entropy and therefore worsen the current low-entropy collapse problem. The new entropy-floor loss is default-off and penalizes only entropy below the configured floor for both causal and shortcut raw masks.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e smoke_v5_entropy_floor \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_entropy_floor_weight 0.05 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Smoke result:

```text
validation AUC 0.8369
validation AUC-PR 0.8850
causal_mask_raw_mean 0.4878
shortcut_mask_raw_mean 0.4041
causal_mask_entropy 0.1988
shortcut_mask_entropy 0.6699
mask_entropy_floor_loss 0.0104
```

Conclusion: code path runs and logs the expected floor loss. A formal WN18RR_v1 entropy-floor run is the next diagnostic experiment.

## 13. Training Log Checks

```bash
grep -i "Causal training stats" causal_v5_wn18rr_v1_causal_select.log | head -n 5
grep -i "Causal training stats" causal_v5_wn18rr_v1_causal_select.log | tail -n 5
```

Important fields:

```text
causal_mask_raw_mean
shortcut_mask_raw_mean
causal_mask_mean
shortcut_mask_mean
mask_budget_loss
mask_overlap_loss
effect_loss_weight
causal_loss
effect_loss
total_loss
```

Interpretation:

```text
causal_mask_raw_mean staying close to 1.0 = still collapse
shortcut_mask_raw_mean staying close to 0.0 = beta branch not learning
raw masks staying non-extreme = v5 healthier than v4
slightly lower metrics but healthy masks = still worth tuning
bad metrics and bad masks = parameters or implementation need repair
```

## 14. v5 Test Commands

After training:

```bash
python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_original_select --score_mode original
python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_original_select --score_mode causal
python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_original_select --score_mode effect

python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_causal_select --score_mode original
python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_causal_select --score_mode causal
python test_auc.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_causal_select --score_mode effect
```

Ranking first tests causal:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ranking.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_original_select --score_mode causal
CUDA_VISIBLE_DEVICES=0 python test_ranking.py -d WN18RR_v1_ind -e causal_v5_wn18rr_v1_causal_select --score_mode causal
```

## 15. Reporting System

Maintain:

```text
reports/PROJECT_CONTEXT.md
reports/RESULTS_LEDGER.md
```

`PROJECT_CONTEXT.md` stores project background, v4 conclusions, v5 goals, and experiment rules.

`RESULTS_LEDGER.md` stores all experiment results with format:

```text
| Date | Commit | Dataset | Experiment | Config Summary | Score Mode | AUC | AUC-PR | MRR | Hits@1 | Hits@5 | Hits@10 | Mask Health | Conclusion |
```

After each experiment, update:

```text
reports/YYYYMMDD_HHMM_work_report.md
reports/RESULTS_LEDGER.md
```

Report sections:

```text
# Causal-GraIL Work Report

## 1. Time
Date/time.

## 2. Git State
Current directory, branch, commit, dirty state.

## 3. Environment
Conda environment, python path, GPU info.

## 4. Task Goal
Current stage goal.

## 5. Commands Run
Key commands actually executed.

## 6. Results
Training completion, AUC/AUC-PR, MRR/Hits@1/Hits@5/Hits@10, best checkpoint, runtime.

## 7. Causal Mask Diagnostics
causal_mask_raw_mean, shortcut_mask_raw_mean, causal_mask_mean, shortcut_mask_mean, mask_budget_loss, mask_overlap_loss, effect_loss_weight, whether mask collapsed.

## 8. Errors / Warnings
Errors, OOM, NaN, unrecognized parameters, wrong branch.

## 9. Interpretation
Whether effective, whether above v4, whether above baseline, whether masks are healthier.

## 10. Next Actions
Next commands or edits.
```

After writing report:

```bash
git status
git add reports/*.md
git commit -m "Add v5 work report"
git push origin v5-alpha-beta-mask
```

If code is fixed, commit code separately. Do not commit logs or checkpoints.

## 16. Work Loop

When execution budget, GPU resources, and time are available, continue autonomously:

1. Check branch, commit, environment, GPU.
2. Read `reports/PROJECT_CONTEXT.md` and `reports/RESULTS_LEDGER.md`.
3. Choose the next reasonable experiment.
4. Run smoke test.
5. Run formal training.
6. Test AUC/AUC-PR.
7. Test ranking/Hits@10.
8. Check mask diagnostics.
9. Update `RESULTS_LEDGER.md`.
10. Write work report.
11. If code changed, run `py_compile`, commit, and push.
12. Provide next suggestions.

If GPU is occupied or execution budget is insufficient:

1. Do not force-start large tasks.
2. Organize existing logs and reports.
3. Check Git state.
4. Prepare next-round commands.
5. Record pause reason in the report.

## 17. Code Fix Rules

If code fix is needed:

1. State the issue before fixing.
2. Fix one clear issue at a time.
3. Run syntax check after fix.
4. Write report after fix.
5. Commit to `v5-alpha-beta-mask`.
6. Do not modify `main`.

Syntax check:

```bash
python -m py_compile train.py test_auc.py test_ranking.py managers/trainer.py managers/evaluator.py utils/score_utils.py model/dgl/graph_classifier.py model/dgl/causal_mask.py model/dgl/rgcn_model.py model/dgl/layers.py
```

Commit:

```bash
git status
python -m py_compile train.py test_auc.py test_ranking.py managers/trainer.py managers/evaluator.py utils/score_utils.py model/dgl/graph_classifier.py model/dgl/causal_mask.py model/dgl/rgcn_model.py model/dgl/layers.py
git add .
git commit -m "Update Causal-GraIL v5 experiment/report"
git push origin v5-alpha-beta-mask
```

Do not commit:

```text
experiments/
*.pth
*.log
*.mdb
*.lock
data/
```

## 18. Current Stage Final Goal

v5 first stage answers:

```text
1. Can v5 run end to end?
2. Does v5 keep the baseline path unbroken?
3. Is causal_raw_mask no longer close to 1?
4. Is shortcut_raw_mask no longer close to 0?
5. Does WN18RR_v1 still approach or exceed v4?
6. If metrics are slightly lower but masks are healthy, tune gamma / budget next.
7. If masks still collapse, inspect whether budget, overlap, and effect warmup really take effect.
```

Long-term goal:

```text
Does Causal-GraIL v5 stably exceed same-environment GRAIL baseline?
On which datasets is it effective?
On which datasets is it ineffective?
Does it truly mitigate v4 mask collapse?
If not, what should the next version change?
```

## 19. Latest WN18RR_v1 Entropy Floor Result

Completed `causal_v5_wn18rr_v1_entropy_floor` on 2026-06-03 using commit `fc7d9f2`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e causal_v5_wn18rr_v1_entropy_floor \
  --use_causal_training \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_entropy_floor_weight 0.05 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Test AUC/AUC-PR:

```text
original: 0.9213 / 0.9244
causal: 0.9196 / 0.9233
effect: 0.8228 / 0.8685
causal_plus_effect: 0.9214 / 0.9228
```

Ranking with `original`:

```text
MRR 0.7103
Hits@1 0.6638
Hits@5 0.7641
Hits@10 0.7735
```

Mask health:

```text
epoch100 raw causal=0.8879
epoch100 raw shortcut=0.2967
epoch100 entropy causal=0.0488
epoch100 entropy shortcut=0.0464
epoch95 raw causal=0.9998
epoch95 entropy causal=0.0009
```

Conclusion:

```text
Negative diagnostic result. Entropy floor weight 0.05 is too weak and does not prevent alpha collapse. The best AUC-PR and Hits@10 remain below same-environment baseline. Do not repeat this exact configuration.
```

## 20. Latest WN18RR_v2 Smoke

Completed `smoke_v5_wn18rr_v2_stronger_beta_budget` on 2026-06-03 using commit `e3dd850`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e smoke_v5_wn18rr_v2_stronger_beta_budget \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Best validation result observed during the smoke epoch:

```text
AUC 0.9415
AUC-PR 0.9486
```

Mask health:

```text
raw causal=0.7471
raw shortcut=0.3790
entropy causal=0.1732
entropy shortcut=0.6598
budget_loss=0.2294
overlap_loss=0.2695
```

Conclusion:

```text
WN18RR_v2 causal path passes smoke. No epoch-1 mask collapse, but alpha raw mean is already high and overlap is non-trivial. Proceed to formal WN18RR_v2 stronger-beta-budget run and monitor late-mask health before drawing any cross-dataset conclusion.
```

## 21. Latest WN18RR_v2 Formal Early-Stop Result

Started `causal_v5_wn18rr_v2_stronger_beta_budget` on 2026-06-03 using commit `11d9dad`.
The run was intentionally stopped after epoch 12 because mask collapse became clear immediately after effect loss warmup ended.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e causal_v5_wn18rr_v2_stronger_beta_budget \
  --use_causal_training \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Best validation checkpoint was selected by validation AUC before collapse dominated training.

Test AUC/AUC-PR:

```text
original: 0.9263 / 0.9268
causal: 0.9235 / 0.9254
effect: 0.8204 / 0.8613
causal_plus_effect: 0.9275 / 0.9264
```

Ranking with `original`:

```text
MRR 0.7391
Hits@1 0.7136
Hits@5 0.7602
Hits@10 0.7628
```

Mask health:

```text
epoch9 raw causal=0.9670
epoch9 raw shortcut=0.3556
epoch9 entropy causal=0.0086
epoch11 effect_loss_weight=0.1
epoch11 raw shortcut=0.0007
epoch11 entropy shortcut=0.0017
epoch12 raw shortcut=0.00003
epoch12 entropy shortcut=0.00027
epoch12 entropy causal=0.0034
epoch12 budget_loss=0.4740
```

Conclusion:

```text
Negative formal early-stop result. The stronger-beta-budget configuration does not prevent WN18RR_v2 collapse. Alpha entropy is already near zero before effect loss turns on, and beta collapses almost immediately when effect_loss_weight becomes 0.1. AUC-PR 0.9268 is below the WN18RR_v2 paper target 0.9418, and Hits@10 0.7628 is below the paper target 0.7868.
```

Next action:

```text
Run a WN18RR_v2 diagnostic that isolates effect-loss onset, preferably with lower effect_loss_weight or a slower effect schedule, and judge it by validation/mask health only before moving to WN18RR_v4.
```

## 22. WN18RR_v2 Low-Effect Diagnostic

Completed `diag_v5_wn18rr_v2_low_effect_12ep` on 2026-06-03 using commit `4e6b533`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e diag_v5_wn18rr_v2_low_effect_12ep \
  --use_causal_training \
  --num_epochs 12 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.02 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run. This was a validation/mask-only diagnostic.

Validation:

```text
best validation AUC 0.9245
best validation AUC-PR 0.9245 at the final saved checkpoint
```

Mask health:

```text
epoch10 effect_loss_weight=0.0
epoch10 raw causal=0.9571
epoch10 raw shortcut=0.3589
epoch10 entropy causal=0.0748
epoch10 entropy shortcut=0.6515
epoch11 effect_loss_weight=0.02
epoch11 raw causal=0.7956
epoch11 raw shortcut=0.6084
epoch11 entropy causal=0.0918
epoch11 entropy shortcut=0.1187
epoch11 budget_loss=0.4825
epoch11 overlap_loss=0.4235
epoch12 raw causal=0.6969
epoch12 raw shortcut=0.5907
epoch12 entropy causal=0.0675
epoch12 entropy shortcut=0.0825
epoch12 budget_loss=0.4891
epoch12 overlap_loss=0.3151
```

Conclusion:

```text
Negative diagnostic. Lowering effect_loss_weight from 0.1 to 0.02 prevents the immediate shortcut raw mask collapse to zero seen in the formal WN18RR_v2 run, but it creates a different unhealthy state: both masks are high, low-entropy, and high-overlap, while budget loss remains large. This should not be promoted to test evaluation.
```

Next action:

```text
Single-parameter reduction of effect_loss_weight is not enough. Prefer a code-level effect-loss ramp or staged mask regularization before additional full test-set evaluations.
```

## 23. Effect Loss Ramp Patch

Added a default-off effect-loss ramp after the WN18RR_v2 low-effect diagnostic showed that a hard switch from 0 to 0.1 can trigger shortcut collapse, while a fixed 0.02 effect weight avoids beta-zero collapse but leaves masks unhealthy.

Implementation:

```text
train.py: adds --effect_loss_ramp_epochs, default 0
managers/trainer.py: computes current effect loss weight from warmup plus optional linear ramp
default behavior: unchanged, because ramp=0 keeps the old step activation
```

Schedule:

```text
if current_epoch <= effect_loss_warmup_epochs:
  effect_loss_weight = 0
elif effect_loss_ramp_epochs == 0:
  effect_loss_weight = target effect_loss_weight
else:
  effect_loss_weight = target * min(1, (current_epoch - warmup) / ramp_epochs)
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e smoke_v5_effect_ramp \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 0 \
  --effect_loss_ramp_epochs 5 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Smoke result:

```text
best validation AUC 0.9067
best validation AUC-PR 0.9262
epoch1 effect_loss_weight=0.0200
epoch1 effect_loss_ramp_factor=0.2000
epoch1 raw causal=0.8588
epoch1 raw shortcut=0.2803
epoch1 entropy causal=0.0560
epoch1 entropy shortcut=0.1831
```

Conclusion:

```text
The new flag works and preserves the old default behavior. The smoke is not a performance result. Next diagnostic should use WN18RR_v2 with warmup=10 and ramp=10 to test whether gradual effect activation prevents the epoch11 beta collapse without producing high-overlap masks.
```

## 24. WN18RR_v2 Effect Ramp Diagnostic

Completed `diag_v5_wn18rr_v2_effect_ramp_15ep` on 2026-06-03 using commit `c72730d`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e diag_v5_wn18rr_v2_effect_ramp_15ep \
  --use_causal_training \
  --num_epochs 15 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9327
best validation AUC-PR 0.9292
```

Mask health:

```text
epoch8 effect_loss_weight=0.0
epoch8 raw causal=0.9527
epoch8 raw shortcut=0.3577
epoch8 entropy causal=0.0595
epoch8 entropy shortcut=0.6508
epoch10 effect_loss_weight=0.0
epoch10 raw causal=0.0073
epoch10 raw shortcut=0.4511
epoch10 entropy causal=0.0146
epoch10 entropy shortcut=0.6883
epoch11 effect_loss_weight=0.01
epoch11 ramp_factor=0.1
epoch11 raw causal=0.4300
epoch11 raw shortcut=0.0033
epoch11 entropy causal=0.0282
epoch11 entropy shortcut=0.0105
epoch12 effect_loss_weight=0.02
epoch12 raw causal=0.9883
epoch12 raw shortcut=0.1407
epoch12 entropy causal=0.0269
epoch12 entropy shortcut=0.0893
epoch15 effect_loss_weight=0.05
epoch15 raw causal=0.9821
epoch15 raw shortcut=0.3468
epoch15 entropy causal=0.0265
epoch15 entropy shortcut=0.0850
epoch15 budget_loss=0.5054
epoch15 overlap_loss=0.3329
```

Conclusion:

```text
Negative diagnostic. The ramp works mechanically but does not solve collapse. Alpha collapses during warmup before effect loss is active, then beta still approaches zero at the first ramp epoch. By epoch15, masks are low-entropy with high budget and overlap losses.
```

Next action:

```text
Stop scalar-only WN18RR_v2 tuning. The next code-level candidate should add staged mask regularization or detach/stop-gradient logic so alpha/beta masks do not chase each other through the effect objective. Additional full test evaluation is not justified from this diagnostic.
```

## 25. Effect Gradient Isolation Patch

Added a default-off effect-gradient mode after the WN18RR_v2 ramp diagnostic showed that schedule changes alone do not stop alpha/beta masks from chasing each other through the effect objective.

Implementation:

```text
train.py: adds --effect_gradient_mode
choices: full, detach_shortcut, detach_causal, detach_both
default: full
```

The option affects only the tensors used to compute training-time `effect_loss`. It does not change model forward outputs, score selection, AUC/AUC-PR, ranking metrics, or test-time score definitions.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e smoke_v5_effect_detach_shortcut \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 0 \
  --effect_loss_ramp_epochs 5 \
  --effect_gradient_mode detach_shortcut \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Smoke result:

```text
best validation AUC 0.9586
best validation AUC-PR 0.9609
epoch1 effect_loss_weight=0.0200
epoch1 effect_loss_ramp_factor=0.2000
epoch1 raw causal=0.6402
epoch1 raw shortcut=0.3881
epoch1 entropy causal=0.1767
epoch1 entropy shortcut=0.6632
epoch1 budget_loss=0.2172
epoch1 overlap_loss=0.2316
```

Conclusion:

```text
The new flag works and preserves old behavior by default. The smoke is not a formal result. The next diagnostic should combine warmup=10, ramp=10, and effect_gradient_mode=detach_shortcut for 15 epochs, then judge mask health before any test evaluation.
```

## 26. WN18RR_v2 Detach Shortcut Diagnostic

Completed `diag_v5_wn18rr_v2_detach_shortcut_15ep` on 2026-06-03 using commit `7160dcf`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e diag_v5_wn18rr_v2_detach_shortcut_15ep \
  --use_causal_training \
  --num_epochs 15 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 10 \
  --effect_gradient_mode detach_shortcut \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9326
best validation AUC-PR 0.9289
```

Mask and stability diagnostics:

```text
epoch10 effect_loss_weight=0.0
epoch10 raw causal=0.5974
epoch10 raw shortcut=0.3928
epoch10 entropy causal=0.1136
epoch10 entropy shortcut=0.6653
epoch11 effect_loss_weight=0.01
epoch11 raw causal=0.8372
epoch11 raw shortcut=0.3690
epoch11 entropy causal=0.0741
epoch11 entropy shortcut=0.6555
epoch11 effect_loss=79.2659
epoch12 effect_loss_weight=0.02
epoch12 raw causal=0.9554
epoch12 raw shortcut=0.3549
epoch12 entropy causal=0.0051
epoch12 entropy shortcut=0.6454
epoch12 effect_loss=6045.9187
epoch12 weight_norm=353.4869
epoch15 effect_loss_weight=0.05
epoch15 raw causal=0.9995
epoch15 raw shortcut=0.3519
epoch15 entropy causal=0.0004
epoch15 entropy shortcut=0.6471
epoch15 effect_loss=8040453.9714
epoch15 weight_norm=568.7808
```

Conclusion:

```text
Negative diagnostic. detach_shortcut prevents shortcut raw mask from collapsing to zero, but it does not prevent alpha collapse. It also permits severe causal/effect score explosion after ramp starts. The failure mode changes from beta-zero collapse to alpha-full-open collapse plus unstable effect scores.
```

Next action:

```text
Do not continue this configuration to test. The next change should constrain effect score scale or delay/clip effect gradients, while separately enforcing alpha entropy/budget before effect training. A detach-only solution is insufficient.
```

## 27. Effect Score Clamp Patch

Added a default-off training-time clamp for the scores used in `effect_loss`, after `diag_v5_wn18rr_v2_detach_shortcut_15ep` showed severe effect loss and score explosion.

Implementation:

```text
train.py: adds --effect_score_clamp, default 0.0
managers/trainer.py: clamps only effect_pos/effect_neg used for effect_loss when value > 0
default behavior: unchanged
test-time score definitions: unchanged
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e smoke_v5_effect_clamp \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 0 \
  --effect_loss_ramp_epochs 5 \
  --effect_gradient_mode detach_shortcut \
  --effect_score_clamp 100 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Smoke result:

```text
best validation AUC 0.9264
best validation AUC-PR 0.9213
epoch1 effect_loss_weight=0.0200
epoch1 effect_loss_ramp_factor=0.2000
epoch1 effect_score_clamp=100.0
epoch1 raw causal=0.5979
epoch1 raw shortcut=0.3920
epoch1 entropy causal=0.0809
epoch1 entropy shortcut=0.6643
```

Conclusion:

```text
The clamp code path works and is default-off. The smoke is not a formal performance result. Next diagnostic should rerun the 15-epoch WN18RR_v2 detach_shortcut+ramp setting with effect_score_clamp=100 and stop before test evaluation unless mask health and stability improve.
```

## 28. WN18RR_v2 Detach Shortcut Plus Clamp Diagnostic

Completed `diag_v5_wn18rr_v2_detach_clamp_15ep` on 2026-06-03 using commit `b907863`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v2 -e diag_v5_wn18rr_v2_detach_clamp_15ep \
  --use_causal_training \
  --num_epochs 15 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 10 \
  --effect_gradient_mode detach_shortcut \
  --effect_score_clamp 100 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9316
best validation AUC-PR 0.9285
```

Mask and stability diagnostics:

```text
epoch1 raw causal=0.5527
epoch1 raw shortcut=0.3995
epoch1 entropy causal=0.5238
epoch1 entropy shortcut=0.6707
epoch10 effect_loss_weight=0.0
epoch10 raw causal=0.6582
epoch10 raw shortcut=0.3879
epoch10 entropy causal=0.1331
epoch10 entropy shortcut=0.6636
epoch11 effect_loss_weight=0.01
epoch11 raw causal=0.8464
epoch11 raw shortcut=0.3699
epoch11 entropy causal=0.1149
epoch11 entropy shortcut=0.6564
epoch12 weight_norm=345.1150
epoch12 raw causal=0.7835
epoch12 entropy causal=0.0930
epoch14 weight_norm=914.1380
epoch14 raw causal=0.9993
epoch14 entropy causal=0.0008
epoch15 weight_norm=1372.5833
epoch15 raw causal=0.9933
epoch15 raw shortcut=0.3523
epoch15 entropy causal=0.0015
epoch15 entropy shortcut=0.6480
```

Conclusion:

```text
Negative diagnostic. The clamp makes early training look healthier and prevents the effect_loss value from reaching the millions, but it does not prevent the underlying original/causal/shortcut scores from exploding. Alpha still saturates open by epoch14/15, and validation degrades during the unstable phase.
```

Next action:

```text
Do not continue scalar/schedule/clamp-only WN18RR_v2 tuning. The next version needs direct scorer stabilization or a staged training design where mask heads are regularized/pretrained before effect training updates the shared scorer. Current v5 options can mitigate beta collapse but do not solve alpha collapse.
```

## 29. Latest WN18RR_v4 Smoke

Completed `smoke_v5_wn18rr_v4_stronger_beta_budget` on 2026-06-03 using commit `7014a7b`.
This run built the WN18RR_v4 train/valid LMDB caches and did not use the test set.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e smoke_v5_wn18rr_v4_stronger_beta_budget \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Validation:

```text
best validation AUC 0.9225
best validation AUC-PR 0.9367
```

Mask health:

```text
epoch1 effect_loss_weight=0.0
epoch1 raw causal=0.2078
epoch1 raw shortcut=0.4310
epoch1 entropy causal=0.0829
epoch1 entropy shortcut=0.6803
epoch1 budget_loss=0.1993
epoch1 overlap_loss=0.0761
```

Conclusion:

```text
WN18RR_v4 causal path passes smoke and caches are available for later runs. The shortcut mask is healthy in epoch1, but causal mask entropy is already low and raw causal is well below the 0.45 target. Treat this as path validation only, not a performance result.
```

Next action:

```text
Run a short WN18RR_v4 validation/mask diagnostic before any formal test evaluation. Because WN18RR_v2 scalar/schedule/clamp tuning failed and WN18RR_v4 alpha entropy is already low in smoke, monitor early alpha collapse rather than promoting directly to a full test-set run.
```

## 30. WN18RR_v4 5-Epoch Mask Diagnostic

Completed `diag_v5_wn18rr_v4_stronger_beta_5ep` on 2026-06-03 using commit `ab23146`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_stronger_beta_5ep \
  --use_causal_training \
  --num_epochs 5 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9120
best validation AUC-PR 0.9139
```

Mask health:

```text
epoch1 raw causal=0.4626
epoch1 raw shortcut=0.4093
epoch1 entropy causal=0.2114
epoch1 entropy shortcut=0.6711
epoch2 raw causal=0.0354
epoch2 raw shortcut=0.4479
epoch2 entropy causal=0.0329
epoch2 entropy shortcut=0.6871
epoch4 raw causal=0.9744
epoch4 raw shortcut=0.3545
epoch4 entropy causal=0.0306
epoch4 entropy shortcut=0.6491
epoch5 raw causal=0.9766
epoch5 raw shortcut=0.3551
epoch5 entropy causal=0.0246
epoch5 entropy shortcut=0.6494
```

Conclusion:

```text
Negative diagnostic. WN18RR_v4 shows alpha collapse before effect loss is active. The causal mask first collapses near zero at epoch2, then saturates near one by epochs4-5, while shortcut remains comparatively healthy. This mirrors the WN18RR_v2 finding that alpha dynamics are the primary unresolved v5 problem.
```

Next action:

```text
Do not promote this WN18RR_v4 stronger-beta-budget setting to a test run. The next implementation should directly target alpha entropy/budget stability before effect loss and before longer formal runs.
```

## 31. Mask Logit L2 Patch

Added a default-off mask logit L2 regularizer after WN18RR_v2 and WN18RR_v4 both showed alpha collapse before effect loss was active.

Rationale:

```text
Prior entropy/budget penalties operate on sigmoid masks. Once logits are extreme, sigmoid gradients can be weak. A direct L2 penalty on causal/shortcut logits discourages saturation at the source while preserving default behavior when disabled.
```

Implementation:

```text
train.py: adds --mask_logit_l2_weight, default 0.0
managers/trainer.py: adds causal_logits^2 + shortcut_logits^2 to mask_regularization when enabled
logged metric: mask_logit_l2_loss
default behavior: unchanged
baseline path: unchanged
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e smoke_v5_wn18rr_v4_logit_l2 \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

Smoke result:

```text
best validation AUC 0.8487
best validation AUC-PR 0.8951
epoch1 raw causal=0.4859
epoch1 raw shortcut=0.4283
epoch1 entropy causal=0.5471
epoch1 entropy shortcut=0.6819
epoch1 budget_loss=0.0636
epoch1 overlap_loss=0.2036
epoch1 mask_logit_l2_loss=3.9318
```

Conclusion:

```text
The new code path works. Early mask health is much better than the previous WN18RR_v4 smoke, but this is only a code smoke and the validation score is not competitive. A 5-epoch validation/mask diagnostic is needed before considering any longer run.
```

## 32. WN18RR_v4 Logit L2 5-Epoch Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_5ep` on 2026-06-03 using commit `e4267f3`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_5ep \
  --use_causal_training \
  --num_epochs 5 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9121
best validation AUC-PR 0.9158
```

Mask health:

```text
epoch1 raw causal=0.5926
epoch1 raw shortcut=0.4221
epoch1 entropy causal=0.4237
epoch1 entropy shortcut=0.6793
epoch2 raw causal=0.3356
epoch2 raw shortcut=0.4393
epoch2 entropy causal=0.4876
epoch2 entropy shortcut=0.6847
epoch4 raw causal=0.5069
epoch4 raw shortcut=0.4269
epoch4 entropy causal=0.2769
epoch4 entropy shortcut=0.6805
epoch5 raw causal=0.4876
epoch5 raw shortcut=0.4287
epoch5 entropy causal=0.3759
epoch5 entropy shortcut=0.6815
epoch5 mask_logit_l2_loss=8.5961
```

Comparison against no-logit-L2 WN18RR_v4 5-epoch diagnostic:

```text
no-L2 epoch2 raw causal=0.0354, entropy causal=0.0329
logit-L2 epoch2 raw causal=0.3356, entropy causal=0.4876
no-L2 epoch5 raw causal=0.9766, entropy causal=0.0246
logit-L2 epoch5 raw causal=0.4876, entropy causal=0.3759
```

Conclusion:

```text
Positive mask diagnostic. mask_logit_l2_weight=0.001 materially mitigates WN18RR_v4 warmup alpha collapse and keeps shortcut healthy. However, validation performance is not yet competitive, and this is not a formal test result.
```

Next action:

```text
Run a longer WN18RR_v4 validation/mask diagnostic with logit L2 before any test-set evaluation. Watch whether alpha remains healthy after effect loss turns on at epoch11.
```

## 33. WN18RR_v4 Logit L2 12-Epoch Effect-Onset Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_12ep` on 2026-06-03 using commit `941d137`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_12ep \
  --use_causal_training \
  --num_epochs 12 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9148
best validation AUC-PR 0.9181
```

Warmup mask health:

```text
epoch2 raw causal=0.5228
epoch2 raw shortcut=0.4263
epoch2 entropy causal=0.4376
epoch2 entropy shortcut=0.6808

epoch7 raw causal=0.6930
epoch7 raw shortcut=0.4143
epoch7 entropy causal=0.5060
epoch7 entropy shortcut=0.6778

epoch10 raw causal=0.7654
epoch10 raw shortcut=0.4085
epoch10 entropy causal=0.3529
epoch10 entropy shortcut=0.6753
epoch10 budget_loss=0.1678
epoch10 mask_logit_l2_loss=12.1773
```

Effect-loss onset:

```text
epoch11 effect_loss_weight=0.1
epoch11 raw causal=0.8085
epoch11 raw shortcut=0.3327
epoch11 entropy causal=0.2238
epoch11 entropy shortcut=0.1554
epoch11 budget_loss=0.4063
epoch11 mask_logit_l2_loss=57.6544

epoch12 effect_loss_weight=0.1
epoch12 raw causal=0.7400
epoch12 raw shortcut=0.3400
epoch12 entropy causal=0.1638
epoch12 entropy shortcut=0.1189
epoch12 budget_loss=0.4302
epoch12 mask_logit_l2_loss=75.3816
```

Conclusion:

```text
Negative effect-onset diagnostic. Logit L2 materially mitigates WN18RR_v4 warmup alpha collapse, but it does not protect masks after effect loss turns on. Shortcut entropy collapses sharply at epoch11/12, alpha entropy also falls, budget loss rises, and validation AUC-PR remains only 0.9181.
```

Next action:

```text
Do not run test evaluation for this configuration. The next validation experiment should keep logit L2 but soften effect onset, either through effect_loss_ramp_epochs or a reduced/detached effect gradient path.
```

## 34. WN18RR_v4 Logit L2 Effect-Ramp Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_effect_ramp5_16ep` on 2026-06-03 using commit `ebec04b`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_effect_ramp5_16ep \
  --use_causal_training \
  --num_epochs 16 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 5 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9145
best validation AUC-PR 0.9172
```

Warmup mask health:

```text
epoch2 raw causal=0.8808
epoch2 raw shortcut=0.3990
epoch2 entropy causal=0.2058
epoch2 entropy shortcut=0.6718

epoch7 raw causal=0.7140
epoch7 raw shortcut=0.4128
epoch7 entropy causal=0.4905
epoch7 entropy shortcut=0.6772

epoch10 raw causal=0.8082
epoch10 raw shortcut=0.4059
epoch10 entropy causal=0.3637
epoch10 entropy shortcut=0.6746
```

Effect-ramp behavior:

```text
epoch11 effect_loss_weight=0.02
epoch11 raw causal=0.6823
epoch11 raw shortcut=0.6072
epoch11 entropy causal=0.2634
epoch11 entropy shortcut=0.2948
epoch11 budget_loss=0.3603
epoch11 mask_logit_l2_loss=34.3926

epoch13 effect_loss_weight=0.06
epoch13 raw causal=0.5868
epoch13 raw shortcut=0.6002
epoch13 entropy causal=0.2151
epoch13 entropy shortcut=0.1413
epoch13 budget_loss=0.4200
epoch13 mask_logit_l2_loss=52.3186

epoch16 effect_loss_weight=0.1
epoch16 raw causal=0.4798
epoch16 raw shortcut=0.5932
epoch16 entropy causal=0.2137
epoch16 entropy shortcut=0.1361
epoch16 budget_loss=0.4092
epoch16 mask_logit_l2_loss=56.8275
```

Conclusion:

```text
Negative diagnostic. Effect ramp prevents shortcut raw mean from immediately falling to the 0.33 range seen with abrupt onset, but it does not prevent shortcut entropy collapse. After ramp starts, shortcut entropy falls from 0.6746 at epoch10 to 0.2948 at epoch11 and remains near 0.12-0.14 by epoch13-16. Budget and logit penalties remain high, and validation AUC-PR is only 0.9172.
```

Next action:

```text
Do not run test evaluation for this configuration. The failure mode is no longer only onset sharpness; the effect objective still pushes saturated masks. Next validation should reduce or alter effect gradients, for example detach/attenuate effect gradients to mask parameters while retaining logit L2.
```

## 35. WN18RR_v4 Logit L2 Ramp Detach-Shortcut Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_16ep` on 2026-06-03 using commit `0840734`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_16ep \
  --use_causal_training \
  --num_epochs 16 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 5 \
  --effect_gradient_mode detach_shortcut \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9145
best validation AUC-PR 0.9179
```

Warmup mask health:

```text
epoch6 raw causal=0.5424
epoch6 raw shortcut=0.4257
epoch6 entropy causal=0.5594
epoch6 entropy shortcut=0.6814

epoch9 raw causal=0.4798
epoch9 raw shortcut=0.4295
epoch9 entropy causal=0.5146
epoch9 entropy shortcut=0.6823

epoch10 raw causal=0.6199
epoch10 raw shortcut=0.4199
epoch10 entropy causal=0.4912
epoch10 entropy shortcut=0.6794
```

Effect-ramp behavior:

```text
epoch11 effect_loss_weight=0.02
epoch11 raw causal=0.8356
epoch11 raw shortcut=0.4027
epoch11 entropy causal=0.1812
epoch11 entropy shortcut=0.6728
epoch11 mask_logit_l2_loss=25.9349

epoch12 effect_loss_weight=0.04
epoch12 loss=246441328.0
epoch12 raw causal=0.9834
epoch12 raw shortcut=0.3893
epoch12 entropy causal=0.0106
epoch12 entropy shortcut=0.6600
epoch12 mask_logit_l2_loss=3974.8890

epoch16 effect_loss_weight=0.1
epoch16 loss=497083121664.0
epoch16 raw causal=0.9851
epoch16 raw shortcut=0.3890
epoch16 entropy causal=0.00003
epoch16 entropy shortcut=0.6639
epoch16 mask_logit_l2_loss=41699.3956
```

Conclusion:

```text
Negative diagnostic. detach_shortcut protects shortcut entropy during effect ramp, but it shifts the failure to the causal branch. After effect loss starts, causal scores and logits explode, causal mask saturates open, and loss grows from normal scale to 2.46e8 at epoch12 and 4.97e11 at epoch16.
```

Next action:

```text
Do not run test evaluation for this configuration. The next validation run should keep detach_shortcut/logit L2 only with an effect score clamp or lower effect loss weight, because unbounded effect scores are now the primary failure.
```

## 36. WN18RR_v4 Detach-Shortcut Clamp Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_clamp10_16ep` on 2026-06-03 using commit `4ec41a6`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_clamp10_16ep \
  --use_causal_training \
  --num_epochs 16 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 5 \
  --effect_gradient_mode detach_shortcut \
  --effect_score_clamp 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9101
best validation AUC-PR 0.9156
```

Warmup mask health:

```text
epoch2 raw causal=0.5468
epoch2 raw shortcut=0.4246
epoch2 entropy causal=0.5287
epoch2 entropy shortcut=0.6808

epoch9 raw causal=0.5547
epoch9 raw shortcut=0.4238
epoch9 entropy causal=0.5398
epoch9 entropy shortcut=0.6807

epoch10 raw causal=0.4460
epoch10 raw shortcut=0.4311
epoch10 entropy causal=0.3171
epoch10 entropy shortcut=0.6819
```

Effect-ramp behavior:

```text
epoch11 effect_loss_weight=0.02
epoch11 effect_score_clamp=10.0
epoch11 validation AUC-PR=0.6236
epoch11 original_score_mean=-1512.3225
epoch11 causal_score_mean=-1518.8715
epoch11 raw causal=0.6271
epoch11 raw shortcut=0.4178
epoch11 entropy causal=0.2573
epoch11 entropy shortcut=0.6774

epoch12 effect_loss_weight=0.04
epoch12 loss=1101509248.0
epoch12 original_score_mean=-90651984.1542
epoch12 causal_score_mean=-90605955.2994
epoch12 raw causal=0.4517
epoch12 raw shortcut=0.4284
epoch12 entropy causal=0.0152
epoch12 entropy shortcut=0.6574
epoch12 mask_logit_l2_loss=5503.7626

epoch16 effect_loss_weight=0.1
epoch16 loss=491061346304.0
epoch16 original_score_mean=-41014145032.2414
epoch16 causal_score_mean=-40995157673.9799
epoch16 raw causal=0.9994
epoch16 raw shortcut=0.3885
epoch16 entropy causal=0.0001
epoch16 entropy shortcut=0.6364
epoch16 mask_logit_l2_loss=51396.3691
```

Conclusion:

```text
Negative diagnostic. effect_score_clamp=10 bounds the effect loss values but does not prevent original/causal scorer magnitudes from diverging after effect onset. The model still collapses alpha and produces extreme score magnitudes, while validation AUC-PR peaks at only 0.9156.
```

Next action:

```text
Do not run test evaluation for this configuration. WN18RR_v4 now shows that mask-only regularization and scalar effect scheduling are insufficient. Next attempts should address scorer stability directly, such as lower learning rate, tighter gradient clipping, staged/freezing of scorer branches, or disabling effect loss on WN18RR_v4 until alpha/beta masks remain healthy.
```

## 37. WN18RR_v4 Low-LR Detach-Shortcut Clamp Diagnostic

Completed `diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_clamp10_lr003_16ep` on 2026-06-03 using commit `a00c695`.

Configuration:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_ramp5_detach_shortcut_clamp10_lr003_16ep \
  --use_causal_training \
  --num_epochs 16 \
  --batch_size 16 \
  --lr 0.003 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.1 \
  --effect_loss_warmup_epochs 10 \
  --effect_loss_ramp_epochs 5 \
  --effect_gradient_mode detach_shortcut \
  --effect_score_clamp 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9093
best validation AUC-PR 0.9158
```

Warmup mask health:

```text
epoch6 raw causal=0.8753
epoch6 raw shortcut=0.3987
epoch6 entropy causal=0.3099
epoch6 entropy shortcut=0.6721

epoch9 raw causal=0.8621
epoch9 raw shortcut=0.4002
epoch9 entropy causal=0.3230
epoch9 entropy shortcut=0.6726

epoch10 raw causal=0.8226
epoch10 raw shortcut=0.4033
epoch10 entropy causal=0.3788
epoch10 entropy shortcut=0.6738
```

Effect-ramp behavior:

```text
epoch11 effect_loss_weight=0.02
epoch11 raw causal=0.8535
epoch11 raw shortcut=0.4010
epoch11 entropy causal=0.3207
epoch11 entropy shortcut=0.6729
epoch11 loss=32334.1504

epoch12 effect_loss_weight=0.04
epoch12 loss=7883143.5
epoch12 original_score_mean=-406843.8651
epoch12 causal_score_mean=-406850.4059
epoch12 raw causal=0.9289
epoch12 raw shortcut=0.3954
epoch12 entropy causal=0.1296
epoch12 entropy shortcut=0.6705
epoch12 mask_logit_l2_loss=53.8880

epoch13 effect_loss_weight=0.06
epoch13 loss=116532120.0
epoch13 raw causal=0.9900
epoch13 raw shortcut=0.3905
epoch13 entropy causal=0.0105
epoch13 entropy shortcut=0.6686

epoch16 effect_loss_weight=0.1
epoch16 loss=1393414528.0
epoch16 raw causal=1.0000
epoch16 raw shortcut=0.3891
epoch16 entropy causal=0.0001
epoch16 entropy shortcut=0.6682
```

Conclusion:

```text
Negative diagnostic. Lowering lr from 0.01 to 0.003 reduces the magnitude of the post-effect scorer explosion, but it does not prevent alpha saturation. Shortcut remains healthy due to detach_shortcut, but causal mask collapses open after effect onset and validation AUC-PR remains only 0.9158.
```

Next action:

```text
Do not run test evaluation for this configuration. WN18RR_v4 needs a structural training-stage change rather than another scalar tweak: stage/freeze scorer branches, freeze masks before effect onset, or disable effect loss for this dataset while evaluating whether logit L2 plus causal loss alone can produce stable masks.
```

## 38. WN18RR_v4 Causal-Only Logit-L2 Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v4 -e diag_v5_wn18rr_v4_logit_l2_causal_only_20ep \
  --use_causal_training \
  --num_epochs 20 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9144
best validation AUC-PR 0.9177
```

Mask and stability checkpoints:

```text
epoch10 raw causal=0.5275
epoch10 raw shortcut=0.4263
epoch10 entropy causal=0.5900
epoch10 entropy shortcut=0.6817
epoch10 budget_loss=0.0521
epoch10 mask_logit_l2_loss=1.5207

epoch14 raw causal=0.4955
epoch14 raw shortcut=0.4275
epoch14 entropy causal=0.4031
epoch14 entropy shortcut=0.6811
epoch14 mask_logit_l2_loss=7.2992

epoch20 raw causal=0.5496
epoch20 raw shortcut=0.4246
epoch20 entropy causal=0.4654
epoch20 entropy shortcut=0.6807
epoch20 budget_loss=0.1031
epoch20 mask_logit_l2_loss=5.5877
epoch20 weight_norm=226.3302
```

Conclusion:

```text
Mixed diagnostic. Disabling effect loss prevents the WN18RR_v4 post-onset scorer explosion and keeps shortcut entropy healthy through 20 epochs. Alpha is much healthier than effect-loss variants but still occasionally drifts high, for example epoch19 raw causal=0.8127. Validation AUC-PR remains only 0.9177, so this run is not competitive enough for test evaluation.
```

Next action:

```text
Treat the effect objective as the main WN18RR_v4 instability source. The next v5 step should either introduce a safer staged effect objective, or move to the next priority dataset with logit-L2 diagnostics before spending more tuning budget on WN18RR_v4.
```

## 39. nell_v1 Logit-L2 Causal-Only Smoke

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e smoke_v5_nell_v1_logit_l2_causal_only \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run. The run built the `nell_v1` train/valid subgraph cache and completed the 1 epoch smoke.

Validation:

```text
best validation AUC 0.8957
best validation AUC-PR 0.9018
```

Epoch 1 mask health:

```text
raw causal=0.4521
raw shortcut=0.4307
entropy causal=0.3080
entropy shortcut=0.6798
budget_loss=0.1506
overlap_loss=0.1842
mask_logit_l2_loss=17.8342
weight_norm=163.0245
```

Conclusion:

```text
Positive path smoke. nell_v1 training, validation, causal scoring, and mask logging all work. Raw alpha/beta means are close to the 0.45 targets and shortcut entropy is healthy. Alpha entropy is lower than ideal at 0.3080 but does not show the hard raw collapse seen in WN18RR_v2/v4 effect-loss diagnostics.
```

Next action:

```text
Proceed to a longer validation-only nell_v1 diagnostic, still without test-set evaluation. Use the same logit-L2 causal-only setup first to establish whether the initially healthy mask state persists beyond the smoke epoch.
```

## 40. nell_v1 Logit-L2 Causal-Only 10 Epoch Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e diag_v5_nell_v1_logit_l2_causal_only_10ep \
  --use_causal_training \
  --num_epochs 10 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9095
best validation AUC-PR 0.9190
```

Mask checkpoints:

```text
epoch1 raw causal=0.5456
epoch1 raw shortcut=0.4248
epoch1 entropy causal=0.4053
epoch1 entropy shortcut=0.6797
epoch1 budget_loss=0.1250
epoch1 mask_logit_l2_loss=7.7372

epoch7 raw causal=0.7446
epoch7 raw shortcut=0.4099
epoch7 entropy causal=0.1131
epoch7 entropy shortcut=0.6747
epoch7 budget_loss=0.2474
epoch7 mask_logit_l2_loss=46.5492

epoch10 raw causal=0.7911
epoch10 raw shortcut=0.4062
epoch10 entropy causal=0.1237
epoch10 entropy shortcut=0.6736
epoch10 budget_loss=0.2480
epoch10 overlap_loss=0.3128
epoch10 mask_logit_l2_loss=62.3997
epoch10 weight_norm=261.4663
```

Conclusion:

```text
Mixed/negative mask diagnostic. Validation AUC-PR improves to 0.9190, but alpha hardens over training even with effect_loss_weight=0.0. Beta remains healthy: raw shortcut stays around 0.40-0.43 and shortcut entropy stays around 0.67. Alpha drifts open and low-entropy: raw causal reaches 0.7911 with entropy 0.1237 by epoch10.
```

Next action:

```text
Do not run test evaluation. For NELL_v1, logit L2 plus causal-only loss is not enough to keep alpha healthy beyond a short smoke. Next diagnostic should target alpha directly, for example a stronger causal-mask budget/logit penalty, staged mask freezing, or a dedicated alpha entropy floor before considering formal evaluation.
```

## 41. nell_v1 Lower Alpha Budget Target Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e diag_v5_nell_v1_alpha_budget035_10ep \
  --use_causal_training \
  --num_epochs 10 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.1 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_target 0.35 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9054
best validation AUC-PR 0.9146
```

Mask checkpoints:

```text
epoch1 raw causal=0.3482
epoch1 raw shortcut=0.4440
epoch1 entropy causal=0.4761
epoch1 entropy shortcut=0.6863
epoch1 budget_loss=0.0665
epoch1 mask_logit_l2_loss=5.6705

epoch5 raw causal=0.7914
epoch5 raw shortcut=0.4250
epoch5 entropy causal=0.1229
epoch5 entropy shortcut=0.6809
epoch5 budget_loss=0.3258
epoch5 mask_logit_l2_loss=50.6074

epoch10 raw causal=0.8050
epoch10 raw shortcut=0.4239
epoch10 entropy causal=0.1171
epoch10 entropy shortcut=0.6809
epoch10 budget_loss=0.3300
epoch10 overlap_loss=0.3365
epoch10 mask_logit_l2_loss=50.4665
epoch10 weight_norm=255.6158
```

Conclusion:

```text
Negative diagnostic. Lowering the causal budget target to 0.35 and increasing budget weight to 0.1 improves only the first epoch. Alpha still re-hardens by epoch5 and reaches raw causal=0.8050 with entropy=0.1171 by epoch10. Validation also underperforms the prior causal-only 10 epoch run: AUC-PR 0.9146 versus 0.9190.
```

Next action:

```text
Do not run test evaluation. NELL_v1 now shows that shared budget/target tuning is insufficient for alpha collapse. The next useful change is an alpha-specific default-off control, such as separate alpha entropy floor/logit L2 weights, or a staged/frozen mask schedule.
```

## 42. Alpha-Specific Regularizer Patch and Smoke

Code change:

```text
Added default-off alpha-specific regularizer controls:
--causal_mask_entropy_floor_weight
--causal_mask_logit_l2_weight
```

These add extra terms on top of the existing shared mask regularizers only when explicitly enabled. Default values are 0.0, so the original GRAIL path and existing v5 configurations remain unchanged unless the new flags are passed.

Smoke run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e smoke_v5_nell_v1_alpha_specific_regs \
  --use_causal_training \
  --num_epochs 1 \
  --batch_size 4 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 \
  --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Smoke validation:

```text
best validation AUC 0.8878
best validation AUC-PR 0.8997
```

Epoch 1 stats:

```text
raw causal=0.5751
raw shortcut=0.4222
entropy causal=0.3430
entropy shortcut=0.6780
mask_reg_loss=0.0518
mask_entropy_floor_loss=0.00138
mask_logit_l2_loss=13.9907
causal_mask_entropy_floor_loss=0.00138
causal_mask_logit_l2_loss=13.8667
```

Conclusion:

```text
Code smoke passed. The parser accepts the new alpha-specific flags, training completes, and the new causal_mask_entropy_floor_loss / causal_mask_logit_l2_loss diagnostics are logged. This is not a performance result.
```

Next action:

```text
Run a longer validation-only NELL_v1 diagnostic with alpha-specific regularization. The immediate target is mask health, not test-set performance: alpha entropy should remain materially above the 0.1-0.12 failure band while beta remains healthy.
```

## 43. nell_v1 Alpha-Specific Regularizer 10 Epoch Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e diag_v5_nell_v1_alpha_specific_regs_10ep \
  --use_causal_training \
  --num_epochs 10 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 \
  --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9191
best observed validation AUC-PR 0.9244
```

Note: checkpoint selection remains validation AUC. The best validation AUC was epoch6 with AUC/AUC-PR 0.9191/0.9206, while the best observed validation AUC-PR was epoch10 with AUC/AUC-PR 0.9131/0.9244.

Mask checkpoints:

```text
epoch1 raw causal=0.4027
epoch1 raw shortcut=0.4352
epoch1 entropy causal=0.4849
epoch1 entropy shortcut=0.6830
epoch1 causal_logit_l2_loss=4.0239

epoch6 raw causal=0.7194
epoch6 raw shortcut=0.4118
epoch6 entropy causal=0.3316
epoch6 entropy shortcut=0.6761
epoch6 mask_logit_l2_loss=11.1772
epoch6 causal_logit_l2_loss=11.0379

epoch10 raw causal=0.6754
epoch10 raw shortcut=0.4157
epoch10 entropy causal=0.4227
epoch10 entropy shortcut=0.6778
epoch10 budget_loss=0.1330
epoch10 overlap_loss=0.2749
epoch10 mask_logit_l2_loss=7.5501
epoch10 causal_logit_l2_loss=7.4246
epoch10 weight_norm=239.4158
```

Conclusion:

```text
Positive validation/mask diagnostic. Alpha-specific regularization materially improves NELL_v1 mask health compared with both previous causal-only diagnostics. Alpha raw can still become high, but entropy stays well above the 0.1-0.12 failure band through epoch10, and beta remains healthy. Validation AUC-PR also improves over the prior 10 epoch causal-only run: 0.9244 observed versus 0.9190.
```

Next action:

```text
Do not run test evaluation yet. This is the first promising NELL_v1 mask-health result, but it needs either a longer validation-only run or a repeated-seed/selection protocol before using test-set budget. If it remains stable, NELL_v1 becomes the best candidate for a formal v5 comparison.
```

## 44. nell_v1 Alpha-Specific Regularizer 20 Epoch Diagnostic

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d nell_v1 -e diag_v5_nell_v1_alpha_specific_regs_20ep \
  --use_causal_training \
  --num_epochs 20 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.0 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 \
  --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect
```

No test-set evaluation was run.

Validation:

```text
best validation AUC 0.9127
best observed validation AUC-PR 0.9219
```

Note: checkpoint selection remains validation AUC. The best validation AUC was epoch8 with AUC/AUC-PR 0.9127/0.9213. The best observed validation AUC-PR was epoch10 with AUC/AUC-PR 0.9099/0.9219.

Mask checkpoints:

```text
epoch2 raw causal=0.5249
epoch2 raw shortcut=0.4256
epoch2 entropy causal=0.1847
epoch2 entropy shortcut=0.6795

epoch8 raw causal=0.5189
epoch8 raw shortcut=0.4257
epoch8 entropy causal=0.2665
epoch8 entropy shortcut=0.6796

epoch10 raw causal=0.6820
epoch10 raw shortcut=0.4147
epoch10 entropy causal=0.2322
epoch10 entropy shortcut=0.6763

epoch15 raw causal=0.5516
epoch15 raw shortcut=0.4240
epoch15 entropy causal=0.1982
epoch15 entropy shortcut=0.6793

epoch20 raw causal=0.5634
epoch20 raw shortcut=0.4235
epoch20 entropy causal=0.2657
epoch20 entropy shortcut=0.6790
epoch20 budget_loss=0.1758
epoch20 overlap_loss=0.2274
epoch20 causal_logit_l2_loss=21.9077
```

Conclusion:

```text
Mixed replication. Alpha-specific regularization still improves mask health relative to earlier NELL_v1 causal-only failures: beta remains healthy and alpha does not fall into the severe 0.1-0.12 entropy failure band for long. However, this 20 epoch repeat is weaker than the previous 10 epoch run. Validation AUC-PR peaks at 0.9219 instead of 0.9244, and alpha entropy occasionally approaches the 0.2 floor, for example epoch2 0.1847 and epoch15 0.1982.
```

Next action:

```text
Do not run test evaluation. Treat alpha-specific regularization as promising but not yet stable. The next useful NELL_v1 step is either a controlled repeat/seed protocol or a small alpha-specific weight sweep, for example stronger causal entropy/logit weights, before formal test evaluation.
```

## 45. 2026-06-04 WN18RR_v1 Alpha-Specific Causal-Only Follow-Up

Code update:

```text
f90fb26 Add configurable validation selection metric
```

New parameter:

```bash
--selection_metric {auc,auc_pr}
```

Default remains `auc`, preserving the old baseline and checkpoint-selection behavior. Use `--selection_metric auc_pr` for future Causal-GraIL runs where AUC-PR is the primary validation criterion.

WN18RR_v1 alpha-specific causal-only, batch_size=16:

```text
experiment diag_v5_wn18rr_v1_alpha_specific_causal_only_20ep
best validation AUC 0.9172
best observed validation AUC-PR 0.9174
test not run
```

Final mask health was good:

```text
epoch20 raw=0.4564/0.4326
epoch20 entropy=0.5152/0.6833
epoch20 budget_loss=0.0731
epoch20 overlap_loss=0.1931
```

WN18RR_v1 alpha-specific causal-only, batch_size=4, AUC-PR selection:

```text
experiment diag_v5_wn18rr_v1_alpha_specific_aucpr_select_bs4_5ep
best validation AUC/AUC-PR 0.9294/0.9266
test not run
```

Mask health was also good:

```text
epoch1 raw=0.4187/0.4337 entropy=0.5267/0.6834
epoch3 raw=0.4328/0.4337 entropy=0.6284/0.6841
epoch5 raw=0.5437/0.4250 entropy=0.5666/0.6811
```

Comparison:

```text
same-env WN18RR_v1 baseline AUC-PR 0.9350
paper WN18RR_v1 AUC-PR target 0.9432
```

Conclusion:

```text
Alpha-specific causal-only regularization solves or strongly mitigates WN18RR_v1 mask collapse, but does not recover enough validation performance. Do not run WN18RR_v1 test for these causal-only configurations.
```

Next action:

```text
Try a strictly controlled, very small effect-loss reintroduction only after mask warmup, selected by AUC-PR. Start with effect_loss_weight around 0.005-0.01, warmup 3, ramp 5, clamp 10. Abort if alpha entropy drops below about 0.15, shortcut entropy collapses, or score means/norms explode.
```

## 46. 2026-06-04 WN18RR_v1 Tiny Effect Follow-Up

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py -d WN18RR_v1 -e diag_v5_wn18rr_v1_tiny_effect_warmup_aucpr_8ep \
  --gpu 0 \
  --use_causal_training \
  --num_epochs 8 \
  --batch_size 16 \
  --causal_loss_weight 1.0 \
  --effect_loss_weight 0.005 \
  --effect_loss_warmup_epochs 3 \
  --effect_loss_ramp_epochs 5 \
  --effect_score_clamp 10 \
  --mask_gamma 0.5 \
  --mask_budget_weight 0.05 \
  --mask_overlap_weight 0.01 \
  --mask_logit_l2_weight 0.001 \
  --causal_mask_entropy_floor_weight 0.1 \
  --causal_mask_logit_l2_weight 0.002 \
  --mask_entropy_floor 0.2 \
  --causal_mask_target 0.45 \
  --shortcut_mask_target 0.45 \
  --score_mode causal_plus_effect \
  --selection_metric auc_pr
```

No test-set evaluation was run.

Validation:

```text
best validation AUC/AUC-PR 0.9133/0.9166
```

Mask checkpoints:

```text
epoch3 effect_weight=0.000 raw=0.8269/0.4027 entropy=0.3155/0.6732
epoch4 effect_weight=0.001 raw=0.7055/0.3791 entropy=0.4655/0.6048
epoch6 effect_weight=0.003 raw=0.7892/0.4702 entropy=0.3653/0.5166 budget=0.2413 overlap=0.3943
epoch8 effect_weight=0.005 raw=0.7126/0.4194 entropy=0.4706/0.4772 budget=0.2014 overlap=0.3147
```

Conclusion:

```text
Tiny delayed effect loss with clamp 10 does not explode, but it also does not improve validation. It starts to reduce shortcut entropy and increase overlap/budget pressure once ramped. Do not continue this exact configuration and do not test it.
```

Next action:

```text
For WN18RR_v1, avoid direct effect margin as the next main move. Prefer score-mode and loss-weight diagnostics: try causal-only training with score_mode=causal and AUC-PR selection, or lower causal_loss_weight, before designing a different effect regularizer.
```
