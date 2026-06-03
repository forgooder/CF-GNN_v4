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
