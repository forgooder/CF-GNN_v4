# Causal-GraIL Results Ledger

| Date | Commit | Dataset | Experiment | Config Summary | Score Mode | AUC | AUC-PR | MRR | Hits@1 | Hits@5 | Hits@10 | Mask Health | Conclusion |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-06-03 00:08 CST | 83d1728 | WN18RR_v1 | smoke_v5_baseline | 1 epoch, batch_size=4, baseline path, validation only | original | 0.9105 | 0.9263 | - | - | - | - | N/A; causal training disabled | Passed smoke; baseline path runs on v5 branch |
| 2026-06-03 00:08 CST | 83d1728 | WN18RR_v1 | smoke_v5_causal | 1 epoch, batch_size=4, causal_loss=1.0, effect_loss=0.1, warmup=5, gamma=0.7, budget=0.01, overlap=0.01, targets 0.5/0.3, validation only | causal | 0.9194 | 0.9302 | - | - | - | - | Smoke healthy: raw causal=0.2166, raw shortcut=0.2541, entropy causal=0.3070, shortcut=0.5449 | Passed causal smoke; no parameter/shape/DGL error |
