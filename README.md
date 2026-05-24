# Causal-GraIL v5

This repository keeps the original GraIL baseline for inductive relation prediction and adds optional Causal-GraIL paths. The causal path is disabled by default, so the original GraIL training and evaluation commands remain valid.

The implementation is based on the GraIL algorithm described in the ICML'20 paper [Inductive relation prediction by subgraph reasoning](https://arxiv.org/abs/1911.06962).

## Requiremetns

All the required packages can be installed by running `pip install -r requirements.txt`.

## Inductive relation prediction experiments

All train-graph and ind-test-graph pairs of graphs can be found in the `data` folder. We use WN18RR_v1 as a runninng example for illustrating the steps.

### GraIL Baseline
To start training a GraIL model, run the following command. 
`python train.py -d WN18RR_v1 -e grail_wn_v1`

To test GraIL run the following commands.
- `python test_auc.py -d WN18RR_v1_ind -e grail_wn_v1`
- `python test_ranking.py -d WN18RR_v1_ind -e grail_wn_v1`

The trained model and the logs are stored in `experiments` folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating GraIL and use these later to evaluate other baseline models.

### Causal-GraIL v5
Causal-GraIL adds opt-in components:
- optional edge masks in the DGL R-GCN message passing path;
- an alpha/beta two-head `CausalMaskGenerator` that can produce original, causal, shortcut, and causal-effect scores;
- causal ranking losses and selectable validation/test score modes.

In v5, shortcut masks are no longer defined as `1 - causal_mask`. The generator learns two independent raw masks:

`causal_mask = sigmoid(causal_logits)`

`shortcut_mask = sigmoid(shortcut_logits)`

The masks injected into R-GCN use a residual gate:

`effective_mask = 1 - gamma + gamma * raw_mask`

where `--mask_gamma` defaults to `1.0`. Smaller values keep more of the original graph signal.

The default configuration does not enable causal training. To run a small causal smoke training job:

`python train.py -d WN18RR_v1 -e causal_grail_wn_v1_smoke --use_causal_training --num_epochs 1 --score_mode original`

To train with causal losses and v5 budget/overlap regularization:

`python train.py -d WN18RR_v1 -e causal_grail_wn_v1_v5 --use_causal_training --causal_loss_weight 1.0 --effect_loss_weight 0.1 --effect_loss_warmup_epochs 5 --mask_gamma 0.7 --mask_budget_weight 0.01 --mask_overlap_weight 0.01 --causal_mask_target 0.5 --shortcut_mask_target 0.3 --score_mode original`

The v4 sparsity/entropy parameters are still accepted for compatibility, but v5 primarily targets mask collapse with budget and overlap terms:
- `--mask_budget_weight`
- `--mask_overlap_weight`
- `--causal_mask_target`
- `--shortcut_mask_target`
- `--relation_budget_path`

`--relation_budget_path` can point to a JSON file with relation-aware target ratios. Keys may be relation ids or relation names:

```
{
  "causal": {
    "0": 0.5,
    "_hypernym": 0.4
  },
  "shortcut": {
    "0": 0.3,
    "_hypernym": 0.25
  }
}
```

Available score modes for validation and testing are:
- `original`: original GraIL score.
- `causal`: score from the causal masked subgraph.
- `effect`: `score_causal - score_shortcut`.
- `causal_plus_effect`: `score_causal + effect`.

AUC/AUC-PR examples:

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode original`

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode causal`

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode effect`

Ranking examples:

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode original`

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode causal`

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1_v5 --score_mode effect`

No metric improvement is claimed by this repository. Run the experiments on your target server and report results using the unchanged AUC, AUC-PR, and Hits@10 evaluation logic.

### RuleN
RuleN operates in two steps. Rules are first learned from a training graph and then applied on the test graph. Detailed instructions can be found [here](http://web.informatik.uni-mannheim.de/RuleN/).
- Learn rules: source learn_rules.sh WN18RR_v1
- Apply rules:
	- To get AUC: `source auc_apply_rules.sh WN18RR_v1 WN18RR_v1_ind num_of_samples_to_score(=1000)`
	- To get ranking score: `source auc_apply_rules.sh WN18RR_v1 WN18RR_v1_ind num_of_samples_to_score(=1000)`

### NeuralLP and Drum
We use the implementations provided by the authors of the respective papers to evaluate these models.

## Transductive experiments

The full transductive datasets used in these experiments are present in the `data` folder.

### GraIL
The training and testing protocols of GraIL remains the same.

### KGE models
We use the comprehensive implementation provided by authors of RotatE. This gives state-of-the-art results on all datasets. The best configurations can be found [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh). To train these KGE models, navigate to the `kge` folder and run the commands as shown in the above reference. For example, to train TransE on FB237-15k, run the following command.

`bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16`

This will store the trained model and the logs in a folder named `experiments/kge_baselines/TransE_FB15k-237`.

### Ensembling instructions
Once the KGE models are trained, to get ensembling results with GraIL, navigate to the `ensembling` folder and run the following command.
`source get_ensemble_predictions.sh WN18RR TransE`

To get ensenbling among different KGE models, from the `ensembling` folder run the following command.
`source get_kge_predictions.sh WN18RR TransE ComplEx`



If you make use of this code or the GraIL algorithm in your work, please cite the following paper:

	@article{Teru2020InductiveRP,
	  title={Inductive Relation Prediction by Subgraph Reasoning.},
	  author={Komal K. Teru and Etienne Denis and William L. Hamilton},
	  journal={arXiv: Learning},
	  year={2020}
	}
