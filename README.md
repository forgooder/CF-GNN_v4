# Causal-GraIL v4

This repository keeps the original GraIL baseline for inductive relation prediction and adds an optional Causal-GraIL path. The causal path is disabled by default, so the original GraIL training and evaluation commands remain valid.

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

### Causal-GraIL
Causal-GraIL adds three opt-in components:
- optional edge masks in the DGL R-GCN message passing path;
- a `CausalMaskGenerator` that can produce original, causal, shortcut, and causal-effect scores;
- causal ranking losses and selectable validation/test score modes.

The default configuration does not enable causal training. To run a small causal smoke training job:

`python train.py -d WN18RR_v1 -e causal_grail_wn_v1_smoke --use_causal_training --num_epochs 1 --score_mode original`

To train with causal losses:

`python train.py -d WN18RR_v1 -e causal_grail_wn_v1 --use_causal_training --causal_loss_weight 1.0 --effect_loss_weight 1.0 --mask_sparsity_weight 0.001 --mask_entropy_weight 0.001 --score_mode original`

Available score modes for validation and testing are:
- `original`: original GraIL score.
- `causal`: score from the causal masked subgraph.
- `effect`: `score_causal - score_shortcut`.
- `causal_plus_effect`: `score_causal + effect`.

AUC/AUC-PR examples:

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode original`

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode causal`

`python test_auc.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode effect`

Ranking examples:

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode original`

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode causal`

`python test_ranking.py -d WN18RR_v1_ind -e causal_grail_wn_v1 --score_mode effect`

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
