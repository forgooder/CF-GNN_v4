import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from utils.score_utils import SCORE_MODE_CHOICES

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--use_causal_training", action='store_true',
                        help="Enable Causal-GraIL training losses. Default keeps the original GraIL baseline.")
    parser.add_argument("--causal_loss_weight", type=float, default=1.0,
                        help="Weight for causal score margin ranking loss when causal training is enabled")
    parser.add_argument("--causal_loss_warmup_epochs", type=int, default=0,
                        help="Number of initial epochs with causal auxiliary loss disabled during causal training")
    parser.add_argument("--causal_loss_ramp_epochs", type=int, default=0,
                        help="Number of epochs used to linearly ramp causal auxiliary loss after warmup; 0 preserves step activation")
    parser.add_argument("--effect_loss_weight", type=float, default=1.0,
                        help="Weight for causal effect margin ranking loss when causal training is enabled")
    parser.add_argument("--effect_loss_warmup_epochs", type=int, default=0,
                        help="Number of initial epochs with effect loss disabled during causal training")
    parser.add_argument("--effect_loss_ramp_epochs", type=int, default=0,
                        help="Number of epochs used to linearly ramp effect loss after warmup; 0 preserves step activation")
    parser.add_argument("--effect_gradient_mode", type=str,
                        choices=['full', 'detach_shortcut', 'detach_causal', 'detach_both'],
                        default='full',
                        help="Gradient path used only for effect loss; default full preserves existing behavior")
    parser.add_argument("--effect_score_clamp", type=float, default=0.0,
                        help="Optional absolute clamp for training-time effect-loss scores; <=0 disables")
    parser.add_argument("--masked_aux_gradient_mode", type=str,
                        choices=['full', 'mask_only'],
                        default='full',
                        help="Gradient path for masked causal/shortcut auxiliary losses; default full preserves existing behavior")
    parser.add_argument("--shortcut_penalty_weight", type=float, default=0.0,
                        help="Optional penalty that discourages shortcut scores from separating positive and negative samples")
    parser.add_argument("--score_l2_weight", type=float, default=0.0,
                        help="Optional training-time L2 penalty on causal-training scorer magnitudes; default off")
    parser.add_argument("--relation_loss_weight_path", type=str, default='',
                        help="Optional JSON file with relation-specific ranking-loss weights for causal training")
    parser.add_argument("--mask_sparsity_weight", type=float, default=0.0,
                        help="Backward-compatible optional sparsity penalty on raw causal edge masks")
    parser.add_argument("--mask_entropy_weight", type=float, default=0.0,
                        help="Backward-compatible optional entropy penalty on raw causal edge masks")
    parser.add_argument("--mask_entropy_floor_weight", type=float, default=0.0,
                        help="Weight for entropy-floor loss that discourages alpha/beta mask saturation")
    parser.add_argument("--mask_entropy_floor", type=float, default=0.1,
                        help="Minimum desired entropy for raw causal and shortcut masks")
    parser.add_argument("--mask_logit_l2_weight", type=float, default=0.0,
                        help="Weight for L2 penalty on causal/shortcut mask logits to discourage saturation")
    parser.add_argument("--causal_mask_entropy_floor_weight", type=float, default=0.0,
                        help="Optional extra entropy-floor weight for causal alpha masks only")
    parser.add_argument("--causal_mask_logit_l2_weight", type=float, default=0.0,
                        help="Optional extra L2 penalty weight for causal alpha mask logits only")
    parser.add_argument("--mask_gamma", type=float, default=1.0,
                        help="Residual mask strength: effective_mask = 1 - gamma + gamma * raw_mask")
    parser.add_argument("--mask_budget_weight", type=float, default=0.0,
                        help="Weight for causal/shortcut raw mask target-ratio budget loss")
    parser.add_argument("--mask_overlap_weight", type=float, default=0.0,
                        help="Weight for raw causal-shortcut mask overlap penalty")
    parser.add_argument("--causal_mask_target", type=float, default=0.5,
                        help="Default target ratio for raw causal masks in budget loss")
    parser.add_argument("--shortcut_mask_target", type=float, default=0.5,
                        help="Default target ratio for raw shortcut masks in budget loss")
    parser.add_argument("--relation_budget_path", type=str, default='',
                        help="Optional JSON file with relation-aware causal/shortcut mask target ratios")
    parser.add_argument("--relation_overlap_penalty_path", type=str, default='',
                        help="Optional JSON file with relation weights for extra overlap penalty")
    parser.add_argument("--relation_overlap_penalty_weight", type=float, default=0.0,
                        help="Weight for relation-specific raw causal-shortcut overlap penalty")
    parser.add_argument("--relation_shortcut_floor_path", type=str, default='',
                        help="Optional JSON file with relation-specific shortcut raw mask floors")
    parser.add_argument("--relation_shortcut_floor_weight", type=float, default=0.0,
                        help="Weight for relation-specific shortcut raw mask floor penalty")
    parser.add_argument("--score_mode", type=str, choices=SCORE_MODE_CHOICES, default='original',
                        help="Score used for validation and training AUC logging")
    parser.add_argument("--selection_metric", type=str, choices=['auc', 'auc_pr'], default='auc',
                        help="Validation metric used to save the best checkpoint; default preserves existing AUC selection")
    parser.add_argument("--log_all_score_modes_validation", action='store_true',
                        help="Log validation AUC/AUC-PR for every score mode; default off and does not affect checkpoint selection")
    parser.add_argument("--log_relation_metrics_validation", action='store_true',
                        help="Log validation AUC/AUC-PR grouped by target relation for the selected score mode; default off")
    parser.add_argument("--log_relation_score_stats_validation", action='store_true',
                        help="Log validation positive/negative score distribution stats grouped by target relation; default off")
    parser.add_argument("--log_relation_pair_stats_validation", action='store_true',
                        help="Log validation paired positive/negative score and graph-size stats grouped by target relation; default off")
    parser.add_argument("--log_relation_mask_validation", action='store_true',
                        help="Log validation mask diagnostics grouped by target relation; default off")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument("--add_ht_interaction_features", action='store_true',
                        help="Add head-tail product and absolute-difference features to the graph scorer; default off")
    parser.add_argument("--add_graph_maxpool_features", action='store_true',
                        help="Add max-pooled graph features to the scorer input; default off")
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument("--score_hidden_dim", type=int, default=0,
                        help="Optional hidden dimension for a nonlinear graph scorer; 0 keeps the original linear scorer")
    parser.add_argument("--score_dropout", type=float, default=0.0,
                        help="Dropout used by the optional nonlinear graph scorer")

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
