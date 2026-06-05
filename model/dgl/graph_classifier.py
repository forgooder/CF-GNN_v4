from .rgcn_model import RGCN
from .causal_mask import CausalMaskGenerator
from dgl import mean_nodes
import torch.nn as nn
import torch
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.causal_mask_generator = CausalMaskGenerator(params)

        if self.params.add_ht_emb:
            scorer_input_dim = 3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim
        else:
            scorer_input_dim = self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim

        score_hidden_dim = getattr(self.params, 'score_hidden_dim', 0)
        if score_hidden_dim and score_hidden_dim > 0:
            score_dropout = getattr(self.params, 'score_dropout', 0.0)
            self.fc_layer = nn.Sequential(
                nn.Linear(scorer_input_dim, score_hidden_dim),
                nn.ReLU(),
                nn.Dropout(score_dropout),
                nn.Linear(score_hidden_dim, 1)
            )
        else:
            self.fc_layer = nn.Linear(scorer_input_dim, 1)

    def _score(self, g, rel_labels, edge_mask=None):
        g.ndata['h'] = self.gnn(g, edge_mask=edge_mask)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output

    def _get_causal_mask_generator(self, device):
        if (
            not hasattr(self, 'causal_mask_generator')
            or not hasattr(self.causal_mask_generator, 'shared_mlp')
            or not hasattr(self.causal_mask_generator, 'shortcut_head')
        ):
            self.causal_mask_generator = CausalMaskGenerator(self.params).to(device=device)
        return self.causal_mask_generator

    def forward(self, data, edge_mask=None, mode='original', return_masks=False):
        g, rel_labels = data

        if mode == 'original':
            return self._score(g, rel_labels, edge_mask=edge_mask)

        if edge_mask is not None:
            raise ValueError("edge_mask can only be passed directly when mode='original'")

        causal_mask_generator = self._get_causal_mask_generator(g.ndata['feat'].device)
        mask_outputs = causal_mask_generator(g, rel_labels)
        causal_edge_mask = mask_outputs['causal_edge_mask']
        shortcut_edge_mask = mask_outputs['shortcut_edge_mask']

        if mode == 'causal':
            output = self._score(g, rel_labels, edge_mask=causal_edge_mask)
            if return_masks:
                return {
                    'causal': output,
                    'causal_mask': causal_edge_mask,
                    'causal_raw_mask': mask_outputs['causal_raw_mask'],
                    'causal_logits': mask_outputs['causal_logits'],
                    'mask_stats': mask_outputs['mask_stats']
                }
            return output

        if mode == 'shortcut':
            output = self._score(g, rel_labels, edge_mask=shortcut_edge_mask)
            if return_masks:
                return {
                    'shortcut': output,
                    'shortcut_mask': shortcut_edge_mask,
                    'shortcut_raw_mask': mask_outputs['shortcut_raw_mask'],
                    'shortcut_logits': mask_outputs['shortcut_logits'],
                    'mask_stats': mask_outputs['mask_stats']
                }
            return output

        if mode == 'all':
            score_original = self._score(g, rel_labels)
            score_causal = self._score(g, rel_labels, edge_mask=causal_edge_mask)
            score_shortcut = self._score(g, rel_labels, edge_mask=shortcut_edge_mask)
            return {
                'original': score_original,
                'causal': score_causal,
                'shortcut': score_shortcut,
                'effect': score_causal - score_shortcut,
                'causal_mask': causal_edge_mask,
                'shortcut_mask': shortcut_edge_mask,
                'causal_raw_mask': mask_outputs['causal_raw_mask'],
                'shortcut_raw_mask': mask_outputs['shortcut_raw_mask'],
                'causal_logits': mask_outputs['causal_logits'],
                'shortcut_logits': mask_outputs['shortcut_logits'],
                'target_rel_labels': mask_outputs['target_rel_labels'],
                'mask_stats': mask_outputs['mask_stats']
            }

        raise ValueError(f"Unknown graph classifier mode: {mode}")
