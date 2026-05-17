import torch
import torch.nn as nn


class CausalMaskGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()

        rel_emb_dim = getattr(params, 'causal_mask_rel_emb_dim', params.rel_emb_dim)
        hidden_dim = getattr(params, 'causal_mask_hidden_dim', params.emb_dim)
        dropout = getattr(params, 'causal_mask_dropout', 0.0)

        self.edge_type_emb = nn.Embedding(params.aug_num_rels, rel_emb_dim)
        self.target_rel_emb = nn.Embedding(params.num_rels, rel_emb_dim)

        self.mask_mlp = nn.Sequential(
            nn.Linear(2 * params.inp_dim + 2 * rel_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _batch_num_edges(self, g):
        batch_num_edges = getattr(g, 'batch_num_edges', None)
        if batch_num_edges is None:
            return None
        return batch_num_edges() if callable(batch_num_edges) else batch_num_edges

    def _target_labels_for_edges(self, g, target_rel_labels):
        if target_rel_labels is None:
            return g.edata['label']

        batch_num_edges = self._batch_num_edges(g)
        if batch_num_edges is None:
            return g.edata['label']

        batch_num_edges = torch.as_tensor(batch_num_edges, device=target_rel_labels.device, dtype=torch.long)
        if len(batch_num_edges) != len(target_rel_labels):
            return g.edata['label']

        return torch.repeat_interleave(target_rel_labels, batch_num_edges).to(device=g.edata['type'].device)

    def forward(self, g, target_rel_labels=None):
        tmp_key = '_causal_mask_logits'
        target_key = '_causal_target_rel'
        had_tmp = tmp_key in g.edata
        previous_tmp = g.edata[tmp_key] if had_tmp else None
        had_target = target_key in g.edata
        previous_target = g.edata[target_key] if had_target else None
        target_edge_labels = self._target_labels_for_edges(g, target_rel_labels)

        def edge_score(edges):
            edge_repr = torch.cat([
                edges.src['feat'],
                edges.dst['feat'],
                self.edge_type_emb(edges.data['type']),
                self.target_rel_emb(edges.data[target_key])
            ], dim=1)
            return {tmp_key: self.mask_mlp(edge_repr)}

        try:
            g.edata[target_key] = target_edge_labels
            g.apply_edges(edge_score)
            causal_edge_mask = torch.sigmoid(g.edata[tmp_key])
        finally:
            if had_tmp:
                g.edata[tmp_key] = previous_tmp
            elif tmp_key in g.edata:
                g.edata.pop(tmp_key)
            if had_target:
                g.edata[target_key] = previous_target
            elif target_key in g.edata:
                g.edata.pop(target_key)

        shortcut_edge_mask = 1.0 - causal_edge_mask
        return causal_edge_mask, shortcut_edge_mask
