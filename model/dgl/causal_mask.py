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
        self.mask_gamma = getattr(params, 'mask_gamma', 1.0)

        self.shared_mlp = nn.Sequential(
            nn.Linear(2 * params.inp_dim + 2 * rel_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.causal_head = nn.Linear(hidden_dim, 1)
        self.shortcut_head = nn.Linear(hidden_dim, 1)

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

    def _apply_residual_mask(self, raw_mask):
        gamma = max(0.0, min(1.0, float(self.mask_gamma)))
        return 1.0 - gamma + gamma * raw_mask

    def _stats(self, causal_raw_mask, shortcut_raw_mask, causal_edge_mask, shortcut_edge_mask):
        eps = 1e-8
        causal_entropy = -(
            causal_raw_mask * torch.log(causal_raw_mask + eps)
            + (1 - causal_raw_mask) * torch.log(1 - causal_raw_mask + eps)
        ).mean()
        shortcut_entropy = -(
            shortcut_raw_mask * torch.log(shortcut_raw_mask + eps)
            + (1 - shortcut_raw_mask) * torch.log(1 - shortcut_raw_mask + eps)
        ).mean()
        return {
            'causal_raw_mean': causal_raw_mask.mean(),
            'shortcut_raw_mean': shortcut_raw_mask.mean(),
            'causal_effective_mean': causal_edge_mask.mean(),
            'shortcut_effective_mean': shortcut_edge_mask.mean(),
            'causal_entropy': causal_entropy,
            'shortcut_entropy': shortcut_entropy,
            'overlap': (causal_raw_mask * shortcut_raw_mask).mean()
        }

    def forward(self, g, target_rel_labels=None):
        logits_key = '_causal_mask_logits'
        target_key = '_causal_target_rel'
        had_logits = logits_key in g.edata
        previous_logits = g.edata[logits_key] if had_logits else None
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
            hidden = self.shared_mlp(edge_repr)
            return {
                logits_key: torch.cat([
                    self.causal_head(hidden),
                    self.shortcut_head(hidden)
                ], dim=1)
            }

        try:
            g.edata[target_key] = target_edge_labels
            g.apply_edges(edge_score)
            mask_logits = g.edata[logits_key]
            causal_logits = mask_logits[:, 0:1]
            shortcut_logits = mask_logits[:, 1:2]
            causal_raw_mask = torch.sigmoid(causal_logits)
            shortcut_raw_mask = torch.sigmoid(shortcut_logits)
            causal_edge_mask = self._apply_residual_mask(causal_raw_mask)
            shortcut_edge_mask = self._apply_residual_mask(shortcut_raw_mask)
        finally:
            if had_logits:
                g.edata[logits_key] = previous_logits
            elif logits_key in g.edata:
                g.edata.pop(logits_key)
            if had_target:
                g.edata[target_key] = previous_target
            elif target_key in g.edata:
                g.edata.pop(target_key)

        return {
            'causal_edge_mask': causal_edge_mask,
            'shortcut_edge_mask': shortcut_edge_mask,
            'causal_raw_mask': causal_raw_mask,
            'shortcut_raw_mask': shortcut_raw_mask,
            'causal_logits': causal_logits,
            'shortcut_logits': shortcut_logits,
            'target_rel_labels': target_edge_labels,
            'mask_stats': self._stats(causal_raw_mask, shortcut_raw_mask, causal_edge_mask, shortcut_edge_mask)
        }
