import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.score_utils import forward_for_score, select_score


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_mode = getattr(self.params, 'score_mode', 'original')
                score_pos = forward_for_score(self.graph_classifier, data_pos, score_mode)
                score_neg = forward_for_score(self.graph_classifier, data_neg, score_mode)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}

    def eval_score_modes(self, score_modes):
        mode_scores = {
            score_mode: {
                'pos_scores': [],
                'neg_scores': [],
                'pos_labels': [],
                'neg_labels': []
            }
            for score_mode in score_modes
        }
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                outputs_pos = self.graph_classifier(data_pos, mode='all')
                outputs_neg = self.graph_classifier(data_neg, mode='all')

                for score_mode, scores in mode_scores.items():
                    score_pos = select_score(outputs_pos, score_mode)
                    score_neg = select_score(outputs_neg, score_mode)
                    scores['pos_scores'] += score_pos.view(-1).detach().cpu().tolist()
                    scores['neg_scores'] += score_neg.view(-1).detach().cpu().tolist()
                    scores['pos_labels'] += targets_pos.tolist()
                    scores['neg_labels'] += targets_neg.tolist()

        results = {}
        for score_mode, scores in mode_scores.items():
            labels = scores['pos_labels'] + scores['neg_labels']
            predicted_scores = scores['pos_scores'] + scores['neg_scores']
            results[score_mode] = {
                'auc': metrics.roc_auc_score(labels, predicted_scores),
                'auc_pr': metrics.average_precision_score(labels, predicted_scores)
            }

        return results

    def eval_by_relation(self, score_mode=None):
        score_mode = score_mode or getattr(self.params, 'score_mode', 'original')
        relation_scores = {}
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = forward_for_score(self.graph_classifier, data_pos, score_mode).view(-1).detach().cpu()
                score_neg = forward_for_score(self.graph_classifier, data_neg, score_mode).view(-1).detach().cpu()
                rel_pos = data_pos[1].view(-1).detach().cpu().tolist()
                rel_neg_tensor = data_neg[1].view(-1).detach().cpu()
                if len(rel_neg_tensor) != len(score_neg):
                    repeat_factor = int(len(score_neg) / max(1, len(rel_neg_tensor)))
                    rel_neg_tensor = rel_neg_tensor.repeat_interleave(repeat_factor)
                rel_neg = rel_neg_tensor.tolist()

                for rel_id, score, label in zip(rel_pos, score_pos.tolist(), targets_pos.detach().cpu().tolist()):
                    bucket = relation_scores.setdefault(int(rel_id), {'scores': [], 'labels': []})
                    bucket['scores'].append(float(score))
                    bucket['labels'].append(int(label))
                for rel_id, score, label in zip(rel_neg, score_neg.tolist(), targets_neg.detach().cpu().tolist()):
                    bucket = relation_scores.setdefault(int(rel_id), {'scores': [], 'labels': []})
                    bucket['scores'].append(float(score))
                    bucket['labels'].append(int(label))

        id2relation = getattr(self.data, 'id2relation', {})
        results = []
        for rel_id, bucket in relation_scores.items():
            labels = bucket['labels']
            predicted_scores = bucket['scores']
            positives = int(sum(labels))
            negatives = int(len(labels) - positives)
            if positives == 0 or negatives == 0:
                auc = None
                auc_pr = None
            else:
                auc = float(metrics.roc_auc_score(labels, predicted_scores))
                auc_pr = float(metrics.average_precision_score(labels, predicted_scores))
            results.append({
                'rel_id': int(rel_id),
                'relation': id2relation.get(int(rel_id), str(rel_id)),
                'support': int(len(labels)),
                'positives': positives,
                'negatives': negatives,
                'auc': auc,
                'auc_pr': auc_pr
            })

        return sorted(
            results,
            key=lambda item: (-1.0 if item['auc_pr'] is None else item['auc_pr'], item['rel_id'])
        )

    def eval_score_stats_by_relation(self, score_mode=None):
        score_mode = score_mode or getattr(self.params, 'score_mode', 'original')
        relation_scores = {}
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = forward_for_score(self.graph_classifier, data_pos, score_mode).view(-1).detach().cpu()
                score_neg = forward_for_score(self.graph_classifier, data_neg, score_mode).view(-1).detach().cpu()
                rel_pos = data_pos[1].view(-1).detach().cpu().tolist()
                rel_neg_tensor = data_neg[1].view(-1).detach().cpu()
                if len(rel_neg_tensor) != len(score_neg):
                    repeat_factor = int(len(score_neg) / max(1, len(rel_neg_tensor)))
                    rel_neg_tensor = rel_neg_tensor.repeat_interleave(repeat_factor)
                rel_neg = rel_neg_tensor.tolist()

                for rel_id, score in zip(rel_pos, score_pos.tolist()):
                    bucket = relation_scores.setdefault(int(rel_id), {'pos_scores': [], 'neg_scores': []})
                    bucket['pos_scores'].append(float(score))
                for rel_id, score in zip(rel_neg, score_neg.tolist()):
                    bucket = relation_scores.setdefault(int(rel_id), {'pos_scores': [], 'neg_scores': []})
                    bucket['neg_scores'].append(float(score))

        id2relation = getattr(self.data, 'id2relation', {})
        results = []
        for rel_id, bucket in relation_scores.items():
            pos_scores = np.asarray(bucket['pos_scores'], dtype=np.float64)
            neg_scores = np.asarray(bucket['neg_scores'], dtype=np.float64)
            pos_mean = float(pos_scores.mean()) if len(pos_scores) else None
            neg_mean = float(neg_scores.mean()) if len(neg_scores) else None
            score_gap = None if pos_mean is None or neg_mean is None else pos_mean - neg_mean
            result = {
                'rel_id': int(rel_id),
                'relation': id2relation.get(int(rel_id), str(rel_id)),
                'positives': int(len(pos_scores)),
                'negatives': int(len(neg_scores)),
                'pos_mean': pos_mean,
                'pos_std': float(pos_scores.std()) if len(pos_scores) else None,
                'neg_mean': neg_mean,
                'neg_std': float(neg_scores.std()) if len(neg_scores) else None,
                'score_gap': score_gap
            }
            if len(pos_scores) and len(pos_scores) == len(neg_scores):
                margins = pos_scores - neg_scores
                result.update({
                    'margin_mean': float(margins.mean()),
                    'margin_p10': float(np.percentile(margins, 10)),
                    'margin_p50': float(np.percentile(margins, 50)),
                    'margin_p90': float(np.percentile(margins, 90)),
                    'pos_gt_neg_rate': float((margins > 0).mean())
                })
            results.append(result)

        return sorted(
            results,
            key=lambda item: (float('inf') if item['score_gap'] is None else item['score_gap'], item['rel_id'])
        )

    def _add_relation_mask_stats(self, relation_masks, outputs):
        rel_labels = outputs.get('target_rel_labels')
        if rel_labels is None:
            return

        eps = 1e-8
        rel_labels = rel_labels.view(-1).detach().cpu()
        causal_raw = outputs['causal_raw_mask'].view(-1).detach().cpu()
        shortcut_raw = outputs['shortcut_raw_mask'].view(-1).detach().cpu()
        causal_entropy = -(
            causal_raw * torch.log(causal_raw + eps)
            + (1 - causal_raw) * torch.log(1 - causal_raw + eps)
        )
        shortcut_entropy = -(
            shortcut_raw * torch.log(shortcut_raw + eps)
            + (1 - shortcut_raw) * torch.log(1 - shortcut_raw + eps)
        )
        overlap = causal_raw * shortcut_raw

        for rel_id, causal, shortcut, c_entropy, s_entropy, ov in zip(
            rel_labels.tolist(),
            causal_raw.tolist(),
            shortcut_raw.tolist(),
            causal_entropy.tolist(),
            shortcut_entropy.tolist(),
            overlap.tolist()
        ):
            bucket = relation_masks.setdefault(int(rel_id), {
                'count': 0,
                'causal_raw_sum': 0.0,
                'shortcut_raw_sum': 0.0,
                'causal_entropy_sum': 0.0,
                'shortcut_entropy_sum': 0.0,
                'overlap_sum': 0.0
            })
            bucket['count'] += 1
            bucket['causal_raw_sum'] += float(causal)
            bucket['shortcut_raw_sum'] += float(shortcut)
            bucket['causal_entropy_sum'] += float(c_entropy)
            bucket['shortcut_entropy_sum'] += float(s_entropy)
            bucket['overlap_sum'] += float(ov)

    def eval_mask_by_relation(self):
        relation_masks = {}
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                outputs_pos = self.graph_classifier(data_pos, mode='all')
                outputs_neg = self.graph_classifier(data_neg, mode='all')
                self._add_relation_mask_stats(relation_masks, outputs_pos)
                self._add_relation_mask_stats(relation_masks, outputs_neg)

        id2relation = getattr(self.data, 'id2relation', {})
        results = []
        for rel_id, bucket in relation_masks.items():
            count = max(1, bucket['count'])
            results.append({
                'rel_id': int(rel_id),
                'relation': id2relation.get(int(rel_id), str(rel_id)),
                'edge_count': int(bucket['count']),
                'causal_raw_mean': bucket['causal_raw_sum'] / count,
                'shortcut_raw_mean': bucket['shortcut_raw_sum'] / count,
                'causal_entropy': bucket['causal_entropy_sum'] / count,
                'shortcut_entropy': bucket['shortcut_entropy_sum'] / count,
                'overlap': bucket['overlap_sum'] / count
            })

        return sorted(results, key=lambda item: item['rel_id'])
