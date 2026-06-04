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
