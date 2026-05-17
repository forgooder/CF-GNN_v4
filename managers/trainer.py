import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics
from utils.score_utils import select_score


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0
        self.use_causal_training = getattr(params, 'use_causal_training', False)
        if self.use_causal_training and hasattr(self.graph_classifier, '_get_causal_mask_generator'):
            self.graph_classifier._get_causal_mask_generator(params.device)

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def ranking_loss(self, score_pos, score_neg):
        score_pos = score_pos.view(-1)
        score_neg = score_neg.view(len(score_pos), -1).mean(dim=1)
        target = torch.ones_like(score_pos, device=self.params.device)
        return self.criterion(score_pos, score_neg, target)

    def mask_regularization(self, outputs_pos, outputs_neg):
        mask_sparsity_weight = getattr(self.params, 'mask_sparsity_weight', 0.0)
        mask_entropy_weight = getattr(self.params, 'mask_entropy_weight', 0.0)

        masks = torch.cat([
            outputs_pos['causal_mask'].view(-1),
            outputs_neg['causal_mask'].view(-1)
        ])
        sparsity = masks.mean()
        eps = 1e-8
        entropy = -(masks * torch.log(masks + eps) + (1 - masks) * torch.log(1 - masks + eps)).mean()
        reg_loss = mask_sparsity_weight * sparsity + mask_entropy_weight * entropy
        return reg_loss, sparsity.item(), entropy.item()

    def shortcut_penalty(self, outputs_pos, outputs_neg):
        shortcut_penalty_weight = getattr(self.params, 'shortcut_penalty_weight', 0.0)
        if shortcut_penalty_weight == 0:
            return torch.tensor(0.0, device=self.params.device)

        shortcut_pos = outputs_pos['shortcut'].view(-1)
        shortcut_neg = outputs_neg['shortcut'].view(len(shortcut_pos), -1).mean(dim=1)
        return shortcut_penalty_weight * torch.abs(shortcut_pos - shortcut_neg).mean()

    def causal_training_step(self, data_pos, data_neg):
        outputs_pos = self.graph_classifier(data_pos, mode='all')
        outputs_neg = self.graph_classifier(data_neg, mode='all')

        original_loss = self.ranking_loss(outputs_pos['original'], outputs_neg['original'])
        causal_loss = self.ranking_loss(outputs_pos['causal'], outputs_neg['causal'])
        effect_loss = self.ranking_loss(outputs_pos['effect'], outputs_neg['effect'])
        mask_reg_loss, mask_sparsity, mask_entropy = self.mask_regularization(outputs_pos, outputs_neg)
        shortcut_loss = self.shortcut_penalty(outputs_pos, outputs_neg)

        total_loss = (
            original_loss
            + getattr(self.params, 'causal_loss_weight', 1.0) * causal_loss
            + getattr(self.params, 'effect_loss_weight', 1.0) * effect_loss
            + mask_reg_loss
            + shortcut_loss
        )

        score_mode = getattr(self.params, 'score_mode', 'original')
        score_pos = select_score(outputs_pos, score_mode)
        score_neg = select_score(outputs_neg, score_mode)

        stats = {
            'original_loss': original_loss.item(),
            'causal_loss': causal_loss.item(),
            'effect_loss': effect_loss.item(),
            'mask_reg_loss': mask_reg_loss.item(),
            'shortcut_penalty': shortcut_loss.item(),
            'total_loss': total_loss.item(),
            'original_score_mean': outputs_pos['original'].mean().item(),
            'causal_score_mean': outputs_pos['causal'].mean().item(),
            'shortcut_score_mean': outputs_pos['shortcut'].mean().item(),
            'effect_score_mean': outputs_pos['effect'].mean().item(),
            'causal_mask_mean': outputs_pos['causal_mask'].mean().item(),
            'shortcut_mask_mean': outputs_pos['shortcut_mask'].mean().item(),
            'mask_sparsity': mask_sparsity,
            'mask_entropy': mask_entropy
        }
        return total_loss, score_pos, score_neg, stats

    def aggregate_stats(self, stats):
        if not stats:
            return {}
        keys = stats[0].keys()
        return {key: float(np.mean([item[key] for item in stats])) for key in keys}

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        epoch_stats = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()

            if self.use_causal_training:
                loss, score_pos, score_neg, stats = self.causal_training_step(data_pos, data_neg)
                epoch_stats.append(stats)
            else:
                score_pos = self.graph_classifier(data_pos)
                score_neg = self.graph_classifier(data_neg)
                loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))
            # print(score_pos, score_neg, loss)
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.view(-1).detach().cpu().tolist() + score_neg.view(-1).detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm, self.aggregate_stats(epoch_stats)

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm, stats = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')
            if stats:
                logging.info('Causal training stats: ' + str(stats))

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
