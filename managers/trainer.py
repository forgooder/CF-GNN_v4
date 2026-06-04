import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import contextmanager

from sklearn import metrics
from utils.score_utils import SCORE_MODE_CHOICES, select_score


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0
        self.current_epoch = 0
        self.use_causal_training = getattr(params, 'use_causal_training', False)
        if self.use_causal_training and hasattr(self.graph_classifier, '_get_causal_mask_generator'):
            self.graph_classifier._get_causal_mask_generator(params.device)
        self.relation_budget = self.load_relation_budget()

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

    def load_relation_budget(self):
        relation_budget_path = getattr(self.params, 'relation_budget_path', '')
        if not relation_budget_path:
            return {'causal': {}, 'shortcut': {}}

        with open(relation_budget_path) as f:
            raw_budget = json.load(f)

        relation2id = getattr(self.graph_classifier, 'relation2id', {})

        def normalize_budget(section):
            budget = {}
            for key, value in raw_budget.get(section, {}).items():
                if key in relation2id:
                    rel_id = relation2id[key]
                else:
                    rel_id = int(key)
                budget[int(rel_id)] = float(value)
            return budget

        return {
            'causal': normalize_budget('causal'),
            'shortcut': normalize_budget('shortcut')
        }

    def ranking_loss(self, score_pos, score_neg):
        score_pos = score_pos.view(-1)
        score_neg = score_neg.view(len(score_pos), -1).mean(dim=1)
        target = torch.ones_like(score_pos, device=self.params.device)
        return self.criterion(score_pos, score_neg, target)

    def mask_targets(self, target_rel_labels, default_target, budget_map):
        targets = torch.full_like(target_rel_labels.float(), float(default_target), device=self.params.device)
        for rel_id, target in budget_map.items():
            targets = torch.where(
                target_rel_labels == int(rel_id),
                torch.full_like(targets, float(target)),
                targets
            )
        return targets

    def mask_regularization(self, outputs_pos, outputs_neg):
        mask_sparsity_weight = getattr(self.params, 'mask_sparsity_weight', 0.0)
        mask_entropy_weight = getattr(self.params, 'mask_entropy_weight', 0.0)
        mask_entropy_floor_weight = getattr(self.params, 'mask_entropy_floor_weight', 0.0)
        mask_entropy_floor = getattr(self.params, 'mask_entropy_floor', 0.1)
        mask_logit_l2_weight = getattr(self.params, 'mask_logit_l2_weight', 0.0)
        causal_entropy_floor_weight = getattr(self.params, 'causal_mask_entropy_floor_weight', 0.0)
        causal_logit_l2_weight = getattr(self.params, 'causal_mask_logit_l2_weight', 0.0)
        mask_budget_weight = getattr(self.params, 'mask_budget_weight', 0.0)
        mask_overlap_weight = getattr(self.params, 'mask_overlap_weight', 0.0)

        causal_masks = torch.cat([
            outputs_pos['causal_raw_mask'].view(-1),
            outputs_neg['causal_raw_mask'].view(-1)
        ])
        shortcut_masks = torch.cat([
            outputs_pos['shortcut_raw_mask'].view(-1),
            outputs_neg['shortcut_raw_mask'].view(-1)
        ])
        causal_logits = torch.cat([
            outputs_pos['causal_logits'].view(-1),
            outputs_neg['causal_logits'].view(-1)
        ])
        shortcut_logits = torch.cat([
            outputs_pos['shortcut_logits'].view(-1),
            outputs_neg['shortcut_logits'].view(-1)
        ])
        target_rel_labels = torch.cat([
            outputs_pos['target_rel_labels'].view(-1),
            outputs_neg['target_rel_labels'].view(-1)
        ]).to(device=self.params.device)

        causal_targets = self.mask_targets(
            target_rel_labels,
            getattr(self.params, 'causal_mask_target', 0.5),
            self.relation_budget['causal']
        )
        shortcut_targets = self.mask_targets(
            target_rel_labels,
            getattr(self.params, 'shortcut_mask_target', 0.5),
            self.relation_budget['shortcut']
        )

        causal_sparsity = causal_masks.mean()
        shortcut_sparsity = shortcut_masks.mean()
        eps = 1e-8
        causal_entropy = -(causal_masks * torch.log(causal_masks + eps) + (1 - causal_masks) * torch.log(1 - causal_masks + eps)).mean()
        shortcut_entropy = -(shortcut_masks * torch.log(shortcut_masks + eps) + (1 - shortcut_masks) * torch.log(1 - shortcut_masks + eps)).mean()
        budget_loss = (
            F.mse_loss(causal_masks, causal_targets)
            + F.mse_loss(shortcut_masks, shortcut_targets)
        )
        overlap_loss = (causal_masks * shortcut_masks).mean()
        entropy_floor = torch.tensor(float(mask_entropy_floor), device=self.params.device)
        entropy_floor_loss = (
            torch.relu(entropy_floor - causal_entropy).pow(2)
            + torch.relu(entropy_floor - shortcut_entropy).pow(2)
        )
        causal_entropy_floor_loss = torch.relu(entropy_floor - causal_entropy).pow(2)
        logit_l2_loss = causal_logits.pow(2).mean() + shortcut_logits.pow(2).mean()
        causal_logit_l2_loss = causal_logits.pow(2).mean()
        legacy_reg = mask_sparsity_weight * causal_sparsity + mask_entropy_weight * causal_entropy
        reg_loss = (
            legacy_reg
            + mask_budget_weight * budget_loss
            + mask_overlap_weight * overlap_loss
            + mask_entropy_floor_weight * entropy_floor_loss
            + mask_logit_l2_weight * logit_l2_loss
            + causal_entropy_floor_weight * causal_entropy_floor_loss
            + causal_logit_l2_weight * causal_logit_l2_loss
        )

        stats = {
            'mask_reg_loss': reg_loss.item(),
            'causal_mask_raw_mean': causal_sparsity.item(),
            'shortcut_mask_raw_mean': shortcut_sparsity.item(),
            'causal_mask_entropy': causal_entropy.item(),
            'shortcut_mask_entropy': shortcut_entropy.item(),
            'mask_budget_loss': budget_loss.item(),
            'mask_overlap_loss': overlap_loss.item(),
            'mask_entropy_floor_loss': entropy_floor_loss.item(),
            'mask_logit_l2_loss': logit_l2_loss.item(),
            'causal_mask_entropy_floor_loss': causal_entropy_floor_loss.item(),
            'causal_mask_logit_l2_loss': causal_logit_l2_loss.item()
        }
        return reg_loss, stats

    def shortcut_penalty(self, outputs_pos, outputs_neg):
        shortcut_penalty_weight = getattr(self.params, 'shortcut_penalty_weight', 0.0)
        if shortcut_penalty_weight == 0:
            return torch.tensor(0.0, device=self.params.device)

        shortcut_pos = outputs_pos['shortcut'].view(-1)
        shortcut_neg = outputs_neg['shortcut'].view(len(shortcut_pos), -1).mean(dim=1)
        return shortcut_penalty_weight * torch.abs(shortcut_pos - shortcut_neg).mean()

    def current_effect_loss_weight(self):
        target_weight = getattr(self.params, 'effect_loss_weight', 1.0)
        warmup_epochs = getattr(self.params, 'effect_loss_warmup_epochs', 0)
        ramp_epochs = max(0, getattr(self.params, 'effect_loss_ramp_epochs', 0))
        if self.current_epoch <= warmup_epochs:
            return 0.0, 0.0
        if ramp_epochs == 0:
            return target_weight, 1.0
        ramp_progress = min(1.0, float(self.current_epoch - warmup_epochs) / float(ramp_epochs))
        return target_weight * ramp_progress, ramp_progress

    def effect_scores_for_loss(self, outputs_pos, outputs_neg):
        mode = getattr(self.params, 'effect_gradient_mode', 'full')
        causal_pos = outputs_pos['causal']
        causal_neg = outputs_neg['causal']
        shortcut_pos = outputs_pos['shortcut']
        shortcut_neg = outputs_neg['shortcut']

        if mode in ['detach_causal', 'detach_both']:
            causal_pos = causal_pos.detach()
            causal_neg = causal_neg.detach()
        if mode in ['detach_shortcut', 'detach_both']:
            shortcut_pos = shortcut_pos.detach()
            shortcut_neg = shortcut_neg.detach()

        effect_pos = causal_pos - shortcut_pos
        effect_neg = causal_neg - shortcut_neg
        effect_score_clamp = getattr(self.params, 'effect_score_clamp', 0.0)
        if effect_score_clamp and effect_score_clamp > 0:
            effect_pos = torch.clamp(effect_pos, -effect_score_clamp, effect_score_clamp)
            effect_neg = torch.clamp(effect_neg, -effect_score_clamp, effect_score_clamp)

        return effect_pos, effect_neg

    @contextmanager
    def freeze_non_mask_parameters(self):
        changed_params = []
        for name, param in self.graph_classifier.named_parameters():
            if name.startswith('causal_mask_generator.'):
                continue
            if param.requires_grad:
                param.requires_grad_(False)
                changed_params.append(param)
        try:
            yield
        finally:
            for param in changed_params:
                param.requires_grad_(True)

    def causal_training_outputs(self, data):
        aux_gradient_mode = getattr(self.params, 'masked_aux_gradient_mode', 'full')
        if aux_gradient_mode == 'full':
            return self.graph_classifier(data, mode='all')
        if aux_gradient_mode != 'mask_only':
            raise ValueError(f"Unknown masked_aux_gradient_mode: {aux_gradient_mode}")

        original_score = self.graph_classifier(data, mode='original')
        with self.freeze_non_mask_parameters():
            outputs = self.graph_classifier(data, mode='all')
        outputs['original'] = original_score
        return outputs

    def causal_training_step(self, data_pos, data_neg):
        outputs_pos = self.causal_training_outputs(data_pos)
        outputs_neg = self.causal_training_outputs(data_neg)

        original_loss = self.ranking_loss(outputs_pos['original'], outputs_neg['original'])
        causal_loss = self.ranking_loss(outputs_pos['causal'], outputs_neg['causal'])
        effect_pos, effect_neg = self.effect_scores_for_loss(outputs_pos, outputs_neg)
        effect_loss = self.ranking_loss(effect_pos, effect_neg)
        mask_reg_loss, mask_stats = self.mask_regularization(outputs_pos, outputs_neg)
        shortcut_loss = self.shortcut_penalty(outputs_pos, outputs_neg)
        effect_loss_weight, effect_loss_ramp_factor = self.current_effect_loss_weight()

        total_loss = (
            original_loss
            + getattr(self.params, 'causal_loss_weight', 1.0) * causal_loss
            + effect_loss_weight * effect_loss
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
            'effect_loss_weight': effect_loss_weight,
            'effect_loss_ramp_factor': effect_loss_ramp_factor,
            'effect_score_clamp': getattr(self.params, 'effect_score_clamp', 0.0),
            'masked_aux_gradient_mode': 0.0 if getattr(self.params, 'masked_aux_gradient_mode', 'full') == 'full' else 1.0,
            'mask_reg_loss': mask_reg_loss.item(),
            'shortcut_penalty': shortcut_loss.item(),
            'total_loss': total_loss.item(),
            'original_score_mean': outputs_pos['original'].mean().item(),
            'causal_score_mean': outputs_pos['causal'].mean().item(),
            'shortcut_score_mean': outputs_pos['shortcut'].mean().item(),
            'effect_score_mean': outputs_pos['effect'].mean().item(),
            'causal_mask_mean': outputs_pos['causal_mask'].mean().item(),
            'shortcut_mask_mean': outputs_pos['shortcut_mask'].mean().item(),
            'causal_mask_raw_mean': outputs_pos['causal_raw_mask'].mean().item(),
            'shortcut_mask_raw_mean': outputs_pos['shortcut_raw_mask'].mean().item(),
            'mask_overlap': outputs_pos['mask_stats']['overlap'].item()
        }
        stats.update(mask_stats)
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
                if getattr(self.params, 'log_all_score_modes_validation', False):
                    all_mode_result = self.valid_evaluator.eval_score_modes(SCORE_MODE_CHOICES)
                    logging.info('\nValidation all score modes:' + str(all_mode_result))

                selection_metric = getattr(self.params, 'selection_metric', 'auc')
                current_metric = result[selection_metric]
                if current_metric >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = current_metric
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = current_metric

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm, self.aggregate_stats(epoch_stats)

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            self.current_epoch = epoch
            time_start = time.time()
            loss, auc, auc_pr, weight_norm, stats = self.train_epoch()
            time_elapsed = time.time() - time_start
            selection_metric = getattr(self.params, 'selection_metric', 'auc')
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation {selection_metric}: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')
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
        logging.info('Better model found w.r.t validation metric. Saved it!')
