#!/usr/bin/env python3

"""
configurable loss functions for training;
note this code only runs when configured with `custom_loss: !!bool true`
"""

import fnmatch

import numpy as np
import torch as t

import wandb
from cotorra.logger import Logger


class Loss:
    def __init__(self, cfg=None, tkzr_cfg=None):

        self.cfg = cfg
        self.tkzr_cfg = tkzr_cfg
        self.vocab = np.array(
            sorted(self.tkzr_cfg.lookup, key=self.tkzr_cfg.lookup.get)
        )
        self.grokked_outcome_tokens = [
            x.item()
            for x in self.vocab
            if any(
                fnmatch.fnmatch(x, p)
                for p in self.cfg.label_weighted_loss.tokens_of_interest
            )
        ]
        self.logger = Logger()
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )

        if "label_weighted_loss" in self.cfg:
            self.toi_flag = np.isin(self.vocab, self.grokked_outcome_tokens)
            self.weights = t.tensor(
                (self.cfg.label_weighted_loss.toi_weight - 1) * self.toi_flag + 1
            )

        if "quantile_token_loss" in self.cfg:
            self.q_type = np.array(
                [
                    v.endswith(tuple(f"Q{i}" for i in range(self.tkzr_cfg.cfg.n_bins)))
                    for v in self.vocab
                ]
            )
            self.qt_cats, self.qt_vals = map(
                np.array,
                zip(*np.char.rsplit(self.vocab[self.q_type], sep="Q", maxsplit=1)),
            )
            self.qt_nums = (
                t.tensor(self.qt_vals.astype(int) + 0.5) / self.tkzr_cfg.cfg.n_bins
            ).to(dtype=t.float32)
            self.label_to_q = t.full((len(self.vocab),), float("nan"))
            self.label_to_q[self.q_type] = self.qt_nums
            self.label_to_cat = t.full((len(self.vocab),), -1)
            self.label_to_cat[self.q_type] = t.tensor(
                np.unique(self.qt_cats, return_inverse=True)[1]
            )
            self.n_cats: int = self.label_to_cat.max().item() + 1

    def quantile_token_loss(self, outputs, labels, **kwargs):
        loss = 0.0
        shift_logits = outputs.get("logits")[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        for i in range(self.n_cats):
            mask = self.label_to_cat.to(device=labels.device)[shift_labels] == i
            if not mask.any():
                continue
            cat_labels = shift_labels[mask]
            cat_logits = shift_logits[mask][:, self.label_to_cat == i].to(
                dtype=t.float32
            )
            cat_preds = t.softmax(cat_logits, dim=-1) @ (
                self.label_to_q[self.label_to_cat == i]
            ).to(device=cat_logits.device, dtype=t.float32)
            cat_true = self.label_to_q.to(device=cat_labels.device)[cat_labels]
            loss += t.nn.MSELoss(reduction='sum')(cat_preds, cat_true).to(dtype=t.float32)
        return loss

    def quantile_token_loss(self, outputs, labels, **kwargs):
            loss = 0.0
            shift_logits = outputs.get("logits")[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            for i in range(self.n_cats):
                mask = self.label_to_cat.to(device=labels.device)[shift_labels] == i
                if not mask.any():
                    continue
                cat_labels = shift_labels[mask]
                cat_logits = shift_logits[mask][:, self.label_to_cat == i].to(
                    dtype=t.float32
                )
                cat_true = self.label_to_q.to(device=cat_labels.device)[cat_labels] # shape: N
                q = (self.label_to_q[self.label_to_cat == i]
                    ).to(device=cat_logits.device, dtype=t.float32).view(-1)        # shape: num_cats
                cat_errors = (cat_true.unsqueeze(-1) - q).abs()                     # shape: N x num_cats                 
                cat_probs = t.softmax(cat_logits, dim=-1)                           # shape: N x num_cats
                loss += (cat_errors * cat_probs).sum(dim=-1)                        # shape: N
            return loss

    def label_weighted_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss(
            weight=self.weights.to(logits.device, dtype=logits.dtype)
        )(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(
            dtype=t.float32
        )

    def x_ent_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss(reduction='sum')(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).to(dtype=t.float32)

    def custom_loss(self, outputs, labels, **kwargs):
        loss = 0.0
        log = dict()
        if "label_weighted_loss" in self.cfg:
            label_weighted_loss = self.label_weighted_loss(outputs, labels)
            log |= {"label_weighted_loss": label_weighted_loss.item()}
            loss += label_weighted_loss
        else:
            x_ent_loss = self.x_ent_loss(outputs, labels)
            log |= {"x_ent_loss": x_ent_loss.item()}
            loss += x_ent_loss
        if "quantile_token_loss" in self.cfg:
            quantile_token_loss = self.quantile_token_loss(outputs, labels)
            log |= {"quantile_token_loss": quantile_token_loss.item()}
            loss += self.cfg.quantile_token_loss.qt_weight * quantile_token_loss
        if "was_token_loss" in self.cfg:
            quantile_token_loss = self.was_token_loss(outputs, labels)
            log |= {"was_token_loss": quantile_token_loss.item()}
            loss += self.cfg.was_tokenLoss.qt_weight * quantile_token_loss
        if wandb.run is not None:
            log |= {"custom_loss": loss.item()}
            wandb.log(log)
        return loss


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loss(cfg=trainer.cfg, tkzr_cfg=trainer.tkzr_cfg)
    # breakpoint()
