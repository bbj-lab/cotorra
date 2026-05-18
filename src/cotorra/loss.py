#!/usr/bin/env python3

"""
configurable loss functions for training;
note this code only runs when configured with `custom_loss: !!bool true`
"""

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
        self.logger = Logger()
        self.pool_cats = False
        self.kernels = {"cubic": lambda x, a: 0.5 + a*4*(x-0.5)**3 + (1-a) * (x - 0.5),
                        "atanh": lambda x, a: 0.5 + 1/(2*a)*t.atanh(a*(2*x - 1)),
                        "linear": lambda x, a: x
                        }
        
        self.loss_functions = {
            "mse": t.nn.MSELoss,
            "mae": t.nn.L1Loss,
            "smooth_mae": t.nn.SmoothL1Loss,
        }

        if "label_weighted_loss" in self.cfg:
            self.toi_flag = np.isin(
                self.vocab, self.cfg.label_weighted_loss.tokens_of_interest
            )
            self.weights = t.tensor(
                (self.cfg.label_weighted_loss.toi_weight - 1) * self.toi_flag + 1
            )

        if "quantile_token_loss" in self.cfg:
            if self.tkzr_cfg.cfg.fused:
                self.logger.warn(
                    "Quantile token loss is still experimental for fused tokenizers."
                )

            if "pool_categories" in self.cfg.quantile_token_loss:
                self.pool_cats = True

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
                t.tensor(self.qt_vals.astype(int)) / (self.tkzr_cfg.cfg.n_bins - 1)
            ).to(dtype=t.float32)
            self.label_to_q = t.full((len(self.vocab),), float("nan"))
            self.label_to_q[self.q_type] = self.qt_nums
            self.label_to_cat = t.full((len(self.vocab),), -1)
            self.label_to_cat[self.q_type] = t.tensor(
                np.unique(self.qt_cats, return_inverse=True)[1]
            )
            self.n_cats: int = self.label_to_cat.max().item() + 1
            if "order" in self.cfg.quantile_token_loss:
                self.order = self.cfg.quantile_token_loss.order
            else:
                self.order = "aggregate_first"
            if "loss_type" in self.cfg.quantile_token_loss:
                self.loss_function = self.loss_functions[self.cfg.quantile_token_loss.loss_type]
            else:
                self.loss_function = t.nn.MSELoss
            if "kernel" in self.cfg.quantile_token_loss:
                if "type" in self.cfg.quantile_token_loss.kernel:
                    self.kernel_type = self.cfg.quantile_token_loss.kernel.type
                else:
                    self.kernel_type = "linear"
                self.kernel_factor = 1.0
                if "factor" in self.cfg.quantile_token_loss.kernel:
                    self.kernel_factor = self.cfg.quantile_token_loss.kernel.factor


    def quantile_token_loss(self, outputs, labels, **kwargs):
        total_loss = t.zeros((), device=labels.device)
        shift_logits = outputs.get("logits")[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        total_tokens = shift_labels.numel()
        total_num_tokens = 0
        def aggregate_first(cat_logits, cat_labels, i):
            cat_preds = t.softmax(cat_logits, dim=-1) @ (
                self.label_to_q[self.label_to_cat == i]
            ).to(device=cat_logits.device)
            cat_true = self.label_to_q.to(device=cat_labels.device)[cat_labels]
            if self.cfg.quantile_token_loss.kernel.type in self.kernels:
                kernel = self.kernels[self.kernel_type]
                cat_preds = kernel(cat_preds, self.kernel_factor)
                cat_true = kernel(cat_true, self.kernel_factor)
            return self.loss_function()(cat_preds, cat_true)
        def loss_first(cat_logits, cat_labels, i):
            cat_true = self.label_to_q.to(device=cat_labels.device)[cat_labels]  # (N,)
            values = (self.label_to_q[self.label_to_cat == i]).to(device=cat_logits.device)  # (K,)
            if self.cfg.quantile_token_loss.kernel.type in self.kernels:
                kernel = self.kernels[self.kernel_type]
                values = kernel(values, self.kernel_factor)
                cat_true = kernel(cat_true, self.kernel_factor)
            N, K = cat_true.shape[0], values.shape[0]
            cat_true_full = cat_true.unsqueeze(-1).expand(N, K)       # (N, K)
            values_full = values.unsqueeze(0).expand(N, K)             # (N, K)
            losses = self.loss_function(reduction='none')(values_full, cat_true_full)
            return (t.softmax(cat_logits, dim=-1) * losses).sum(dim=-1).mean()
            
        for i in range(self.n_cats):
            mask = self.label_to_cat.to(device=labels.device)[shift_labels] == i
            n_i = mask.sum().item()
            if n_i == 0:
                continue
            cat_labels = shift_labels[mask]
            cat_logits = shift_logits[mask][:, self.label_to_cat == i]
            if self.order == 'loss_first':
                loss_del = loss_first(cat_logits, cat_labels, i)
            else: 
                loss_del = aggregate_first(cat_logits, cat_labels, i)
            total_loss += loss_del * n_i
            total_num_tokens += n_i
        return total_loss / max(total_tokens,1), total_num_tokens/total_tokens
    
    def x_ent_loss_cat(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")
        assert logits.shape[:2] == labels.shape, \
            f"logits {logits.shape} and labels {labels.shape} disagree on (batch, seq)"
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        total_loss = t.zeros((), device=labels.device)
        total_tokens = 0
        label_to_cat = self.label_to_cat.to(device=labels.device)
        token_cats = label_to_cat[shift_labels]

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_token_cats = token_cats.view(-1)
        log_probs = t.log_softmax(flat_logits, dim=-1) 

        for i in range(self.n_cats):
            mask = flat_token_cats == i
            n_i = mask.sum().item()
            if n_i == 0:
                continue
            cat_cols = (self.label_to_cat == i).to(labels.device)
            log_p_cat = t.logsumexp(log_probs[mask][:, cat_cols], dim=-1)
            loss_i = -log_p_cat.mean()
            total_loss += loss_i * n_i
            total_tokens += n_i

        non_numeric_mask = ~t.isin(flat_token_cats, t.arange(self.n_cats, device=labels.device))
        n_non = non_numeric_mask.sum().item()
        if n_non > 0:
            loss_non = t.nn.functional.nll_loss(log_probs[non_numeric_mask], flat_labels[non_numeric_mask])
            total_loss += loss_non * n_non
            total_tokens += n_non

        return total_loss / max(total_tokens, 1)
    
    #TODO Implement category pooled label_weighted_loss

    def label_weighted_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss(
            weight=self.weights.to(logits.device, dtype=logits.dtype)
        )(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def x_ent_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    def custom_loss(self, outputs, labels, **kwargs):
        loss = 0.0
        log = dict()
        if "label_weighted_loss" in self.cfg:
            label_weighted_loss = self.label_weighted_loss(outputs, labels)
            log |= {"label_weighted_loss": label_weighted_loss.item()}
            loss += label_weighted_loss
        else:
            x_ent_loss = self.x_ent_loss(outputs, labels) if (not self.pool_cats) else self.x_ent_loss_cat(outputs, labels)
            log |= {"x_ent_loss": x_ent_loss.item()}
            loss += x_ent_loss
        if "quantile_token_loss" in self.cfg:
            quantile_token_loss, frac_numeric = self.quantile_token_loss(outputs, labels)
            log |= {"quantile_token_loss": quantile_token_loss.item()/frac_numeric, "frac_numeric": frac_numeric}
            loss += self.cfg.quantile_token_loss.qt_weight * quantile_token_loss
            loss = loss/2
        if wandb.run is not None:
            log |= {"custom_loss": loss.item()}
            wandb.log(log)
        return loss


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loss(cfg=trainer.cfg, tkzr_cfg=trainer.tkzr_cfg)
