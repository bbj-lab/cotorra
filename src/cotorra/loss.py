#!/usr/bin/env python3

"""
configurable loss functions for training;
note this code only runs when configured with `custom_loss: !!bool true`
"""

import fnmatch
import re

import numpy as np
import torch as t

from cotorra.logger import Logger
from cotorra.model import HEADS, head_options


class Loss:
    def __init__(self, cfg=None, tkzr_cfg=None):

        self.cfg = cfg
        self.tkzr_cfg = tkzr_cfg
        self.vocab = np.array(
            sorted(self.tkzr_cfg.lookup, key=self.tkzr_cfg.lookup.get)
        )
        self.logger = Logger()
        self.heads = head_options(self.cfg, self.tkzr_cfg.lookup)

        # deprecated -- `Trainer` warns -- but still honored, so the configs that
        # use it train as they always did
        if "label_weighted_loss" in self.cfg:
            self.grokked_outcome_tokens = [
                x.item()
                for x in self.vocab
                if any(
                    fnmatch.fnmatch(x, p)
                    for p in self.cfg.get("label_weighted_loss", {}).get(
                        "tokens_of_interest", {}
                    )
                )
            ]
            self.logger.info(
                f"Processed expressions to generate {self.grokked_outcome_tokens=}"
            )
            self.toi_flag = np.isin(self.vocab, self.grokked_outcome_tokens)
            self.weights = t.tensor(
                (self.cfg.label_weighted_loss.toi_weight - 1) * self.toi_flag + 1
            )

        if "quantile_token_loss" in self.cfg:
            n_bins = self.tkzr_cfg.cfg.n_bins
            # cocoa fuses a bin onto its code (`LAB//sodium_Q3`) or, unfused, emits
            # a bare `Q3` after it; the bare bins then form a single category, the
            # code they belong to being the token before them
            bins = {
                i: (m["code"] or "", int(m["q"]))
                for tok, i in self.tkzr_cfg.lookup.items()
                if (m := re.fullmatch(r"(?:(?P<code>.*)_)?Q(?P<q>\d+)", tok))
                and int(m["q"]) < n_bins
            }
            codes = sorted({code for code, _ in bins.values()})
            self.n_cats: int = len(codes)
            if not self.n_cats:
                self.logger.warning(
                    "`quantile_token_loss` is configured but the vocabulary holds no "
                    "quantile tokens, so the term is always zero"
                )
            # row c holds the token id of each of category c's bins, -1 where the
            # vocabulary lacks one (tied breaks skip bins, as can winnowing)
            self.qt_table = t.full((self.n_cats, n_bins), -1)
            self.label_to_cat = t.full((len(self.vocab),), -1)
            # each bin stands for the midpoint of its slice of the quantile scale;
            # a non-quantile token gets 0 rather than nan, which would poison the
            # gradient through the `t.where` that masks it out
            self.qt_vals = (t.arange(n_bins, dtype=t.float32) + 0.5) / n_bins
            self.label_to_q = t.zeros(len(self.vocab))
            for i, (code, q) in bins.items():
                self.qt_table[codes.index(code), q] = i
                self.label_to_cat[i] = codes.index(code)
                self.label_to_q[i] = self.qt_vals[q]

        if "balanced_toi_loss" in self.cfg:
            patterns = (self.cfg.balanced_toi_loss or {}).get("tokens_of_interest", [])
            matched = [
                x.item()
                for x in self.vocab
                if any(fnmatch.fnmatch(x, p) for p in patterns)
            ]
            if not 0 < len(matched) < len(self.vocab):
                raise ValueError(
                    "`balanced_toi_loss.tokens_of_interest` has to match some of the "
                    f"vocabulary but not all of it; {list(patterns)!r} matched "
                    f"{len(matched)} of {len(self.vocab)} tokens"
                )
            self.logger.info(f"Processed expressions to generate {matched=}")
            self.balanced_toi_flag = t.zeros(len(self.vocab), dtype=t.bool)
            self.balanced_toi_flag[[self.tkzr_cfg.lookup[x] for x in matched]] = True

    def quantile_token_loss(self, outputs, labels, **kwargs):
        """
        squared error between the bin midpoint the model expects for a quantile
        token, from its softmax over that code's bins alone, and the midpoint of
        the bin that came, averaged over quantile tokens. Cross-entropy counts Q9
        for a true Q3 as no worse than Q4; this teaches the model the bins are
        ordered. It squares the error of the expected bin, not the expected
        squared error, which would reward piling mass onto one bin, so the true
        distribution still minimizes cross-entropy plus this term; and confined
        to a code's bins, it moves mass among them without making the code
        likelier
        """
        logits = outputs.get("logits")[:, :-1]
        # a tensor rather than a bare 0.0, since `custom_loss` calls `.detach()`
        if not self.n_cats:
            return t.zeros((), device=logits.device, dtype=t.float32)
        shift_labels = labels[:, 1:].to(logits.device)
        cat = self.label_to_cat.to(logits.device)[shift_labels]
        is_q = cat >= 0
        # gathered at every position, so no shape depends on the data and nothing
        # waits on the host; a position without a quantile token borrows category
        # 0's bins and drops out in the `t.where`
        ids = self.qt_table.to(logits.device)[cat.clamp(min=0)]
        bin_logits = (
            logits.gather(-1, ids.clamp(min=0))
            .to(dtype=t.float32)
            .masked_fill(ids < 0, -t.inf)
        )
        pred = bin_logits.softmax(dim=-1) @ self.qt_vals.to(logits.device)
        true = self.label_to_q.to(logits.device)[shift_labels]
        sq_err = t.where(is_q, (pred - true) ** 2, 0.0)
        return sq_err.sum() / is_q.sum().clamp(min=1)

    def label_weighted_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss(
            weight=self.weights.to(logits.device, dtype=logits.dtype)
        )(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(
            dtype=t.float32
        )

    def balanced_toi_loss(self, outputs, labels, **kwargs):
        """
        binary cross-entropy on whether the next token is a token of interest,
        scored with the total probability the softmax puts on those tokens. A
        proper scoring rule -- minimized by the true probability, as
        cross-entropy itself is -- so adding it makes getting those tokens right
        count for more without making them likelier, which the deprecated
        `label_weighted_loss` (cross-entropy weighted by its target) did. Both
        sides come straight from the logits, so neither degrades to
        `log(1 - p)` as the other nears certainty
        """
        logits = outputs.get("logits")[:, :-1].to(dtype=t.float32)
        flag = self.balanced_toi_flag.to(logits.device)
        is_toi = flag[labels[:, 1:].to(logits.device)]
        lse_toi = logits[..., flag].logsumexp(dim=-1)
        lse_rest = logits.masked_fill(flag, -t.inf).logsumexp(dim=-1)
        lse_all = t.logaddexp(lse_toi, lse_rest)
        return -t.where(is_toi, lse_toi - lse_all, lse_rest - lse_all).mean()

    def x_ent_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).to(dtype=t.float32)

    def head_loss(self, outputs, name: str):
        """a secondary head's term, which the model computes itself -- it owns the
        head -- and hands back alongside the logits. It gets folded into the
        objective here rather than in the model because `TrainerWithCustomLoss`
        pops `labels` before the forward call, so the model's own weighted sum
        is never formed"""
        if (loss := outputs.get(f"{name}_loss")) is None:
            raise ValueError(
                f"`{name}_objective` is configured but the model returned no "
                f"`{name}_loss`: the term needs a model carrying a `{name}` head "
                "(what `Trainer.model_init` builds when the block is present) fed "
                f"a batch carrying `{HEADS[name].target}`"
            )
        return loss.to(dtype=t.float32)

    def custom_loss(self, outputs, labels, terms: dict | None = None, **kwargs):
        """
        the configured objective: cross-entropy plus each weighted term its blocks
        ask for. Given a dict `terms`, it also records each term there, unweighted
        and detached, under the name it gets logged by; `TrainerWithCustomLoss`
        averages them for its logs, left on the device so that no batch waits on
        the host
        """
        loss = 0.0
        terms = dict() if terms is None else terms
        if "label_weighted_loss" in self.cfg:
            label_weighted_loss = self.label_weighted_loss(outputs, labels)
            terms["label_weighted_loss"] = label_weighted_loss.detach()
            loss += label_weighted_loss
        else:
            x_ent_loss = self.x_ent_loss(outputs, labels)
            terms["x_ent_loss"] = x_ent_loss.detach()
            loss += x_ent_loss
        if "balanced_toi_loss" in self.cfg:
            balanced_toi_loss = self.balanced_toi_loss(outputs, labels)
            terms["balanced_toi_loss"] = balanced_toi_loss.detach()
            bce_weight = self.cfg.balanced_toi_loss.get("bce_weight", 1.0)
            loss += bce_weight * balanced_toi_loss
        if "quantile_token_loss" in self.cfg:
            quantile_token_loss = self.quantile_token_loss(outputs, labels)
            terms["quantile_token_loss"] = quantile_token_loss.detach()
            loss += self.cfg.quantile_token_loss.qt_weight * quantile_token_loss
        for name, options in self.heads.items():
            head_loss = self.head_loss(outputs, name)
            terms[f"{name}_loss"] = head_loss.detach()
            loss += options["weight"] * head_loss
        return loss


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loss(cfg=trainer.cfg, tkzr_cfg=trainer.tkzr_cfg)
    # breakpoint()
