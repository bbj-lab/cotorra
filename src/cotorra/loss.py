#!/usr/bin/env python3

"""
configurable loss functions for training;
note this code only runs when configured with `custom_loss: !!bool true`
"""

import fnmatch
import math

import numpy as np
import torch as t
import torch.nn.functional as F
import wandb

from cotorra import crps
from cotorra.logger import Logger


class Loss:
    def __init__(self, cfg=None, tkzr_cfg=None, basis_vocab=None):
        self.cfg = cfg
        self.tkzr_cfg = tkzr_cfg
        self.vocab = np.array(
            sorted(self.tkzr_cfg.lookup, key=self.tkzr_cfg.lookup.get)
        )

        # basis_vocab (see cotorra.basis_blended.build_basis_vocab), when
        # provided, is the *same* collapsed-vocab object the model was built
        # from -- category/basis-id assignment can never disagree between the
        # embedding blend and the loss target because both read from this one
        # source of truth (threaded through by Trainer.__init__).
        self.basis_vocab = basis_vocab
        if self.basis_vocab is not None:
            self._category_base_id_t = t.tensor(
                self.basis_vocab["category_base_id"], dtype=t.long
            )
        if "basis_blended_tokens" in self.cfg and self.cfg.basis_blended_tokens.get(
            "crps_loss", False
        ):
            assert self.cfg.basis_blended_tokens.get("mixture_nll_loss", False), (
                "basis_blended_tokens.crps_loss: true requires "
                "basis_blended_tokens.mixture_nll_loss: true -- crps_loss "
                "replaces the NLL term's contribution inside that loss, it "
                "isn't a standalone alternative (see basis_blended_token_loss)"
            )
            n_quad = int(
                self.cfg.basis_blended_tokens.get(
                    "crps_quad_points", crps.DEFAULT_OUTER_POINTS
                )
            )
            self._crps_gl_out = crps.gl_rule(n_quad, dtype=t.float32)
            self._crps_de_in = crps.tanh_sinh_rule(dtype=t.float32)
        self.grokked_outcome_tokens = [
            x.item()
            for x in self.vocab
            if any(
                fnmatch.fnmatch(x, p)
                for p in self.cfg.get("label_weighted_loss", {}).get(
                    "tokens_of_interest", []
                )
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
            loss += t.nn.MSELoss(reduction="sum")(cat_preds, cat_true).to(
                dtype=t.float32
            )
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
            cat_true = self.label_to_q.to(device=cat_labels.device)[
                cat_labels
            ]  # shape: N
            q = (
                (self.label_to_q[self.label_to_cat == i])
                .to(device=cat_logits.device, dtype=t.float32)
                .view(-1)
            )  # shape: num_cats
            cat_errors = (cat_true.unsqueeze(-1) - q).abs()  # shape: N x num_cats
            cat_probs = t.softmax(cat_logits, dim=-1)  # shape: N x num_cats
            loss += (cat_errors * cat_probs).sum(dim=-1)  # shape: N
        return loss

    def basis_blended_token_loss(
        self, outputs, labels, category_ids=None, ranks=None, **kwargs
    ):
        """
        replaces one-hot next-token-prediction CE with a soft target for
        numeric (category, rank) positions: P(other categories / non-numeric)
        = 0, P(this category's i-th basis token) = w_i(rank) -- see "Loss
        function" in fuzzy_token_planning.md. Non-numeric positions keep
        ordinary one-hot CE. Sum-reduced (not mean), matching the rest of
        this module, so batches with more/fewer numeric tokens contribute
        proportionally more/less gradient signal.

        `outputs["mixture_weights"]` is read directly from the model's own
        forward pass (see BasisBlendedCausalLM) rather than recomputed here
        from log_alpha/log_beta, so the loss target and the input embedding
        blend can never disagree, and this keeps working unmodified under
        Opacus's GradSampleModule wrapping (which would otherwise require
        unwrapping the model just to read its parameters).

        `basis_blended_tokens.kl_loss: true` switches the numeric-position
        term from CE(w, p) to KL(w || p), computed directly via torch's own
        F.kl_div rather than the CE(w, p) - H(w) identity used originally
        -- that indirect form was producing NaNs in training (wandb logs
        showed frequent nan entries), traced to t.xlogy(w, w): xlogy only
        special-cases x == 0 *exactly*, and a softmax-derived w can land on
        a tiny *negative* floating-point residual at a nominal zero (e.g.
        -1e-8), where log(x) is NaN and xlogy doesn't catch it.

        Switching to F.kl_div alone wasn't sufficient, though (still saw
        nan grad_norm, not loss, after that first fix): its *forward* value
        at target == 0 is correctly 0, but its *backward* pass is NaN
        there whenever the target requires grad -- which shift_w does here,
        unlike F.kl_div's typical use case of a frozen reference
        distribution. This isn't a library bug so much as a real
        singularity: d/dw[w*log(w)] = log(w) + 1 diverges to -inf as
        w -> 0+, so *any* correct implementation of this term's gradient
        blows up at an exact zero. clamp_min(1e-8) (rather than
        clamp_min(0)) keeps log(w) finite -- a floor small enough not to
        perceptibly change the loss value, but large enough to keep the
        gradient finite too.

        KL(w || p) is exactly 0 for a one-hot target, so this is a no-op
        for non-numeric positions -- CE and KL coincide there, hence the
        nonnum branch above is unaffected either way. For numeric
        positions, though, w is a function of the model's own (trainable,
        when train_beta_params/train_importance_scale are on)
        log_alpha/log_beta/log_importance, not a fixed label -- so the
        -H(w) component implicit in KL carries real gradient into those
        parameters, rewarding mixtures that stay spread across basis
        elements (high entropy) over ones that collapse onto a single basis
        element (low entropy), on top of whatever CE alone would do.

        `basis_blended_tokens.mixture_nll_loss: true` (mutually exclusive
        with kl_loss) replaces both CE(w, p) and KL(w || p) with a
        different objective entirely -- see fuzzy_token_planning.md point
        16. Both alternatives above are a *classification*-style match
        between two discrete k-way distributions: a fixed target w derived
        from the true rank, and the model's own prediction p, with the
        true rank itself never evaluated against anything. mixture_nll_loss
        instead scores the true rank directly: treat the model's own
        (renormalized) predicted distribution over the k basis tokens as
        *mixture weights* ŵ_i, and take the negative log-likelihood of the
        observed rank r under the resulting Beta mixture, sum_i ŵ_i *
        Beta_pdf(r; alpha_i, beta_i) -- a mixture-density-network-style
        loss. Decomposes additively (in log-space) into:
          -log P(next token is one of category c's k basis tokens)   [Z_c]
          -log( sum_i ŵ_i * Beta_pdf_i(r) )                          [NLL]
        The first term is exactly log_probs_numeric's own normalizer
        (t.logsumexp over the k gathered log-probs); the second reuses
        beta_log_pdf (the *unscaled* density, no log_importance -- see
        BasisBlendedCausalLMOutput) so "which Beta each component is" is
        never recomputed independently of what forward() actually used.
        Both terms are computed via t.logsumexp end to end, never
        exponentiating and summing in linear space first: a Beta density
        can be large near its mode and near-zero elsewhere, so a naive
        sum-then-log risks underflowing every component to exactly 0 and
        hitting log(0) = -inf, the same class of bug already hit and fixed
        for kl_loss above. No collapse-guarding (e.g. a floor on
        alpha/beta) is added beyond the existing clamp(1e-3, 1e4) in
        _mixture_weights -- watch for it empirically before adding more.

        `basis_blended_tokens.mixture_nll_alpha` (default 1.0, the original
        unweighted formulation) scales the [NLL] term only:
        L = -log(Z_c) - alpha * log(sum_i ŵ_i * Beta_pdf_i(r)). alpha > 1
        pushes more of the loss's gradient weight onto matching the
        observed rank's density (the "which value" task) relative to
        picking category c at all (the "which category" task); alpha < 1
        the reverse. See nll_alpha_sweep.md for the motivating question
        (does rebalancing these two sub-tasks change downstream rep-based
        AUC) and results.

        `basis_blended_tokens.crps_loss: true` (requires mixture_nll_loss:
        true; errors in __init__ otherwise) replaces the [NLL] term with
        an equal-weighted blend of itself and CRPS -- the continuous
        ranked probability score of the mixture's own predicted CDF
        (built from ŵ_i and the Beta(a_i,b_i) shape params, via
        crps.mixture_crps) against the true rank r, treated as a point
        observation (CDF = a unit step at r). Unlike the density-based
        NLL, CRPS scores the *entire* predicted distribution's shape
        against where r actually falls, not just the density at r itself:
        L = -log(Z_c) - 0.5*alpha*log_likelihood + 0.5*crps_weight*CRPS.
        `alpha` keeps scaling the NLL half exactly as before;
        `crps_weight` (default 1.0) is a separate, independent knob for
        the CRPS half -- at both defaults this is the literal 1/2 CRPS +
        1/2 NLL blend. `crps_quad_points` (default
        crps.DEFAULT_OUTER_POINTS) tunes the outer quadrature's
        accuracy/cost; see crps.py for the full numerical derivation
        (nested Gauss-Legendre/tanh-sinh quadrature -- torch has no
        incomplete beta function to get the mixture CDF from directly).
        """
        shift_logits = outputs.get("logits")[:, :-1].contiguous().to(dtype=t.float32)
        shift_labels = labels[:, 1:].contiguous()
        shift_cat = category_ids[:, 1:].contiguous()
        log_probs = t.log_softmax(shift_logits, dim=-1)
        numeric = shift_cat >= 0

        loss = t.zeros((), dtype=t.float32, device=shift_logits.device)

        nonnum = ~numeric
        if nonnum.any():
            nn_labels = shift_labels[nonnum].unsqueeze(-1)
            loss = loss - log_probs[nonnum].gather(-1, nn_labels).squeeze(-1).sum()

        if numeric.any():
            mixture_weights = outputs.get("mixture_weights")
            assert mixture_weights is not None, (
                "basis_blended_tokens requires the model to return mixture_weights"
            )
            shift_w = mixture_weights[:, 1:].contiguous()[numeric].to(dtype=t.float32)
            cat_num = shift_cat[numeric]
            k = shift_w.shape[-1]
            basis_ids = self._category_base_id_t.to(cat_num.device)[cat_num].unsqueeze(
                -1
            ) + t.arange(k, device=cat_num.device)
            log_probs_numeric = log_probs[numeric].gather(-1, basis_ids)
            if self.cfg.basis_blended_tokens.get("kl_loss", False):
                # clamp_min(0) alone isn't enough: d/dw[w*log(w)] = log(w)+1
                # diverges to -inf as w->0+, a genuine mathematical
                # singularity, not a numerical artifact -- so F.kl_div's
                # *forward* value at target==0 is fine (0, verified), but
                # its *backward* pass is NaN there whenever target requires
                # grad, which shift_w does here (see basis_blended_token_loss
                # docstring). A small positive floor keeps log(w) finite
                # without perceptibly changing the loss value.
                loss = loss + F.kl_div(
                    log_probs_numeric,
                    shift_w.clamp_min(1e-8),
                    reduction="sum",
                    log_target=False,
                )
            elif self.cfg.basis_blended_tokens.get("mixture_nll_loss", False):
                beta_log_pdf = outputs.get("beta_log_pdf")
                assert beta_log_pdf is not None, (
                    "mixture_nll_loss requires the model to return beta_log_pdf"
                )
                shift_log_pdf = (
                    beta_log_pdf[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                )
                # term 1: -log P(next token is one of category c's k basis
                # tokens) -- log_probs_numeric's own full-vocab normalizer.
                log_z_c = t.logsumexp(log_probs_numeric, dim=-1)
                # term 2: -log likelihood of the *true* rank under a Beta
                # mixture whose weights are the model's own predicted
                # (renormalized-over-k) distribution, not the rank-derived
                # target used by CE/KL above. Entirely log-space -- see
                # docstring for why a linear-space sum-then-log isn't safe.
                log_w_hat = log_probs_numeric - log_z_c.unsqueeze(-1)
                log_likelihood = t.logsumexp(log_w_hat + shift_log_pdf, dim=-1)
                # mixture_nll_alpha (default 1.0, i.e. unweighted -- the
                # original formulation) scales term 2 only: L = -log_z_c -
                # alpha*log_likelihood. >1 pushes more gradient weight onto
                # matching the observed rank's density relative to getting
                # the category right; <1 the reverse. See
                # nll_alpha_sweep.md for the motivation and results of
                # sweeping this.
                alpha = float(
                    self.cfg.basis_blended_tokens.get("mixture_nll_alpha", 1.0)
                )
                if self.cfg.basis_blended_tokens.get("crps_loss", False):
                    # basis_blended_tokens.crps_loss: true replaces this
                    # term's NLL half with an equal-weighted blend of CRPS
                    # and NLL -- see crps.py's module docstring for the
                    # full derivation (nested Gauss-Legendre/tanh-sinh
                    # quadrature against the Beta mixture's own predicted
                    # CDF) and crps_quad_points to tune its accuracy/cost.
                    # crps_weight (default 1.0) scales the CRPS half
                    # independently of alpha, which keeps scaling the NLL
                    # half exactly as it always has.
                    beta_a = outputs.get("beta_a")
                    beta_b = outputs.get("beta_b")
                    assert beta_a is not None and beta_b is not None, (
                        "crps_loss requires the model to return beta_a/beta_b"
                    )
                    shift_a = (
                        beta_a[:, 1:]
                        .contiguous()[numeric]
                        .to(dtype=t.float32)
                        .clamp(*crps.SHAPE_PARAM_CLAMP)
                    )
                    shift_b = (
                        beta_b[:, 1:]
                        .contiguous()[numeric]
                        .to(dtype=t.float32)
                        .clamp(*crps.SHAPE_PARAM_CLAMP)
                    )
                    shift_ranks = ranks[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                    w_hat = log_w_hat.exp()
                    gl_out = tuple(x.to(w_hat.device) for x in self._crps_gl_out)
                    de_in = tuple(x.to(w_hat.device) for x in self._crps_de_in)
                    sample_crps = crps.mixture_crps(
                        shift_ranks, w_hat, shift_a, shift_b, gl_out, de_in
                    )
                    crps_weight = float(
                        self.cfg.basis_blended_tokens.get("crps_weight", 1.0)
                    )
                    loss = (
                        loss
                        - log_z_c.sum()
                        - 0.5 * alpha * log_likelihood.sum()
                        + 0.5 * crps_weight * sample_crps.sum()
                    )
                else:
                    loss = loss - log_z_c.sum() - alpha * log_likelihood.sum()
            else:
                loss = loss - (shift_w * log_probs_numeric).sum()

        return loss.to(dtype=t.float32)

    def numerical_basis_model_loss(
        self, outputs, labels, category_ids=None, ranks=None, **kwargs
    ):
        """
        loss for basis_blended_tokens.numerical_basis_model: true (see
        fuzzy_token_planning.md point 18) -- a from-scratch reproduction of
        a different paper's approach, NOT the k-basis Beta-mixture scheme
        basis_blended_token_loss implements. Two additive, sum-reduced
        terms (matching this module's convention: a batch with more
        numeric tokens contributes proportionally more gradient signal):

        Term 1 -- plain next-token CE over the *truncated* vocab (see
        build_gaussian_vocab: num_non_numeric + num_categories, one slot
        per category, no k-way sub-choice). Unlike basis_blended_token_loss,
        this needs no special-casing for numeric vs. non-numeric positions:
        a numeric position's label already *is* its category's single
        collapsed token id, so this is ordinary one-hot CE uniformly across
        every position -- structurally identical to x_ent_loss, just over
        the smaller vocab.

        Term 2 -- Gaussian NLL of the observed normalized value. For every
        position, BasisBlendedCausalLM's gaussian_head reads the model's
        own final hidden state (the same one the LM head reads) and
        predicts (a_c, b_c) for *every* category c -- outputs.gaussian_params,
        shape (B, T, 2*n_cat), unshifted, exactly like logits. This term
        gathers the (a_c, b_c) pair for whichever category the *actual*
        next token turns out to be (shift by one position, same as
        shift_labels/shift_logits below), constructs Normal(a_c,
        softplus(b_c)), and takes the NLL of the true next value
        (ranks, shifted the same way) under it -- only at positions whose
        actual next token is numeric. softplus keeps sigma positive without
        a hard floor; clamp_min(1e-6) on top guards the 1/sigma term below
        from blowing up if b_c drifts very negative during training.
        """
        shift_logits = outputs.get("logits")[:, :-1].contiguous().to(dtype=t.float32)
        shift_labels = labels[:, 1:].contiguous()
        shift_cat = category_ids[:, 1:].contiguous()
        numeric = shift_cat >= 0

        log_probs = t.log_softmax(shift_logits, dim=-1)
        loss = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1).sum()

        if numeric.any():
            gaussian_params = outputs.get("gaussian_params")
            assert gaussian_params is not None, (
                "numerical_basis_model requires the model to return gaussian_params"
            )
            shift_gauss = (
                gaussian_params[:, :-1].contiguous()[numeric].to(dtype=t.float32)
            )
            n_cat = shift_gauss.shape[-1] // 2
            cat_num = shift_cat[numeric]
            a = shift_gauss.gather(-1, cat_num.unsqueeze(-1)).squeeze(-1)
            b = shift_gauss.gather(-1, (cat_num + n_cat).unsqueeze(-1)).squeeze(-1)
            sigma = F.softplus(b).clamp_min(1e-6)
            v_true = ranks[:, 1:].contiguous()[numeric].to(dtype=t.float32)
            nll = (
                0.5 * math.log(2 * math.pi)
                + sigma.log()
                + 0.5 * ((v_true - a) / sigma).pow(2)
            )
            loss = loss + nll.sum()

        return loss.to(dtype=t.float32)

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
        return t.nn.CrossEntropyLoss(reduction="sum")(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).to(dtype=t.float32)

    def custom_loss(self, outputs, labels, **kwargs):
        loss = 0.0
        log = dict()
        if "basis_blended_tokens" in self.cfg and self.cfg.basis_blended_tokens.get(
            "numerical_basis_model", False
        ):
            numerical_basis_model_loss = self.numerical_basis_model_loss(
                outputs, labels, **kwargs
            )
            log |= {"numerical_basis_model_loss": numerical_basis_model_loss.item()}
            loss += numerical_basis_model_loss
        elif "basis_blended_tokens" in self.cfg:
            basis_blended_token_loss = self.basis_blended_token_loss(
                outputs, labels, **kwargs
            )
            log |= {"basis_blended_token_loss": basis_blended_token_loss.item()}
            loss += basis_blended_token_loss
        elif "label_weighted_loss" in self.cfg:
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

        # the sum-reductions above are load-bearing (see x_ent_loss /
        # basis_blended_token_loss): they're what makes a batch with more
        # numeric tokens contribute proportionally more gradient signal.
        # Dividing the already-combined scalar by the token count here only
        # rescales overall magnitude -- it doesn't touch that relative
        # weighting -- so loss/gradient scale stops depending on
        # batch_size * seq_len and becomes a comparable per-token average.
        n_tokens = labels[:, 1:].numel()
        loss = loss / n_tokens
        log = {k: v / n_tokens for k, v in log.items()}

        if wandb.run is not None:
            log |= {"custom_loss": loss.item(), "n_tokens": n_tokens}
            wandb.log(log)
        return loss


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loss(cfg=trainer.cfg, tkzr_cfg=trainer.tkzr_cfg)
    # breakpoint()
