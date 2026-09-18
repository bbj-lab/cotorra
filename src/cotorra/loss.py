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
        # overwritten per call by custom_loss; True so a direct call to one
        # of the loss methods (self-tests, notebooks) behaves like training
        self._is_train = True
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
            "interval_nll_loss", False
        ):
            assert self.cfg.basis_blended_tokens.get("mixture_nll_loss", False), (
                "basis_blended_tokens.interval_nll_loss: true requires "
                "basis_blended_tokens.mixture_nll_loss: true -- it replaces "
                "that loss's density term with the interval's probability "
                "mass, it isn't a standalone alternative"
            )
            assert not self.cfg.basis_blended_tokens.get("crps_loss", False), (
                "basis_blended_tokens.interval_nll_loss and crps_loss are "
                "mutually exclusive -- both replace the NLL term"
            )
            # 32 points, not crps.DEFAULT_OUTER_POINTS (8): measured
            # against scipy's exact incomplete beta over 400 random
            # mixtures, 8 points leaves up to 2.6e-2 nats of error on the
            # widest intervals (~0.95, where a Beta shape < 1 makes the
            # integrand steep at an endpoint) -- the coarse ordinal
            # categories this loss exists to fix are exactly the wide ones.
            # 32 points brings that to <=2.4e-3 nats, negligible against
            # the effects of interest, at a cost comparable to crps_loss's
            # nested quadrature.
            self._interval_gl = crps.gl_rule(
                int(self.cfg.basis_blended_tokens.get("interval_quad_points", 32)),
                dtype=t.float32,
            )
        if "basis_blended_tokens" in self.cfg and (
            self.cfg.basis_blended_tokens.get("crps_loss", False)
            or float(self.cfg.basis_blended_tokens.get("interval_crps_weight", 0.0)) > 0
        ):
            assert self.cfg.basis_blended_tokens.get("mixture_nll_loss", False), (
                "basis_blended_tokens.crps_loss / interval_crps_weight require "
                "basis_blended_tokens.mixture_nll_loss: true -- both act inside "
                "that loss, neither is a standalone alternative (see "
                "basis_blended_token_loss)"
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

        # the bin-vocabulary lookups every decile-baseline numeric loss needs.
        # Built whenever ANY of them is configured: a was_token_loss-only (or
        # numeric_loss-only) config used to leave these unbuilt and die with
        # AttributeError on the first batch.
        self._baseline_numeric = (
            any(
                key in self.cfg
                for key in ("quantile_token_loss", "was_token_loss", "crps_token_loss")
            )
            or self.cfg.get("numeric_loss", None) is not None
        )
        if self._baseline_numeric and "basis_blended_tokens" not in self.cfg:
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
        elif self._baseline_numeric:
            # These read the RAW cocoa (code, decile) vocabulary off
            # tkzr_cfg.lookup, but under basis_blended_tokens the model's
            # logits are over the COLLAPSED basis vocabulary
            # (num_non_numeric + n_cat*k). The column index sets do not
            # correspond, so at coincident sizes this would score arbitrary
            # slots without erroring. The basis arm's equivalents are
            # numeric_loss: was / mse under the tied continuous head, which
            # score the head's own density; only the legacy per-token blocks
            # are refused outright.
            assert not any(
                key in self.cfg
                for key in ("quantile_token_loss", "was_token_loss", "crps_token_loss")
            ), (
                "quantile_token_loss / was_token_loss / crps_token_loss are "
                "decile-baseline losses over the RAW vocabulary and cannot be "
                "combined with basis_blended_tokens, whose logits are over "
                "the collapsed basis vocabulary. Use numeric_loss: was / mse "
                "with tied_continuous_head instead."
            )

        kind, _w = self._numeric_loss()
        if kind is not None:
            assert not any(
                key in self.cfg
                for key in ("quantile_token_loss", "was_token_loss", "crps_token_loss")
            ), (
                "numeric_loss and the legacy quantile_token_loss / "
                "was_token_loss / crps_token_loss blocks both set the same "
                "weight -- use exactly one. The blocks are the pre-Optuna "
                "form; numeric_loss is the tunable one."
            )
            if "basis_blended_tokens" in self.cfg:
                assert self.cfg.basis_blended_tokens.get(
                    "tied_continuous_head", False
                ), (
                    "numeric_loss under basis_blended_tokens requires "
                    "tied_continuous_head: true -- both terms are statistics "
                    "of p(r|h) = exp(g)/Z, which only that head predicts"
                )
                assert not (
                    kind == "crps"
                    and float(
                        self.cfg.basis_blended_tokens.get("interval_crps_weight", 0.0)
                    )
                    > 0
                ), (
                    "numeric_loss: crps and interval_crps_weight add the SAME "
                    "term twice -- use interval_crps_weight for a fixed "
                    "weight or numeric_loss for the tunable one, not both"
                )

    def _qt_index(self, device):
        """per-category (column indices, bin midpoints), cached per device.

        Two reasons this is not rebuilt per batch. The masks built in
        __init__ are CPU tensors, and indexing CUDA logits with a CPU boolean
        mask raises -- the defect the three losses below used to share, and
        one a CPU-only self-test cannot catch. And rebuilding 2*n_cats masks
        every step is pure waste at the 151 categories the real vocabulary
        has. Same lazy per-device cache as _interval_quadrature/_active_masks.
        """
        cache = self.__dict__.setdefault("_qt_index_cache", {})
        hit = cache.get(str(device))
        if hit is not None:
            return hit
        cols, mids = [], []
        for i in range(self.n_cats):
            sel = (self.label_to_cat == i).nonzero(as_tuple=True)[0]
            cols.append(sel.to(device))
            mids.append(self.label_to_q[sel].to(device=device, dtype=t.float32))
        cache[str(device)] = (cols, mids)
        return cache[str(device)]

    def _bin_terms(self, outputs, labels, ranks=None):
        """(bin probabilities, bin midpoints, target rank) per category.

        The shared core of the three decile-baseline numeric losses, so they
        cannot drift apart in masking, device handling or target convention.
        Yields nothing for a category absent from the batch.

        TARGET. By default the bin MIDPOINT, which is the literal NTL
        formulation over a bin vocabulary -- but the tied head's equivalents
        score the EXACT rank, so a cross-arm comparison of the same loss would
        otherwise differ in its target as well as its backbone. Set
        baseline_exact_ranks to score the exact rank here too; see
        Trainer.collate_fn, which only then puts `ranks` in the batch.
        """
        shift_logits = outputs.get("logits")[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        assert shift_logits.shape[-1] == len(self.vocab), (
            f"logits are over {shift_logits.shape[-1]} slots but the raw "
            f"vocabulary has {len(self.vocab)} -- these losses index the raw "
            "(code, decile) vocabulary; see the basis_blended_tokens assert "
            "in Loss.__init__"
        )
        exact = bool(self.cfg.get("baseline_exact_ranks", False))
        assert not exact or ranks is not None, (
            "baseline_exact_ranks: true but the batch has no `ranks` -- "
            "Trainer.collate_fn must collate them for this arm"
        )
        l2c = self.label_to_cat.to(device=labels.device)
        l2q = self.label_to_q.to(device=labels.device)
        cols, mids = self._qt_index(shift_logits.device)
        shift_ranks = ranks[:, 1:].contiguous() if exact else None
        for i in range(self.n_cats):
            mask = l2c[shift_labels] == i
            if not mask.any():
                continue
            probs = t.softmax(
                shift_logits[mask][:, cols[i]].to(dtype=t.float32), dim=-1
            )
            true = (
                shift_ranks[mask].to(dtype=t.float32)
                if exact
                else l2q[shift_labels[mask]].to(dtype=t.float32)
            )
            yield probs, mids[i], true

    def quantile_token_loss(self, outputs, labels, **kwargs):
        """NTL-MSE over the decile-bin vocabulary: squared error between the
        predicted distribution's MEAN and the target rank.

        Sum-reduced like every other term here; custom_loss divides by
        n_tokens exactly once."""
        loss = outputs.get("logits").new_zeros((), dtype=t.float32)
        for probs, q, true in self._bin_terms(outputs, labels, kwargs.get("ranks")):
            loss = loss + (probs @ q - true).pow(2).sum()
        return loss

    def was_token_loss(self, outputs, labels, **kwargs):
        """NTL-WAS over the decile-bin vocabulary: W1 between the predicted
        bin distribution and the empirical one-hot at the target rank, which
        for a discrete predictive distribution is E_p|X - r| with the bins
        placed at their midpoints."""
        loss = outputs.get("logits").new_zeros((), dtype=t.float32)
        for probs, q, true in self._bin_terms(outputs, labels, kwargs.get("ranks")):
            loss = loss + ((true.unsqueeze(-1) - q).abs() * probs).sum()
        return loss

    def crps_token_loss(self, outputs, labels, **kwargs):
        """CRPS over the decile-bin vocabulary, in closed form.

        The bins partition the RANK axis into n_bins equal-width intervals by
        construction (a decile holds 10% of the mass, so its rank extent is
        0.1 wide), so the predictive distribution is piecewise uniform: within
        bin i, F(x) = C_i + p_i * (x - lo_i)/w with C_i the mass strictly
        below it. Integrating (F - 1{x >= r})^2 over each bin gives, with
        d_i = C_i - 1 and f the position of r inside its own bin,

            below r:  w * (C_i^2 + C_i p_i + p_i^2/3)
            above r:  w * (d_i^2 + d_i p_i + p_i^2/3)
            r's bin:  w * [ f C^2 + f^2 C p + f^3 p^2/3
                          + (1-f) d^2 + (1-f^2) d p + (1-f^3) p^2/3 ]

        which is exact rather than quadrature -- the discrete analogue of the
        tied head's continuous CRPS, so the two arms score the same event.
        """
        loss = outputs.get("logits").new_zeros((), dtype=t.float32)
        for probs, q, true in self._bin_terms(outputs, labels, kwargs.get("ranks")):
            w = 1.0 / float(self.tkzr_cfg.cfg.n_bins)
            lo = q - w / 2  # (n_bin,) bin lower edges on the rank axis
            C = probs.cumsum(-1) - probs  # mass strictly below each bin
            d = C - 1.0
            r = true.unsqueeze(-1)
            f = ((r - lo) / w).clamp(0.0, 1.0)  # 0 below r's bin, 1 above it
            third = probs.pow(2) / 3.0
            term = (
                f * C.pow(2)
                + f.pow(2) * C * probs
                + f.pow(3) * third
                + (1 - f) * d.pow(2)
                + (1 - f.pow(2)) * d * probs
                + (1 - f.pow(3)) * third
            )
            loss = loss + w * term.sum()
        return loss

    # numerical floor on the rank-interval width. NOT the resolution floor
    # (delta_min) discussed as a possible fix for over-granular gradients --
    # this exists only so a genuinely degenerate interval can't produce
    # log(0). Width is exactly 0 for values outside the training range,
    # where cocoa pins p1 and p2 to the same boundary (see that tokenizer's
    # _add_exact_rank); those positions are counted and logged as
    # interval_degenerate_frac so we can see whether they matter.
    INTERVAL_WIDTH_EPS = 1e-6
    # keeps quadrature nodes off the 0/1 endpoints, where a Beta density
    # with a shape parameter < 1 diverges.
    INTERVAL_ENDPOINT_EPS = 1e-6

    def _extra_numeric_term(self, outputs, numeric, shift_ranks, is_tch):
        """the additive numeric term under numeric_loss, or None.

        Shared by the anchor arms and the FiLM arm: both read a statistic the
        MODEL computed on the same quadrature nodes as log P, so neither
        recomputes the density here and neither can drift onto a different
        one. Sets self._numeric_diag; see _numeric_loss for the weighting.

        Kinds:
          "was"  W1(p, delta_r) = E_p|X - r|. Distance-aware like CRPS but
                 first-order; the quantity NTL-WAS uses over a bin
                 vocabulary, taken here in the continuum.
          "mse"  (E_p[X] - r)^2. NTL-MSE. A point-estimate term, so unlike
                 W1 and CRPS it is NOT a proper scoring rule for the density
                 -- it constrains only the first moment. Here because it is
                 the NTL baseline, not because it is the better rule.
          "crps" the same CRPS interval_crps_weight adds, reached through the
                 tunable knob instead of that fixed weight.
        "crps_z" that CRPS in standardised VALUE units instead of rank units
                 -- the same integral weighted by the category's own
                 quantile-function slope, so the tails count for what they
                 are worth clinically. See
                 BasisBlendedConfig.continuous_crps_z.
        """
        kind, num_w = self._numeric_loss()
        if kind is None or num_w <= 0.0:
            return None
        if not getattr(self, "_is_train", True):
            # eval: the primary objective only -- see custom_loss
            return None
        assert is_tch, (
            f"numeric_loss: {kind} is tied-continuous-head only -- it scores "
            "p(r|h) = exp(g)/Z. The arithmetic mixture head predicts a "
            "different density, and the decile baseline's equivalents are "
            "the *_token_loss methods."
        )
        field = {"crps": "continuous_crps", "crps_z": "continuous_crps_z"}.get(
            kind, "continuous_w1"
        )
        need = field if kind in ("crps", "crps_z") else "continuous_moments"
        stat = outputs.get(field)
        assert stat is not None, (
            f"numeric_loss: {kind} needs a model built with {need}: true -- "
            "Trainer derives that from the PRESENCE of numeric_loss, so the "
            "key must be in the config the model was built from"
        )
        s_stat = stat[:, 1:].contiguous()[numeric].to(dtype=t.float32)
        if kind in ("crps", "crps_z"):
            self._numeric_diag = {
                field: s_stat.mean().item(),
                "numeric_loss_weight": num_w,
            }
            return num_w * s_stat.sum()
        c_mean = outputs.get("continuous_mean")
        s_mean = c_mean[:, 1:].contiguous()[numeric].to(dtype=t.float32)
        s_err = s_mean - shift_ranks
        s_mse = s_err.pow(2)
        # both terms logged whichever is active, so the "was" and "mse" arms
        # are directly comparable; the bias is signed, > 0 meaning the
        # predicted mean sits above the truth on average.
        self._numeric_diag = {
            "continuous_w1": s_stat.mean().item(),
            "continuous_mse": s_mse.mean().item(),
            "continuous_mean_bias": s_err.mean().item(),
            # the searched knob, so wandb shows which dose a trial is at
            # from its first logged step rather than at trial end
            "numeric_loss_weight": num_w,
        }
        return num_w * (s_stat if kind == "was" else s_mse).sum()

    def film_continuous_loss(
        self, outputs, labels, category_ids=None, ranks=None, rank_widths=None, **kwargs
    ):
        """FiLM value embedding scored by the tied continuous head.

        ONE vocabulary slot per concept, so the discrete term is ordinary
        next-token CE over the collapsed vocabulary -- at a numeric position
        that IS -log z_c, there being a single slot to choose, so it is the
        same quantity the anchor arms separate out. The value term is the
        head's own interval log-probability, the same
        -log P(r in [lo,hi] | h) those arms score, at the same
        mixture_nll_alpha and with the same optional numeric_loss on top.

        No mixture weights, no beta_log_pdf, no k-way gather: all three
        describe the anchor blend, which this parameterisation replaces.
        """
        self._interval_diag = {}
        self._numeric_diag = {}
        shift_logits = outputs.get("logits")[:, :-1].contiguous().to(dtype=t.float32)
        shift_labels = labels[:, 1:].contiguous()
        shift_cat = category_ids[:, 1:].contiguous()
        numeric = shift_cat >= 0
        log_probs = t.log_softmax(shift_logits, dim=-1)
        loss = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1).sum()

        if numeric.any():
            clp = outputs.get("continuous_logprob")
            assert clp is not None, (
                "film_value_embed with tied_continuous_head requires the "
                "model to return continuous_logprob -- forward() emits it "
                "only when category_ids AND ranks are passed"
            )
            alpha = float(self.cfg.basis_blended_tokens.get("mixture_nll_alpha", 1.0))
            lp = clp[:, 1:].contiguous()[numeric].to(dtype=t.float32)
            loss = loss - alpha * lp.sum()
            width = (
                rank_widths[:, 1:]
                .contiguous()[numeric]
                .to(t.float32)
                .clamp_min(self.INTERVAL_WIDTH_EPS)
            )
            base_rate_nats = -t.log(width)
            self._interval_diag = {
                "interval_mean_width": width.mean().item(),
                "interval_baserate_nats": base_rate_nats.mean().item(),
                # log(w/P): < 0 beats base rate, 0 ties it, > 0 is worse --
                # the same convention, so it is comparable across arms.
                "interval_skill_nats": (-lp - base_rate_nats).mean().item(),
            }
            shift_ranks = ranks[:, 1:].contiguous()[numeric].to(dtype=t.float32)
            extra = self._extra_numeric_term(outputs, numeric, shift_ranks, True)
            if extra is not None:
                loss = loss + extra
        return loss.to(dtype=t.float32)

    def _component_family(self):
        """
        "beta" (default) or "truncnorm" -- read from basis_blended_tokens.
        Under truncnorm the model's beta_a/beta_b outputs carry (mu, sigma)
        rather than Beta shapes; see BasisBlendedConfig.component_family.
        """
        bb = self.cfg.get("basis_blended_tokens", {}) or {}
        return bb.get("component_family", "beta")

    # the extra numeric term's kinds. "crps" is basis-arm-only through
    # interval_crps_weight (which predates this knob) and baseline-only
    # through crps_token_loss, so it is not offered as a numeric_loss kind
    # for the tied head -- see basis_blended_token_loss.
    NUMERIC_LOSS_KINDS = ("was", "mse", "crps", "crps_z")

    def _numeric_loss(self):
        """(kind, w2) for the single additive numeric term.

        The total loss is
            base categorical term + w1 * primary numeric + w2 * extra numeric
        with w1 fixed (mixture_nll_alpha, default 1.0) so that w2 alone is
        tunable -- one scalar, which is what lets it go in an Optuna search
        space. Additive rather than a convex blend on purpose: see the
        mixture_align_loss note in basis_blended_token_loss for the measured
        cost of the convex form. At w2 = 0 nothing is added and the arm is
        bit-identical to one with no numeric_loss key at all.
        """
        kind = self.cfg.get("numeric_loss", None)
        assert kind is None or kind in self.NUMERIC_LOSS_KINDS, (
            f"numeric_loss: {kind!r} is not one of "
            f"{self.NUMERIC_LOSS_KINDS} or null -- a typo here would "
            "silently train with no numeric term at all"
        )
        return kind, float(self.cfg.get("numeric_loss_weight", 0.0))

    def _interval_log_prob(self, log_w_hat, shift_a, shift_b, lo, hi, n_quad=None):
        """
        log of the mixture's probability mass on [lo, hi]:

            log sum_i w_i * (I_beta(hi; a_i, b_i) - I_beta(lo; a_i, b_i))

        Per-component masses come from crps.beta_component_interval_log_mass,
        the same helper BasisBlendedCausalLM._mixture_weights uses for the
        input blend under interval_mixture_weights -- one implementation so
        the encoder and the objective can't drift apart on what an
        observation is.
        """
        if self._component_family() == "truncnorm":
            # shift_a/shift_b carry (mu, sigma) here -- closed form, exact.
            comp = crps.trunc_normal_interval_log_mass(shift_a, shift_b, lo, hi)
        else:
            nodes, gl_w = self._interval_gl
            nodes = nodes.to(lo.device, lo.dtype)
            gl_w = gl_w.to(lo.device, lo.dtype)
            comp = crps.beta_component_interval_log_mass(
                shift_a, shift_b, lo, hi, nodes, gl_w, self.INTERVAL_ENDPOINT_EPS
            )
        return t.logsumexp(log_w_hat + comp, dim=-1)

    def basis_blended_token_loss(
        self, outputs, labels, category_ids=None, ranks=None, rank_widths=None, **kwargs
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
        # cleared each call: these are stashed for custom_loss to merge AFTER
        # its per-token division, and a branch that stops firing must stop
        # logging rather than keep re-logging the previous batch's numbers --
        # which is what a tuner trial at weight 0, or an eval batch with no
        # numeric positions, would otherwise do.
        self._interval_diag = {}
        self._align_diag = {}
        self._crps_diag = {}
        self._numeric_diag = {}

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
                if self.cfg.basis_blended_tokens.get("interval_nll_loss", False):
                    # score the probability MASS the mixture puts on the
                    # rank interval [p1, p2] the midpoint came from, rather
                    # than the density AT the midpoint. Mass is bounded by
                    # 1, so this term is bounded below by 0; the density is
                    # unbounded, which let a model drive the loss
                    # arbitrarily low by spiking on the handful of distinct
                    # values a tied/discrete variable actually takes (see
                    # experiments/loss_breakdown/ -- 11 numeric categories,
                    # all coarse ordinal scales, had reached losses only
                    # attainable that way).
                    assert rank_widths is not None, (
                        "interval_nll_loss requires rank_widths -- retokenize "
                        "with a cocoa build that emits exact_rank_widths"
                    )
                    beta_a = outputs.get("beta_a")
                    beta_b = outputs.get("beta_b")
                    assert beta_a is not None and beta_b is not None, (
                        "interval_nll_loss requires the model to return "
                        "beta_a/beta_b (the mixture must be evaluated away "
                        "from the single observed rank)"
                    )
                    shift_a = (
                        beta_a[:, 1:]
                        .contiguous()[numeric]
                        .to(dtype=t.float32)
                        .clamp(
                            *(
                                crps.MU_CLAMP
                                if self._component_family() == "truncnorm"
                                else crps.SHAPE_PARAM_CLAMP
                            )
                        )
                    )
                    shift_b = (
                        beta_b[:, 1:]
                        .contiguous()[numeric]
                        .to(dtype=t.float32)
                        .clamp(
                            *(
                                crps.SIGMA_CLAMP
                                if self._component_family() == "truncnorm"
                                else crps.SHAPE_PARAM_CLAMP
                            )
                        )
                    )
                    shift_ranks = ranks[:, 1:].contiguous()[numeric].to(t.float32)
                    width = (
                        rank_widths[:, 1:]
                        .contiguous()[numeric]
                        .to(t.float32)
                        .clamp_min(self.INTERVAL_WIDTH_EPS)
                    )
                    lo = (shift_ranks - width / 2).clamp(0.0, 1.0)
                    hi = (shift_ranks + width / 2).clamp(0.0, 1.0)
                    clp = outputs.get("continuous_logprob")
                    vlp = outputs.get("value_logprob")
                    if clp is not None:
                        # TIED CONTINUOUS HEAD. The model returned
                        # log P(r in [lo,hi] | h) directly, from scoring the
                        # hidden state against the input embedding function
                        # over the whole rank axis and normalising by
                        # quadrature. Same quantity this branch otherwise
                        # computes from mixture weights, but without the
                        # arithmetic mixture's envelope ceiling -- see
                        # BasisBlendedConfig.tied_continuous_head.
                        #
                        # Level 1 (which category) is unchanged: still the
                        # log_z_c term from the existing LM head.
                        log_interval_prob = (
                            clp[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                        )
                    elif vlp is not None:
                        # VALUE-VOCABULARY CE. The model already returned
                        # log P2(entry | category) for every target position;
                        # use it directly in place of interval mass.
                        #
                        # These are the same quantity in kind -- both are a
                        # log-probability of the observed value under the
                        # model -- but the CE version is optimised by a hidden
                        # state equal to the EXPECTED next embedding, whereas
                        # interval mass is optimised by a one-hot over
                        # components (a single anchor). That asymmetry is the
                        # reason for the switch; see BasisBlendedConfig's
                        # value_vocab_file note and
                        # docs/value_vocabulary_ce.md.
                        #
                        # Level 1 (which category) is unchanged: it is still
                        # the log_z_c term computed from the existing LM head.
                        log_interval_prob = (
                            vlp[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                        )
                    else:
                        log_interval_prob = self._interval_log_prob(
                            log_w_hat, shift_a, shift_b, lo, hi, None
                        )
                    # Reported-only decomposition -- no gradient effect.
                    # exact_ranks come from the empirical CDF, so the
                    # marginal rank distribution is uniform and a calibrated
                    # but UNINFORMATIVE model puts exactly `width` of mass on
                    # [p1, p2]. That splits this term into
                    #     -log P = (-log w)      <- base rate, data only
                    #            + log(w / P)    <- skill, model only
                    # and only the second half is comparable across
                    # variables. Measured on processed_basis_blended_iv3 the
                    # base-rate half averages 3.538 nats, but its per-token
                    # mean ranges 0.033 (MED-INT//acetaminophen_Q9, a
                    # near-constant ordinal) to 13.436
                    # (MED-CTS//sodium_chloride_Q7, a near-continuous
                    # infusion rate) -- a 403x spread that swamps model
                    # quality when comparing tokens or runs.
                    #
                    # NB it is subtracted, not divided. -log w depends on the
                    # REALIZED outcome, so dividing by it reweights outcomes
                    # within a context and makes the rule improper: the
                    # optimum moves from pi_b to pi_b/(-log w_b) normalized,
                    # i.e. it pays to over-report wide/common bins (with
                    # widths .70/.25/.05 and truth .30/.30/.40 the optimum
                    # becomes .71/.18/.11). Subtracting a data-only constant
                    # leaves every gradient bit-identical, so this stays a
                    # proper scoring rule and remains directly comparable to
                    # every run trained before it.
                    base_rate_nats = -t.log(width)
                    skill_nats = -log_interval_prob - base_rate_nats

                    # mixture_align_loss: blend the interval NLL with an
                    # alignment term between the encoder's own blend w_enc
                    # (target) and the head's predicted mixture w_hat.
                    #
                    # WHY. Under the interval NLL alone the encoder and the
                    # decoder solve different problems. w_enc is a posterior
                    # over components whose prior is the static
                    # log_importance a; the interval NLL's gradient pushes
                    # the hidden state toward a posterior whose prior is the
                    # head's own context-dependent w_hat. Those agree only
                    # when w_hat ~ a, i.e. only when the model predicts
                    # nothing -- measured on iv3, cos(gradient target, the
                    # embedding that actually arrives next) falls 0.971 ->
                    # 0.946 between the least- and most-informative
                    # quartiles, so the misalignment is CAUSED by predictive
                    # skill. Pulling w_hat and w_enc together is the step
                    # toward the expected-embedding property.
                    #
                    # "kl" (default) vs "ce": CE(p,q) = KL(p||q) + H(p), and
                    # the target p = w_enc carries gradient, so CE adds a
                    # live -H(p) pressure that sharpens the encoder. That
                    # sounds attractive (soft binning hardening where
                    # hardening is predictable) but it CONTAMINATES model
                    # selection: H(p) needs no agreement with anything, so
                    # lowering it lowers eval loss whether or not prediction
                    # improved, and eval_loss is what
                    # metric_for_best_model/Optuna minimise. Measured over
                    # the first 3 trials of a ce run: corr(encoder eff_k,
                    # loss) = +0.997, with ce_nats spanning 0.69 nats against
                    # only 0.09 nats of interval_skill_nats -- ~88% of the
                    # between-trial signal was encoder entropy rather than
                    # predictive quality. KL is bounded below by 0 and only
                    # falls when the two distributions actually agree, so it
                    # cannot be lowered by sharpening alone. Default "kl";
                    # "ce" is kept only to reproduce that finding.
                    #
                    # The target is NOT detached. Detaching would leave only
                    # w_hat -> w_enc, so the encoder would never move toward
                    # the prediction and the expected-embedding property
                    # could never be reached; mutual movement is the point.
                    #
                    # -log_z_c ("next token is one of category c's slots") is
                    # deliberately OUTSIDE the blend: it is the discrete
                    # category term, common to both halves, and halving it
                    # would quietly down-weight getting the category right.
                    # Only the two value-side terms are mixed.
                    bb = self.cfg.basis_blended_tokens
                    align_mode = bb.get("mixture_align_loss", None)
                    align_w = (
                        float(bb.get("mixture_align_weight", 0.1))
                        if align_mode
                        else 0.0
                    )
                    # ADDITIVE, not a convex blend. An earlier version spent
                    # (1 - align_w) on the interval term, which at 0.5 cost
                    # ~0.12 nats of real predictive skill: interval_skill_nats
                    # median -0.725 on the interval-only control vs -0.607
                    # over the first trials of the blended run. The interval
                    # NLL is the objective the model exists to optimise; the
                    # alignment term is a regulariser and is priced like one.
                    loss = loss - log_z_c.sum() - alpha * log_interval_prob.sum()
                    if align_w > 0.0:
                        # mixture_align_detach picks which way the alignment
                        # pressure flows:
                        #   "none"       both move (a standard tied
                        #                transformer moves the embedding AND
                        #                the head, so this is the analogue)
                        #   "prediction" detach w_hat -> only the ENCODER
                        #                moves: "embed values the way they
                        #                are actually predicted". Attacks the
                        #                unpredictable part of the encoding,
                        #                which is exactly what breaks the
                        #                expected-embedding property, at the
                        #                cost of encoding fidelity. NB with
                        #                train_importance_scale false the
                        #                encoder has no learned prior, so
                        #                this can only move the Beta shapes,
                        #                not add context dependence.
                        #   "target"     detach w_enc -> only the HEAD moves:
                        #                "predict values the way they are
                        #                embedded". The encoding stays ground
                        #                truth; w_hat can only reach
                        #                E[w_enc | context].
                        detach = bb.get("mixture_align_detach", "none")
                        p_enc = shift_w.detach() if detach == "target" else shift_w
                        q_log = log_w_hat
                        if detach == "prediction":
                            q_log = log_w_hat.detach()
                        # log_w_hat is -inf on slots switched off by
                        # k_per_category and shift_w is exactly 0 there, so a
                        # naive product is 0 * -inf = NaN -- logsumexp absorbs
                        # -inf in the interval term but an explicit multiply
                        # does not. Mask those entries exactly rather than
                        # leaning on a floor; clamp what survives so a live
                        # slot the head has written off costs a large but
                        # finite penalty.
                        lwh = q_log.clamp_min(-1e4)
                        live = shift_w > 0
                        if align_mode == "ce":
                            term = -(p_enc * lwh)
                        else:  # "kl"
                            term = p_enc * (p_enc.clamp_min(1e-8).log() - lwh)
                        align_term = t.where(live, term, t.zeros_like(term)).sum(-1)
                        loss = loss + align_w * align_term.sum()
                        ent = -(shift_w * shift_w.clamp_min(1e-30).log()).sum(-1)
                        self._align_diag = {
                            f"{align_mode}_mixture_nats": align_term.mean().item(),
                            "encoder_blend_entropy": ent.mean().item(),
                            "encoder_blend_eff_k": t.exp(ent).mean().item(),
                        }
                    # interval_crps_weight: CRPS ADDED to the interval term,
                    # not swapped for it (that is what crps_loss does, and why
                    # crps_loss and interval_nll_loss are mutually exclusive).
                    #
                    # Both existing terms are "which component" losses: the
                    # interval mass and the KL both score the weight vector,
                    # and neither knows that component 2 is NEARER to
                    # component 3 than component 9 is. A prediction that puts
                    # its mass one bin away is penalised exactly as hard as
                    # one that puts it at the opposite end of the range. CRPS
                    # integrates (F(x) - 1{x >= r})^2 over x, so its penalty
                    # grows with DISTANCE -- it is the ordinal term the other
                    # two are missing.
                    crps_w = float(
                        self.cfg.basis_blended_tokens.get("interval_crps_weight", 0.0)
                    )
                    if crps_w > 0.0 and clp is not None:
                        # TIED CONTINUOUS HEAD: CRPS of p(r|h) = exp(g)/Z
                        # itself, from the model, on the node set log P uses.
                        # The truncnorm mixture below is NOT what this head
                        # predicts; scoring it would pull the k logits toward
                        # a second prediction rule instead of adding an
                        # ordinal term to this one.
                        ccrps = outputs.get("continuous_crps")
                        assert ccrps is not None, (
                            "interval_crps_weight under tied_continuous_head "
                            "needs a model built with continuous_crps: true "
                            "(Trainer derives it from interval_crps_weight)"
                        )
                        sample_crps = (
                            ccrps[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                        )
                        loss = loss + crps_w * sample_crps.sum()
                        self._crps_diag = {"interval_crps": sample_crps.mean().item()}
                    elif crps_w > 0.0:
                        gl_out = tuple(
                            x.to(log_w_hat.device) for x in self._crps_gl_out
                        )
                        de_in = tuple(x.to(log_w_hat.device) for x in self._crps_de_in)
                        if self._component_family() == "truncnorm":
                            # closed-form CDF -> outer quadrature only
                            sample_crps = crps.truncnorm_mixture_crps(
                                shift_ranks, log_w_hat.exp(), shift_a, shift_b, gl_out
                            )
                        else:
                            sample_crps = crps.mixture_crps(
                                shift_ranks,
                                log_w_hat.exp(),
                                shift_a,
                                shift_b,
                                gl_out,
                                de_in,
                            )
                        loss = loss + crps_w * sample_crps.sum()
                        self._crps_diag = {"interval_crps": sample_crps.mean().item()}
                    # numeric_loss: an NTL-style ordinal term ADDED to the
                    # interval NLL and priced exactly like interval_crps_weight
                    # above -- additive, not a convex blend (see the
                    # mixture_align_loss note for the measured reason: an
                    # earlier convex version cost ~0.12 nats of real skill).
                    # Both kinds read statistics of the SAME density log P
                    # scores, computed by the model on the same node set; see
                    # BasisBlendedConfig.continuous_moments.
                    #
                    #   "was"  W1(p, delta_r) = E_p|X - r|. Distance-aware like
                    #          CRPS but first-order; the quantity NTL-WAS uses
                    #          over a bin vocabulary, taken here in the
                    #          continuum.
                    #   "mse"  (E_p[X] - r)^2. NTL-MSE. A point-estimate term,
                    #          so unlike W1 and CRPS it is NOT a proper scoring
                    #          rule for the density -- it constrains only the
                    #          first moment. Here because it is the NTL
                    #          baseline, not because it is the better rule.
                    #   "crps" the same CRPS interval_crps_weight adds, reached
                    #          through the tunable knob instead of that fixed
                    #          weight; the two are mutually exclusive.
                    _extra = self._extra_numeric_term(
                        outputs, numeric, shift_ranks, clp is not None
                    )
                    if _extra is not None:
                        loss = loss + _extra
                    # stashed rather than logged here: the wandb `log`
                    # dict lives in custom_loss, which merges this in after
                    # its own per-token normalization (these are already
                    # fractions/means, so they must not be divided again).
                    self._interval_diag = {
                        "interval_degenerate_frac": (
                            (
                                rank_widths[:, 1:].contiguous()[numeric]
                                <= self.INTERVAL_WIDTH_EPS
                            )
                            .to(t.float32)
                            .mean()
                            .item()
                        ),
                        "interval_mean_width": width.mean().item(),
                        # -log w: what a calibrated but uninformative model
                        # would pay. Data-only, so a change here between runs
                        # means the tokenization moved, not the model.
                        "interval_baserate_nats": base_rate_nats.mean().item(),
                        # log(w/P): the only width-comparable number here.
                        # < 0 beats base rate, 0 ties it, > 0 is worse.
                        "interval_skill_nats": skill_nats.mean().item(),
                    }
                elif self.cfg.basis_blended_tokens.get("crps_loss", False):
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
                        .clamp(
                            *(
                                crps.MU_CLAMP
                                if self._component_family() == "truncnorm"
                                else crps.SHAPE_PARAM_CLAMP
                            )
                        )
                    )
                    shift_b = (
                        beta_b[:, 1:]
                        .contiguous()[numeric]
                        .to(dtype=t.float32)
                        .clamp(
                            *(
                                crps.SIGMA_CLAMP
                                if self._component_family() == "truncnorm"
                                else crps.SHAPE_PARAM_CLAMP
                            )
                        )
                    )
                    shift_ranks = ranks[:, 1:].contiguous()[numeric].to(dtype=t.float32)
                    w_hat = log_w_hat.exp()
                    gl_out = tuple(x.to(w_hat.device) for x in self._crps_gl_out)
                    de_in = tuple(x.to(w_hat.device) for x in self._crps_de_in)
                    sample_crps = (
                        crps.truncnorm_mixture_crps(
                            shift_ranks, w_hat, shift_a, shift_b, gl_out
                        )
                        if self._component_family() == "truncnorm"
                        else crps.mixture_crps(
                            shift_ranks, w_hat, shift_a, shift_b, gl_out, de_in
                        )
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
            v_true = ranks[:, 1:].contiguous()[numeric].to(dtype=t.float32)
            if self.cfg.basis_blended_tokens.get("xval_head", False):
                # LITERAL xVal: one scalar prediction for every position,
                # scored by MSE. No per-category gather (the head emits a
                # single column) and no sigma -- the published method
                # regresses the value directly rather than modelling its
                # spread. See BasisBlendedConfig.xval_head.
                assert shift_gauss.shape[-1] == 1, (
                    "xval_head expects a scalar regression head, but the "
                    f"model returned {shift_gauss.shape[-1]} columns"
                )
                loss = loss + (shift_gauss.squeeze(-1) - v_true).pow(2).sum()
            else:
                n_cat = shift_gauss.shape[-1] // 2
                cat_num = shift_cat[numeric]
                a = shift_gauss.gather(-1, cat_num.unsqueeze(-1)).squeeze(-1)
                b = shift_gauss.gather(-1, (cat_num + n_cat).unsqueeze(-1)).squeeze(-1)
                sigma = F.softplus(b).clamp_min(1e-6)
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
        # TRAIN WITH THE EXTRA NUMERIC TERM, EVALUATE WITHOUT IT.
        #
        # numeric_loss adds a non-negative term to the loss, so reporting it
        # in eval_loss would make a SMALLER numeric_loss_weight look better
        # mechanically -- at w2=10 with crps_z the added term is ~2.1 nats
        # against an eval_loss of ~2.1, while real between-arm differences
        # are ~0.003-0.09. Optuna minimises eval_loss and
        # metric_for_best_model selects checkpoints by it, so both would have
        # chased the smallest weight rather than the best model.
        #
        # Dropping it at eval makes eval_loss the PRIMARY objective alone:
        # comparable across every arm, and a fair arbiter of whether the
        # extra term earned its place. The term still shapes training; it
        # just does not get to score itself.
        self._is_train = bool(kwargs.pop("training", True))
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
        elif "basis_blended_tokens" in self.cfg and self.cfg.basis_blended_tokens.get(
            "film_value_embed", False
        ):
            film_loss = self.film_continuous_loss(outputs, labels, **kwargs)
            log |= {"film_continuous_loss": film_loss.item()}
            loss += film_loss
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
        # the three decile-baseline numeric terms. Each takes **kwargs so it
        # can score the exact rank under baseline_exact_ranks -- without them
        # the batch's `ranks` never reach the loss.
        for key, fn in (
            ("quantile_token_loss", self.quantile_token_loss),
            ("was_token_loss", self.was_token_loss),
            ("crps_token_loss", self.crps_token_loss),
        ):
            if key in self.cfg:
                term = fn(outputs, labels, **kwargs)
                log |= {key: term.item()}
                loss += float(self.cfg[key].get("qt_weight", 1.0)) * term
        # ... and the same three reached through the tunable knob. The basis
        # arm's equivalents live inside basis_blended_token_loss, where the
        # density they score is available.
        kind, num_w = self._numeric_loss()
        if (
            kind is not None
            and num_w > 0.0
            and self._is_train
            and "basis_blended_tokens" not in self.cfg
        ):
            fn = {
                "was": self.was_token_loss,
                "mse": self.quantile_token_loss,
                "crps": self.crps_token_loss,
            }[kind]
            term = fn(outputs, labels, **kwargs)
            log |= {f"numeric_loss_{kind}": term.item()}
            loss += num_w * term

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
        # already-normalized diagnostics from interval_nll_loss (see
        # basis_blended_token_loss) -- merged after the division above
        log |= getattr(self, "_interval_diag", {})
        # encoder_blend_eff_k is the one to watch under mixture_align_loss: it
        # should FALL over training as CE sharpens the blend toward hard
        # binning. If it collapses to ~1 the CE half has won outright and
        # the soft blend is gone; if it never moves, the CE half is inert.
        log |= getattr(self, "_align_diag", {})
        log |= getattr(self, "_crps_diag", {})
        # continuous_w1 / continuous_mse / continuous_mean_bias under
        # numeric_loss: the last two are logged whichever kind is active, so
        # the arms stay comparable.
        log |= getattr(self, "_numeric_diag", {})

        if wandb.run is not None:
            log |= {"custom_loss": loss.item(), "n_tokens": n_tokens}
            wandb.log(log)
        return loss


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loss(cfg=trainer.cfg, tkzr_cfg=trainer.tkzr_cfg)
    # breakpoint()
