#!/usr/bin/env python3

"""
basis blended tokens: wraps a base causal LM so that numeric (category, exact
rank) entries are embedded as a per-category shared vector plus a
Beta-mixture blend of k learned basis elements per category, and predicted
via a matching soft-label loss target (see loss.py's basis_blended_token_loss),
instead of a single fused bin token
"""

import dataclasses
import json
import math
import pathlib
import re

import numpy as np
import torch as t
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from cotorra import crps

_RANK_EPS = 1e-4
# degenerate-interval floor. MUST equal Loss.INTERVAL_WIDTH_EPS (loss.py:177):
# the two code paths score the same positions and loss.py derives
# base_rate_nats from its own value, so a mismatch both biases the objective
# between arms and corrupts the logged interval_skill_nats.
_INTERVAL_WIDTH_EPS = 1e-6
# floor on the deviation norm before rescaling to the magnitude channel:
# sum_i delta_i = 0, so a uniform w gives an exactly-zero deviation.
_MAGNITUDE_EPS = 1e-6


_BIN_TOKEN_RE = re.compile(r"(Q|B)(\d+)$")


def _detect_numeric_categories(tkzr_cfg) -> tuple:
    """
    shared category-detection logic for build_basis_vocab/build_gaussian_vocab:
    collapse cocoa's fused bin vocabulary (num_non_numeric + per-category bin
    tokens -- "Q<i>" for quantile binning, "B<i>" for Bayesian Blocks, see
    cocoa's Tokenizer.bin_data) down to (raw lookup, raw vocab size, the
    sorted list of numeric categories, a category->index map, a per-raw-id
    category index (-1 for non-numeric), and a lookup of every non-numeric
    label to its own collapsed id). Callers add their own numeric-category
    token slot(s) -- 1 for build_gaussian_vocab, k for build_basis_vocab --
    on top of this shared base.

    Detects a bin token by regex (a trailing "Q<digits>" or "B<digits>"),
    not a fixed n_bins-bounded range: quantile bins are a uniform n_bins
    across every category, but Bayesian Blocks bins are *not* -- each
    category gets its own count, uncapped, so there's no single upper
    bound to check against. (An earlier version of this function checked
    `v.endswith(tuple(f"Q{i}" for i in range(n_bins)))`, which both
    silently ignored every "B<i>" token entirely -- treating the whole
    numeric vocabulary as non-numeric, disabling the basis-blended
    mechanism completely with no error -- and would have under-counted
    quantile categories with more than n_bins observed bins too.)
    """
    lookup = dict(tkzr_cfg.lookup)
    raw_vocab_size = len(lookup)
    vocab = np.array(sorted(lookup, key=lookup.get))  # index == raw token id

    matches = [_BIN_TOKEN_RE.search(v) for v in vocab]
    q_type = np.array([m is not None for m in matches])
    if q_type.any():
        qt_cats = np.array(
            [v[: m.start()] for v, m in zip(vocab, matches) if m is not None]
        )
    else:
        qt_cats = np.array([], dtype=vocab.dtype)

    categories = sorted(set(qt_cats.tolist()))
    cat_index = {c: i for i, c in enumerate(categories)}

    raw_to_category = np.full(raw_vocab_size, -1, dtype=np.int64)
    for label, cat in zip(vocab[q_type], qt_cats):
        raw_to_category[lookup[label]] = cat_index[cat]

    non_numeric_labels = [str(v) for v, is_q in zip(vocab, q_type) if not is_q]
    basis_lookup = {label: i for i, label in enumerate(non_numeric_labels)}
    num_non_numeric = len(basis_lookup)

    return (
        lookup,
        raw_vocab_size,
        categories,
        raw_to_category,
        basis_lookup,
        num_non_numeric,
    )


def build_basis_vocab(tkzr_cfg, k: int) -> dict:
    """
    collapse cocoa's fused bin vocabulary down to the basis blended
    vocabulary (num_non_numeric + num_categories * k) -- k learned basis
    token slots per numeric category; single source of truth shared by the
    model config and the loss. See _detect_numeric_categories for the
    shared category-detection this builds on.
    """
    (
        lookup,
        raw_vocab_size,
        categories,
        raw_to_category,
        basis_lookup,
        num_non_numeric,
    ) = _detect_numeric_categories(tkzr_cfg)

    category_base_id = []
    for i, cat in enumerate(categories):
        base = num_non_numeric + i * k
        category_base_id.append(base)
        for j in range(k):
            # "K<j>" (not "Q<j>"/"B<j>"): a synthetic, display-only label for
            # this basis slot (see Trainer.reverse_lookup) -- it isn't a raw
            # vocabulary token in either binning scheme, so it shouldn't be
            # spelled like one.
            basis_lookup[f"{cat}K{j}"] = base + j

    raw_to_collapsed = np.empty(raw_vocab_size, dtype=np.int64)
    for label, raw_id in lookup.items():
        cat = raw_to_category[raw_id]
        raw_to_collapsed[raw_id] = (
            basis_lookup[label] if cat < 0 else category_base_id[cat]
        )

    return {
        "k": k,
        "num_non_numeric": num_non_numeric,
        "categories": categories,
        "category_base_id": category_base_id,
        "vocab_size": num_non_numeric + len(categories) * k,
        "basis_lookup": basis_lookup,
        "raw_to_category": raw_to_category.tolist(),
        "raw_to_collapsed": raw_to_collapsed.tolist(),
        "bos_token_id": basis_lookup["BOS"],
        "eos_token_id": basis_lookup["EOS"],
    }


def build_gaussian_vocab(tkzr_cfg) -> dict:
    """
    collapse cocoa's fused bin vocabulary down to (num_non_numeric +
    num_categories) -- exactly *one* output-vocab slot per numeric
    category, for the numerical_basis_model reproduction (see
    fuzzy_token_planning.md point 18 and BasisBlendedCausalLM's
    numerical_basis_model branch). Unlike build_basis_vocab's k basis
    slots, there's no k-way sub-choice here: next-token CE over this
    truncated vocab predicts *which category* comes next (or which
    non-numeric token), and a separate small Gaussian head -- not part of
    this collapsed vocab at all -- predicts that category's normalized
    value. Reuses the "category_base_id" field name from build_basis_vocab
    (here, a category's one and only vocab slot) so BasisBlendedConfig/
    BasisBlendedCausalLM can address either vocab scheme through the same
    field.
    """
    (
        lookup,
        raw_vocab_size,
        categories,
        raw_to_category,
        basis_lookup,
        num_non_numeric,
    ) = _detect_numeric_categories(tkzr_cfg)

    category_base_id = []
    for i, cat in enumerate(categories):
        tok_id = num_non_numeric + i
        category_base_id.append(tok_id)
        # "NUM" (not "Q<j>"/"B<j>"/"K<j>"): a synthetic, display-only label
        # for this category's single numeric-value slot -- see the "K<j>"
        # comment in build_basis_vocab for why this shouldn't be spelled
        # like a raw vocabulary token.
        basis_lookup[f"{cat}NUM"] = tok_id

    raw_to_collapsed = np.empty(raw_vocab_size, dtype=np.int64)
    for label, raw_id in lookup.items():
        cat = raw_to_category[raw_id]
        raw_to_collapsed[raw_id] = (
            basis_lookup[label] if cat < 0 else category_base_id[cat]
        )

    return {
        "num_non_numeric": num_non_numeric,
        "categories": categories,
        "category_base_id": category_base_id,
        "vocab_size": num_non_numeric + len(categories),
        "basis_lookup": basis_lookup,
        "raw_to_category": raw_to_category.tolist(),
        "raw_to_collapsed": raw_to_collapsed.tolist(),
        "bos_token_id": basis_lookup["BOS"],
        "eos_token_id": basis_lookup["EOS"],
    }


@dataclasses.dataclass
class BasisBlendedCausalLMOutput(CausalLMOutputWithPast):
    """
    CausalLMOutputWithPast plus mixture_weights/beta_log_pdf as *declared*
    dataclass fields (not dynamically-added dict keys) -- ModelOutput
    reconstruction (e.g. Accelerate's bf16 `convert_outputs_to_fp32`, which
    rebuilds via `type(data)({k: v for k, v in data.items()})`) only
    preserves declared fields, silently dropping anything else.

    beta_log_pdf is the *unscaled* per-component Beta log-density (no
    log_importance folded in, unlike mixture_weights) at each numeric
    position's rank -- needed by Loss.basis_blended_token_loss's
    mixture_nll_loss option, which evaluates the likelihood of the true
    rank under a mixture whose *weights* come from the model's own
    next-token predictions rather than from this rank-derived density
    (see that method's docstring). Exposed separately from mixture_weights
    (which already folds log_importance in and is softmax-normalized) so
    the loss can build that likelihood without recomputing the Beta
    density itself and risking it drifting out of sync with what forward()
    actually used, per the existing "loss target read from the model's own
    forward pass" precedent this file follows for mixture_weights.

    gaussian_params is the numerical_basis_model reproduction's own output
    (see BasisBlendedCausalLM's numerical_basis_model branch and
    Loss.numerical_basis_model_loss): unscaled (a_c, b_c) parameters -- one
    pair per numeric category -- of a per-position Normal distribution
    predicting that category's normalized value, shape (B, T, 2*n_cat), read
    from the model's own final hidden state exactly like logits are (via a
    small separate linear head), *not* shifted for next-token alignment
    here (that happens in the loss, same as logits/labels).

    beta_a/beta_b are the clamped per-position, per-basis-component Beta
    shape parameters themselves (a_c,i, b_c,i -- exponentiated log_alpha/
    log_beta, gathered by category, same values beta_log_pdf was built
    from), shape (B, T, k). Not needed by mixture_nll_loss (which only
    needs the density at the single true rank, already in beta_log_pdf),
    but needed by crps_loss (basis_blended_tokens.crps_loss: true) to
    evaluate the mixture CDF at quadrature points other than the true rank
    -- see Loss.crps_loss.
    """

    mixture_weights: t.Tensor | None = None
    beta_log_pdf: t.Tensor | None = None
    gaussian_params: t.Tensor | None = None
    beta_a: t.Tensor | None = None
    beta_b: t.Tensor | None = None
    # log P2(value entry | category) for each target position, present only
    # when basis_blended_tokens.value_vocab_file is set. Column 0 is unused;
    # index [:, 1:] to align with ranks/rank_widths, exactly as the other
    # target-aligned tensors here are used.
    value_logprob: t.Tensor | None = None
    # log P(r in [lo,hi] | h) under tied_continuous_head, same contract as
    # value_logprob above: shape (B, T), column 0 unused.
    continuous_logprob: t.Tensor | None = None
    # CRPS(p(.|h), r) of that same density, present only when continuous_crps
    # is set. Same (B, T) contract, column 0 unused.
    continuous_crps: t.Tensor | None = None
    # W1(p(.|h), delta_r) = E_p|X - r| and E_p[X], present only when
    # continuous_moments is set. Same (B, T) contract, column 0 unused.
    continuous_w1: t.Tensor | None = None
    continuous_mean: t.Tensor | None = None
    # CRPS of the same density in standardised VALUE units; present only
    # when continuous_crps_z is set. Same (B, T) contract.
    continuous_crps_z: t.Tensor | None = None


class BasisBlendedConfig(PretrainedConfig):
    """composite config: wraps the base model's config plus the collapsed
    basis vocabulary, so save/load round-trips everything needed to rebuild
    the model with no dependency on cocoa's tokenizer.yaml at reload time"""

    model_type = "basis_blended"

    def __init__(
        self,
        base_model_type: str = "llama",
        base_config: dict | None = None,
        k: int = 8,
        train_beta_params: bool = True,
        component_family: str = "beta",
        untie_output_basis: bool = False,
        untie_continuous_head: bool = False,
        continuous_category_logits: bool = False,
        value_vocab_curve: bool = False,
        beta_mu_kappa_param: bool = False,
        fixed_uniform_component: bool = False,
        value_vocab_file: str | None = None,
        train_importance_scale: bool = True,
        train_category_embed: bool = True,
        numerical_basis_model: bool = False,
        xval_head: bool = False,
        xval_value_scale: float = 1.0,
        film_value_embed: bool = False,
        film_hidden: int = 64,
        interval_mixture_weights: bool = False,
        num_non_numeric: int = 0,
        categories: list[str] | None = None,
        category_base_id: list[int] | None = None,
        basis_lookup: dict | None = None,
        raw_to_category: list[int] | None = None,
        raw_to_collapsed: list[int] | None = None,
        k_per_category: list[int] | None = None,
        share_basis_embed_init: bool = False,
        decoupled_magnitude: bool = False,
        magnitude_target: float | None = None,
        poly_curve_basis: bool = False,
        bspline_weights: bool = False,
        bspline_degree: int = 3,
        tied_continuous_head: bool = False,
        continuous_quad_panels: int = 16,
        continuous_quad_nodes: int = 8,
        continuous_interval_panels: int = 8,
        continuous_logit_bound: float = 8.0,
        continuous_interval_nodes: int = 8,
        continuous_crps: bool = False,
        continuous_moments: bool = False,
        continuous_crps_z: bool = False,
        value_quantile_file: str | None = None,
        value_quantile_points: int = 0,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.base_model_type = base_model_type
        self.base_config = base_config or {}
        self.k = k
        self.train_beta_params = train_beta_params
        # "beta" (default, original) or "truncnorm". See
        # crps.py's truncated-normal section for why: Beta's density is
        # EXACTLY zero at r=1 whenever b>1 and at r=0 whenever a>1, so no
        # weighting can put mass at the ends of the rank range, which is
        # where extreme labs and vitals live. Under "truncnorm" the two
        # stored parameters are reinterpreted: log_alpha holds mu DIRECTLY
        # (unconstrained -- mu may sit at or beyond a boundary, which is
        # the whole point) and log_beta holds log sigma. Storage names are
        # kept so checkpoints, _init_weights and the train_beta_params
        # freeze knob all keep working unchanged.
        self.component_family = component_family
        # untie_output_basis: give the OUTPUT side its own per-component
        # embeddings and distribution parameters, so the input blend and the
        # predictive head are no longer forced to share a geometry.
        #
        # This deliberately reverses the original design, which forced
        # tie_word_embeddings=True precisely so the two could not diverge
        # ("Why the tie is structurally guaranteed", fuzzy_token_planning.md).
        # The evidence for reversing it: with the value loss switched off
        # entirely (mixture_nll_alpha 0) the input geometry still trains well
        # -- on glucose |rho| 0.770 -> 0.879 and step ratio 0.775 -> 0.612 --
        # so the embedding does not need the value objective to organise
        # itself. Meanwhile every refinement to the SHARED value machinery
        # (CRPS, truncnorm, three KL variants) moved AUC by nothing. The tie
        # makes one set of vectors serve two jobs: be a useful input encoding
        # AND be a sharp predictive target. Untying lets each specialise.
        self.untie_output_basis = untie_output_basis
        # continuous_category_logits: make level 1 ("which concept comes
        # next") use the SAME continuous measure as level 2, instead of the
        # discrete anchor sum inherited from the mixture head.
        #
        # The coherent joint over {non-numeric tokens} u (u_c [0,1]) is
        #
        #   p(c,r) = exp(h.e_c(r)) / N
        #   N      = sum_v exp(h.e_v) + sum_c Z_c,  Z_c = int_0^1 exp(h.e_c(x))dx
        #
        # so P(c) is proportional to the INTEGRAL Z_c. What the LM head gives
        # instead is sum_i exp(h.e_c,i) over the k anchors. Writing
        # h.e_c,i = h.ebar_c + z_i, the centroid appears in both and cancels,
        # so the whole discrepancy is logsumexp_i(z_i) - log int exp(g).
        #
        # WHY IT MATTERS. For any curve, with p_c = exp(g)/Z_c,
        #
        #   log Z_c = E_{p_c}[g] + H(p_c)                    (exact)
        #
        # -- fit plus BREADTH. On [0,1] the uniform density maximises
        # differential entropy at 0, so H <= 0 always: committing to a narrow
        # value range COSTS category probability, and is repaid only when the
        # rank actually lands there. That is Bayesian Occam's razor, with the
        # rank as the marginalised variable. The discrete score has no such
        # term: logsumexp_i(z_i) - max_i z_i lies in [0, log k], always a
        # bonus, never a penalty, and capped however sharp the curve gets. So
        # under the LM head, spiking one anchor to win the category is free at
        # level 1 and is punished only through level 2's observed rank -- a
        # sampling penalty rather than a structural one. Measured on trained
        # tch_plain: the gap has mean 4.20 nats, sd 3.21, and correlates
        # -0.77 with the density's entropy (sharpest quartile 7.99 nats,
        # most diffuse 2.24).
        #
        # IMPLEMENTATION. forward() OVERWRITES the k logits of each category
        # with log Z_c - log k_active_c, so their logsumexp is exactly log Z_c
        # and the ordinary log_softmax over the full vocabulary then yields
        # log P(c) = log Z_c - log N with the non-numeric slots normalised
        # against the same N. loss.py needs no change: its existing
        # logsumexp-over-a-category's-slots IS log P(c). The per-slot split is
        # even (not all mass on slot 0) so nothing downstream sees -inf.
        #
        # CONSEQUENCE. log_w_hat (log_probs_numeric - log_z_c) becomes uniform
        # and meaningless -- it is unused under the tied head, which takes
        # level 2 from continuous_logprob, and this flag requires that head.
        # Generative sampling of a numeric token from these logits no longer
        # identifies a component; rep-based extraction and scoring read hidden
        # states and are unaffected.
        #
        # COST. The normaliser needs Z_c for EVERY category at EVERY position,
        # not just the observed one. The per-row [0,lo]u[lo,hi]u[hi,1]
        # partition is only needed for the observed category's interval term,
        # so the normaliser uses a FIXED grid shared by all categories and the
        # blend weights become a (n_cat,R,k) tensor built once per forward.
        # See _category_logz.
        self.continuous_category_logits = continuous_category_logits
        # value_vocab_curve: score the TCH curve on the value vocabulary's own
        # entries instead of normalising it by quadrature.
        #
        #   logit_b = g_c(m_b) + log w_b      m_b, w_b the entry's midpoint
        #   P(b|h)  = softmax_b(logit)        and width
        #
        # This IS the integral the quadrature approximates -- a midpoint
        # rectangle rule on the data's OWN partition rather than Gauss-Legendre
        # nodes. The difference is that the entries tile [0,1] exactly and the
        # observation always falls in one, so it is the exact likelihood of a
        # discrete target rather than an approximation of a continuous one.
        # P <= 1 holds by construction, as it did for the shared-node
        # partition, so the loss stays bounded above by 0.
        #
        # WHY log w_b. Entry widths sum to 1 per category and exact_ranks come
        # from the empirical CDF, so the rank marginal is uniform: a calibrated
        # but UNINFORMATIVE model puts exactly w_b on entry b. The term makes
        # g == const reproduce that base rate exactly, so g only ever learns
        # the DEVIATION -- which is what interval_skill_nats already reports.
        # Without it the model spends capacity relearning the marginal.
        #
        # COST. (positions, entries) per category instead of (positions, R, k)
        # over 192 quadrature nodes. At min5/cap500 the median category has 121
        # entries and the worst 500, i.e. 0.63x the quadrature at the median
        # and 2.6x at the worst.
        self.value_vocab_curve = value_vocab_curve
        # untie_continuous_head: the untied ABLATION of the tied continuous
        # head -- "CH". The head keeps scoring h against a curve over the rank
        # axis, but against its OWN curve e_out_c(r) instead of the input
        # embedding function, so the arm measures what the tie is worth while
        # holding the head and the curve construction fixed.
        #
        # Requires untie_output_basis, and reuses its basis_out_embed as the
        # output anchor table rather than allocating a third one. That is what
        # keeps the two LEVELS of one prediction consistent: level 1 (log_z_c,
        # from the LM head's spliced basis columns) and level 2 (the curve)
        # then read the same geometry, exactly as the tied model has both read
        # the input table. Two separate output tables would reintroduce the
        # disagreement this flag exists to fix.
        #
        # WHICH PARAMETERS ARE PART OF THE CURVE. Under the mixture head the
        # predicted weights come from the LM logits, so log_importance only
        # reweights the INPUT blend -- the reason it was given no output copy
        # above. Under the tied continuous head that reasoning does not hold:
        # log_importance enters _blend_weights_rowwise, which is what builds
        # w(r) for the head itself, so it shapes the OUTPUT curve and needs a
        # copy. Same for basis_log_magnitude under decoupled_magnitude and
        # knot_logits under bspline_weights. category_embed is deliberately
        # excluded: h . e_hat_c is constant in r and cancels in the
        # normalisation, so an output copy of it could not change any
        # prediction.
        self.untie_continuous_head = untie_continuous_head
        # beta_mu_kappa_param: store the Beta components as (logit mu, log
        # kappa) instead of (log a, log b), with a = mu*kappa, b = (1-mu)*kappa.
        #
        # Same family, same loss surface -- ONLY the coordinates change, and
        # the init maps exactly (logit mu_i = log((i+1)/(k-i)), log kappa =
        # log(k+1) reproduces a=i+1, b=k-i), so this starts bit-identical to
        # the Bernstein model.
        #
        # Why: measured on the trained control, d(loss)/d(log_alpha) and
        # d(loss)/d(log_beta) are anticorrelated at -0.882, so their SUM --
        # the only thing that moves concentration a+b -- survives at 0.436 of
        # either individual magnitude. Under Adam each of log_alpha/log_beta
        # is normalised by an RMS set by the large opposing LOCATION
        # component, so both take near-full steps in opposite directions and
        # concentration moves only by the residual. Observed consequence:
        # median concentration crept 11 -> 12.0 (1.09x) over training, and
        # the mode-heavy categories that need it most (acetaminophen 97.4%
        # modal, propofol 73.6%) sat at exactly 1.0x. Giving concentration
        # its own coordinate lets Adam take a full-size step in it.
        self.beta_mu_kappa_param = beta_mu_kappa_param
        # fixed_uniform_component: SLOT 0 of every category is pinned to
        # Beta(1,1) -- the uniform density -- and never trains. The remaining
        # k-1 slots tile [0,1] with the Bernstein basis of degree k-2.
        #
        # Why: safe mass everywhere currently has to be bought two ways at
        # once -- keep components from sharpening, AND spread softmax weight
        # across many of them. Both fight the model's need for sharp,
        # confident components. A pinned uniform lets it buy the same
        # insurance with ONE softmax dimension, leaving the other k-1 free to
        # specialise. Slot 0 (not k-1) so that a category's active slots
        # [0, k_c) stay contiguous under the m_c cap.
        #
        # Evidence this is the right target: ~200 held-out targets (0.34%)
        # get assigned effectively zero probability by EVERY basis arm --
        # bin NLL pinned at the -ln(1e-12) = 27.63 floor -- and those alone
        # account for a third of the bin-NLL gap to the decile baseline
        # (+0.2792 -> +0.1897 with them excluded). The baseline scores those
        # same targets at mean 1.061, so they are a basis-specific coverage
        # hole, not intrinsically hard. 93% of them are the same targets
        # across arms.
        self.fixed_uniform_component = fixed_uniform_component
        # value_vocab_file: switch the numeric objective from interval mass
        # over the observed rank interval to plain CROSS-ENTROPY over a merged
        # value vocabulary, with each entry's output embedding GENERATED by
        # the same basis blend that produces the input embedding.
        #
        # Why. -log sum_i w_hat_i * mass_i([lo,hi]) is minimised by a one-hot
        # w_hat -- a hidden state aligned with a single anchor -- while the
        # INPUT embedding of that same observation is a diffuse blend of the
        # same masses. So the loss-optimal hidden state is not the embedding
        # of the token being predicted, which breaks the property a
        # tied-embedding LM depends on. Measured on a toy at H=64: the
        # interval optimum's implied embedding has cosine 0.10 with the
        # target's mean embedding, while the CE optimum matches it to 1.4e-04
        # relative (cosine 1.000000).
        #
        # The distribution factorises exactly, P(token) = P1(head) *
        # P2(value | category), and LEVEL 1 NEEDS NO NEW HEAD: log P1(c) is
        # logsumexp over category c's k slots under the existing LM head --
        # the log_z_c term Loss already computes. Only level 2 is new.
        self.value_vocab_file = value_vocab_file
        self.train_importance_scale = train_importance_scale
        self.train_category_embed = train_category_embed
        # interval_mixture_weights: build the input embedding blend from
        # each basis component's probability MASS over the observation's
        # rank interval [p1, p2], instead of its point density at the
        # interval midpoint. Symmetry with interval_nll_loss, which scores
        # the model's prediction the same way -- without this the encoder
        # represents an observation as a point while the objective treats
        # it as an interval. Degrades to the point-density behavior for
        # narrow intervals (the width cancels in the normalization), so it
        # only changes coarse/tied variables. Requires rank_widths at
        # forward() time, at training AND extraction -- persisted here so
        # the extractor can read it off the checkpoint.
        self.interval_mixture_weights = interval_mixture_weights
        # numerical_basis_model: a from-scratch reproduction of a different
        # paper's approach (see fuzzy_token_planning.md point 18), NOT
        # composable with k/train_beta_params/train_importance_scale/
        # train_category_embed/mixture_nll_loss/kl_loss -- when true, those
        # are ignored entirely; this mode builds its own embedding
        # (e_c,1 + v*e_c,2 per category, see BasisBlendedCausalLM.__init__)
        # and loss (CE over a 1-slot-per-category truncated vocab plus a
        # Gaussian NLL head, see Loss.numerical_basis_model_loss) from
        # scratch, on top of a vocab built by build_gaussian_vocab instead
        # of build_basis_vocab.
        self.numerical_basis_model = numerical_basis_model
        # xval_head: the LITERAL xVal variant of numerical_basis_model.
        # Requires it, and changes exactly the three things that separate
        # this codebase's reproduction from the published method:
        #   1. ONE shared scaled vector for every numeric token, not a
        #      per-category e_c,2 -- xVal has a single [NUM] embedding whose
        #      magnitude carries the value;
        #   2. a SCALAR regression head trained with MSE, replacing the
        #      2-parameter Gaussian head and its NLL (see
        #      Loss.numerical_basis_model_loss);
        #   3. xval_value_scale, applied to v before it multiplies that
        #      vector, because xVal scales values into roughly [-5, 5] while
        #      this project's rank_column is a rank in [0, 1]. Left at 1.0
        #      the embedding is unchanged, so the knob is a no-op until an
        #      arm sets it; with rank_column: zscore_ranks the values are
        #      already z-scored and 1.0 is the right setting.
        # e_c,1 stays per-category and tied: it is the category token's own
        # vocabulary row, which is xVal's per-code embedding, not part of the
        # numeric encoding.
        self.xval_head = xval_head
        self.xval_value_scale = float(xval_value_scale)
        # film_value_embed: the THIRD way this class builds the numeric
        # token's embedding, alongside the k-anchor blend and the xVal line.
        # The concept picks an affine transform and applies it to a learned
        # function of the value:
        #     e_cont = W2 relu(W1 v + b1) + b2          (a 2-layer MLP)
        #     gamma  = W_g e_cat + b_g,  beta = W_b e_cat + b_b
        #     e      = gamma * e_cont + beta            (elementwise)
        # with a NaN value falling back to e_cat alone. Like the xVal family
        # it needs ONE vocabulary slot per concept (build_gaussian_vocab),
        # not k -- there are no anchors to address.
        #
        # AS A CURVE. r -> e_c(r) is a curve in embedding space exactly as
        # the anchor blend is, so the tied continuous head scores it with no
        # change: g(r) = h . e_c(r), normalised by the same quadrature. That
        # makes the three parameterisations directly comparable under one
        # head -- which is the point of having it. The alternative pairing,
        # film_value_embed with numerical_basis_model, keeps FiLM's embedding
        # and scores values with the Gaussian (or xval_head scalar) head
        # instead.
        #
        # film_hidden is the MLP's width, and it is the cost knob under the
        # tied head: g(r) collapses to u . relu(w1 r + b1) with u in R^D (see
        # _continuous_logprob), so the quadrature tensor is (N, R, D) rather
        # than (N, R, H).
        self.film_value_embed = film_value_embed
        self.film_hidden = int(film_hidden)
        self.num_non_numeric = num_non_numeric
        self.categories = categories or []
        self.category_base_id = category_base_id or []
        self.basis_lookup = basis_lookup or {}
        self.raw_to_category = raw_to_category or []
        self.raw_to_collapsed = raw_to_collapsed or []
        # k_per_category: how many of the k allocated slots category c may
        # actually use, k_c <= k. None (default) = every category uses all k,
        # i.e. the uniform-k behaviour every run before this one had.
        #
        # Why a mask over a uniform-k layout rather than ragged per-category
        # widths: the vocabulary stays `num_non_numeric + n_cat * k` with a
        # fixed stride, so category_base_id, raw_to_collapsed, basis_ids and
        # every checkpoint-compat path are untouched; only the surplus slots
        # are switched off. Costs some idle embedding rows, buys a change
        # that cannot silently corrupt token indexing.
        #
        # The bound worth enforcing is k_c <= m_c, the number of DISTINCT
        # training values for c. interval_nll_loss reads the predicted
        # density only through the mass it places on the observation's rank
        # interval, and c has exactly m_c such intervals -- so the loss can
        # only ever see a point in the m_c-simplex, a k-component mixture
        # spans at most k-1 free dimensions of it, and components beyond
        # m_c cannot change anything the objective observes. Same on the
        # encoder side, whose blend is a softmax over per-component interval
        # mass: m_c distinct values admit only m_c distinct blends.
        self.k_per_category = k_per_category
        # share_basis_embed_init: give every basis slot of a category the
        # SAME initial embedding row (plus a small jitter), instead of k
        # independent draws.
        #
        # The numeric input embedding is e = sum_i w_i * e_{c,i} with the
        # weights summing to 1. With k iid N(0, sigma^2) rows that is a
        # weighted average of independent vectors, so
        # ||e|| ~ sigma*sqrt(H) * sqrt(sum_i w_i^2) = sigma*sqrt(H)/sqrt(k_eff)
        # -- it SHRINKS as the blend spreads. A non-numeric token keeps a
        # whole row, so numeric positions enter the transformer quieter than
        # non-numeric ones by a factor that depends on k and nothing else.
        # Measured on iv3 at init: ratio 0.492 at k=10 and 0.423 at k=32, a
        # further 14% attenuation bought by nothing but the component count,
        # which handicaps exactly the arm a large-k experiment is testing.
        #
        # Sharing the row removes it: sum_i w_i * e_c = e_c, so the norm is
        # independent of k AND of how spread the blend is. The jitter only
        # exists as insurance -- the slots are already asymmetric through
        # their differing Beta components, so identical rows would separate
        # under training anyway, but exact ties are a bad thing to rely on.
        self.share_basis_embed_init = share_basis_embed_init
        # decoupled_magnitude: give the blend's MAGNITUDE its own channel,
        # instead of letting it fall out of the anchor geometry.
        #
        #   e(r) = e_hat_c + m_c(r) * dir(sum_i w_i(r) * delta_c,i)
        #   m_c(r) = sum_i w_i(r) * softplus(mu_c,i),   delta_c,i = e_c,i - ebar_c
        #
        # WHY. Under the plain blend e = sum_i w_i e_c,i the magnitude is
        # tied to how spread the weights are: with independent anchors
        # ||e|| ~ sigma*sqrt(H)*||w||_2 = sigma*sqrt(H)/sqrt(k_eff), so a
        # value sitting on top of a component enters at full strength and one
        # falling BETWEEN components enters attenuated -- a magnitude signal
        # that encodes position relative to the component grid rather than
        # anything about the data. Measured on iv3's trained k=10 models, the
        # rank-carrying deviation swings 1.7x-4.2x across the rank axis, and
        # on sparse categories (LAB-RES//wbc_) 75-98% of ranks come out
        # WEAKER than the weakest single anchor.
        #
        # INIT FAIRNESS. Every mu equal at init gives
        # m_c(r) = s0 * sum_i w_i(r) = s0 exactly, for every r and every k,
        # since the weights sum to one -- the same mechanism by which equal
        # component probabilities under a Bernstein basis give a uniform
        # density. No rank is privileged and the sqrt(k) attenuation cannot
        # arise, so share_basis_embed_init is no longer needed to hold the
        # numeric/non-numeric norm ratio at parity: that ratio is s0 by
        # construction, whether the anchors are shared- or iid-initialised.
        #
        # EXPRESSIVENESS. After training mu is free, so magnitude is a
        # function of rank independent of where the anchors sit: tall at a
        # component, tall BETWEEN two components (set both flanking mu high
        # -- they reinforce, so this reaches higher than a single component
        # can), or quiet over a band.
        #
        # The anchor centroid ebar_c is deliberately NOT added back: it is a
        # per-category constant, which is exactly what e_hat_c already is, and
        # keeping both would leave ||e(r)|| = ||ebar_c + dev(r)|| varying with
        # the angle between them -- flat in the deviation but not in the
        # embedding, which is not what was asked for. Dropping it makes
        # ||e(r) - e_hat_c|| = m_c(r) exactly, and e_hat_c is zero at init, so
        # ||e(r)|| == s0 for every rank. Nothing is lost: e_hat_c is trainable
        # and spans whatever ebar_c was contributing.
        #
        # PAIR WITH IID ANCHORS. share_basis_embed_init puts every anchor on
        # one row plus 0.1*sigma jitter, so delta_c,i is jitter alone and the
        # direction here would be pure initialisation noise amplified to full
        # magnitude s0. That flag exists only to hold the numeric norm at
        # parity, which this one now does by construction, so leave it off.
        #
        # magnitude_target sets s0; None means initializer_range*sqrt(H),
        # i.e. the norm of an ordinary freshly-initialised token row.
        #
        # Applies to the INPUT embedding only. value_vocab_file's generated
        # output embeddings still use the plain blend.
        self.decoupled_magnitude = decoupled_magnitude
        self.magnitude_target = magnitude_target
        # tied_continuous_head: predict a density over the WHOLE rank axis by
        # scoring the hidden state against the input embedding function, the
        # same way ordinary next-token prediction scores it against a token
        # row -- only the candidate set is the continuum [0,1] rather than a
        # finite vocabulary, so the softmax denominator becomes an integral.
        #
        #   g_c(r;h) = h . e_c(r) = h.e_hat_c + m_c(r) * <h, u_c(r)>
        #   p_c(r|h) = exp(g_c(r;h)) / Z_c(h),  Z_c(h) = int_0^1 exp(g_c)
        #
        # h.e_hat_c does not depend on r and cancels in the normalisation, so
        # only the second term is ever formed. Writing z_c,i = <h, delta_c,i>
        # and G_c = D_c D_c' (D_c the matrix of delta_c,i),
        #
        #   <h, u_c(r)> = (sum_i w_i(r) z_c,i) / sqrt(w(r)' G_c w(r))
        #
        # so k projections -- the existing basis logits, category-centred --
        # determine g at EVERY r. No new head, no new parameters, and the
        # logit count does not grow with the resolution of the prediction.
        #
        # WHAT IT FIXES. The current head softmaxes those same k logits into
        # mixture weights and predicts sum_i what_i f_i(r), an ARITHMETIC
        # mixture whose density can never exceed max_i f_i(r). That envelope
        # is a structural ceiling: measured on iv3, 29/151 categories under
        # truncnorm and 71/151 under the value-vocab arm have a rank region
        # where no weighting can reach even uniform density. exp(g) has no
        # such ceiling -- on CRRT//scuf_ (envelope -23.68 nats) the mixture
        # can put at most 5.2e-11 density at the hole while this head reaches
        # 1.195. It also makes the output side USE the magnitude channel,
        # which the mixture head structurally cannot.
        #
        # COST. Z is no longer free. A simplex-weighted mixture of normalised
        # densities is normalised by construction; here the normaliser is a
        # quadrature. continuous_quad_panels x continuous_quad_nodes is a
        # COMPOSITE Gauss-Legendre rule on [0,1] -- composite, not a single
        # high-order rule, because a peaked exp(g) is exactly where a flat
        # rule fails and an UNDER-estimated Z rewards the model for spiking, a
        # divergence the loss cannot see. Measured relative error in Z as the
        # density sharpens from peak 1.6 to peak 49: flat 32-node Gauss-
        # Legendre 0.01% -> 22.51%, while 16x8 composite stays at 0.00-0.01%
        # and 32x8 at 0.00%.
        #
        # continuous_interval_panels x continuous_interval_nodes is the rule
        # for the numerator int_lo^hi exp(g). It is composite for the same
        # reason, and the reason is not hypothetical: rank widths reach 1.0
        # (median 0.104, p90 0.314), and a FLAT rule over a wide interval
        # containing a peak fails exactly as it does for Z. Measured at peak
        # density 33: flat GL-16 gives 13.0% error at width 0.31 and 17.6% at
        # width 1.0, and flat GL-32 is WORSE at width 1.0 (21.96%) -- the
        # non-monotonicity that says a rule is stepping over the peak. The
        # 8x8 composite holds 0.049% across every width and sharpness tried.
        #
        # Requires decoupled_magnitude: m_c(r) is that flag's parameter.
        # bspline_weights: build the blend weights from a clamped B-SPLINE
        # basis instead of a softmax over component densities.
        #
        # WHY. Under tied_continuous_head the f_i are no longer a probability
        # model -- the predictive density is exp(sum_i w_i(r) z_i)/Z, and the
        # components enter ONLY through w(r). Their whole job is to define a
        # path through the simplex, which the anchors map into embedding
        # space. Softmax-over-truncnormals is a roundabout way to specify that
        # path, and a measured-inefficient one: on trained models the curve
        # e(r) needs 3 dimensions for 90% of its variance and 4-6 for 99%,
        # while k=10 anchors hand it a 9-dimensional hull, and mean k_eff is
        # only 2.4. The 151 per-category curves do NOT share a global
        # subspace (a 64-dim shared basis leaves 68% reconstruction error), so
        # the redundancy is within each category, not across them -- hence
        # fewer control points rather than a factorised basis.
        #
        # WHAT IT BUYS. B-spline bases are non-negative and sum to exactly one
        # on the knot span, so partition of unity is a CONSTRUCTION rather
        # than something softmax enforces: no exponentials, no truncated-
        # normal normalisers, no MU_CLAMP/SIGMA_CLAMP, no Gauss-Legendre
        # interval mass. They are piecewise polynomials of degree
        # bspline_degree, so interval averages are exact under a small fixed
        # rule. They have LOCAL support -- each basis function is nonzero over
        # only degree+1 spans -- so a gradient at one rank cannot move the
        # curve at a distant one, which softmax weights do not give.
        #
        # PARAMETERS. Shape parameters drop from 2k (mu, sigma) to k-degree-1
        # free interior knots, stored as knot_logits through a softmax-
        # cumsum so they are automatically ordered and inside (0,1). Zero
        # init gives uniform knots, which is already matched to the data:
        # r is the empirical CDF, so its marginal is uniform.
        #
        # Requires tied_continuous_head. log_alpha/log_beta still exist for
        # checkpoint compatibility but no longer drive the weights, so the
        # mixture-density losses would be reading parameters the blend does
        # not use.
        # poly_curve_basis: parameterise the CURVE directly and drop the
        # simplex. e_c(r) = e_hat_c + sum_j T_j(r) v_c,j with T_j the shifted
        # ORTHONORMAL LEGENDRE polynomials on [0,1] and v_c,j learned.
        #
        # Legendre specifically because r is the empirical CDF, so its
        # marginal is uniform and the shifted Legendre polynomials are exactly
        # the orthonormal basis of L^2([0,1], uniform) -- the inner product the
        # data actually carries. Chebyshev would overweight the endpoints.
        #
        # WHAT GOES AWAY. Every shape parameter: no mu, sigma, importance,
        # knots, softmax, or interval-mass quadrature. Interval averaging is
        # EXACT under a k//2+1-node Gauss-Legendre rule, since ceil(k/2) nodes
        # integrate a degree-(k-1) polynomial exactly.
        #
        # WHY k=6. The curve measures dim90 = 3 and dim99 = 4-6 across all 151
        # categories, and here it has EXACTLY k dimensions -- you buy the
        # dimensionality you measured rather than provisioning a (k-1)-dim hull
        # the blend only partly explores (bspline k=6 ran at k_eff 2.23 of 6,
        # the softmax arms at ~2.0 of 10).
        #
        # WHAT IT COSTS. The coefficients are signed and do not sum to one, so
        # this is not a blend: no convex hull, and `reach` and the density
        # envelope stop being defined. The basis is GLOBAL -- every coefficient
        # moves every rank -- where B-splines give local support. And because
        # sum_j T_j(r) is not constant, the centring the tied head applies to
        # blend weights is invalid here and is skipped; see _continuous_logprob.
        #
        # Requires tied_continuous_head, and share_basis_embed_init must be
        # OFF: identical v_j would collapse the curve to one fixed scalar
        # profile times a single shared vector.
        self.poly_curve_basis = poly_curve_basis
        self.bspline_weights = bspline_weights
        self.bspline_degree = bspline_degree
        self.tied_continuous_head = tied_continuous_head
        self.continuous_quad_panels = continuous_quad_panels
        self.continuous_quad_nodes = continuous_quad_nodes
        # continuous_logit_bound: cap |g| via g <- B*tanh(g/B).
        #
        # NOT optional. ||h|| is exactly sqrt(H) = 32, fixed by the final
        # RMSNorm, so |<h,u(r)>| <= 32 and with m ~ 0.8 the raw logit range is
        # +-25 nats: exp(g) spans 1e16. An ordinary softmax tolerates logits
        # that large because its denominator is a discrete sum; here the
        # denominator is an integral, and no fixed-node rule integrates a
        # function with that dynamic range. Left unbounded, training drives a
        # spike narrower than a quadrature panel, Z is under-estimated by up
        # to 21 nats, and -log P goes deeply NEGATIVE -- observed directly:
        # eval_loss fell to 0.7306 with grad_norm 116 while one position in
        # 1215 reported log P = +25.77 (implied P = 1.6e11).
        #
        # tanh is smooth and monotone, so ordering and gradients survive, and
        # it is near-identity well inside the bound. At B=8 the density can
        # still span e^16 ~ 9e6 peak-to-trough, far beyond what the mixture
        # head's envelope permitted. Measured effect on quadrature accuracy
        # (coarse vs fine rule, worst position): 20.985 nats -> 1.665.
        self.continuous_logit_bound = continuous_logit_bound
        self.continuous_interval_panels = continuous_interval_panels
        self.continuous_interval_nodes = continuous_interval_nodes
        # continuous_crps: also return CRPS(p(.|h), r) for the tied head's
        # density, computed on the SAME node set as log P, for Loss to add at
        # basis_blended_tokens.interval_crps_weight. Set by Trainer from that
        # weight, never by hand. Loss's own interval CRPS scores the
        # arithmetic truncnorm mixture of the k logits, which is not what this
        # head predicts -- using it here would train the logits toward a
        # second prediction rule rather than add an ordinal term to this one.
        self.continuous_crps = continuous_crps
        # continuous_moments: also return W1(p, delta_r) = E_p|X - r| and
        # E_p[X] for the tied head's density, on the same nodes as log P, for
        # Loss to add at numeric_loss: "was" / "mse". Set by Trainer from the
        # PRESENCE of numeric_loss, never by hand -- transformers builds the
        # model before Optuna suggests a weight, so a value-gated flag would
        # leave every tuned trial without the statistic its loss then asks
        # for. Both quantities are statistics of the head's OWN density; the
        # arithmetic mixture is a different distribution, which is why these
        # are computed here rather than in Loss.
        self.continuous_moments = continuous_moments
        # continuous_crps_z / value_quantile_file: CRPS in VALUE units rather
        # than rank units, for numeric_loss: crps_z.
        #
        # Rank-space CRPS weights the middle of a lab's distribution like its
        # tails. Pushing the quadrature nodes through the category's training
        # quantile function Q_c first restores the weighting that matters --
        # measured, the TCH-vs-decile gap is 4.5% in rank units and ~19% in
        # value units. Equivalently this is a threshold-weighted CRPS with
        # w(r) = Q'_c(r)/scale_c; the two are the same integral.
        #
        # Q_c is monotone, so mapping the nodes PRESERVES their order and the
        # sorted-cumsum identity the CRPS kernel relies on still holds -- the
        # whole change is "evaluate the same estimator at v = Q(x)".
        #
        # The table is built by experiments/value_scale/build_value_quantiles.py
        # from TRAINING values only, clipped to 0.1-99.9% (above that sit
        # sentinel values like heart rate 10^7 and pH 999999; below it sit
        # real critical values like K+ 7.8), and scaled by each category's
        # winsorised sd. It rides in a persistent buffer so a checkpoint keeps
        # scoring identically when the file is not on the machine.
        #
        # NB the median is NOT subtracted: CRPS is translation-equivariant, so
        # a per-category offset cancels exactly and omitting it removes a
        # thing that could silently disagree between table and model.
        self.continuous_crps_z = continuous_crps_z
        self.value_quantile_file = value_quantile_file
        self.value_quantile_points = int(value_quantile_points)
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    def magnitude_init(self, hidden_size: int) -> float:
        """mu_0 such that softplus(mu_0) is the target blend magnitude s0.

        s0 defaults to initializer_range*sqrt(H) -- the expected norm of an
        ordinary freshly-initialised token row -- so numeric positions enter
        the transformer at parity with non-numeric ones at every rank. See
        decoupled_magnitude."""
        s0 = self.magnitude_target
        if s0 is None:
            s0 = float(self.base_config.get("initializer_range", 0.02)) * math.sqrt(
                hidden_size
            )
        return math.log(math.expm1(float(s0)))

    def __getattr__(self, name):
        # Fallback for inner-model-only fields (num_hidden_layers, head_dim,
        # etc.) that generic HF utilities read directly off `model.config`
        # without knowing it's a composite config -- e.g. DynamicCache's
        # __init__ reads config.num_hidden_layers during .generate(). Only
        # invoked when normal attribute lookup fails, so this can't shadow
        # any field this class (or PretrainedConfig) actually sets.
        base_config = self.__dict__.get("base_config")
        if base_config is not None and name in base_config:
            return base_config[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class BasisBlendedCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BasisBlendedConfig
    # NB: must not be "base_model" -- that name collides with PreTrainedModel's
    # own `base_model` property (getattr(self, base_model_prefix) would then
    # recurse into the property itself instead of reaching the submodule).
    base_model_prefix = "inner_model"

    def __init__(self, config: BasisBlendedConfig):
        super().__init__(config)

        base_config = {k: v for k, v in config.base_config.items() if k != "model_type"}
        inner_cfg = AutoConfig.for_model(config.base_model_type, **base_config)
        inner_cfg.vocab_size = config.vocab_size
        inner_cfg.tie_word_embeddings = True  # required -- see fuzzy_token_planning.md
        if config.bos_token_id is not None:
            inner_cfg.bos_token_id = config.bos_token_id
        if config.eos_token_id is not None:
            inner_cfg.eos_token_id = config.eos_token_id
        self.inner_model = AutoModelForCausalLM.from_config(inner_cfg)

        n_cat = max(len(config.categories), 1)
        film = getattr(config, "film_value_embed", False)
        if config.numerical_basis_model or film:
            # both of these address ONE vocab slot per category, so neither
            # allocates the k-wide anchor machinery below.
            if film:
                _D = int(getattr(config, "film_hidden", 64))
                _H = inner_cfg.hidden_size
                # MLP(v): kept as two explicit Linears rather than a
                # Sequential because _continuous_logprob needs W1/b1 and W2
                # separately to evaluate the curve in D dimensions instead
                # of H. See BasisBlendedConfig.film_value_embed.
                self.film_in = nn.Linear(1, _D)
                self.film_out = nn.Linear(_D, _H)
                self.film_gamma = nn.Linear(_H, _H)
                self.film_beta = nn.Linear(_H, _H)
        if config.numerical_basis_model:
            # numerical_basis_model reproduction (fuzzy_token_planning.md
            # point 18): 2 trainable embeddings per category instead of k
            # Beta-mixture basis elements. e_c,1 is the category's *tied*
            # single vocab-slot embedding (looked up via category_base_id_t
            # in forward(), exactly like embed(basis_ids) is reused on both
            # the input and output side elsewhere in this class); e_c,2 is
            # this untied "slope" -- e = e_c,1 + v*e_c,2, v the position's
            # normalized value (whatever rank_column is configured). Always
            # trainable (no freeze knob, unlike train_beta_params/
            # train_importance_scale/train_category_embed above -- this
            # mode is a fixed-shape reproduction, not a tunable ablation).
            # Zero-init like category_embed, so a freshly-initialized model
            # ignores the numeric value entirely (e = e_c,1) until training
            # moves e_c,2 away from zero.
            # under xval_head a SINGLE shared row, indexed with 0 in
            # forward() instead of by category -- see that flag.
            self.e2 = nn.Parameter(
                t.zeros(
                    1 if getattr(config, "xval_head", False) else n_cat,
                    inner_cfg.hidden_size,
                )
            )
            # second linear layer, h -> 2*n_cat: for every position, (a_c,
            # b_c) for *every* category c, read from the model's own final
            # hidden state as gaussian_params (see BasisBlendedCausalLMOutput
            # and Loss.numerical_basis_model_loss, which shifts this and
            # gathers the true next category's (a_c, b_c) pair to score
            # against the observed value under Normal(a_c, softplus(b_c))).
            # Standard nn.Linear init (no special handling needed -- this is
            # a fresh top-level submodule, already fully initialized by its
            # own __init__ before post_init() runs, same reasoning as
            # inner_model's own layers -- see _init_weights).
            self.gaussian_head = nn.Linear(
                inner_cfg.hidden_size,
                1 if getattr(config, "xval_head", False) else 2 * n_cat,
            )
        elif film:
            pass  # FiLM allocated above; no anchors, no value head of its own
        else:
            k = config.k
            idx = t.arange(k, dtype=t.float32)
            if getattr(config, "component_family", "beta") == "truncnorm":
                # even tiling of [0,1]: mu_i = (i+0.5)/k, sigma = 1/k, so
                # adjacent components overlap at roughly one sigma. Compare
                # the Beta init below, whose means (i+1)/(k+1) are unevenly
                # spaced with widely varying widths.
                log_alpha0 = ((idx + 0.5) / k).unsqueeze(0).expand(n_cat, k).clone()
                log_beta0 = t.full((n_cat, k), math.log(1.0 / k))
            elif getattr(config, "fixed_uniform_component", False):
                # slot 0 pinned to Beta(1,1); slots 1..k-1 carry the Bernstein
                # basis for k-1 components, means j/k for j=1..k-1.
                j = t.arange(1, k, dtype=t.float32)  # 1..k-1
                a0 = t.cat([t.ones(1), j])  # 1, 1,2,...,k-1
                b0 = t.cat([t.ones(1), k - j])  # 1, k-1,...,1
                if getattr(config, "beta_mu_kappa_param", False):
                    log_alpha0 = (a0 / b0).log().unsqueeze(0).expand(n_cat, k).clone()
                    log_beta0 = (a0 + b0).log().unsqueeze(0).expand(n_cat, k).clone()
                else:
                    log_alpha0 = a0.log().unsqueeze(0).expand(n_cat, k).clone()
                    log_beta0 = b0.log().unsqueeze(0).expand(n_cat, k).clone()
            elif getattr(config, "beta_mu_kappa_param", False):
                # exactly the Bernstein components, in (logit mu, log kappa):
                # mu_i = (i+1)/(k+1) and kappa = k+1 give a = i+1, b = k-i.
                log_alpha0 = (
                    ((idx + 1) / (k - idx)).log().unsqueeze(0).expand(n_cat, k).clone()
                )
                log_beta0 = t.full((n_cat, k), math.log(float(k + 1)))
            else:
                log_alpha0 = (idx + 1).log().unsqueeze(0).expand(n_cat, k).clone()
                log_beta0 = (k - idx).log().unsqueeze(0).expand(n_cat, k).clone()
            self.log_alpha = nn.Parameter(
                log_alpha0, requires_grad=config.train_beta_params
            )
            self.log_beta = nn.Parameter(
                log_beta0, requires_grad=config.train_beta_params
            )
            # importance scaling a_c,i (see "Determination of mixture
            # weights" in fuzzy_token_planning.md, extended): reweights each
            # basis element's density contribution independently of its
            # Beta shape. log_a=0 (a=1 uniformly) at init reproduces the
            # plain density-ratio formula exactly, so this can't change
            # behavior until training moves it.
            # k - degree gaps -> cumsum -> k-degree-1 interior knots. Zero
            # init = uniform spacing. See BasisBlendedConfig.bspline_weights.
            if getattr(config, "bspline_weights", False):
                n_gap = max(int(k) - int(config.bspline_degree), 1)
                self.knot_logits = nn.Parameter(t.zeros(n_cat, n_gap))
            self.log_importance = nn.Parameter(
                t.zeros(n_cat, k), requires_grad=config.train_importance_scale
            )
            # untie_output_basis: a SECOND copy of the per-component
            # embeddings and shape parameters, used only on the output side.
            #
            # log_alpha/log_beta keep driving the input blend exactly as
            # before (w_i from the observed value's interval mass); the _out
            # copies are what the loss scores the PREDICTED weights against.
            # basis_out_embed replaces the head's rows for the basis slots
            # only -- the whole-vocab tie stays on, so every non-numeric
            # token keeps sharing its input and output vector as before.
            #
            # log_importance gets no output copy UNDER THE MIXTURE HEAD: there
            # it only reweights the input blend's softmax, because the output
            # weights come from the LM logits. That is false under the tied
            # continuous head, where _blend_weights_rowwise builds the head's
            # own w(r) -- so untie_continuous_head adds the copy, along with
            # the other curve-shaping parameters. See that flag's note.
            if getattr(config, "untie_output_basis", False):
                self.log_alpha_out = nn.Parameter(
                    log_alpha0.clone(), requires_grad=config.train_beta_params
                )
                self.log_beta_out = nn.Parameter(
                    log_beta0.clone(), requires_grad=config.train_beta_params
                )
                self.basis_out_embed = nn.Parameter(
                    t.zeros(n_cat * k, inner_cfg.hidden_size)
                )
                if getattr(config, "untie_continuous_head", False):
                    # freeze flags MIRROR the input side, so the CH arm
                    # differs from the tied head by untying alone rather than
                    # by also unfreezing something.
                    self.log_importance_out = nn.Parameter(
                        t.zeros(n_cat, k), requires_grad=config.train_importance_scale
                    )
                    if getattr(config, "bspline_weights", False):
                        self.knot_logits_out = nn.Parameter(t.zeros(n_cat, n_gap))
                    if getattr(config, "decoupled_magnitude", False):
                        self.basis_log_magnitude_out = nn.Parameter(
                            t.full(
                                (n_cat, k),
                                config.magnitude_init(inner_cfg.hidden_size),
                                dtype=t.float32,
                            )
                        )
            # e_hat_c: a per-category vector added (not weighted) to every
            # numeric position's basis blend, so e = e_hat_c + sum_i w_c,i *
            # e_c,i instead of just sum_i w_c,i * e_c,i -- a shared,
            # trainable part of the embedding tying together every basis
            # element of the same category, on top of each element's own
            # individual embedding. Zero-init like log_importance, so a
            # freshly-initialized model reproduces the plain blend exactly
            # until training moves it.
            self.category_embed = nn.Parameter(
                t.zeros(n_cat, inner_cfg.hidden_size),
                requires_grad=config.train_category_embed,
            )
            # mu_c,i: the magnitude channel. k scalars per category, all
            # equal at init so m_c(r) is flat in r. Created only under the
            # flag so checkpoints of every other mode are unchanged. See
            # BasisBlendedConfig.decoupled_magnitude.
            if getattr(config, "decoupled_magnitude", False):
                self.basis_log_magnitude = nn.Parameter(
                    t.full(
                        (n_cat, k),
                        config.magnitude_init(inner_cfg.hidden_size),
                        dtype=t.float32,
                    )
                )
        if film:
            assert config.tied_continuous_head or config.numerical_basis_model, (
                "film_value_embed builds the numeric token's embedding but "
                "predicts nothing on its own -- pair it with "
                "tied_continuous_head (which scores the FiLM curve) or with "
                "numerical_basis_model (which adds the value head)"
            )
            _nn = getattr(config, "num_non_numeric", None)
            if _nn is not None and getattr(config, "vocab_size", None):
                assert config.vocab_size == _nn + n_cat, (
                    "film_value_embed needs build_gaussian_vocab's one slot "
                    f"per category ({_nn + n_cat} slots), but the vocabulary "
                    f"has {config.vocab_size} -- a k-anchor vocabulary here "
                    "would leave most slots unaddressable"
                )
        uch = getattr(config, "untie_continuous_head", False)
        if not config.numerical_basis_model and not film:
            assert not getattr(config, "xval_head", False), (
                "xval_head is a variant of numerical_basis_model (the "
                "e_c,1 + v*e_c,2 embedding with a value head) and requires "
                "numerical_basis_model: true"
            )
            assert not uch or config.tied_continuous_head, (
                "untie_continuous_head is the untied ablation of the TIED "
                "continuous head -- it does nothing without "
                "tied_continuous_head: true"
            )
            assert not uch or config.untie_output_basis, (
                "untie_continuous_head reuses basis_out_embed as its output "
                "anchor table, so it requires untie_output_basis: true -- "
                "otherwise level 1 (log_z_c, from the LM head) and level 2 "
                "(the curve) would read different geometries"
            )
            assert not (
                config.untie_output_basis and config.tied_continuous_head and not uch
            ), (
                "untie_output_basis + tied_continuous_head WITHOUT "
                "untie_continuous_head is a half-untied hybrid: forward() "
                "splices the LM head's basis columns from basis_out_embed "
                "while _continuous_logprob still scores the INPUT embedding "
                "table and the INPUT blend parameters, so the two levels of "
                "one prediction disagree about what category c's components "
                "are. Set untie_continuous_head: true for the CH ablation, or "
                "untie_output_basis: false for plain TCH."
            )
            vvc = getattr(config, "value_vocab_curve", False)
            assert not vvc or config.tied_continuous_head, (
                "value_vocab_curve scores the tied head's curve, so it "
                "requires tied_continuous_head: true"
            )
            assert not vvc or getattr(config, "value_vocab_file", None), (
                "value_vocab_curve needs the entries to score: set "
                "value_vocab_file to a merged value vocabulary"
            )
            assert not (vvc and getattr(config, "film_value_embed", False)), (
                "value_vocab_curve builds the curve from the anchor blend; "
                "under FiLM e_c(r) is gamma_c * MLP(r) + beta_c and this path "
                "would score the wrong function"
            )
            ccl = getattr(config, "continuous_category_logits", False)
            assert not ccl or config.tied_continuous_head, (
                "continuous_category_logits replaces level 1 with the "
                "continuous head's own normaliser, so it requires "
                "tied_continuous_head: true -- without that head there is no "
                "curve to integrate"
            )
            assert not (ccl and getattr(config, "film_value_embed", False)), (
                "continuous_category_logits is implemented for the anchor "
                "blend only. Under FiLM the r-independent part of "
                "h.e_c(r) is h.beta_c + (h*gamma_c).b2 rather than h.ebar_c, "
                "so _category_logz would silently drop it and mis-scale every "
                "category's Z. Implement that decomposition before pairing "
                "the two."
            )
        self.register_buffer(
            "category_base_id_t", t.tensor(config.category_base_id or [0], dtype=t.long)
        )
        # rank -> standardised value table; see continuous_crps_z. Allocated
        # from the config's point count (not the file) so a checkpoint loads
        # with no file present, and persistent so it travels with the model.
        _nq = int(getattr(config, "value_quantile_points", 0) or 0)
        if _nq > 0:
            self.register_buffer("vq_grid", t.zeros(n_cat, _nq), persistent=True)
            self.register_buffer("vq_scale", t.ones(n_cat), persistent=True)
            _vqf = getattr(config, "value_quantile_file", None)
            if _vqf and pathlib.Path(_vqf).expanduser().exists():
                _z = np.load(pathlib.Path(_vqf).expanduser(), allow_pickle=True)
                _tc = [str(x) for x in _z["cats"]]
                assert _tc == list(config.categories), (
                    "value_quantile_file's categories do not match the model's "
                    "-- the table would scale the wrong labs, silently"
                )
                assert _z["Q"].shape[1] == _nq, (
                    f"value_quantile_points={_nq} but the table has "
                    f"{_z['Q'].shape[1]} points"
                )
                with t.no_grad():
                    self.vq_grid.copy_(t.tensor(_z["Q"], dtype=t.float32))
                    self.vq_scale.copy_(t.tensor(_z["scale"], dtype=t.float32))

        # per-category active-slot masks: see BasisBlendedConfig.
        # k_per_category. Validated here, but BUILT LAZILY in
        # _active_masks() rather than stored as buffers -- under
        # low_cpu_mem_usage HF constructs the module inside
        # init_empty_weights, so anything allocated here lands on the meta
        # device, and a mask that is missing from (or absent from) the
        # checkpoint then materialises as uninitialised memory. A garbage
        # bool mask -inf's arbitrary slots and any fully-masked row returns
        # NaN, which silently corrupted evaluation of older checkpoints.
        # The masks are a pure function of config, so deriving them on first
        # use is both correct and checkpoint-version-proof -- the same
        # reasoning (and the same __dict__ cache) as _interval_quadrature.
        k_per_cat = getattr(config, "k_per_category", None)
        if k_per_cat and not config.numerical_basis_model:
            assert len(k_per_cat) == n_cat, (
                f"k_per_category has {len(k_per_cat)} entries but there are "
                f"{n_cat} categories"
            )
            for c, kc in enumerate(k_per_cat):
                assert 1 <= int(kc) <= config.k, (
                    f"k_per_category[{c}]={kc} outside [1, k={config.k}]"
                )

        self.post_init()
        self._share_basis_embed_init()
        self._init_output_side()

    # jitter as a fraction of the embedding init std -- see
    # BasisBlendedConfig.share_basis_embed_init. Small enough that the
    # blend's norm is still e_c's to within a few percent
    # (the jitter contributes ~JITTER_FRAC/sqrt(k_eff) of it), large enough
    # that no two slots start bit-identical.
    BASIS_EMBED_JITTER_FRAC = 0.1

    # every output-side copy, paired with the input-side parameter it is
    # seeded from. Which ones exist depends on untie_continuous_head /
    # decoupled_magnitude / bspline_weights, so they are looked up rather
    # than referenced. basis_out_embed is seeded from the embedding TABLE
    # rather than a parameter, so it is handled separately below.
    _OUTPUT_SIDE_PAIRS = (
        "log_alpha",
        "log_beta",
        "log_importance",
        "knot_logits",
        "basis_log_magnitude",
    )

    def _init_output_side(self):
        """
        Seed each output-side copy from its input-side counterpart so an
        untied run STARTS identical to the tied model and any divergence is
        learned.

        Called from TWO places, and both are load-bearing:
          * __init__, after _share_basis_embed_init rewrites the input basis
            rows -- the copy has to see the final values;
          * _init_weights, so a checkpoint saved before one of these params
            existed reloads it as the intended seed instead of uninitialised
            memory. This is the missing-key recovery log_importance and
            category_embed already get and the _out params never had: the
            previous version ran only from __init__ and returned early on the
            meta device, i.e. in exactly the low_cpu_mem_usage load path where
            the recovery was needed.

        Each param is skipped if the loader already filled it (the
        _is_hf_initialized convention), so a trained checkpoint is never
        clobbered, and the method is idempotent -- being reached from both
        sites during one construction just re-copies the same values.
        """
        if not getattr(self.config, "untie_output_basis", False):
            return
        if self.config.numerical_basis_model:
            return
        n_cat = max(len(self.config.categories), 1)
        k = self.config.k
        with t.no_grad():
            out = getattr(self, "basis_out_embed", None)
            if (
                out is not None
                and not out.is_meta
                and not getattr(out, "_is_hf_initialized", False)
            ):
                emb = self.get_input_embeddings()
                if emb.weight.is_meta:
                    # the input table is still on the meta device, so there is
                    # nothing to copy FROM. A finite random table is the wrong
                    # seed but a recoverable one; uninitialised memory is
                    # neither, and that is what this branch replaces.
                    out.normal_(
                        0.0, float(self.base_config_get("initializer_range", 0.02))
                    )
                else:
                    start = int(self.category_base_id_t.min())
                    out.copy_(emb.weight[start : start + n_cat * k].detach())
            for name in self._OUTPUT_SIDE_PAIRS:
                src = getattr(self, name, None)
                dst = getattr(self, f"{name}_out", None)
                if src is None or dst is None or src.is_meta or dst.is_meta:
                    continue
                if getattr(dst, "_is_hf_initialized", False):
                    continue
                dst.copy_(src.detach())

    def _value_vocab(self):
        """Merged value entries, loaded and cached on first use."""
        vv = self.__dict__.get("_value_vocab_cache")
        if vv is not None:
            return vv
        path = getattr(self.config, "value_vocab_file", None)
        if not path:
            return None
        blob = json.loads(pathlib.Path(path).read_text())
        v = blob["vocab"]
        lo, hi, cat_of, start, size = [], [], [], [], []
        for ci, c in enumerate(self.config.categories):
            e = v.get(c, {"lo": [], "hi": []})
            start.append(len(lo))
            size.append(len(e["lo"]))
            lo += e["lo"]
            hi += e["hi"]
            cat_of += [ci] * len(e["lo"])
        vv = dict(
            lo=t.tensor(lo, dtype=t.float32),
            hi=t.tensor(hi, dtype=t.float32),
            cat_of=t.tensor(cat_of, dtype=t.long),
            start=t.tensor(start, dtype=t.long),
            size=t.tensor(size, dtype=t.long),
            V=len(lo),
            meta=blob["meta"],
        )
        self.__dict__["_value_vocab_cache"] = vv
        return vv

    def _entry_of(self, vv, cat_ids, ranks):
        """
        Observed (category, rank) -> global entry index. Entries tile [0,1]
        per category, so searchsorted on their upper edges is the lookup; the
        clamp implements the project's existing out-of-range convention
        (above the max -> the max, below the min -> the min).
        """
        out = t.zeros_like(cat_ids)
        for ci in cat_ids.unique():
            m = cat_ids == ci
            s = int(vv["start"][ci])
            n = int(vv["size"][ci])
            if n == 0:
                continue
            edges = vv["hi"][s : s + n].to(ranks.device).to(ranks.dtype).contiguous()
            j = t.searchsorted(edges, ranks[m].contiguous())
            out[m] = s + j.clamp(0, n - 1)
        return out

    def _g_at(self, z, W, s, G):
        """g = m(r) * <h,u(r)> at each grid point.

        z (N,k) the per-position projections <h, delta_c,i>; W (N,R,k) the
        blend weights at the grid; s (N,k) the magnitude scalars; G (N,k,k)
        the deviation Gram. Returns (N,R)."""
        num = t.einsum("nrk,nk->nr", W, z)
        m = t.einsum("nrk,nk->nr", W, s)
        wGw = t.einsum("nrk,nkj,nrj->nr", W, G, W).clamp_min(_MAGNITUDE_EPS**2)
        return m * num / wGw.sqrt()

    def _continuous_logprob(self, hidden, category_ids, ranks, rank_widths):
        """log P(r in [lo,hi] | h) under the tied continuous head.

        Shape (B, T), column 0 unused -- the value at position i is scored
        from the hidden state at i-1, the same contract as _value_logprob.

        Everything is computed on NUMERIC positions only: the gathered
        (N, R, k) weight tensor is the memory high-water mark here, and the
        numeric positions are a minority of a batch.

        Returns a dict with "logprob" always, plus "crps" (config.
        continuous_crps) and "w1"/"mean" (config.continuous_moments), each
        None when its flag is off. A dict rather than a widening tuple
        because this function has TWO returns -- the zero-numeric early exit
        below and the main one -- and a tuple that drifted between them would
        only fail on a batch with no numeric positions, i.e. thousands of
        steps into a run.
        """
        dec = bool(getattr(self.config, "decoupled_magnitude", False))
        want_crps = bool(getattr(self.config, "continuous_crps", False))
        want_mom = bool(getattr(self.config, "continuous_moments", False))
        want_z = bool(getattr(self.config, "continuous_crps_z", False))
        side = self._curve_side()
        B, T, _ = hidden.shape
        out = hidden.new_zeros(B, T, dtype=t.float32)

        def _slot(flag):
            return hidden.new_zeros(B, T, dtype=t.float32) if flag else None

        crps_out, w1_out, mean_out = _slot(want_crps), _slot(want_mom), _slot(want_mom)
        crpsz_out = _slot(want_z)
        cat_t = category_ids[:, 1:]
        sel = cat_t >= 0
        if not bool(sel.any()):
            return {
                "logprob": out,
                "crps": crps_out,
                "w1": w1_out,
                "mean": mean_out,
                "crps_z": crpsz_out,
            }
        h = hidden[:, :-1][sel].float()  # (N,H)
        cat = cat_t[sel]  # (N,)
        r_t = ranks[:, 1:][sel].float()
        k = int(self.config.k)
        film = bool(getattr(self.config, "film_value_embed", False))
        if film:
            # FiLM curve: e_c(r) = gamma_c * MLP(r) + beta_c, so
            #   g(r) = h . e_c(r)
            #        = (h * gamma_c) . (W2 relu(W1 r + b1) + b2) + h . beta_c
            # and BOTH r-independent pieces -- the beta term and the MLP's
            # output bias -- cancel in the normalisation, exactly as
            # h . e_hat_c does for the blend. What is left is
            #   g(r) = u . relu(w1 r + b1),   u = W2^T (h * gamma_c)
            # which is evaluated in D dimensions rather than H: at the real
            # settings that is a (N,R,64) tensor instead of (N,R,1024).
            gam, _bet, _e_cat = self._film_gamma_beta(cat)
            u = t.einsum("nh,hd->nd", (h * gam.float()), self.film_out.weight.float())
            E = act = z = None
            G = s = None
        elif side == "out":
            # the untied ablation: the curve is built from the head's own
            # anchor table, not the input embedding. See
            # untie_continuous_head.
            E = self._out_basis_table()[cat].float()  # (N,k,H)
        else:
            emb = self.get_input_embeddings().weight
            base = self.category_base_id_t[cat].unsqueeze(-1) + t.arange(
                k, device=h.device
            )
            E = emb[base].float()  # (N,k,H)
        if not film:
            basis_active, _ = self._active_masks(h.device)
            act = basis_active[cat].unsqueeze(-1).float()  # (N,k,1)
            # Centring is valid ONLY because the blend weights sum to one, which
            # makes h.ebar_c an r-independent constant that cancels in the
            # normalisation. Under poly_curve_basis sum_j T_j(r) is a function of
            # r, so subtracting the centroid would inject an r-dependent error
            # straight into g. Skip it there.
            if getattr(self.config, "poly_curve_basis", False):
                ebar = t.zeros_like(E[:, 0])
                D = E * act
            else:
                ebar = (E * act).sum(1) / act.sum(1).clamp_min(1.0)  # (N,H)
                D = (E - ebar.unsqueeze(1)) * act  # (N,k,H), inactive slots zeroed
            z = t.einsum("nkh,nh->nk", D, h)  # the k projections
            if dec:
                G = t.einsum("nkh,njh->nkj", D, D)
                s = nn.functional.softplus(
                    self._side_param("basis_log_magnitude", side)[cat].float()
                ) * act.squeeze(-1)
            else:
                # PLAIN BLEND: e_c(r) = e_hat_c + sum_i w_i e_c,i, so
                #   g(r) = h.e_hat_c + sum_i w_i z_i
                # and the first term is constant in r, cancelling in the
                # normalisation. g is then LINEAR in the weights -- no
                # direction normalisation, no magnitude channel, no Gram
                # matrix. Centring z costs nothing (the weights sum to one,
                # so it subtracts a constant) and keeps g near zero, which
                # suits the logit bound.
                G = None
                s = None

        # ---- interval probability on a SHARED node set ----
        # Partition [0,1] into [0,lo] u [lo,hi] u [hi,1] and quadrature each
        # segment, then take Z over ALL nodes and the numerator over the
        # middle segment's nodes only. Because every term is positive and the
        # numerator's terms are a SUBSET of the denominator's, P <= 1 holds
        # exactly -- in floating point, for any rule, however badly it
        # resolves the integrand. Computing the two with independent rules
        # (as an earlier version did) makes P <= 1 an accident of accuracy,
        # and training found that out: it learned a spike narrower than a
        # panel, so Z was under-estimated and -log P went negative.
        if rank_widths is not None:
            half = rank_widths[:, 1:][sel].float() / 2
        else:
            half = t.zeros_like(r_t)
        lo = (r_t - half).clamp(0.0, 1.0)
        hi = (r_t + half).clamp(0.0, 1.0)
        # A zero-width interval (value outside the training range, ~0.5% of
        # numeric positions) would make the numerator -inf. Widen it to the
        # SAME floor Loss uses for the mixture path -- Loss.INTERVAL_WIDTH_EPS,
        # see loss.py:177 and the clamp_min at loss.py:444. An earlier version
        # picked 1e-4 independently, a 200x wider window, which handed this
        # head 0.96 nats per degenerate position over the mixture arms (10.96
        # vs 11.92) and inflated eval_loss by 0.0033 nats/token -- 4% of the
        # measured gap. It also silently corrupted the logged
        # interval_skill_nats, because loss.py computes base_rate_nats from
        # ITS width while this branch scored a different one. Keep the two
        # numbers equal; if loss.py's floor ever moves, move this with it.
        deg = (hi - lo) <= 0
        if bool(deg.any()):
            eps = _INTERVAL_WIDTH_EPS
            lo = t.where(deg, (r_t - eps).clamp(0.0, 1.0), lo)
            hi = t.where(deg, (r_t + eps).clamp(0.0, 1.0), hi)

        Po = int(getattr(self.config, "continuous_quad_panels", 8))
        Pi = int(getattr(self.config, "continuous_interval_panels", 8))
        Q = int(getattr(self.config, "continuous_quad_nodes", 8))
        zeros = t.zeros_like(lo)
        ones = t.ones_like(hi)
        nA, wA = self._segment_rule(zeros, lo, Po, Q)
        nB, wB = self._segment_rule(lo, hi, Pi, Q)
        nC, wC = self._segment_rule(hi, ones, Po, Q)
        pts = t.cat([nA, nB, nC], dim=1)
        logw = t.cat([wA, wB, wC], dim=1)
        if film:
            w1 = self.film_in.weight.view(1, 1, -1).float()
            b1 = self.film_in.bias.view(1, 1, -1).float()
            feats = t.relu(pts.unsqueeze(-1) * w1 + b1)  # (N,R,D)
            g = t.einsum("nrd,nd->nr", feats, u)
        else:
            W = self._blend_weights_rowwise(cat, pts, side).float()
            g = self._g_at(z, W, s, G) if dec else t.einsum("nrk,nk->nr", W, z)
        bound = float(getattr(self.config, "continuous_logit_bound", 8.0))
        if bound > 0:
            g = bound * t.tanh(g / bound)
        gw = g + logw
        log_Z = t.logsumexp(gw, dim=-1)
        a, bnd = nA.shape[1], nA.shape[1] + nB.shape[1]
        log_num = t.logsumexp(gw[:, a:bnd], dim=-1)
        # <= 0 by the subset argument above; the clamp is a belt-and-braces
        # guard against a floating-point tie, not the mechanism.
        lp = (log_num - log_Z).clamp(max=0.0)

        res = t.zeros(B, T - 1, device=h.device, dtype=t.float32)
        res[sel] = lp
        out[:, 1:] = res

        if want_crps or want_mom or want_z:
            # Every statistic below is taken over the SAME nodes and weights
            # as Z, so q sums to one exactly and they all describe the density
            # log P scores -- not some other distribution fitted alongside it.
            q = (gw - log_Z.unsqueeze(-1)).exp()
            # E_q|X - r|. This is W1(p, delta_r), the Wasserstein-1 distance
            # between the predicted density and the empirical one-hot at the
            # observed rank, AND the first half of the CRPS identity below --
            # which is why continuous_moments costs almost nothing.
            e_abs = (q * (pts - r_t.unsqueeze(-1)).abs()).sum(-1)

            def _scatter(v, dest):
                r_ = t.zeros(B, T - 1, device=h.device, dtype=t.float32)
                r_[sel] = v
                dest[:, 1:] = r_

            if want_crps:
                # CRPS(F, r) = E|X - r| - E|X - X'|/2, with X, X' ~ p(.|h)
                # iid. The node set is sorted within each row ([0,lo],
                # [lo,hi], [hi,1], each ascending), which turns the O(R^2)
                # pair term into a cumsum:
                #   E|X - X'| = 2 sum_j q_j x_j (2 C_<j + q_j - 1),
                # C_<j the mass strictly below node j. That identity uses
                # sum q = 1; any rounding along the all-ones direction is
                # projected out by the softmax Jacobian, so gradients are
                # unaffected.
                below = q.cumsum(-1) - q
                e_pair = 2.0 * (q * pts * (2.0 * below + q - 1.0)).sum(-1)
                _scatter(e_abs - 0.5 * e_pair, crps_out)
            if want_mom:
                # the MEAN, not the squared error: this is a statistic of the
                # density like beta_a/beta_b, the squaring is the loss's job,
                # and mean - r is worth logging on its own as a calibration
                # bias. See continuous_moments.
                _scatter(e_abs, w1_out)
                _scatter((q * pts).sum(-1), mean_out)
            if want_z:
                # the SAME estimator, evaluated at v = Q_c(x) instead of x.
                # Q_c is monotone, so the nodes stay sorted and the
                # sorted-cumsum identity below is still valid -- that is why
                # this costs one interpolation rather than a new derivation.
                # See continuous_crps_z.
                self._vq_check()
                v = self._value_map(cat, pts)
                v_true = self._value_map(cat, r_t)
                e_abs_z = (q * (v - v_true.unsqueeze(-1)).abs()).sum(-1)
                below_z = q.cumsum(-1) - q
                e_pair_z = 2.0 * (q * v * (2.0 * below_z + q - 1.0)).sum(-1)
                _scatter(e_abs_z - 0.5 * e_pair_z, crpsz_out)
        return {
            "logprob": out,
            "crps": crps_out,
            "w1": w1_out,
            "mean": mean_out,
            "crps_z": crpsz_out,
        }

    def _segment_rule(self, a, b, panels, nodes):
        """composite Gauss-Legendre on a PER-ROW interval [a,b].

        a, b are (N,); returns nodes (N, panels*nodes) and log weights of the
        same shape. gl_rule gives standard [-1,1] abscissae with weights
        summing to 2, so each panel maps them by x = mid + half*node and
        w = half*weight. A degenerate segment (a == b) gets log weight
        log(1e-30), i.e. contributes nothing to a logsumexp."""
        gn, gw = crps.gl_rule(nodes)
        gn = gn.to(device=a.device, dtype=t.float32).view(1, 1, nodes)
        gw = gw.to(device=a.device, dtype=t.float32).view(1, 1, nodes)
        frac = t.linspace(0.0, 1.0, panels + 1, device=a.device, dtype=t.float32)
        span = (b - a).unsqueeze(-1)
        sub_lo = a.unsqueeze(-1) + span * frac[:-1].view(1, panels)
        sub_hi = a.unsqueeze(-1) + span * frac[1:].view(1, panels)
        mid = (0.5 * (sub_lo + sub_hi)).unsqueeze(-1)
        half = (0.5 * (sub_hi - sub_lo)).unsqueeze(-1)
        n = a.shape[0]
        pts = (mid + half * gn).reshape(n, panels * nodes)
        logw = (half * gw).reshape(n, panels * nodes).clamp_min(1e-30).log()
        return pts, logw

    def _legendre_basis(self, r, p):
        """shifted orthonormal Legendre on [0,1]: r (...,) -> (..., p).

        T_j(r) = sqrt(2j+1) * P_j(2r-1) by the standard three-term recurrence.
        Orthonormal under the UNIFORM measure on [0,1], which is the measure
        the rank axis carries since r is the empirical CDF."""
        x = (2.0 * r.float() - 1.0).unsqueeze(-1)
        cols = [t.ones_like(x)]
        if p > 1:
            cols.append(x)
        for n in range(1, p - 1):
            cols.append(((2 * n + 1) * x * cols[n] - n * cols[n - 1]) / (n + 1))
        P = t.cat(cols[:p], dim=-1)
        scale = t.sqrt(t.arange(p, device=r.device, dtype=P.dtype) * 2.0 + 1.0)
        return P * scale

    def _poly_mixture(
        self, cat, ranks, rank_widths, return_log_pdf, return_beta_params
    ):
        """Legendre curve coefficients, matching _mixture_weights' contract.

        Under interval_mixture_weights these are the basis AVERAGED over the
        observation's rank interval, and that average is EXACT: ceil(k/2)
        Gauss-Legendre nodes integrate a degree-(k-1) polynomial exactly, so
        none of the composite-rule accuracy argument the density path needs
        applies here.

        These are NOT mixture weights -- signed, and they do not sum to one.
        log_pdf / a / b are returned only to satisfy callers that ask for them;
        nothing downstream of poly_curve_basis should read them.
        """
        B, T = cat.shape
        k = int(self.config.k)
        r = ranks.reshape(-1).float()
        if (
            getattr(self.config, "interval_mixture_weights", False)
            and rank_widths is not None
        ):
            half = rank_widths.reshape(-1).float() / 2
            lo = (r - half).clamp(0.0, 1.0)
            hi = (r + half).clamp(0.0, 1.0)
            Q = k // 2 + 1
            gn, gw = crps.gl_rule(Q)
            gn = gn.to(device=r.device, dtype=t.float32).view(1, Q)
            gw = gw.to(device=r.device, dtype=t.float32).view(1, Q)
            mid = (0.5 * (lo + hi)).unsqueeze(-1)
            hf = (0.5 * (hi - lo)).unsqueeze(-1)
            Bv = self._legendre_basis(mid + hf * gn, k)
            den = (hf * gw).sum(-1, keepdim=True).clamp_min(1e-12)
            w = (Bv * (hf * gw).unsqueeze(-1)).sum(1) / den
            deg = (hi - lo) <= 0
            if bool(deg.any()):
                w = t.where(deg.unsqueeze(-1), self._legendre_basis(r, k), w)
        else:
            w = self._legendre_basis(r, k)
        basis_active, _ = self._active_masks(w.device)
        if not bool(basis_active.all()):
            # k_c < k simply gives that category a lower-degree curve. The
            # inactive coefficients are switched off, NOT renormalised: there
            # is no partition of unity here to preserve.
            w = w * basis_active[cat.reshape(-1)].to(w.dtype)
        w = w.reshape(B, T, k)
        out = [w]
        if return_log_pdf:
            out.append(t.zeros_like(w))
        if return_beta_params:
            out.append(self.log_alpha[cat])
            out.append(self.log_beta[cat])
        return tuple(out) if len(out) > 1 else out[0]

    def _bspline_mixture(
        self, cat, ranks, rank_widths, return_log_pdf, return_beta_params
    ):
        """B-spline blend weights, matching _mixture_weights' return contract.

        Under interval_mixture_weights the weights are the basis AVERAGED over
        the observation's rank interval rather than evaluated at its midpoint.
        The basis is piecewise polynomial of degree d, so a composite
        Gauss-Legendre rule with ceil((d+1)/2) nodes per panel is EXACT up to
        the knots it straddles; 4 panels x 4 nodes is comfortably exact for
        d <= 3 and costs a single (N,16,k) evaluation.

        log_pdf / a / b are returned only to satisfy callers that request
        them. Under bspline_weights there are no component densities: the
        weights ARE the model. log_pdf is log w, and a/b are the untouched
        stored parameters -- see BasisBlendedConfig.bspline_weights on why
        the mixture-density losses are not meaningful in this mode.
        """
        B, T = cat.shape
        flat_c = cat.reshape(-1)
        r = ranks.reshape(1, -1).squeeze(0).float()
        if (
            getattr(self.config, "interval_mixture_weights", False)
            and rank_widths is not None
        ):
            half = rank_widths.reshape(-1).float() / 2
            lo = (r - half).clamp(0.0, 1.0)
            hi = (r + half).clamp(0.0, 1.0)
            deg = (hi - lo) <= 0
            lo = t.where(deg, r, lo)
            hi = t.where(deg, r, hi)
            P_, Q_ = 4, 4
            gn, gw = crps.gl_rule(Q_)
            gn = gn.to(device=r.device, dtype=t.float32).view(1, 1, Q_)
            gw = gw.to(device=r.device, dtype=t.float32).view(1, 1, Q_)
            frac = t.linspace(0.0, 1.0, P_ + 1, device=r.device, dtype=t.float32)
            span = (hi - lo).unsqueeze(-1)
            sl = lo.unsqueeze(-1) + span * frac[:-1].view(1, P_)
            sh = lo.unsqueeze(-1) + span * frac[1:].view(1, P_)
            mid = (0.5 * (sl + sh)).unsqueeze(-1)
            hf = (0.5 * (sh - sl)).unsqueeze(-1)
            pts = (mid + hf * gn).reshape(-1, P_ * Q_)
            wt = (hf * gw).reshape(-1, P_ * Q_)
            Bs = self._bspline_basis(flat_c, pts)  # (N, P*Q, k)
            num = (Bs * wt.unsqueeze(-1)).sum(1)
            den = wt.sum(-1, keepdim=True).clamp_min(1e-12)
            w = num / den
            # a zero-width interval collapses the rule; fall back to the
            # point basis there, matching the degenerate-width convention
            # used everywhere else.
            if bool(deg.any()):
                pt = self._bspline_basis(flat_c, r.unsqueeze(-1)).squeeze(1)
                w = t.where(deg.unsqueeze(-1), pt, w)
        else:
            w = self._bspline_basis(flat_c, r.unsqueeze(-1)).squeeze(1)
        basis_active, _ = self._active_masks(w.device)
        if not bool(basis_active.all()):
            w = w * basis_active[flat_c].to(w.dtype)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        k = w.shape[-1]
        w = w.reshape(B, T, k)
        out = [w]
        if return_log_pdf:
            out.append(w.clamp_min(1e-30).log())
        if return_beta_params:
            out.append(self.log_alpha[cat])
            out.append(self.log_beta[cat])
        return tuple(out) if len(out) > 1 else out[0]

    def _bspline_knots(self, cat_idx, side="in"):
        """clamped knot vector per row: (N, k + degree + 1).

        degree+1 repeats at 0, the free interior knots, degree+1 repeats at 1.
        Interior knots come from a softmax over k-degree gaps followed by a
        cumulative sum, so they are strictly increasing and inside (0,1) for
        any value of knot_logits -- no clamping and no ordering penalty."""
        d = int(getattr(self.config, "bspline_degree", 3))
        g = t.softmax(self._side_param("knot_logits", side)[cat_idx].float(), dim=-1)
        interior = g.cumsum(-1)[..., :-1]  # (N, k-d-1), strictly increasing
        n = cat_idx.shape[0]
        z = t.zeros(n, d + 1, device=interior.device, dtype=interior.dtype)
        o = t.ones(n, d + 1, device=interior.device, dtype=interior.dtype)
        return t.cat([z, interior, o], dim=-1)

    def _bspline_basis(self, cat_idx, x, side="in"):
        """Cox-de Boor basis: cat_idx (N,), x (N,R) -> (N,R,k).

        Non-negative and summing to exactly 1 by construction, so this is a
        partition of unity with no softmax. x is clamped just inside 1 because
        the degree-0 spans are half-open [t_i, t_{i+1}) and x == 1 would fall
        outside every one of them."""
        d = int(getattr(self.config, "bspline_degree", 3))
        kn = self._bspline_knots(cat_idx, side)  # (N, K)
        K = kn.shape[-1]
        tt = kn.unsqueeze(1)  # (N,1,K)
        xx = x.float().clamp(0.0, 1.0 - 1e-6).unsqueeze(-1)  # (N,R,1)
        B = ((xx >= tt[..., :-1]) & (xx < tt[..., 1:])).to(xx.dtype)  # (N,R,K-1)
        for p_ in range(1, d + 1):
            m = K - p_ - 1  # number of basis functions at this degree
            t_i = tt[..., :m]
            t_ip = tt[..., p_ : p_ + m]
            t_ip1 = tt[..., p_ + 1 : p_ + 1 + m]
            t_i1 = tt[..., 1 : 1 + m]
            dl = (t_ip - t_i).clamp_min(1e-12)
            dr = (t_ip1 - t_i1).clamp_min(1e-12)
            # a repeated knot gives a zero-width span; its term is defined to
            # be 0, which the masks below enforce rather than the clamps.
            left = t.where(t_ip > t_i, (xx - t_i) / dl, t.zeros_like(dl))
            right = t.where(t_ip1 > t_i1, (t_ip1 - xx) / dr, t.zeros_like(dr))
            B = left * B[..., :m] + right * B[..., 1 : m + 1]
        return B

    def _blend_weights_rowwise(self, cat_idx, pts, side="in"):
        """w_c,i at a per-row grid: cat_idx (N,), pts (N,R) -> (N,R,k).

        Shape contract: component params broadcast as (N,1,k) against the
        grid, and trunc_normal_log_pdf takes x as (N,R) because it appends
        the component axis itself.

        side picks which copy of the shape parameters builds the weights:
        "in" (the default, and what every caller but the untied continuous
        head wants) or "out". Under poly_curve_basis the basis is the fixed
        Legendre system with no shape parameters at all, so side is a
        deliberate no-op there -- basis_out_embed alone unties that curve."""
        if getattr(self.config, "poly_curve_basis", False):
            return self._legendre_basis(pts, int(self.config.k))
        if getattr(self.config, "bspline_weights", False):
            return self._bspline_basis(cat_idx, pts, side)
        fam = getattr(self.config, "component_family", "beta")
        p1 = self._side_param("log_alpha", side)[cat_idx].unsqueeze(1)  # (N,1,k)
        p2 = self._side_param("log_beta", side)[cat_idx].unsqueeze(1)
        if fam == "truncnorm":
            lp = crps.trunc_normal_log_pdf(
                pts, p1.clamp(*crps.MU_CLAMP), p2.exp().clamp(*crps.SIGMA_CLAMP)
            )  # (N,R,k)
        else:
            a, b = self._beta_ab(p1, p2)
            x = pts.unsqueeze(-1).clamp(_RANK_EPS, 1 - _RANK_EPS)  # (N,R,1)
            lp = (
                (a - 1) * x.log()
                + (b - 1) * (1 - x).log()
                - (t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b))
            )
        lp = lp + self._side_param("log_importance", side)[cat_idx].unsqueeze(1)
        lp = t.nan_to_num(lp, nan=-1e4, posinf=1e4, neginf=-1e4)
        basis_active, _ = self._active_masks(lp.device)
        if not bool(basis_active.all()):
            act = basis_active[cat_idx].unsqueeze(1).expand_as(lp)
            lp = lp.masked_fill(~act, float("-inf"))
        return t.softmax(lp, dim=-1)

    def _category_logz(self, hidden, chunk=2048):
        """log Z_c = log int_0^1 exp(h . e_c(x)) dx for EVERY category.

        hidden (B,T,H) -> (B,T,n_cat). No shift: position t's value is built
        from hidden[t] and lands in logits[t], which predicts token t+1, so
        the loss's existing shift handles the alignment.

        Every position is scored against every category because the
        normaliser N = sum_v exp(h.e_v) + sum_c Z_c is needed wherever a next
        token is predicted, whether or not that token is numeric.

        Unlike _continuous_logprob this uses a FIXED grid on [0,1]: the
        per-row [0,lo]u[lo,hi]u[hi,1] partition exists to make the observed
        category's interval probability exact, which the normaliser does not
        need. That makes the blend weights a (n_cat,R,k) tensor built once
        rather than per position.

        Chunked over positions: the (positions, n_cat, R) intermediate is the
        peak allocation here (~600MB unchunked at B=16, T=1024, R=64) and it
        is reduced away immediately by the logsumexp.
        """
        cfg = self.config
        n_cat = max(len(cfg.categories), 1)
        k = int(cfg.k)
        dev = hidden.device
        side = self._curve_side()
        if side == "out":
            E = self._out_basis_table().float()  # (n_cat,k,H)
        else:
            emb = self.get_input_embeddings().weight
            start = int(self.category_base_id_t.min())
            E = emb[start : start + n_cat * k].view(n_cat, k, -1).float()
        basis_active, _ = self._active_masks(dev)
        act = basis_active.unsqueeze(-1).float()  # (n_cat,k,1)
        # same centring rule as _continuous_logprob: valid because the blend
        # weights sum to one, which is false under poly_curve_basis.
        if getattr(cfg, "poly_curve_basis", False):
            ebar = t.zeros(n_cat, E.shape[-1], device=dev, dtype=E.dtype)
            D = E * act
        else:
            ebar = (E * act).sum(1) / act.sum(1).clamp_min(1.0)  # (n_cat,H)
            D = (E - ebar.unsqueeze(1)) * act  # (n_cat,k,H)
        # WHICH MEASURE. Level 1 must integrate the SAME thing level 2
        # normalises by, or the two levels disagree about what the measure is
        # -- the exact inconsistency continuous_category_logits exists to
        # remove. So when value_vocab_curve is on, Z_c is the sum over that
        # category's entries, matching _value_logprob_curve exactly; otherwise
        # it is the quadrature integral.
        vvc = bool(getattr(cfg, "value_vocab_curve", False))
        bound = float(getattr(cfg, "continuous_logit_bound", 8.0))
        if vvc:
            W, logw, valid, _ = self._entry_tables(dev, side)  # (n_cat,Emax,*)
        else:
            Po = int(getattr(cfg, "continuous_quad_panels", 8))
            Q = int(getattr(cfg, "continuous_quad_nodes", 8))
            pts, logw = self._segment_rule(
                t.zeros(n_cat, device=dev), t.ones(n_cat, device=dev), Po, Q
            )
            W = self._blend_weights_rowwise(
                t.arange(n_cat, device=dev), pts, side
            ).float()  # (n_cat,R,k)
            valid = None

        B, T, H = hidden.shape
        flat = hidden.reshape(B * T, H).float()
        # (chunk, n_cat, R) is the peak allocation; size the chunk to it rather
        # than fixing it, since R is 192 quadrature nodes but Emax entries under
        # value_vocab_curve and the two need not match.
        R = W.shape[1]
        chunk = max(1, min(chunk, int(2e7 // max(n_cat * R, 1))))
        out = []
        for i in range(0, flat.shape[0], chunk):
            hc = flat[i : i + chunk]  # (n,H)
            z = t.einsum("ckh,nh->nck", D, hc)  # (n,n_cat,k)
            g = t.einsum("crk,nck->ncr", W, z)  # (n,n_cat,R)
            if bound > 0:
                g = bound * t.tanh(g / bound)
            g = g + logw.unsqueeze(0)
            if valid is not None:
                g = g.masked_fill(~valid.unsqueeze(0), float("-inf"))
            out.append(t.logsumexp(g, dim=-1) + hc @ ebar.t())
        return t.cat(out, dim=0).view(B, T, n_cat)

    def _value_logprob(self, hidden, category_ids, ranks):
        """
        log P2(entry of the token at position i | its category), computed from
        the hidden state at i-1. Returned shape (B, T) with column 0 unused,
        so callers index [:, 1:] exactly as they do for ranks/rank_widths.

        Low-rank: beta = h @ E_c^T costs k dot products, then every entry's
        logit is a fixed combination of those k scalars -- V*k work rather
        than V*H. No (B, T, V) tensor is ever formed; the softmax is taken
        per category over that category's entries only.
        """
        vv = self._value_vocab()
        B, T, _ = hidden.shape
        out = hidden.new_zeros(B, T)
        cat_t = category_ids[:, 1:]
        rank_t = ranks[:, 1:].to(hidden.dtype)
        h = hidden[:, :-1]
        emb = self.get_input_embeddings().weight
        fam = getattr(self.config, "component_family", "beta")
        k = self.config.k
        # Blend weights for EVERY entry in one vectorised call, not once per
        # category inside the loop. The interval-mass evaluation dominates the
        # cost, and 151 small launches of it ran ~3x slower than the interval
        # objective end to end (4.3 it/s vs ~13). B does not depend on the
        # batch, so it is built once per forward and sliced per category.
        cat_all = vv["cat_of"].to(hidden.device)
        lo_all = vv["lo"].to(hidden.device)
        hi_all = vv["hi"].to(hidden.device)
        la_all = self.log_alpha[cat_all]
        lb_all = self.log_beta[cat_all]
        if fam == "truncnorm":
            p1 = la_all.clamp(*crps.MU_CLAMP)
            p2 = lb_all.exp().clamp(*crps.SIGMA_CLAMP)
            lm_all = crps.trunc_normal_interval_log_mass(p1, p2, lo_all, hi_all)
        else:
            a_all, b_all = self._beta_ab(la_all, lb_all)
            nodes, glw = self._interval_quadrature(hidden.device, a_all.dtype)
            lm_all = crps.beta_component_interval_log_mass(
                a_all, b_all, lo_all, hi_all, nodes, glw, 1e-6
            )
        B_all = t.softmax(
            t.nan_to_num(
                lm_all + self.log_importance[cat_all], nan=-1e4, posinf=1e4, neginf=-1e4
            ),
            dim=-1,
        )
        for ci in cat_t[cat_t >= 0].unique():
            ci = int(ci)
            s = int(vv["start"][ci])
            n = int(vv["size"][ci])
            if n <= 1:
                continue  # nothing to predict: a single entry is P=1
            m = cat_t == ci
            if not bool(m.any()):
                continue
            Bw = B_all[s : s + n]
            base = int(self.category_base_id_t[ci])
            E_c = emb[base : base + k].to(hidden.dtype)
            # Chunk over positions when a category is wide enough that the
            # (n_sel, V_c) logit block would be large. With the unmerged
            # vocabulary one category has 539,262 entries, so a batch that
            # happened to concentrate there could otherwise allocate tens of
            # GB. 4e7 elements ~ 160 MB fp32 per chunk.
            hm = h[m]
            local = self._entry_of(vv, cat_t[m], rank_t[m]) - s
            Bt = Bw.to(hidden.dtype).t()
            step = max(1, int(4e7 // max(n, 1)))
            vals = []
            for i0 in range(0, hm.shape[0], step):
                lg = (hm[i0 : i0 + step] @ E_c.t()) @ Bt
                lp = t.log_softmax(lg.float(), dim=-1)
                vals.append(
                    lp.gather(
                        1, local[i0 : i0 + step].clamp(0, n - 1).unsqueeze(1)
                    ).squeeze(1)
                )
            out[:, 1:][m] = t.cat(vals).to(out.dtype)
        return out

    def _entry_tables(self, device, side):
        """(Wpad, logw_pad, valid, mid_pad) for every category, padded to the
        widest one. Built per forward, not cached: the blend weights depend on
        log_alpha/log_beta, which train.

        Wpad (n_cat, Emax, k) is the curve evaluated at every entry midpoint,
        in ONE _blend_weights_rowwise call rather than one per category. Padded
        slots carry a harmless midpoint and are masked out by `valid`.
        """
        vv = self._value_vocab()
        n_cat = max(len(self.config.categories), 1)
        sizes = vv["size"].to(device)
        emax = int(sizes.max())
        lo = vv["lo"].to(device).float()
        hi = vv["hi"].to(device).float()
        start = vv["start"].to(device)
        ar = t.arange(emax, device=device).unsqueeze(0)  # (1,Emax)
        valid = ar < sizes.unsqueeze(-1)  # (n_cat,Emax)
        gidx = (start.unsqueeze(-1) + ar).clamp(0, lo.numel() - 1)
        lo_p = t.where(valid, lo[gidx], t.zeros_like(lo[gidx]))
        hi_p = t.where(valid, hi[gidx], t.ones_like(hi[gidx]))
        mid = (0.5 * (lo_p + hi_p)).clamp(_RANK_EPS, 1.0 - _RANK_EPS)
        # NOT _INTERVAL_WIDTH_EPS (1e-6): that is the loss's floor on a
        # DEGENERATE observation interval, and applying it here inflates every
        # entry narrower than it -- 67 of 32,921 at min5/cap500, min width
        # 7.15e-07 -- stealing mass from the rest and breaking the base-rate
        # property the log w term exists for. Entry widths are strictly
        # positive by construction, so this guards log(0) and must never bind.
        logw = (hi_p - lo_p).clamp_min(1e-12).log()
        W = self._blend_weights_rowwise(
            t.arange(n_cat, device=device), mid, side
        ).float()  # (n_cat,Emax,k)
        return W, logw, valid, mid

    def _value_logprob_curve(self, hidden, category_ids, ranks):
        """log P(observed entry | h) from the TCH curve, with NO quadrature.

        Returned (B, T) with column 0 unused, exactly like _value_logprob and
        _continuous_logprob, so the loss indexes [:, 1:] unchanged.

        For each merged entry b of category c, midpoint m_b and width w_b:

            logit_b = g_c(m_b) + log w_b,    P(b|h) = softmax_b(logit)

        See BasisBlendedConfig.value_vocab_curve for why the log w_b term is
        there and why this is exact rather than approximate.

        g is formed exactly as _continuous_logprob forms it -- projected on the
        centroid-centred anchors, then tanh-bounded -- so this arm differs from
        plain TCH by the NORMALISER alone. The centroid is constant across a
        category's entries and cancels in the softmax regardless; it is
        subtracted so the bound applies to the same quantity in both paths.

        VECTORISED ACROSS CATEGORIES. The obvious implementation loops over the
        categories present in the batch, which is what _value_logprob does --
        and its own docstring records that 151 small launches ran ~3x slower
        than the objective they replaced. Measured here: the loop took 206
        ms/step against the quadrature's 92, despite evaluating FEWER nodes
        (median 121 entries vs 192 quadrature points). Padding every category
        to the widest and masking costs one (N, Emax, k) gather instead.
        """
        vv = self._value_vocab()
        assert vv is not None, (
            "value_vocab_curve requires value_vocab_file to be set and loadable"
        )
        Bsz, T, _ = hidden.shape
        # float32, matching _continuous_logprob. hidden is bf16 in training, and
        # a bf16 log-probability carries ~8 mantissa bits: at log P ~ -3.3 that
        # is ~0.013 nats of quantisation, the order of the between-arm
        # differences this arm exists to measure. (_value_logprob still uses
        # the bf16 default -- pre-existing, and worth revisiting.)
        out = hidden.new_zeros(Bsz, T, dtype=t.float32)
        cat_t = category_ids[:, 1:]
        sel = cat_t >= 0
        if not bool(sel.any()):
            return out
        dev = hidden.device
        side = self._curve_side()
        k = int(self.config.k)
        bound = float(getattr(self.config, "continuous_logit_bound", 8.0))
        W, logw, valid, _ = self._entry_tables(dev, side)

        n_cat = max(len(self.config.categories), 1)
        if side == "out":
            table = self._out_basis_table().float()
        else:
            st0 = int(self.category_base_id_t.min())
            table = (
                self.get_input_embeddings()
                .weight[st0 : st0 + n_cat * k]
                .view(n_cat, k, -1)
                .float()
            )
        basis_active, _ = self._active_masks(dev)
        act = basis_active.unsqueeze(-1).float()  # (n_cat,k,1)
        ebar = (table * act).sum(1) / act.sum(1).clamp_min(1.0)  # (n_cat,H)
        D = (table - ebar.unsqueeze(1)) * act  # (n_cat,k,H)

        cat = cat_t[sel]  # (N,)
        h = hidden[:, :-1][sel].float()  # (N,H)
        rank = ranks[:, 1:][sel].float()
        tgt = self._entry_of(vv, cat, rank) - vv["start"].to(dev)[cat]  # (N,)
        emax = W.shape[1]
        # chunk over positions: (chunk, Emax, k) is the peak allocation
        step = max(1, int(2e7 // max(emax * k, 1)))
        vals = []
        for i0 in range(0, h.shape[0], step):
            hc, cc = h[i0 : i0 + step], cat[i0 : i0 + step]
            z = t.einsum("nkh,nh->nk", D[cc], hc)  # (chunk,k)
            g = t.einsum("nek,nk->ne", W[cc], z)  # (chunk,Emax)
            if bound > 0:
                g = bound * t.tanh(g / bound)
            g = (g + logw[cc]).masked_fill(~valid[cc], float("-inf"))
            lp = t.log_softmax(g, dim=-1)
            vals.append(
                lp.gather(
                    1, tgt[i0 : i0 + step].clamp(0, emax - 1).unsqueeze(1)
                ).squeeze(1)
            )
        out[:, 1:][sel] = t.cat(vals)
        return out

    def _beta_ab(self, p1, p2):
        """
        Stored Beta parameters -> (a, b), honouring beta_mu_kappa_param.

        Default: p1 = log a, p2 = log b.
        mu/kappa: p1 = logit mu, p2 = log kappa, a = mu*kappa, b = (1-mu)*kappa.
        Clamps are applied to (a, b) either way, so the admissible region is
        identical between the two parameterisations.
        """
        if getattr(self.config, "beta_mu_kappa_param", False):
            mu = t.sigmoid(p1)
            kappa = p2.exp().clamp(2e-3, 2e4)
            a, b = mu * kappa, (1.0 - mu) * kappa
        else:
            a, b = p1.exp(), p2.exp()
        a, b = a.clamp(1e-3, 1e4), b.clamp(1e-3, 1e4)
        if getattr(self.config, "fixed_uniform_component", False):
            # Slot 0 == Beta(1,1) exactly, as a CONSTANT: t.where against
            # ones_like means no gradient reaches p1[...,0]/p2[...,0], while
            # every other slot keeps its own gradient path untouched.
            sel = t.zeros(a.shape[-1], dtype=t.bool, device=a.device)
            sel[0] = True
            one = t.ones_like(a)
            a = t.where(sel, one, a)
            b = t.where(sel, one, b)
        return a, b

    def _output_components(self, category_ids, ranks):
        """
        The OUTPUT side's (p1, p2, log-density-at-the-true-rank), from the
        _out parameters. These become beta_a/beta_b/beta_log_pdf, i.e. the
        observation model the loss grades the head's predicted weights with.
        The input blend is untouched and still uses log_alpha/log_beta.
        """
        cat = category_ids.clamp(min=0)
        fam = getattr(self.config, "component_family", "beta")
        if fam == "truncnorm":
            p1 = self.log_alpha_out[cat].clamp(*crps.MU_CLAMP)
            p2 = self.log_beta_out[cat].exp().clamp(*crps.SIGMA_CLAMP)
            r = ranks.to(p2.dtype).clamp(_RANK_EPS, 1 - _RANK_EPS)
            return p1, p2, crps.trunc_normal_log_pdf(r, p1, p2)
        p1, p2 = self._beta_ab(self.log_alpha_out[cat], self.log_beta_out[cat])
        r = ranks.to(p1.dtype).clamp(_RANK_EPS, 1 - _RANK_EPS).unsqueeze(-1)
        log_pdf = (
            (p1 - 1) * r.log()
            + (p2 - 1) * (1 - r).log()
            - (t.lgamma(p1) + t.lgamma(p2) - t.lgamma(p1 + p2))
        )
        return p1, p2, log_pdf

    def _share_basis_embed_init(self):
        """
        Point every basis slot of a category at one shared initial row, so
        the blended numeric embedding sum_i w_i e_{c,i} = e_c has a norm
        that does not depend on k. See BasisBlendedConfig.
        share_basis_embed_init for why the default (independent rows)
        penalises large k.

        Runs at construction only. from_pretrained builds the module and
        THEN loads the checkpoint over it, so a loaded model keeps its
        trained embeddings and this is a no-op for it; under the
        meta-device load path the rows aren't materialised here at all,
        hence the is_meta bail-out.
        """
        cfg = self.config
        if not getattr(cfg, "share_basis_embed_init", False):
            return
        if cfg.numerical_basis_model or not cfg.category_base_id:
            return
        emb = self.get_input_embeddings()
        if emb is None or emb.weight.is_meta:
            return
        sigma = float(self.base_config_get("initializer_range", 0.02))
        k = int(cfg.k)
        with t.no_grad():
            for base in cfg.category_base_id:
                base = int(base)
                shared = emb.weight[base].clone()
                jitter = t.randn(
                    k,
                    emb.weight.shape[1],
                    dtype=emb.weight.dtype,
                    device=emb.weight.device,
                ) * (self.BASIS_EMBED_JITTER_FRAC * sigma)
                emb.weight[base : base + k] = shared.unsqueeze(0) + jitter

    def _decoupled_blend(self, blended, anchors, w, category_ids):
        """rescale the blend's deviation to m_c(r), the magnitude channel.

        `blended` is sum_i w_i e_c,i, and since the weights sum to one that
        equals ebar_c + sum_i w_i delta_c,i -- so the deviation is recovered
        by subtracting the centroid, no second pass over the anchors. Only
        ACTIVE slots enter the centroid: under k_per_category the unused rows
        exist in the vocabulary but are never trained, so averaging them in
        would drag ebar_c toward untouched initialisation noise. The weights
        need no such care -- _mixture_weights masks inactive slots with -inf,
        so they carry exactly zero.

        ebar_c itself is NOT returned -- see the config note. The caller adds
        e_hat_c, which is the per-category constant, so what leaves here is
        exactly the rank-carrying part at length m_c(r).

        The direction is left alone; only its length is set. sum_i delta_i = 0
        by construction, so the deviation vanishes as w approaches uniform
        and the direction is a 0/0 there -- hence the clamp, and hence
        k_eff is worth logging if train_importance_scale is ever enabled
        alongside this (importance can flatten w toward that singularity).
        See BasisBlendedConfig.decoupled_magnitude."""
        cat = category_ids.clamp(min=0)
        basis_active, _ = self._active_masks(w.device)
        act = basis_active[cat].unsqueeze(-1).to(anchors.dtype)  # (B,T,k,1)
        ebar = (anchors * act).sum(dim=2) / act.sum(dim=2).clamp_min(1.0)  # (B,T,H)
        dev = (blended - ebar).float()  # sum_i w_i delta_c,i
        s = nn.functional.softplus(self.basis_log_magnitude[cat].float())  # (B,T,k)
        m = (w.float() * s).sum(dim=-1, keepdim=True)  # (B,T,1)
        # norm in float32: a bf16 reduction over H=1024 loses enough
        # precision to show up in the rescale factor.
        den = dev.norm(dim=-1, keepdim=True).clamp_min(_MAGNITUDE_EPS)
        return (dev * (m / den)).to(blended.dtype)

    def _curve_side(self):
        """which parameters build e_c(r): "out" once the continuous head has
        its own curve, "in" otherwise. See untie_continuous_head."""
        return "out" if getattr(self.config, "untie_continuous_head", False) else "in"

    def _side_param(self, name, side):
        """log_alpha vs log_alpha_out and friends, picked by side.

        Fails loudly when an output copy is missing rather than falling back
        to the input side: a silent fallback is exactly the half-untied
        behaviour untie_continuous_head exists to remove."""
        if side == "in":
            return getattr(self, name)
        p = getattr(self, f"{name}_out", None)
        assert p is not None, (
            f"{name}_out is missing while the continuous head is untied -- the "
            "output curve would silently read the input side's parameter"
        )
        return p

    def _out_basis_table(self):
        """basis_out_embed viewed as (n_cat, k, H).

        The basis slots are one contiguous block by construction (see
        build_basis_vocab: num_non_numeric first, then k slots per category in
        order), which is what makes both this view and forward()'s logit
        splice legitimate -- assert it here so the two readers share one
        check."""
        n_cat = max(len(self.config.categories), 1)
        k = int(self.config.k)
        start = int(self.category_base_id_t.min())
        assert int(self.category_base_id_t.max()) + k == start + n_cat * k, (
            "basis slots are not contiguous; untie_output_basis assumes "
            "one block [start, start + n_cat*k)"
        )
        return self.basis_out_embed.view(n_cat, k, -1)

    def _active_masks(self, device):
        """
        (basis_active (n_cat,k), vocab_active (vocab_size,)) for
        k_per_category, built on first use and cached. See __init__ for why
        these are not buffers.
        """
        cached = self.__dict__.get("_active_masks_cache")
        if cached is not None and cached[0].device == device:
            return cached
        cfg = self.config
        n_cat = max(len(cfg.categories), 1)
        k = int(cfg.k) if cfg.k else 1
        basis = t.ones(n_cat, k, dtype=t.bool, device=device)
        vocab = t.ones(int(cfg.vocab_size), dtype=t.bool, device=device)
        k_per_cat = getattr(cfg, "k_per_category", None)
        if k_per_cat and not cfg.numerical_basis_model:
            for c, kc in enumerate(k_per_cat):
                kc = int(kc)
                if kc < k:
                    basis[c, kc:] = False
                    base = int(cfg.category_base_id[c])
                    vocab[base + kc : base + k] = False
        self.__dict__["_active_masks_cache"] = (basis, vocab)
        return basis, vocab

    def base_config_get(self, name, default=None):
        """read a field off the wrapped inner-model config dict"""
        return (self.config.base_config or {}).get(name, default)

    def _init_magnitude(self):
        """(re)fill the magnitude channel unless the loader supplied it.

        Same missing-key recovery as log_importance/category_embed, but the
        param only exists under decoupled_magnitude, so it is looked up
        rather than referenced. See BasisBlendedConfig.decoupled_magnitude."""
        p = getattr(self, "basis_log_magnitude", None)
        if p is None or getattr(p, "_is_hf_initialized", False):
            return
        p.fill_(self.config.magnitude_init(int(self.base_config_get("hidden_size", 1))))

    def _init_film(self):
        """gamma = 1 and beta = identity at init, so a fresh model's numeric
        token is e_cat + MLP(v): the concept's own row plus a value offset,
        the same additive shape numerical_basis_model and category_embed
        start from.

        nn.Linear's default init would instead start gamma at small noise
        around 0, which multiplies the whole value channel by ~0 and leaves
        the MLP with almost no gradient -- the FiLM analogue of the
        zero-init that makes e2 inert until training moves it, except that
        here it would be a trap rather than a deliberate no-op."""
        if not getattr(self.config, "film_value_embed", False):
            return
        with t.no_grad():
            g, b = self.film_gamma, self.film_beta
            if not g.weight.is_meta and not getattr(
                g.weight, "_is_hf_initialized", False
            ):
                g.weight.zero_()
                g.bias.fill_(1.0)
            if not b.weight.is_meta and not getattr(
                b.weight, "_is_hf_initialized", False
            ):
                b.weight.copy_(t.eye(b.weight.shape[0], device=b.weight.device))
                b.bias.zero_()

    def _value_map(self, cat, r):
        """rank -> standardised value, per category: r (N,) or (N,R) -> same.

        The table is a UNIFORM grid over [0,1], so the lookup is a floor plus
        a lerp rather than a searchsorted, and it is indexed flat to avoid
        materialising an (N, points) gather. See continuous_crps_z."""
        g = self.vq_grid
        nq = g.shape[1]
        flat = g.reshape(-1)
        one_d = r.dim() == 1
        x = (r.unsqueeze(-1) if one_d else r).clamp(0.0, 1.0).float() * (nq - 1)
        i0 = x.floor().long().clamp(0, nq - 2)
        frac = x - i0.to(x.dtype)
        base = (cat.unsqueeze(-1) * nq).expand_as(i0)
        v0 = flat.gather(0, (base + i0).reshape(-1)).view_as(x)
        v1 = flat.gather(0, (base + i0 + 1).reshape(-1)).view_as(x)
        v = (v0 + (v1 - v0) * frac) / self.vq_scale[cat].unsqueeze(-1)
        return v.squeeze(-1) if one_d else v

    def _vq_check(self):
        """fail loudly if the value table was never filled -- an all-zero
        table would score every prediction against a constant and look like a
        merely disappointing result rather than a bug."""
        if self.__dict__.get("_vq_ok"):
            return
        g = getattr(self, "vq_grid", None)
        assert g is not None and g.numel() > 0 and float(g.abs().max()) > 0, (
            "numeric_loss: crps_z needs a filled value-quantile table -- set "
            "basis_blended_tokens.value_quantile_file (and "
            "value_quantile_points), or load a checkpoint that carries one"
        )
        self.__dict__["_vq_ok"] = True

    def _film_gamma_beta(self, cat):
        """(gamma_c, beta_c, e_cat) for each position, from the concept's own
        embedding row -- the FiLM conditioning. cat (...,) -> three (..., H).

        The input is cast to the LINEAR's dtype, not to a fixed one: during
        training the embedding table can be bf16 while these Linears are
        fp32, and at extraction the whole model is cast to bf16 so it is the
        other way round. Either mismatch raises rather than downcasting
        silently, and both directions have been hit for real. e_cat comes
        back in its own dtype; the caller casts the finished embedding."""
        e_cat = self.get_input_embeddings()(self.category_base_id_t[cat])
        e_in = e_cat.to(self.film_gamma.weight.dtype)
        return self.film_gamma(e_in), self.film_beta(e_in), e_cat

    def _film_value_mlp(self, v):
        """MLP(v) = W2 relu(W1 v + b1) + b2: v (...,) -> (..., H).

        Computed in the MLP's own dtype, for the reason in _film_gamma_beta.
        _continuous_logprob evaluates the same curve in float32 regardless,
        by casting the weights there: g(r) is a difference of nearby curve
        values under the quadrature, which bf16's ~3 decimal digits would
        not resolve."""
        w = self.film_in.weight
        return self.film_out(t.relu(self.film_in(v.to(w.dtype).unsqueeze(-1))))

    def _init_derived(self):
        """the params that exist only under a flag, so _init_weights cannot
        reference them directly: the magnitude channel and every output-side
        copy. One call site per _init_weights return path -- adding a new
        flag-gated param means extending one of these two, not all four."""
        self._init_magnitude()
        self._init_output_side()

    def _init_weights(self, module):
        # HF's from_pretrained calls this for `self` (this whole module) any
        # time even one of its direct params is missing from the checkpoint
        # (e.g. an old checkpoint saved before log_importance/category_embed
        # existed) -- under the low_cpu_mem_usage/meta-device load path,
        # skipping this override leaves such params as uninitialized garbage
        # memory rather than the values __init__ intends. But the call is
        # per-*module*, not per-param, and log_alpha/log_beta/log_importance/
        # category_embed are 4 separate params of this one module -- so a
        # blanket overwrite here
        # would also clobber params that *were* correctly loaded (they sit
        # on the same module). Each individually-loaded param is swapped in
        # by the loader as a fresh tensor carrying `_is_hf_initialized=True`
        # (see transformers.core_model_loading.set_param_for_module); only
        # touch a param here if it's missing that flag. inner_model's own
        # layers are already fully initialized by AutoModelForCausalLM.
        # from_config before this ever runs, so only handle top-level params.
        if module is not self:
            return
        n_cat = max(len(self.config.categories), 1)
        if getattr(self.config, "film_value_embed", False):
            # FiLM addresses one vocab slot per category and has no anchors,
            # so none of the log_alpha/log_beta recovery below applies. Its
            # own conditioning init does, and missing it would reload gamma
            # as noise around 0 -- see _init_film.
            self._init_film()
            if not self.config.numerical_basis_model:
                return
        with t.no_grad():
            if self.config.numerical_basis_model:
                if not getattr(self.e2, "_is_hf_initialized", False):
                    self.e2.zero_()
                # gaussian_head's weight/bias aren't given the same
                # missing-key recovery as e2 above: there's no pre-existing
                # checkpoint format for this brand-new mode that could be
                # missing them, so (unlike log_alpha/log_beta/log_importance/
                # category_embed, all added incrementally to an
                # already-shipped mode) this gap doesn't affect any real
                # checkpoint today -- revisit if that changes.
                return
            idx = t.arange(self.config.k, dtype=t.float32)
            # component_family must be honoured HERE as well as in __init__:
            # post_init() calls this after the constructor, so a family-aware
            # __init__ alone is silently overwritten by the Beta defaults.
            if getattr(self.config, "component_family", "beta") == "truncnorm":
                if not getattr(self.log_alpha, "_is_hf_initialized", False):
                    self.log_alpha.copy_(
                        ((idx + 0.5) / self.config.k)
                        .unsqueeze(0)
                        .expand(n_cat, self.config.k)
                    )
                if not getattr(self.log_beta, "_is_hf_initialized", False):
                    self.log_beta.fill_(math.log(1.0 / self.config.k))
                if not getattr(self.log_importance, "_is_hf_initialized", False):
                    self.log_importance.zero_()
                if not getattr(self.category_embed, "_is_hf_initialized", False):
                    self.category_embed.zero_()
                self._init_derived()
                return
            if getattr(self.config, "fixed_uniform_component", False):
                kk = self.config.k
                j = t.arange(1, kk, dtype=t.float32)
                a0 = t.cat([t.ones(1), j])
                b0 = t.cat([t.ones(1), kk - j])
                if getattr(self.config, "beta_mu_kappa_param", False):
                    p1v, p2v = (a0 / b0).log(), (a0 + b0).log()
                else:
                    p1v, p2v = a0.log(), b0.log()
                if not getattr(self.log_alpha, "_is_hf_initialized", False):
                    self.log_alpha.copy_(p1v.unsqueeze(0).expand(n_cat, kk))
                if not getattr(self.log_beta, "_is_hf_initialized", False):
                    self.log_beta.copy_(p2v.unsqueeze(0).expand(n_cat, kk))
                if not getattr(self.log_importance, "_is_hf_initialized", False):
                    self.log_importance.zero_()
                if not getattr(self.category_embed, "_is_hf_initialized", False):
                    self.category_embed.zero_()
                self._init_derived()
                return
            if getattr(self.config, "beta_mu_kappa_param", False):
                # same components, (logit mu, log kappa) coordinates
                if not getattr(self.log_alpha, "_is_hf_initialized", False):
                    self.log_alpha.copy_(
                        ((idx + 1) / (self.config.k - idx))
                        .log()
                        .unsqueeze(0)
                        .expand(n_cat, self.config.k)
                    )
                if not getattr(self.log_beta, "_is_hf_initialized", False):
                    self.log_beta.fill_(math.log(float(self.config.k + 1)))
                if not getattr(self.log_importance, "_is_hf_initialized", False):
                    self.log_importance.zero_()
                if not getattr(self.category_embed, "_is_hf_initialized", False):
                    self.category_embed.zero_()
                self._init_derived()
                return
            if not getattr(self.log_alpha, "_is_hf_initialized", False):
                self.log_alpha.copy_(
                    (idx + 1).log().unsqueeze(0).expand(n_cat, self.config.k)
                )
            if not getattr(self.log_beta, "_is_hf_initialized", False):
                self.log_beta.copy_(
                    (self.config.k - idx)
                    .log()
                    .unsqueeze(0)
                    .expand(n_cat, self.config.k)
                )
            if not getattr(self.log_importance, "_is_hf_initialized", False):
                self.log_importance.zero_()
            if not getattr(self.category_embed, "_is_hf_initialized", False):
                self.category_embed.zero_()
            self._init_derived()

    def get_input_embeddings(self):
        return self.inner_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.inner_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.inner_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.inner_model.set_output_embeddings(value)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.inner_model.prepare_inputs_for_generation(*args, **kwargs)

    def _interval_quadrature(self, device, dtype):
        """
        Gauss-Legendre rule for interval_mixture_weights, built on first use
        and cached per (device, dtype).

        Deliberately NOT constructed in __init__: from_pretrained runs
        __init__ under a meta device, so plain (non-parameter, non-buffer)
        tensors created there materialize as meta tensors and raise
        "Cannot copy out of meta tensor" on first use at extraction time.
        Building lazily sidesteps that without needing a persistent buffer
        for what is a fixed constant.

        32 points matches Loss's default, validated to <=2.4e-3 nats
        against scipy's exact incomplete beta on the widest intervals.
        """
        key = (str(device), str(dtype))
        cache = self.__dict__.setdefault("_interval_gl_cache", {})
        if key not in cache:
            nodes, weights = crps.gl_rule(
                int(getattr(self.config, "interval_quad_points", 32)), dtype=t.float32
            )
            cache[key] = (
                nodes.to(device=device, dtype=dtype),
                weights.to(device=device, dtype=dtype),
            )
        return cache[key]

    def _mixture_weights(
        self,
        category_ids: t.Tensor,
        ranks: t.Tensor,
        return_log_pdf: bool = False,
        return_beta_params: bool = False,
        rank_widths: t.Tensor | None = None,
    ) -> t.Tensor | tuple:
        """log-space softmax over the k basis elements' *importance-scaled*
        Beta densities at `ranks`: w_c,i = a_c,i*f(r,i) / sum_j a_c,j*f(r,j),
        for whatever category each position belongs to; see "Determination
        of mixture weights" in fuzzy_token_planning.md. Scaling by a_c,i in
        log-space before the softmax is exactly this ratio (softmax(x)_i =
        exp(x_i)/sum_j exp(x_j) with x = log(a) + log_pdf), so a_c,i can
        reweight each component's overall influence independently of its
        Beta shape.

        return_log_pdf=True additionally returns the *unscaled* log-density
        f(r,i) itself (no log_importance folded in) -- used by
        mixture_nll_loss (see BasisBlendedCausalLMOutput.beta_log_pdf and
        Loss.basis_blended_token_loss).

        return_beta_params=True additionally returns the clamped (a_c,i,
        b_c,i) themselves -- already computed below regardless, just not
        otherwise surfaced -- used by Loss.crps_loss to evaluate the Beta
        CDF at quadrature points other than the single true rank `ranks`
        covers (see BasisBlendedCausalLMOutput.beta_a/beta_b).

        Both default False, preserving the original single-tensor return
        for existing callers (forward()'s embedding blend, and direct
        notebook/analysis use). Return shape: `w` alone, `(w, log_pdf)`,
        `(w, a, b)`, or `(w, log_pdf, a, b)`, matching which flags are set."""
        cat = category_ids.clamp(min=0)
        if getattr(self.config, "poly_curve_basis", False):
            return self._poly_mixture(
                cat, ranks, rank_widths, return_log_pdf, return_beta_params
            )
        if getattr(self.config, "bspline_weights", False):
            return self._bspline_mixture(
                cat, ranks, rank_widths, return_log_pdf, return_beta_params
            )
        # alpha/beta are unconstrained during training and can drift into
        # ranges where lgamma is poorly conditioned (very small or very
        # large); clamp defensively rather than let that surface as NaN here.
        family = getattr(self.config, "component_family", "beta")
        if family == "truncnorm":
            # log_alpha holds mu, log_beta holds log sigma (see config).
            a = self.log_alpha[cat].clamp(*crps.MU_CLAMP)  # (B,T,k) -- mu
            b = self.log_beta[cat].exp().clamp(*crps.SIGMA_CLAMP)  # sigma
            r_pt = ranks.to(a.dtype).clamp(_RANK_EPS, 1 - _RANK_EPS)
            log_pdf = crps.trunc_normal_log_pdf(r_pt, a, b)
        else:
            # alpha/beta are unconstrained during training and can drift into
            # ranges where lgamma is poorly conditioned (very small or very
            # large); clamp defensively rather than let that surface as NaN.
            a, b = self._beta_ab(self.log_alpha[cat], self.log_beta[cat])  # (B,T,k)
            r = ranks.to(a.dtype).clamp(_RANK_EPS, 1 - _RANK_EPS).unsqueeze(-1)
            log_pdf = (
                (a - 1) * r.log()
                + (b - 1) * (1 - r).log()
                - (t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b))
            )
        log_importance = self.log_importance[cat]  # (B,T,k)
        # interval_mixture_weights: weight each component by its mass over
        # the observation's rank interval rather than its density at the
        # midpoint (see BasisBlendedConfig). log_pdf itself is left alone --
        # it is returned as beta_log_pdf for mixture_nll_loss, which wants
        # the point density.
        if getattr(self.config, "interval_mixture_weights", False):
            assert rank_widths is not None, (
                "interval_mixture_weights requires rank_widths at forward() "
                "time -- for extraction this comes from cocoa's "
                "exact_rank_widths_past; see Extractor.collate_fn"
            )
            w_half = rank_widths.to(a.dtype).unsqueeze(-1) / 2
            r_c = ranks.to(a.dtype).unsqueeze(-1)
            lo = (r_c - w_half).clamp(0.0, 1.0).squeeze(-1)
            hi = (r_c + w_half).clamp(0.0, 1.0).squeeze(-1)
            if family == "truncnorm":
                # closed form in erf -- no quadrature, no approximation error
                comp = crps.trunc_normal_interval_log_mass(a, b, lo, hi)
            else:
                nodes, gl_w = self._interval_quadrature(a.device, a.dtype)
                comp = crps.beta_component_interval_log_mass(a, b, lo, hi, nodes, gl_w)
            # width 0 (value outside the training range, ~0.2% of numeric
            # positions) leaves a degenerate interval -- fall back to the
            # point density there rather than log(0).
            degenerate = (hi - lo).unsqueeze(-1) <= 0
            log_weight_basis = t.where(degenerate, log_pdf, comp)
        else:
            log_weight_basis = log_pdf
        logits = log_weight_basis + log_importance
        # last-resort sanitation: never let a NaN/inf reach the softmax,
        # whatever its cause -- this is a plain terminal function of finite
        # inputs, so any non-finite value here is a bug, not real signal.
        logits = t.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        # switch off slots this category may not use. Applied AFTER
        # nan_to_num on purpose: -inf here survives to the softmax and gives
        # those slots exactly zero weight, whereas masking first would be
        # rewritten to -1e4 and leak a small but nonzero blend contribution.
        basis_active, _ = self._active_masks(logits.device)
        if not bool(basis_active.all()):
            logits = logits.masked_fill(~basis_active[cat], float("-inf"))
        w = t.softmax(logits, dim=-1)
        out = [w]
        if return_log_pdf:
            out.append(log_pdf)
        if return_beta_params:
            out.append(a)
            out.append(b)
        return tuple(out) if len(out) > 1 else out[0]

    def forward(
        self,
        input_ids: t.Tensor,
        category_ids: t.Tensor | None = None,
        ranks: t.Tensor | None = None,
        rank_widths: t.Tensor | None = None,
        position_ids: t.Tensor | None = None,
        attention_mask: t.Tensor | None = None,
        output_hidden_states: bool = False,
        labels=None,  # unused: loss is computed externally by cotorra.loss
        inputs_embeds=None,  # unused/ignored -- see comment below
        **kwargs,
    ) -> BasisBlendedCausalLMOutput:
        # inputs_embeds must be captured as a named parameter (not left to
        # fall through into **kwargs) even though this model never uses it:
        # this model always builds its own embeddings from input_ids (it
        # needs the raw ids to look up category/numeric status), so
        # embed(input_ids) below is unconditional regardless of what's
        # passed here. But recent transformers versions include an
        # inputs_embeds key (typically None) in the dict
        # prepare_inputs_for_generation builds during .generate()'s prefill
        # step -- if that key isn't consumed by a named parameter here, it
        # rides through in **kwargs and collides with the explicit
        # inputs_embeds=base_embeds passed to self.inner_model below
        # ("got multiple values for keyword argument 'inputs_embeds'").
        embed = self.get_input_embeddings()
        base_embeds = embed(input_ids)

        if category_ids is None:
            category_ids = t.full_like(input_ids, -1)
        numeric = category_ids >= 0

        w = None
        log_pdf = None
        beta_a = None
        beta_b = None
        if numeric.any():
            if ranks is None:
                ranks = t.zeros_like(input_ids, dtype=base_embeds.dtype)
            if getattr(self.config, "film_value_embed", False):
                # FiLM: the concept picks an affine transform and applies it
                # to a learned function of the value.
                #   e = gamma_c * MLP(v) + beta_c,   NaN v -> e_cat
                # See BasisBlendedConfig.film_value_embed.
                cat = category_ids.clamp(min=0)
                gam, bet, e_cat = self._film_gamma_beta(cat)
                v = ranks.to(self.film_in.weight.dtype)
                blended = (gam * self._film_value_mlp(v) + bet).to(e_cat.dtype)
                # a missing value carries no information to modulate with, so
                # the token falls back to the concept embedding alone.
                blended = t.where(t.isnan(v).unsqueeze(-1), e_cat, blended)
            elif self.config.numerical_basis_model:
                # numerical_basis_model reproduction: e = e_c,1 + v*e_c,2 --
                # e_c,1 is the category's single tied vocab-slot embedding
                # (category_base_id_t here holds exactly one id per
                # category, from build_gaussian_vocab, unlike the k-wide
                # basis_ids below), e_c,2 the untied "slope" parameter, v
                # the position's normalized value (whatever rank_column was
                # configured). See __init__ and fuzzy_token_planning.md
                # point 18.
                cat = category_ids.clamp(min=0)
                e1 = embed(self.category_base_id_t[cat])  # (B,T,H), tied
                xv = getattr(self.config, "xval_head", False)
                e2 = self.e2[t.zeros_like(cat) if xv else cat].to(e1.dtype)
                v = ranks.to(e1.dtype).unsqueeze(-1)
                if xv:
                    v = v * float(getattr(self.config, "xval_value_scale", 1.0))
                blended = e1 + v * e2
            else:
                w, log_pdf, beta_a, beta_b = self._mixture_weights(
                    category_ids,
                    ranks,
                    return_log_pdf=True,
                    return_beta_params=True,
                    rank_widths=rank_widths,
                )  # (B,T,k), (B,T,k), (B,T,k), (B,T,k)
                if getattr(self.config, "untie_output_basis", False):
                    # w stays as computed above -- the INPUT blend still
                    # derives its weights from log_alpha/log_beta. Only the
                    # tensors the loss uses to SCORE the head's prediction
                    # are swapped to the _out parameters.
                    beta_a, beta_b, log_pdf = self._output_components(
                        category_ids, ranks
                    )
                k_dim = w.shape[-1]
                basis_ids = self.category_base_id_t[
                    category_ids.clamp(min=0)
                ].unsqueeze(-1) + t.arange(k_dim, device=input_ids.device)  # (B,T,k)
                anchors = embed(basis_ids)  # (B,T,k,H)
                blended = (w.unsqueeze(-1) * anchors).sum(dim=2)
                if getattr(self.config, "decoupled_magnitude", False):
                    blended = self._decoupled_blend(blended, anchors, w, category_ids)
                cat_embed = self.category_embed[category_ids.clamp(min=0)]  # (B,T,H)
                blended = blended + cat_embed.to(blended.dtype)
            base_embeds = t.where(
                numeric.unsqueeze(-1), blended.to(base_embeds.dtype), base_embeds
            )

        # numerical_basis_model needs the transformer's own final hidden
        # state to feed gaussian_head (the same state the LM head reads for
        # next-token logits) -- force it internally regardless of what the
        # caller asked for, then strip it back out of the returned object
        # below if the caller didn't request it, preserving the normal
        # output_hidden_states contract for callers.
        untie = getattr(self.config, "untie_output_basis", False) and not (
            self.config.numerical_basis_model
        )
        use_vv = bool(getattr(self.config, "value_vocab_file", None)) and (
            category_ids is not None and ranks is not None
        )
        use_tch = getattr(self.config, "tied_continuous_head", False) and (
            category_ids is not None and ranks is not None
        )
        use_ccl = bool(getattr(self.config, "continuous_category_logits", False))
        need_hidden = (
            self.config.numerical_basis_model
            or untie
            or use_vv
            or use_tch
            or use_ccl  # needs h at EVERY position, targets or not
        )
        outputs = self.inner_model(
            inputs_embeds=base_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or need_hidden,
            **kwargs,
        )

        gaussian_params = None
        # guard on the MODE, not on need_hidden -- untie_output_basis also
        # forces hidden states, but has no gaussian_head to feed.
        if self.config.numerical_basis_model:
            # cast to float32 for the same numerical-stability reason
            # logits/beta_log_pdf are cast to float32 downstream in
            # loss.py -- this head's output feeds a Gaussian NLL directly.
            last_hidden = outputs.hidden_states[-1].to(dtype=t.float32)
            gaussian_params = self.gaussian_head(last_hidden)  # (B,T,2*n_cat)

        outputs_dict = dict(outputs.items())
        use_vvc = bool(getattr(self.config, "value_vocab_curve", False))
        if use_vv:
            # log P2(value entry | category) for each target position, from
            # the hidden state one step earlier. Column 0 is unused so callers
            # index [:, 1:] exactly as for ranks/rank_widths.
            outputs_dict["value_logprob"] = (
                self._value_logprob_curve(
                    outputs.hidden_states[-1], category_ids, ranks
                )
                if use_vvc
                else self._value_logprob(outputs.hidden_states[-1], category_ids, ranks)
            )
        # under value_vocab_curve the head's own normaliser is REPLACED by the
        # value-vocabulary softmax, so continuous_logprob must not be emitted:
        # Loss prefers it over value_logprob and would silently score the
        # quadrature path instead, i.e. train plain TCH under this arm's name.
        if use_tch and not use_vvc:
            ch = self._continuous_logprob(
                outputs.hidden_states[-1], category_ids, ranks, rank_widths
            )
            outputs_dict["continuous_logprob"] = ch["logprob"]
            for _key, _field in (
                ("crps", "continuous_crps"),
                ("w1", "continuous_w1"),
                ("mean", "continuous_mean"),
                ("crps_z", "continuous_crps_z"),
            ):
                if ch[_key] is not None:
                    outputs_dict[_field] = ch[_key]
        if untie and outputs_dict.get("logits") is not None:
            # Replace the tied head's basis columns with ones read off
            # basis_out_embed. Only [start, start+n_cat*k) is touched, so
            # every non-numeric token keeps the tied weight it always had.
            # The basis slots are one contiguous block by construction (see
            # build_basis_vocab: num_non_numeric first, then k slots per
            # category in order), which is what makes a slice legitimate --
            # _out_basis_table asserts it, for this reader and for the untied
            # curve alike.
            n_cat = max(len(self.config.categories), 1)
            k = self.config.k
            start = int(self.category_base_id_t.min())
            h = outputs.hidden_states[-1]
            w_out = self._out_basis_table().flatten(0, 1).to(h.dtype)
            lg = outputs_dict["logits"]
            basis_logits = t.matmul(h, w_out.t()).to(lg.dtype)
            lg = t.cat(
                [lg[..., :start], basis_logits, lg[..., start + n_cat * k :]], dim=-1
            )
            outputs_dict["logits"] = lg
        if need_hidden and not output_hidden_states:
            outputs_dict.pop("hidden_states", None)

        if use_ccl and outputs_dict.get("logits") is not None:
            # LEVEL 1 FROM THE CONTINUOUS MEASURE. Overwrite each category's k
            # logits with log Z_c - log k_active_c, so that logsumexp over
            # them is exactly log Z_c and the ordinary log_softmax over the
            # whole vocabulary normalises the categories against the
            # non-numeric tokens with one N. loss.py is unchanged: its
            # logsumexp-over-a-category's-slots becomes log P(c).
            #
            # Split EVENLY over the active slots rather than putting the mass
            # on slot 0: that keeps every live logit finite, so anything
            # downstream that still divides by them (log_w_hat) sees a uniform
            # distribution instead of -inf/nan. Inactive slots are written too
            # and then masked to -inf just below, which is why the divisor is
            # k_active_c and not k.
            n_cat = max(len(self.config.categories), 1)
            k = self.config.k
            start = int(self.category_base_id_t.min())
            lg = outputs_dict["logits"]
            logz = self._category_logz(outputs.hidden_states[-1])  # (B,T,n_cat)
            basis_active, _ = self._active_masks(lg.device)
            n_act = basis_active.sum(-1).clamp_min(1).float()  # (n_cat,)
            per = (logz - n_act.log().view(1, 1, -1)).unsqueeze(-1)
            rep = per.expand(-1, -1, -1, k).reshape(*lg.shape[:-1], n_cat * k)
            # FLOAT32, not lg.dtype. The head's logits are bf16, whose ~3
            # decimal digits quantise log Z_c by up to 0.0156 nats -- the same
            # order as the between-arm differences this model exists to
            # measure (0.003-0.09). An ordinary anchor logit is a single dot
            # product and tolerates that; log Z_c is h.ebar_c plus a
            # logsumexp, i.e. a difference of larger quantities, so it does
            # not. loss.py upcasts the logits to float32 anyway, so this only
            # moves the cast earlier -- non-numeric slots are unchanged by it.
            outputs_dict["logits"] = t.cat(
                [
                    lg[..., :start].float(),
                    rep.float(),
                    lg[..., start + n_cat * k :].float(),
                ],
                dim=-1,
            )

        # mask the LM head over slots no category may use. The loss takes a
        # log_softmax over the FULL collapsed vocabulary before gathering a
        # category's k slots, so an unmasked dead slot would still absorb
        # probability -- and worse, would enter log_z_c, the "next token is
        # one of category c's slots" term. -inf makes those slots exactly
        # zero-probability everywhere downstream (loss, extraction,
        # generation) from one place, rather than each caller re-deriving it.
        # Non-numeric slots are always active, so no row is fully masked.
        lg = outputs_dict.get("logits")
        if lg is not None:
            _, vocab_active = self._active_masks(lg.device)
            if not bool(vocab_active.all()):
                outputs_dict["logits"] = lg.masked_fill(~vocab_active, float("-inf"))

        return BasisBlendedCausalLMOutput(
            **outputs_dict,
            mixture_weights=w,
            beta_log_pdf=log_pdf,
            gaussian_params=gaussian_params,
            beta_a=beta_a,
            beta_b=beta_b,
        )


AutoConfig.register("basis_blended", BasisBlendedConfig, exist_ok=True)
AutoModelForCausalLM.register(BasisBlendedConfig, BasisBlendedCausalLM, exist_ok=True)


if __name__ == "__main__":
    import tempfile

    from omegaconf import OmegaConf

    # ---- build_basis_vocab: synthetic 2-category, n_bins=4 fused vocab ----
    tkzr_cfg = OmegaConf.create(
        {
            "cfg": {"n_bins": 4},
            "lookup": {
                "UNK": 0,
                "BOS": 1,
                "EOS": 2,
                "AGE//age_Q0": 3,
                "AGE//age_Q1": 4,
                "AGE//age_Q2": 5,
                "AGE//age_Q3": 6,
                "VTL//hr_Q0": 7,
                "VTL//hr_Q1": 8,
                "VTL//hr_Q2": 9,
                "VTL//hr_Q3": 10,
            },
        }
    )
    k = 3
    bv = build_basis_vocab(tkzr_cfg, k)
    assert bv["categories"] == ["AGE//age_", "VTL//hr_"]
    assert bv["num_non_numeric"] == 3  # UNK, BOS, EOS
    assert bv["vocab_size"] == 3 + 2 * k
    assert bv["raw_to_category"][3:7] == [0, 0, 0, 0]
    assert bv["raw_to_category"][7:11] == [1, 1, 1, 1]
    assert bv["raw_to_category"][:3] == [-1, -1, -1]
    assert bv["raw_to_collapsed"][3:7] == [bv["category_base_id"][0]] * 4
    print("build_basis_vocab: OK", bv["basis_lookup"])

    # ---- KNOWN LIMITATION: category detection cannot reliably distinguish
    # a real cocoa bin token ("<code>_Q<i>"/"<code>_B<i>") from an ordinary
    # categorical label that happens to end in the same shape by coincidence
    # -- e.g. a real drug name like "vitamin_B12". _BIN_TOKEN_RE only sees
    # the flattened tokenizer.yaml label; cocoa's own code/binned_value
    # column split (tokenizer.py's bin_data/get_pretokenized) is gone by
    # then, so this is a structural gap, not a simple regex bug. Two
    # tempting fixes were tried and rejected against the real MIMIC
    # tokenizer.yaml (1356 raw tokens, 151 real categories) before writing
    # this test, so don't reintroduce them without re-checking:
    #   - requiring an "_" immediately before Q/B: "vitamin_B12" already has
    #     one ("vitamin" + "_" + "B12"), so this doesn't help.
    #   - requiring every detected category to contain an index-0 member
    #     ("_Q0"/"_B0"): 57 of the 151 real categories (mostly MED-CTS
    #     continuous infusions, e.g. propofol_/fentanyl_/heparin_) have NO
    #     index-0 member in the real vocab -- their lowest quantile bin was
    #     never observed in training data -- so this raises false alarms on
    #     genuine categories far more often than it catches real collisions.
    # This test pins down today's actual (silent-corruption) behavior as an
    # intentional, visible characterization rather than an unnoticed gap. A
    # real fix needs cocoa to expose which raw `code`s are numeric directly
    # (e.g. a `numeric_codes` list in tokenizer.yaml) instead of cotorra
    # re-deriving it from string shape -- if that ever lands and this
    # assertion starts failing, update/delete this test then, not before.
    collision_tkzr_cfg = OmegaConf.create(
        {
            "cfg": {"n_bins": 4},
            "lookup": {
                "UNK": 0,
                "BOS": 1,
                "EOS": 2,
                "VTL//hr_Q0": 3,
                "VTL//hr_Q1": 4,
                "VTL//hr_Q2": 5,
                "VTL//hr_Q3": 6,
                "MED-CTS//vitamin_B12": 7,  # real drug name, not a bin token
            },
        }
    )
    collision_bv = build_basis_vocab(collision_tkzr_cfg, k=3)
    assert "MED-CTS//vitamin_" in collision_bv["categories"], (
        "expected (known-limitation) misdetection of 'vitamin_B12' as a "
        "numeric category didn't happen -- detection logic changed; see "
        "comment above before deleting this test"
    )
    assert "MED-CTS//vitamin_B12" not in collision_bv["basis_lookup"], (
        "the raw collision token should no longer appear as its own "
        "non-numeric vocab entry once (mis)absorbed into a category"
    )
    print(
        "KNOWN LIMITATION confirmed (see comment above): 'vitamin_B12'-style "
        "labels are silently misdetected as numeric bin tokens -- no "
        "delimiter or index-0 heuristic fixes this without false alarms on "
        "real categories"
    )

    # ---- config + model construction, init formula, mixture weights ----
    cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        train_beta_params=True,
        **bv,
    )
    mdl = BasisBlendedCausalLM(cfg)

    exp_log_alpha = t.log(t.arange(1, k + 1).float())
    exp_log_beta = t.log(t.arange(k, 0, -1).float())
    assert t.allclose(mdl.log_alpha[0], exp_log_alpha)
    assert t.allclose(mdl.log_beta[0], exp_log_beta)
    assert t.allclose(mdl.log_alpha[0], mdl.log_alpha[1])  # same init, both cats
    assert t.allclose(mdl.log_importance, t.zeros_like(mdl.log_importance))
    assert mdl.category_embed.shape == (max(len(bv["categories"]), 1), 16)
    assert t.allclose(mdl.category_embed, t.zeros_like(mdl.category_embed))
    print("order-statistic init (incl. log_importance/category_embed=0): OK")

    B, T = 2, 6
    input_ids = t.randint(0, bv["num_non_numeric"], (B, T))
    category_ids = t.full((B, T), -1, dtype=t.long)
    category_ids[:, 2] = 0
    category_ids[:, 4] = 1
    ranks = t.zeros(B, T)
    ranks[:, 2] = 0.13
    ranks[:, 4] = 0.9
    out = mdl(input_ids=input_ids, category_ids=category_ids, ranks=ranks)
    w = out.mixture_weights
    assert w is not None and w.shape == (B, T, k)
    assert t.allclose(w.sum(-1), t.ones(B, T), atol=1e-5)
    print("mixture weights sum to 1: OK")

    log_pdf_out = out.beta_log_pdf
    assert log_pdf_out is not None and log_pdf_out.shape == (B, T, k)
    assert t.isfinite(log_pdf_out).all()
    # log_importance=0 at init, so softmax(beta_log_pdf) must equal
    # mixture_weights exactly -- they're the same quantity before/after the
    # (zero, at init) importance term is added.
    assert t.allclose(t.softmax(log_pdf_out, dim=-1), w, atol=1e-5)
    print("beta_log_pdf: present, finite, consistent with mixture_weights: OK")

    loss = out.logits.sum()
    loss.backward()
    assert mdl.log_alpha.grad is not None and mdl.log_alpha.grad.abs().sum() > 0
    assert (
        mdl.log_importance.grad is not None and mdl.log_importance.grad.abs().sum() > 0
    )
    assert (
        mdl.category_embed.grad is not None and mdl.category_embed.grad.abs().sum() > 0
    )
    print("gradient reaches log_alpha/log_beta/log_importance/category_embed: OK")

    # beta_a/beta_b (added for crps_loss, see loss.py's basis_blended_token_loss)
    # must be the per-position, per-component Beta shape params actually used
    # to build mixture_weights/beta_log_pdf above -- i.e. each category's
    # (log_alpha, log_beta) broadcast out to (B, T, k) and clamped the same
    # way _mixture_weights clamps them internally.
    assert out.beta_a is not None and out.beta_a.shape == (B, T, k)
    assert out.beta_b is not None and out.beta_b.shape == (B, T, k)
    cat = category_ids.clamp(min=0)
    manual_a = mdl.log_alpha[cat].exp().clamp(1e-3, 1e4)
    manual_b = mdl.log_beta[cat].exp().clamp(1e-3, 1e4)
    assert t.allclose(out.beta_a, manual_a)
    assert t.allclose(out.beta_b, manual_b)
    print("beta_a/beta_b: present, correct shape, match log_alpha/log_beta: OK")

    # category_embed=0 (init) must reproduce the plain basis blend exactly:
    # e = sum_i w_c,i * e_c,i, with no shared per-category offset. Verified
    # directly against hidden_states[0] (the actual embeddings forward()
    # built and fed to inner_model), not just re-derived math, so this
    # catches a forward() wiring bug the manual re-derivation alone
    # wouldn't.
    with t.no_grad():
        embed = mdl.get_input_embeddings()
        numeric_mask = category_ids >= 0
        basis_ids = mdl.category_base_id_t[category_ids.clamp(min=0)].unsqueeze(
            -1
        ) + t.arange(k)
        manual_blend = (w.unsqueeze(-1) * embed(basis_ids)).sum(dim=2)
        manual_embeds = t.where(
            numeric_mask.unsqueeze(-1), manual_blend, embed(input_ids)
        )
        out_hs = mdl(
            input_ids=input_ids,
            category_ids=category_ids,
            ranks=ranks,
            output_hidden_states=True,
        )
    assert t.allclose(out_hs.hidden_states[0], manual_embeds, atol=1e-4)
    print("category_embed=0 reproduces the plain basis blend (e = sum w*e): OK")

    # a non-zero category_embed must shift *only* that category's numeric
    # positions, by exactly the added vector -- not the mixture weights, not
    # non-numeric positions, not the other category.
    with t.no_grad():
        shift = t.randn(16)
        mdl.category_embed[0] += shift
        out_hs_shifted = mdl(
            input_ids=input_ids,
            category_ids=category_ids,
            ranks=ranks,
            output_hidden_states=True,
        )
        delta = out_hs_shifted.hidden_states[0] - out_hs.hidden_states[0]
    assert t.allclose(delta[:, 2], shift.expand(B, 16), atol=1e-4)  # category 0
    assert t.allclose(delta[:, 4], t.zeros(B, 16), atol=1e-4)  # category 1, untouched
    assert t.allclose(
        delta[:, [0, 1, 3, 5]], t.zeros(B, 4, 16), atol=1e-4
    )  # non-numeric, untouched
    with t.no_grad():
        mdl.category_embed[0] -= shift  # restore for subsequent tests
    print("non-zero category_embed shifts only its own category's positions: OK")

    # log_importance=0 (init) must reproduce the plain density-ratio formula
    # exactly: w_c,i = f(r,i)/sum_j f(r,j), with no importance reweighting.
    with t.no_grad():
        cat = category_ids.clamp(min=0)
        a = mdl.log_alpha[cat].exp()
        b = mdl.log_beta[cat].exp()
        r = ranks.clamp(_RANK_EPS, 1 - _RANK_EPS).unsqueeze(-1)
        log_pdf = (
            (a - 1) * r.log()
            + (b - 1) * (1 - r).log()
            - (t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b))
        )
        w_plain = t.softmax(log_pdf, dim=-1)
        w_scaled = mdl._mixture_weights(category_ids, ranks)
    assert t.allclose(w_plain, w_scaled, atol=1e-5)
    print("log_importance=0 reproduces the plain density-ratio formula: OK")

    with t.no_grad():
        _, log_pdf_returned = mdl._mixture_weights(
            category_ids, ranks, return_log_pdf=True
        )
    assert t.allclose(log_pdf_returned, log_pdf, atol=1e-5)
    print("_mixture_weights(return_log_pdf=True) matches the manual density: OK")

    with t.no_grad():
        mdl.log_importance[0, 0] += 5.0  # heavily favor basis 0 for category 0
        w_after = mdl._mixture_weights(category_ids, ranks)
    assert w_after[0, 2, 0] > w_scaled[0, 2, 0]  # position 2 is category 0
    print("non-zero log_importance measurably reweights mixture weights: OK")

    # regression: alpha/beta combinations seen in real trained models (e.g.
    # a mix of small (<1) and large (>20) values across the k basis elements
    # of one category) previously produced NaN mixture weights via lgamma
    # ill-conditioning. Exercise a similarly wide spread directly.
    with t.no_grad():
        mdl.log_alpha[0] = t.log(
            t.tensor([0.33, 0.80, 1.79, 3.34, 13.85, 18.70, 20.97, 19.56])[:k]
        )
        mdl.log_beta[0] = t.log(
            t.tensor([24.72, 21.73, 16.28, 10.70, 1.45, 8.48, 0.79, 0.49])[:k]
        )
        for rv in (0.001, 0.1, 0.5, 0.9, 0.999):
            w_extreme = mdl._mixture_weights(
                t.zeros(1, 1, dtype=t.long), t.full((1, 1), rv)
            )
            assert t.isfinite(w_extreme).all(), f"non-finite mixture weight at r={rv}"
            assert t.allclose(w_extreme.sum(), t.tensor(1.0), atol=1e-4)
    print("wide alpha/beta spread stays finite (no NaN mixture weights): OK")

    # same regression, but for beta_log_pdf specifically -- this is the
    # tensor mixture_nll_loss evaluates a raw (unsoftmaxed) likelihood from,
    # so it's the one most exposed to an extreme alpha/beta spike producing
    # a huge-but-finite log-density that overflows on exponentiation
    # (t.logsumexp handles this safely; a naive t.exp(...).sum().log() would
    # not).
    with t.no_grad():
        for rv in (0.001, 0.1, 0.5, 0.9, 0.999):
            _, log_pdf_extreme = mdl._mixture_weights(
                t.zeros(1, 1, dtype=t.long), t.full((1, 1), rv), return_log_pdf=True
            )
            assert t.isfinite(log_pdf_extreme).all(), (
                f"non-finite beta_log_pdf at r={rv}"
            )
            # a NLL loss built from this must itself stay finite even under
            # this spread, for an arbitrary (here: uniform) predicted w_hat
            uniform_log_w_hat = t.full_like(log_pdf_extreme, -t.log(t.tensor(float(k))))
            nll = -t.logsumexp(uniform_log_w_hat + log_pdf_extreme, dim=-1)
            assert t.isfinite(nll).all(), f"non-finite mixture NLL at r={rv}"
    print("wide alpha/beta spread stays finite (no NaN beta_log_pdf / NLL): OK")

    mdl2 = BasisBlendedCausalLM(BasisBlendedConfig(**cfg.to_dict()))
    for pn, p in mdl.named_parameters():
        mdl2.state_dict()[pn].copy_(p.detach())
    with tempfile.TemporaryDirectory() as d:
        mdl2.save_pretrained(d)
        reloaded = AutoModelForCausalLM.from_pretrained(d)
    assert isinstance(reloaded, BasisBlendedCausalLM)
    assert t.allclose(reloaded.log_alpha, mdl2.log_alpha)
    assert t.allclose(reloaded.category_embed, mdl2.category_embed)
    assert reloaded.config.categories == bv["categories"]
    print("save_pretrained / from_pretrained (via AutoModelForCausalLM): OK")

    # regression: a checkpoint saved before log_importance/category_embed
    # existed (or any future param added the same way) must reload it as
    # the intended zero init, not uninitialized garbage -- see
    # _init_weights.
    with tempfile.TemporaryDirectory() as d:
        mdl2.save_pretrained(d)
        import safetensors.torch as _st

        sd = _st.load_file(f"{d}/model.safetensors")
        del sd["log_importance"]
        del sd["category_embed"]
        _st.save_file(sd, f"{d}/model.safetensors", metadata={"format": "pt"})
        reloaded_missing = AutoModelForCausalLM.from_pretrained(d)
    assert t.isfinite(reloaded_missing.log_importance).all(), (
        "log_importance not finite after reloading a checkpoint missing it"
    )
    assert t.isfinite(reloaded_missing.category_embed).all(), (
        "category_embed not finite after reloading a checkpoint missing it"
    )
    assert t.allclose(
        reloaded_missing.log_importance, t.zeros_like(reloaded_missing.log_importance)
    )
    assert t.allclose(
        reloaded_missing.category_embed, t.zeros_like(reloaded_missing.category_embed)
    )
    assert t.allclose(reloaded_missing.log_alpha, mdl2.log_alpha)
    print("missing log_importance/category_embed reloads as zero, not garbage: OK")

    frozen_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=cfg.base_config,
        train_beta_params=False,
        train_importance_scale=False,
        train_category_embed=False,
        **bv,
    )
    frozen = BasisBlendedCausalLM(frozen_cfg)
    assert not frozen.log_alpha.requires_grad and not frozen.log_beta.requires_grad
    assert not frozen.log_importance.requires_grad
    assert not frozen.category_embed.requires_grad
    print(
        "train_beta_params=False / train_importance_scale=False / "
        "train_category_embed=False freeze params: OK"
    )

    # config must expose inner-model-only fields (e.g. num_hidden_layers) for
    # generic HF utilities that read model.config directly -- DynamicCache's
    # __init__ does this during .generate(), so exercise that exact path.
    assert cfg.num_hidden_layers == 1
    mdl.generate(
        t.tensor([[bv["bos_token_id"]]]),
        max_length=8,
        do_sample=True,
        top_k=bv["vocab_size"],
    )
    print("config falls back to inner base_config fields, .generate() works: OK")

    # regression: some transformers versions' prepare_inputs_for_generation
    # includes an inputs_embeds=None key in the dict passed to forward()
    # during .generate()'s prefill step. If inputs_embeds isn't consumed by
    # a named parameter, it falls through **kwargs and collides with the
    # explicit inputs_embeds=base_embeds this wrapper passes to
    # self.inner_model -- "got multiple values for keyword argument
    # 'inputs_embeds'". Exercise that exact call shape directly rather than
    # relying on hitting the right transformers version/code path via
    # .generate() alone.
    mdl(
        input_ids=t.tensor([[bv["bos_token_id"], bv["eos_token_id"]]]),
        inputs_embeds=None,
    )
    print("forward() tolerates an explicit inputs_embeds=None kwarg: OK")

    # ---- numerical_basis_model reproduction (fuzzy_token_planning.md
    # point 18): build_gaussian_vocab, e_c,1 + v*e_c,2 embedding, gaussian
    # head, and Loss.numerical_basis_model_loss ----
    from cotorra.loss import Loss

    gv = build_gaussian_vocab(tkzr_cfg)
    assert gv["categories"] == ["AGE//age_", "VTL//hr_"]
    assert gv["num_non_numeric"] == 3  # UNK, BOS, EOS
    assert gv["vocab_size"] == 3 + 2  # num_non_numeric + num_categories, no k
    assert gv["category_base_id"] == [3, 4]  # one slot per category
    print("build_gaussian_vocab: OK", gv["basis_lookup"])

    gauss_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=cfg.base_config,
        numerical_basis_model=True,
        **gv,
    )
    gauss_mdl = BasisBlendedCausalLM(gauss_cfg)
    n_cat = len(gv["categories"])
    assert gauss_mdl.e2.shape == (n_cat, 16)
    assert t.allclose(gauss_mdl.e2, t.zeros_like(gauss_mdl.e2))
    assert not hasattr(gauss_mdl, "log_alpha")
    assert not hasattr(gauss_mdl, "category_embed")
    print("numerical_basis_model init: e2=0, no k-mixture params exist: OK")

    g_input_ids = t.randint(0, gv["num_non_numeric"], (B, T))
    g_category_ids = t.full((B, T), -1, dtype=t.long)
    g_category_ids[:, 2] = 0
    g_category_ids[:, 4] = 1
    g_ranks = t.zeros(B, T)
    g_ranks[:, 2] = 0.3
    g_ranks[:, 4] = 0.8
    g_out = gauss_mdl(input_ids=g_input_ids, category_ids=g_category_ids, ranks=g_ranks)
    assert g_out.logits.shape[-1] == gv["vocab_size"]
    assert g_out.gaussian_params is not None
    assert g_out.gaussian_params.shape == (B, T, 2 * n_cat)
    assert g_out.mixture_weights is None and g_out.beta_log_pdf is None
    print("numerical_basis_model forward() shapes: OK")

    # e2=0 at init must reproduce e_c,1-only embedding exactly, verified
    # directly against hidden_states[0] (same pattern as category_embed's
    # equivalent check above).
    with t.no_grad():
        g_embed = gauss_mdl.get_input_embeddings()
        g_numeric = g_category_ids >= 0
        g_e1 = g_embed(gauss_mdl.category_base_id_t[g_category_ids.clamp(min=0)])
        g_manual = t.where(g_numeric.unsqueeze(-1), g_e1, g_embed(g_input_ids))
        g_out_hs = gauss_mdl(
            input_ids=g_input_ids,
            category_ids=g_category_ids,
            ranks=g_ranks,
            output_hidden_states=True,
        )
    assert t.allclose(g_out_hs.hidden_states[0], g_manual, atol=1e-4)
    assert g_out.hidden_states is None  # not leaked when not requested
    print("e2=0 reproduces e_c,1-only embedding; hidden_states not leaked: OK")

    g_loss_fn = Loss(
        cfg=OmegaConf.create({"basis_blended_tokens": {"numerical_basis_model": True}}),
        tkzr_cfg=tkzr_cfg,
        basis_vocab=gv,
    )
    g_labels = g_input_ids.clone()
    g_labels[:, 3] = gv["category_base_id"][0]
    g_labels[:, 5] = gv["category_base_id"][1]
    g_loss = g_loss_fn.numerical_basis_model_loss(
        g_out, g_labels, category_ids=g_category_ids, ranks=g_ranks
    )
    assert t.isfinite(g_loss)
    g_loss.backward()
    assert gauss_mdl.e2.grad is not None and gauss_mdl.e2.grad.abs().sum() > 0
    assert (
        gauss_mdl.gaussian_head.weight.grad is not None
        and gauss_mdl.gaussian_head.weight.grad.abs().sum() > 0
    )
    print("numerical_basis_model_loss finite, gradient reaches e2/gaussian_head: OK")

    gauss_mdl2 = BasisBlendedCausalLM(BasisBlendedConfig(**gauss_cfg.to_dict()))
    for pn, p in gauss_mdl.named_parameters():
        gauss_mdl2.state_dict()[pn].copy_(p.detach())
    with tempfile.TemporaryDirectory() as d:
        gauss_mdl2.save_pretrained(d)
        g_reloaded = AutoModelForCausalLM.from_pretrained(d)
    assert isinstance(g_reloaded, BasisBlendedCausalLM)
    assert t.allclose(g_reloaded.e2, gauss_mdl2.e2)
    assert g_reloaded.config.numerical_basis_model is True
    print("numerical_basis_model save_pretrained / from_pretrained: OK")

    # ---- xval_head: the LITERAL xVal variant ----
    xv_cfg = BasisBlendedConfig(
        **{**gauss_cfg.to_dict(), "xval_head": True, "xval_value_scale": 2.0}
    )
    xv_mdl = BasisBlendedCausalLM(xv_cfg)
    # ONE shared scaled vector and a SCALAR head, against the reproduction's
    # per-category e_c,2 and 2*n_cat Gaussian head.
    assert xv_mdl.e2.shape == (1, 16), f"xVal e2 must be shared, got {xv_mdl.e2.shape}"
    assert xv_mdl.gaussian_head.out_features == 1
    xv_out = xv_mdl(input_ids=g_input_ids, category_ids=g_category_ids, ranks=g_ranks)
    assert xv_out.gaussian_params.shape == (B, T, 1)
    # xval_value_scale multiplies v before it scales the shared vector, so
    # doubling it must double the value's whole contribution to the
    # embedding. Checked against the 1.0 model on identical weights.
    xv_one = BasisBlendedCausalLM(
        BasisBlendedConfig(**{**xv_cfg.to_dict(), "xval_value_scale": 1.0})
    )
    xv_one.load_state_dict(xv_mdl.state_dict())
    with t.no_grad():
        xv_mdl.e2.normal_(0.0, 0.1)
        xv_one.e2.copy_(xv_mdl.e2)
        _emb = xv_mdl.get_input_embeddings()
        _cat = g_category_ids.clamp(min=0)
        _e1 = _emb(xv_mdl.category_base_id_t[_cat])
        _v = g_ranks.unsqueeze(-1)
        _b2 = _e1 + 2.0 * _v * xv_mdl.e2[0]
        _b1 = _e1 + 1.0 * _v * xv_one.e2[0]
        assert t.allclose(_b2 - _e1, 2.0 * (_b1 - _e1), atol=1e-6)
    # the loss is MSE on that scalar, not a Gaussian NLL
    xv_loss_fn = Loss(
        cfg=OmegaConf.create(
            {
                "custom_loss": True,
                "basis_blended_tokens": {
                    "numerical_basis_model": True,
                    "xval_head": True,
                },
            }
        ),
        tkzr_cfg=tkzr_cfg,
        basis_vocab=gv,
    )
    xv_l = xv_loss_fn.numerical_basis_model_loss(
        xv_out, g_labels, category_ids=g_category_ids, ranks=g_ranks
    )
    _num = g_category_ids[:, 1:] >= 0
    _pred = xv_out.gaussian_params[:, :-1].squeeze(-1)[_num]
    _tru = g_ranks[:, 1:][_num]
    _ce = (
        -t.log_softmax(xv_out.logits[:, :-1].float(), dim=-1)
        .gather(-1, g_labels[:, 1:].unsqueeze(-1))
        .squeeze(-1)
        .sum()
    )
    assert abs(float(xv_l - _ce) - float((_pred - _tru).pow(2).sum())) < 1e-4, (
        "xval_head's value term must be exactly the MSE of the scalar head"
    )
    assert t.isfinite(xv_l)
    xv_l.backward()
    assert xv_mdl.e2.grad is not None and xv_mdl.e2.grad.abs().sum() > 0
    _raised_xv = False
    try:
        BasisBlendedCausalLM(
            BasisBlendedConfig(
                base_model_type="llama",
                base_config=cfg.base_config,
                xval_head=True,
                **bv,
            )
        )
    except AssertionError:
        _raised_xv = True
    assert _raised_xv, "xval_head without numerical_basis_model must be refused"
    print(
        "xval_head: one shared scaled vector, scalar MSE head, value scale "
        "applies, gradient reaches e2, refused without "
        "numerical_basis_model: OK"
    )

    # ---- film_value_embed: the third curve parameterisation ----
    # e_c(r) = gamma_c * MLP(r) + beta_c, over the same one-slot-per-category
    # vocabulary the xVal family uses, scored by the same tied continuous
    # head the anchor blend uses.
    fm_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=cfg.base_config,
        film_value_embed=True,
        film_hidden=8,
        tied_continuous_head=True,
        component_family="truncnorm",
        **gv,
    )
    fm_mdl = BasisBlendedCausalLM(fm_cfg)
    assert not hasattr(fm_mdl, "log_alpha"), "FiLM must allocate no anchors"
    assert not hasattr(fm_mdl, "e2")
    assert fm_mdl.film_out.weight.shape == (16, 8)
    # gamma = 1 and beta = identity at init, so the numeric token starts as
    # e_cat + MLP(v) rather than with the value channel multiplied by ~0.
    assert t.allclose(fm_mdl.film_gamma.bias, t.ones_like(fm_mdl.film_gamma.bias))
    assert t.allclose(fm_mdl.film_gamma.weight, t.zeros_like(fm_mdl.film_gamma.weight))
    assert t.allclose(fm_mdl.film_beta.weight, t.eye(16))
    _fc = t.zeros(1, dtype=t.long)
    _gam, _bet, _ecat = fm_mdl._film_gamma_beta(_fc)
    assert t.allclose(_bet, _ecat, atol=1e-6), "beta = e_cat at init"

    # the embedding: gamma * MLP(v) + beta, and a NaN value falls back to the
    # concept embedding alone.
    f_ids = t.tensor(
        [[gv["bos_token_id"], gv["category_base_id"][0], gv["eos_token_id"]]]
    )
    f_cat = t.tensor([[-1, 0, -1]])
    f_out = fm_mdl(
        input_ids=f_ids, category_ids=f_cat, ranks=t.tensor([[0.0, 0.4, 0.0]])
    )
    assert f_out.logits.shape == (1, 3, gv["vocab_size"])
    _emb_nan = fm_mdl(
        input_ids=f_ids,
        category_ids=f_cat,
        ranks=t.tensor([[0.0, float("nan"), 0.0]]),
        output_hidden_states=True,
    )
    assert t.isfinite(_emb_nan.logits).all(), "a NaN value must not poison the batch"

    # THE DECOMPOSITION. _continuous_logprob evaluates g in film_hidden
    # dimensions by dropping two r-independent terms (h.beta_c and the MLP's
    # output bias). Those cancel in a DIFFERENCE of g at two ranks, which is
    # all the normalised density depends on -- so check the cheap form
    # against the full h . e_c(r) exactly there.
    _hh = t.randn(16)
    _u = (_hh * _gam[0]) @ fm_mdl.film_out.weight

    def _g_direct(rr):
        return float(_hh @ (_gam[0] * fm_mdl._film_value_mlp(rr)[0] + _bet[0]))

    def _g_quick(rr):
        return float(
            _u @ t.relu(fm_mdl.film_in.weight.view(-1) * rr + fm_mdl.film_in.bias)
        )

    _r1, _r2 = t.tensor([0.2]), t.tensor([0.9])
    _dd = (_g_direct(_r1) - _g_direct(_r2)) - (_g_quick(_r1) - _g_quick(_r2))
    assert abs(_dd) < 1e-4, f"FiLM g decomposition off by {_dd:.2e}"

    # ...and the head over that curve still normalises: P(r in [0,1]) = 1.
    def fm_lp(width):
        return fm_mdl(
            input_ids=f_ids,
            category_ids=f_cat,
            ranks=t.tensor([[0.0, 0.4, 0.0]]),
            rank_widths=t.tensor([[0.0, width, 0.0]]),
        )["continuous_logprob"]

    with t.no_grad():
        fm_mdl.film_gamma.weight.normal_(0.0, 0.2)
        fm_mdl.film_in.weight.normal_(0.0, 1.0)
        fm_mdl.film_out.weight.normal_(0.0, 0.3)
    _fm_full, _fm_narrow = fm_lp(2.0), fm_lp(0.05)
    assert abs(float(_fm_full[0, 1])) < 1e-4, (
        f"FiLM curve: P(r in [0,1]) must be 1, got log P {float(_fm_full[0, 1]):.2e}"
    )
    assert float(_fm_narrow[0, 1]) < 0
    fm_mdl.zero_grad()
    fm_lp(0.05).sum().backward()
    for _n in ("film_in", "film_out", "film_gamma"):
        _gp = getattr(fm_mdl, _n).weight.grad
        assert _gp is not None and _gp.abs().sum() > 0, f"no gradient to {_n}"
    # bf16 REGRESSION. Training can have an fp32 FiLM against a bf16
    # embedding table, and `cotorra extract` casts the whole model to bf16 --
    # the mismatch raises in opposite directions in the two settings, and
    # both were hit for real. Exercise the cast path itself.
    fm_bf = BasisBlendedCausalLM(BasisBlendedConfig(**fm_cfg.to_dict())).to(t.bfloat16)
    _bf = fm_bf(
        input_ids=f_ids,
        category_ids=f_cat,
        ranks=t.tensor([[0.0, 0.4, 0.0]]),
        rank_widths=t.tensor([[0.0, 0.05, 0.0]]),
    )
    assert t.isfinite(_bf.logits).all(), "bf16 FiLM forward must not raise or NaN"
    assert float(_bf["continuous_logprob"][0, 1]) < 0
    _raised_fm = False
    try:
        BasisBlendedCausalLM(
            BasisBlendedConfig(
                base_model_type="llama",
                base_config=cfg.base_config,
                film_value_embed=True,
                **gv,
            )
        )
    except AssertionError:
        _raised_fm = True
    assert _raised_fm, "FiLM with no head to score it must be refused"
    print(
        "film_value_embed: no anchors, gamma=1/beta=e_cat at init, NaN falls "
        "back to e_cat, g decomposition exact, tied head normalises over the "
        "FiLM curve, gradients flow, headless config refused: OK"
    )

    # ---- decoupled_magnitude ----
    # The two properties the flag exists for: magnitude is FLAT across the
    # rank axis at init (equal mu, weights sum to one), and it is free to
    # vary once mu moves -- including peaking BETWEEN two components.
    H_dm = 16
    # k=3 is too coarse to tell "peaked between two modes" from "peaked on
    # one": the modes sit 0.5 apart. Build the same vocabulary at k=10.
    k_dm = 10
    bv_dm = build_basis_vocab(tkzr_cfg, k_dm)
    dm_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=dict(
            hidden_size=H_dm,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            initializer_range=0.02,
        ),
        train_beta_params=True,
        decoupled_magnitude=True,
        **bv_dm,
    )
    dm_mdl = BasisBlendedCausalLM(dm_cfg)
    s0 = 0.02 * math.sqrt(H_dm)
    assert dm_mdl.basis_log_magnitude.shape == (max(len(bv_dm["categories"]), 1), k_dm)
    assert t.allclose(
        nn.functional.softplus(dm_mdl.basis_log_magnitude),
        t.full_like(dm_mdl.basis_log_magnitude, s0),
        atol=1e-6,
    ), "every mu must map to the same s0 at init"

    # iid anchors, so the PLAIN blend would attenuate by 1/sqrt(k_eff)
    with t.no_grad():
        emb_dm = dm_mdl.get_input_embeddings().weight
        emb_dm.normal_(0.0, 0.02)
        dm_mdl.category_embed.zero_()
    n_r = 40
    dm_ids = t.zeros(1, n_r, dtype=t.long)  # UNK, a non-numeric row
    dm_cat = t.zeros(1, n_r, dtype=t.long)
    dm_ranks = t.linspace(0.02, 0.98, n_r).unsqueeze(0)
    dm_out = dm_mdl(
        input_ids=dm_ids, category_ids=dm_cat, ranks=dm_ranks, output_hidden_states=True
    )
    got = dm_out.hidden_states[0][0]  # the embedding layer's own output
    nrm = got.norm(dim=-1)
    assert t.allclose(nrm, t.full_like(nrm, s0), atol=1e-4), (
        f"magnitude must be flat at init: got {nrm.min():.5f}-{nrm.max():.5f}, "
        f"expected {s0:.5f}"
    )
    print(f"decoupled_magnitude: flat at init ||e(r)|| == s0 == {s0:.5f}: OK")

    # same anchors, flag OFF -> the attenuation the flag exists to remove
    off_cfg = BasisBlendedConfig(**{**dm_cfg.to_dict(), "decoupled_magnitude": False})
    off_mdl = BasisBlendedCausalLM(off_cfg)
    with t.no_grad():
        off_mdl.get_input_embeddings().weight.copy_(emb_dm)
        off_mdl.category_embed.zero_()
    off_nrm = (
        off_mdl(
            input_ids=dm_ids,
            category_ids=dm_cat,
            ranks=dm_ranks,
            output_hidden_states=True,
        )
        .hidden_states[0][0]
        .norm(dim=-1)
    )
    assert off_nrm.max() / off_nrm.min() > 1.2, (
        "the plain blend should visibly vary in magnitude across ranks with "
        "iid anchors -- if it does not, this test is no longer measuring "
        "what it was written to measure"
    )
    print(
        f"  plain blend over the same anchors: {off_nrm.min():.5f}-"
        f"{off_nrm.max():.5f} ({off_nrm.max() / off_nrm.min():.2f}x swing)"
    )

    # expressiveness: raise two ADJACENT mu and the magnitude peak must land
    # strictly between their component modes, not on one of them.
    lo_i = k_dm // 2 - 1
    with t.no_grad():
        dm_mdl.basis_log_magnitude[0, lo_i] = 3.0
        dm_mdl.basis_log_magnitude[0, lo_i + 1] = 3.0
    peak_nrm = (
        dm_mdl(
            input_ids=dm_ids,
            category_ids=dm_cat,
            ranks=dm_ranks,
            output_hidden_states=True,
        )
        .hidden_states[0][0]
        .norm(dim=-1)
    )
    r_at_max = float(dm_ranks[0, int(peak_nrm.argmax())])
    modes = (t.arange(k_dm).float()) / (k_dm - 1)
    dist = float((modes - r_at_max).abs().min())
    assert peak_nrm.max() / peak_nrm.min() > 1.5, "mu must move the magnitude"
    assert dist > 0.25 / (k_dm - 1), (
        f"peak at r={r_at_max:.3f} sits on a component mode (dist {dist:.3f}) "
        "-- magnitude is not decoupled from anchor position"
    )
    print(
        f"decoupled_magnitude: peak BETWEEN components at r={r_at_max:.3f} "
        f"(nearest mode {dist:.3f} away), swing "
        f"{peak_nrm.max() / peak_nrm.min():.2f}x: OK"
    )

    # gradient must reach the new channel
    dm_mdl.zero_grad()
    dm_mdl(
        input_ids=dm_ids, category_ids=dm_cat, ranks=dm_ranks
    ).logits.sum().backward()
    assert (
        dm_mdl.basis_log_magnitude.grad is not None
        and dm_mdl.basis_log_magnitude.grad.abs().sum() > 0
    )
    print("decoupled_magnitude: gradient reaches basis_log_magnitude: OK")

    dm_mdl2 = BasisBlendedCausalLM(BasisBlendedConfig(**dm_cfg.to_dict()))
    for pn, p in dm_mdl.named_parameters():
        dm_mdl2.state_dict()[pn].copy_(p.detach())
    with tempfile.TemporaryDirectory() as d:
        dm_mdl2.save_pretrained(d)
        dm_reloaded = AutoModelForCausalLM.from_pretrained(d)
    assert t.allclose(dm_reloaded.basis_log_magnitude, dm_mdl2.basis_log_magnitude)
    assert dm_reloaded.config.decoupled_magnitude is True
    print("decoupled_magnitude: save_pretrained / from_pretrained: OK")

    # ---- tied_continuous_head ----
    # p(r|h) = exp(g)/Z with g = h . e_c(r): a density over the WHOLE rank
    # axis from k logits. Checked against the normalisation identity, which
    # is the one property a bug in either quadrature would break.
    tch_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            initializer_range=0.02,
        ),
        component_family="truncnorm",
        decoupled_magnitude=True,
        tied_continuous_head=True,
        **bv_dm,
    )
    tch_mdl = BasisBlendedCausalLM(tch_cfg)
    with t.no_grad():
        tch_mdl.get_input_embeddings().weight.normal_(0.0, 0.05)
        tch_mdl.basis_log_magnitude.normal_(0.0, 0.5)
    Bc, Tc = 1, 5
    c_ids = t.zeros(Bc, Tc, dtype=t.long)
    c_cat = t.full((Bc, Tc), -1, dtype=t.long)
    c_cat[0, 1] = 0
    c_cat[0, 3] = 1
    c_rank = t.zeros(Bc, Tc)
    c_rank[0, 1], c_rank[0, 3] = 0.27, 0.81

    def tch_lp(width):
        w = t.zeros(Bc, Tc)
        w[0, 1] = w[0, 3] = width
        return tch_mdl(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=w
        )["continuous_logprob"]

    narrow = tch_lp(0.06)
    assert narrow.shape == (Bc, Tc)
    assert bool((narrow[:, 0] == 0).all()), "column 0 must stay unused"
    assert bool((narrow[0, [1, 3]] < 0).all()), "a narrow interval must have P < 1"
    # a width that spans all of [0,1] must give log P = 0 exactly: this is
    # the numerator quadrature and the Z quadrature agreeing, computed by
    # DIFFERENT rules (8x8 vs 16x8), so it fails loudly if either is wrong.
    full = tch_lp(2.0)
    assert full[0, [1, 3]].abs().max() < 1e-4, (
        f"P(r in [0,1]) must be 1: got log P {full[0, 1]:.3e}, {full[0, 3]:.3e}"
    )
    # monotone in width, since the interval only grows
    mid = tch_lp(0.6)
    assert bool((narrow[0, 1] < mid[0, 1] < full[0, 1]).all())
    print(
        f"tied_continuous_head: log P(r in [0,1]) = {float(full[0, 1]):+.2e}, "
        f"monotone in width ({float(narrow[0, 1]):.3f} -> "
        f"{float(mid[0, 1]):.3f} -> {float(full[0, 1]):.3f}): OK"
    )

    tch_mdl.zero_grad()
    tch_lp(0.06).sum().backward()
    for _pn in ("basis_log_magnitude", "log_alpha", "log_beta"):
        _g = getattr(tch_mdl, _pn).grad
        assert _g is not None and _g.abs().sum() > 0, f"no gradient to {_pn}"
    _ge = tch_mdl.get_input_embeddings().weight.grad
    assert _ge is not None and _ge.abs().sum() > 0
    print(
        "tied_continuous_head: gradient reaches basis_log_magnitude/log_alpha/"
        "log_beta/embed_tokens: OK"
    )

    off = BasisBlendedCausalLM(
        BasisBlendedConfig(**{**tch_cfg.to_dict(), "tied_continuous_head": False})
    )
    assert (
        off(input_ids=c_ids, category_ids=c_cat, ranks=c_rank).get("continuous_logprob")
        is None
    ), "flag off must not emit continuous_logprob"
    print("tied_continuous_head: absent when the flag is off: OK")

    # continuous_crps. With every anchor identical the deviations vanish, so
    # g = 0 and p(r|h) is exactly uniform, whose CRPS is closed form:
    #   CRPS(U, r) = (r^2 + (1-r)^2)/2 - 1/6.
    # That pins the cumsum pair-term identity. Then a non-uniform density must
    # agree between the training rule and a much finer one.
    cr_cfg = BasisBlendedConfig(
        **{**tch_cfg.to_dict(), "decoupled_magnitude": False, "continuous_crps": True}
    )
    cr_mdl = BasisBlendedCausalLM(cr_cfg)
    _w = t.zeros(Bc, Tc)
    _w[0, 1] = _w[0, 3] = 0.06
    with t.no_grad():
        cr_mdl.get_input_embeddings().weight.fill_(0.01)
        _uni = cr_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _exact = t.tensor([(r * r + (1 - r) ** 2) / 2 - 1 / 6 for r in (0.27, 0.81)])
    _got = _uni["continuous_crps"][0, [1, 3]]
    assert (_got - _exact).abs().max() < 1e-5, f"uniform CRPS {_got} vs {_exact}"
    assert bool((_uni["continuous_crps"][:, 0] == 0).all())
    with t.no_grad():
        cr_mdl.get_input_embeddings().weight.normal_(0.0, 0.5)
    _coarse = cr_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    cr_fine = BasisBlendedCausalLM(
        BasisBlendedConfig(
            **{
                **cr_cfg.to_dict(),
                "continuous_quad_panels": 128,
                "continuous_interval_panels": 64,
                "continuous_quad_nodes": 16,
                "continuous_interval_nodes": 16,
            }
        )
    )
    cr_fine.load_state_dict(cr_mdl.state_dict())
    with t.no_grad():
        _fine = cr_fine(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w
        )["continuous_crps"]
    _dc = (_coarse["continuous_crps"] - _fine)[0, [1, 3]].abs().max()
    assert _dc < 1e-4, f"CRPS coarse vs fine rule differ by {_dc:.2e}"
    cr_mdl.zero_grad()
    _coarse["continuous_crps"].sum().backward()
    assert cr_mdl.get_input_embeddings().weight.grad.abs().sum() > 0
    assert cr_mdl.log_alpha.grad is not None and cr_mdl.log_alpha.grad.abs().sum() > 0
    assert (
        off(input_ids=c_ids, category_ids=c_cat, ranks=c_rank).get("continuous_crps")
        is None
    )
    _eu = float((_got - _exact).abs().max())
    print(
        f"continuous_crps: uniform exact (err {_eu:.1e}),"
        f" coarse vs fine rule {float(_dc):.1e}, gradient flows: OK"
    )

    # ---- continuous_moments: W1 and the predictive mean ----
    mom_cfg = BasisBlendedConfig(
        **{**cr_cfg.to_dict(), "continuous_crps": False, "continuous_moments": True}
    )
    mom_mdl = BasisBlendedCausalLM(mom_cfg)
    with t.no_grad():
        mom_mdl.get_input_embeddings().weight.fill_(0.01)
        _mu = mom_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    # identical anchors -> the deviations vanish -> g = 0 -> p is exactly
    # uniform on [0,1], where E[X] = 1/2 and
    #   W1(U, delta_r) = int_0^1 |x - r| dx = (r^2 + (1-r)^2)/2.
    _w1_exact = t.tensor([(r * r + (1 - r) ** 2) / 2 for r in (0.27, 0.81)])
    _got_w1 = _mu["continuous_w1"][0, [1, 3]]
    _got_mean = _mu["continuous_mean"][0, [1, 3]]
    assert (_got_w1 - _w1_exact).abs().max() < 1e-5, f"uniform W1 {_got_w1}"
    assert (_got_mean - 0.5).abs().max() < 1e-5, f"uniform mean {_got_mean}"
    assert bool((_mu["continuous_w1"][:, 0] == 0).all()), "column 0 must stay unused"
    assert (
        cr_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w).get(
            "continuous_w1"
        )
        is None
    ), "continuous_w1 must be absent when continuous_moments is off"

    both_mdl = BasisBlendedCausalLM(
        BasisBlendedConfig(**{**mom_cfg.to_dict(), "continuous_crps": True})
    )
    with t.no_grad():
        both_mdl.get_input_embeddings().weight.fill_(0.01)
        _bo = both_mdl(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w
        )
    # E|X - X'| = 1/3 for iid uniforms, so CRPS = W1 - 1/6 EXACTLY. This is
    # the cross-check that the two statistics come from ONE q on ONE node
    # set, rather than from two separately computed densities.
    _id = (_bo["continuous_crps"] - (_bo["continuous_w1"] - 1 / 6))[0, [1, 3]]
    assert _id.abs().max() < 1e-5, f"CRPS != W1 - 1/6 on the uniform: {_id}"
    with t.no_grad():
        both_mdl.get_input_embeddings().weight.normal_(0.0, 0.5)
        _nb = both_mdl(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w
        )
    # E|X - X'| >= 0, so CRPS <= W1 for ANY density, not just the uniform
    assert bool(
        (
            _nb["continuous_crps"][0, [1, 3]] <= _nb["continuous_w1"][0, [1, 3]] + 1e-6
        ).all()
    )
    both_mdl.zero_grad()
    _gm = both_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    (_gm["continuous_w1"].sum() + _gm["continuous_mean"].sum()).backward()
    assert both_mdl.get_input_embeddings().weight.grad.abs().sum() > 0
    assert (
        both_mdl.log_alpha.grad is not None and both_mdl.log_alpha.grad.abs().sum() > 0
    )
    print(
        f"continuous_moments: uniform W1/mean exact "
        f"(err {float((_got_w1 - _w1_exact).abs().max()):.1e}), "
        f"CRPS == W1 - 1/6 to {float(_id.abs().max()):.1e}, gradients flow: OK"
    )

    # ---- continuous_crps_z: CRPS in value units ----
    # Two exact identities pin the transform. With an IDENTITY table the
    # value axis IS the rank axis, so crps_z must equal crps bit for bit;
    # with a LINEAR table Q(r) = a + b r and scale s, CRPS is
    # translation-equivariant and scale-homogeneous, so crps_z must be
    # exactly (b/s) * crps. Together they test the interpolation, the
    # scaling, and that the estimator was not silently re-derived.
    _NQ = 101
    zz_cfg = BasisBlendedConfig(
        **{
            **mom_cfg.to_dict(),
            "continuous_crps": True,
            "continuous_crps_z": True,
            "value_quantile_points": _NQ,
        }
    )
    zz_mdl = BasisBlendedCausalLM(zz_cfg)
    assert zz_mdl.vq_grid.shape == (max(len(bv_dm["categories"]), 1), _NQ)
    with t.no_grad():
        zz_mdl.get_input_embeddings().weight.normal_(0.0, 0.5)
        _ramp = t.linspace(0.0, 1.0, _NQ)
        zz_mdl.vq_grid.copy_(_ramp.expand_as(zz_mdl.vq_grid))
        zz_mdl.vq_scale.fill_(1.0)
        _zi = zz_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _d_id = (_zi["continuous_crps_z"] - _zi["continuous_crps"])[0, [1, 3]].abs().max()
    assert _d_id < 1e-6, f"identity table: crps_z != crps by {_d_id:.2e}"
    with t.no_grad():
        zz_mdl.vq_grid.copy_((3.0 + 2.0 * _ramp).expand_as(zz_mdl.vq_grid))
        zz_mdl.vq_scale.fill_(4.0)
        _zl = zz_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _exp = _zl["continuous_crps"][0, [1, 3]] * (2.0 / 4.0)
    _d_lin = (_zl["continuous_crps_z"][0, [1, 3]] - _exp).abs().max()
    assert _d_lin < 1e-6, f"linear table: crps_z != (b/s)*crps by {_d_lin:.2e}"
    zz_mdl.zero_grad()
    _zl2 = zz_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _zl2["continuous_crps_z"].sum().backward()
    assert zz_mdl.get_input_embeddings().weight.grad.abs().sum() > 0
    # an unfilled table must fail loudly rather than score everything against
    # a constant -- the failure mode the _out params used to have.
    _empty = BasisBlendedCausalLM(BasisBlendedConfig(**zz_cfg.to_dict()))
    _raised_vq = False
    try:
        _empty(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    except AssertionError:
        _raised_vq = True
    assert _raised_vq, "an all-zero value table must raise, not score silently"
    print(
        f"continuous_crps_z: identity table == crps ({float(_d_id):.1e}), linear "
        f"table == (b/s)*crps ({float(_d_lin):.1e}), gradients flow, empty "
        "table refused: OK"
    )

    # ---- untie_continuous_head: the CH ablation ----
    ch_base = {
        **tch_cfg.to_dict(),
        "decoupled_magnitude": False,
        "continuous_crps": False,
        "continuous_moments": False,
    }
    _raised = False
    try:
        BasisBlendedCausalLM(
            BasisBlendedConfig(**{**ch_base, "untie_output_basis": True})
        )
    except AssertionError:
        _raised = True
    assert _raised, (
        "untie_output_basis + tied_continuous_head without "
        "untie_continuous_head is the half-untied hybrid and must be refused"
    )
    ch_cfg = BasisBlendedConfig(
        **{**ch_base, "untie_output_basis": True, "untie_continuous_head": True}
    )
    ch_mdl = BasisBlendedCausalLM(ch_cfg)
    _n_cat_ch = max(len(bv_dm["categories"]), 1)
    _start_ch = int(ch_mdl.category_base_id_t.min())
    for _n in ("log_alpha_out", "log_beta_out", "log_importance_out"):
        assert hasattr(ch_mdl, _n), f"{_n} missing under untie_continuous_head"
    assert t.allclose(ch_mdl.log_alpha_out, ch_mdl.log_alpha)
    assert t.allclose(ch_mdl.log_importance_out, ch_mdl.log_importance)
    assert t.allclose(
        ch_mdl._out_basis_table().flatten(0, 1),
        ch_mdl.get_input_embeddings().weight[_start_ch : _start_ch + _n_cat_ch * k_dm],
    )
    # the curve is scored from a FIXED hidden state on purpose: through a
    # full forward the input-side parameters would also move h (they build
    # the numeric token's input embedding), so the comparison would say
    # nothing about which curve the HEAD reads.
    _h_ch = t.randn(Bc, Tc, 32)

    def ch_curve(m):
        return m._continuous_logprob(_h_ch, c_cat, c_rank, _w)["logprob"]

    tied_ref = BasisBlendedCausalLM(BasisBlendedConfig(**ch_base))
    tied_ref.load_state_dict(
        {
            _k: _v
            for _k, _v in ch_mdl.state_dict().items()
            if not _k.endswith("_out") and _k != "basis_out_embed"
        }
    )
    _lp_ch = ch_curve(ch_mdl)
    assert (_lp_ch - ch_curve(tied_ref)).abs().max() < 1e-6, (
        "seeded equal, the untied curve must reproduce the tied one exactly "
        "-- a forgotten _out substitution shows up here"
    )
    with t.no_grad():
        ch_mdl.log_alpha.normal_(0.0, 0.3)
        ch_mdl.log_importance.normal_(0.0, 0.3)
        ch_mdl.get_input_embeddings().weight.normal_(0.0, 0.3)
    assert (ch_curve(ch_mdl) - _lp_ch).abs().max() < 1e-6, (
        "input-side parameters must NOT move the untied curve"
    )
    with t.no_grad():
        ch_mdl.log_alpha_out.normal_(0.0, 0.3)
        ch_mdl.basis_out_embed.normal_(0.0, 0.3)
    assert (ch_curve(ch_mdl) - _lp_ch).abs().max() > 1e-4, (
        "output-side parameters MUST move the untied curve"
    )
    ch_mdl.zero_grad()
    ch_curve(ch_mdl).sum().backward()
    for _n in ("basis_out_embed", "log_alpha_out", "log_beta_out"):
        _g3 = getattr(ch_mdl, _n).grad
        assert _g3 is not None and _g3.abs().sum() > 0, f"no gradient to {_n}"
    assert ch_mdl.log_alpha.grad is None or ch_mdl.log_alpha.grad.abs().sum() == 0, (
        "the input-side shape params must get no gradient from the curve"
    )

    with tempfile.TemporaryDirectory() as d:
        ch_mdl.save_pretrained(d)
        ch_re = AutoModelForCausalLM.from_pretrained(d)
        assert ch_re.config.untie_continuous_head is True
        assert t.allclose(ch_re.log_alpha_out, ch_mdl.log_alpha_out)
        assert t.allclose(ch_re.basis_out_embed, ch_mdl.basis_out_embed)
        import safetensors.torch as _st_ch

        _sd_ch = _st_ch.load_file(f"{d}/model.safetensors")
        del _sd_ch["log_importance_out"]
        del _sd_ch["basis_out_embed"]
        _st_ch.save_file(_sd_ch, f"{d}/model.safetensors", metadata={"format": "pt"})
        ch_missing = AutoModelForCausalLM.from_pretrained(d)
    # the _out params never had missing-key recovery: _init_output_basis ran
    # only from __init__ and returned early on the meta device, i.e. exactly
    # the load path where it was needed, so this used to reload uninitialised
    # memory. See _init_output_side.
    assert t.isfinite(ch_missing.log_importance_out).all()
    assert t.isfinite(ch_missing.basis_out_embed).all()
    assert t.allclose(ch_missing.log_importance_out, ch_missing.log_importance)
    assert t.allclose(
        ch_missing.basis_out_embed,
        ch_missing.get_input_embeddings().weight[
            _start_ch : _start_ch + _n_cat_ch * k_dm
        ],
    ), "a missing basis_out_embed must reload as the input-side seed"
    print(
        "untie_continuous_head: seeded from the input side, reproduces TCH "
        "when equal, only _out params move the curve, gradients flow, "
        "half-untied refused, missing _out keys reload finite: OK"
    )

    # ---- continuous_category_logits: level 1 from the same measure ----
    ccl_cfg = BasisBlendedConfig(**{**ch_base, "continuous_category_logits": True})
    ccl_mdl = BasisBlendedCausalLM(ccl_cfg)
    _n_ccl = max(len(ccl_cfg.categories), 1)
    _k_ccl = int(ccl_cfg.k)
    _st_ccl = int(ccl_mdl.category_base_id_t.min())
    _bnd_ccl = float(getattr(ccl_cfg, "continuous_logit_bound", 8.0))
    with t.no_grad():
        _o_ccl = ccl_mdl(
            input_ids=c_ids,
            category_ids=c_cat,
            ranks=c_rank,
            rank_widths=_w,
            output_hidden_states=True,
        )
    _lg_ccl = _o_ccl.logits.float()
    _lz_ccl = ccl_mdl._category_logz(_o_ccl.hidden_states[-1]).float()
    _ids_ccl = (
        t.arange(_n_ccl).unsqueeze(-1) * _k_ccl + _st_ccl + t.arange(_k_ccl)
    ).reshape(-1)
    _lse_ccl = t.logsumexp(
        _lg_ccl[..., _ids_ccl].view(*_lg_ccl.shape[:-1], _n_ccl, _k_ccl), dim=-1
    )
    # 1. the k slots of a category carry exactly log Z_c between them
    _d1_ccl = float((_lse_ccl - _lz_ccl).abs().max())
    assert _d1_ccl < 1e-4, f"logsumexp over a category's slots != log Z_c ({_d1_ccl})"
    # 2. so the induced P(c) is the coherent Z_c / N, and the vocabulary is
    #    still a proper distribution
    _lp_ccl = t.log_softmax(_lg_ccl, dim=-1)
    _lP_ccl = t.logsumexp(
        _lp_ccl[..., _ids_ccl].view(*_lp_ccl.shape[:-1], _n_ccl, _k_ccl), dim=-1
    )
    _lN_ccl = t.logsumexp(t.cat([_lg_ccl[..., :_st_ccl], _lz_ccl], dim=-1), dim=-1)
    _d2_ccl = float((_lP_ccl - (_lz_ccl - _lN_ccl.unsqueeze(-1))).abs().max())
    assert _d2_ccl < 1e-4, f"P(c) is not Z_c/N ({_d2_ccl})"
    assert t.allclose(_lp_ccl.exp().sum(-1), t.ones_like(_lN_ccl), atol=1e-5), (
        "the vocabulary must still normalise to 1"
    )
    # 3. log Z_c against an INDEPENDENT integral: uniform grid + Simpson,
    #    neither the composite Gauss-Legendre rule nor its node set.
    with t.no_grad():
        _G = 2001
        _x = t.linspace(0.0, 1.0, _G, dtype=t.float64)
        _W_ccl = ccl_mdl._blend_weights_rowwise(
            t.arange(_n_ccl), _x.float().unsqueeze(0).expand(_n_ccl, _G), "in"
        ).double()
        _E_ccl = (
            ccl_mdl.get_input_embeddings()
            .weight[_st_ccl : _st_ccl + _n_ccl * _k_ccl]
            .view(_n_ccl, _k_ccl, -1)
            .double()
        )
        _ba_ccl, _ = ccl_mdl._active_masks(_E_ccl.device)
        _act_ccl = _ba_ccl.unsqueeze(-1).double()
        _eb_ccl = (_E_ccl * _act_ccl).sum(1) / _act_ccl.sum(1).clamp_min(1.0)
        _D_ccl = (_E_ccl - _eb_ccl.unsqueeze(1)) * _act_ccl
        _h_ccl = _o_ccl.hidden_states[-1][0].double()
        _g_ccl = t.einsum("crk,ckh,nh->ncr", _W_ccl, _D_ccl, _h_ccl)
        _g_ccl = _bnd_ccl * t.tanh(_g_ccl / _bnd_ccl)
        _sw = t.ones(_G, dtype=t.float64)
        _sw[1:-1:2], _sw[2:-1:2] = 4.0, 2.0
        _sw = _sw * (1.0 / (_G - 1)) / 3.0
        _ref_ccl = (
            t.logsumexp(_g_ccl + _sw.log().view(1, 1, -1), dim=-1)
            + _h_ccl @ _eb_ccl.t()
        )
    _d3_ccl = float((_lz_ccl[0].double() - _ref_ccl).abs().max())
    assert _d3_ccl < 5e-3, f"log Z_c disagrees with a Simpson integral ({_d3_ccl})"
    # 4. it must actually change the logits -- the silent-no-op check
    ccl_mdl.config.continuous_category_logits = False
    with t.no_grad():
        _off_ccl = ccl_mdl(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w
        ).logits.float()
    ccl_mdl.config.continuous_category_logits = True
    _num_ccl = slice(_st_ccl, _st_ccl + _n_ccl * _k_ccl)
    assert not t.allclose(_lg_ccl[..., _num_ccl], _off_ccl[..., _num_ccl], atol=1e-3), (
        "the flag did not change the numeric logits"
    )
    assert t.allclose(_lg_ccl[..., :_st_ccl], _off_ccl[..., :_st_ccl], atol=1e-3), (
        "the flag must not touch the non-numeric logits"
    )
    # 5. gradients reach the embedding table through the new path
    _o2_ccl = ccl_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _o2_ccl.logits[..., _num_ccl].sum().backward()
    _ge_ccl = ccl_mdl.get_input_embeddings().weight.grad
    assert _ge_ccl is not None and float(_ge_ccl[_num_ccl].abs().sum()) > 0, (
        "no gradient into the anchors through log Z_c"
    )
    # 6. refused without the head it integrates, and against FiLM
    for _bad, _why in (
        ({"tied_continuous_head": False}, "without tied_continuous_head"),
        ({"film_value_embed": True, "film_hidden": 8}, "with film_value_embed"),
    ):
        _r_ccl = False
        try:
            BasisBlendedCausalLM(
                BasisBlendedConfig(
                    **{**ch_base, "continuous_category_logits": True, **_bad}
                )
            )
        except AssertionError:
            _r_ccl = True
        assert _r_ccl, f"continuous_category_logits must be refused {_why}"

    # ---- value_vocab_curve: the quadrature replaced by the data's own bins ----
    import tempfile as _tf_vvc

    _cats_vvc = list(tch_cfg.categories)
    _vocab_vvc = {}
    for _i, _c in enumerate(_cats_vvc):
        _n = 3 + (_i % 4)  # 3..6 entries, deliberately ragged
        _edges = np.linspace(0.0, 1.0, _n + 1)
        _vocab_vvc[_c] = dict(
            lo=_edges[:-1].tolist(),
            hi=_edges[1:].tolist(),
            rank=((_edges[:-1] + _edges[1:]) / 2).tolist(),
            count=[10] * _n,
        )
    _vvf = pathlib.Path(_tf_vvc.mkdtemp()) / "vv.json"
    _vvf.write_text(json.dumps(dict(meta=dict(min_count=10), vocab=_vocab_vvc)))
    vvc_cfg = BasisBlendedConfig(
        **{**ch_base, "value_vocab_file": str(_vvf), "value_vocab_curve": True}
    )
    vvc_mdl = BasisBlendedCausalLM(vvc_cfg)
    with t.no_grad():
        _o_vvc = vvc_mdl(
            input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w
        )
    _vlp = _o_vvc.get("value_logprob")
    assert _vlp is not None, "value_logprob must be emitted"
    assert _o_vvc.get("continuous_logprob") is None, (
        "continuous_logprob must be SUPPRESSED under value_vocab_curve -- Loss "
        "prefers it and would silently score the quadrature instead"
    )
    assert _vlp.dtype == t.float32, "log-probabilities must not be stored in bf16"
    _nm_vvc = c_cat[:, 1:] >= 0
    _lp_vvc = _vlp[:, 1:][_nm_vvc]
    assert bool((_lp_vvc <= 1e-6).all()), "log P must stay <= 0"
    # the base-rate property: g == 0 must give P(b) = w_b exactly, which is
    # the whole reason the log w_b term is there
    _vv_vvc = vvc_mdl._value_vocab()
    with t.no_grad():
        _z_vvc = vvc_mdl._value_logprob_curve(
            t.zeros(c_ids.shape[0], c_ids.shape[1], vvc_cfg.base_config["hidden_size"]),
            c_cat,
            c_rank,
        )
    _g_vvc = vvc_mdl._entry_of(_vv_vvc, c_cat[:, 1:][_nm_vvc], c_rank[:, 1:][_nm_vvc])
    _w_vvc = _vv_vvc["hi"][_g_vvc] - _vv_vvc["lo"][_g_vvc]
    _e_vvc = float((_z_vvc[:, 1:][_nm_vvc].exp() - _w_vvc).abs().max())
    assert _e_vvc < 1e-5, f"log w must recover the base rate ({_e_vvc})"
    # gradients reach the anchors through the curve
    _o2_vvc = vvc_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=_w)
    _o2_vvc.value_logprob[:, 1:][_nm_vvc].sum().backward()
    _ge_vvc = vvc_mdl.get_input_embeddings().weight.grad
    assert _ge_vvc is not None and float(_ge_vvc.abs().sum()) > 0, "no gradient"
    # refused where it would score the wrong thing
    for _bad_vvc, _why_vvc in (
        ({"tied_continuous_head": False}, "without tied_continuous_head"),
        ({"value_vocab_file": None}, "without value_vocab_file"),
        ({"film_value_embed": True, "film_hidden": 8}, "with film_value_embed"),
    ):
        _r_vvc = False
        try:
            BasisBlendedCausalLM(
                BasisBlendedConfig(
                    **{
                        **ch_base,
                        "value_vocab_file": str(_vvf),
                        "value_vocab_curve": True,
                        **_bad_vvc,
                    }
                )
            )
        except AssertionError:
            _r_vvc = True
        assert _r_vvc, f"value_vocab_curve must be refused {_why_vvc}"

    # ---- value_vocab_curve + continuous_category_logits: ONE measure ----
    # The combination is only coherent if level 1 integrates what level 2
    # normalises by. _category_logz switches to the entry sum when
    # value_vocab_curve is on; without that it would integrate a continuous
    # density at level 1 while level 2 summed a discrete one -- the very
    # mismatch continuous_category_logits exists to remove.
    vc_cfg = BasisBlendedConfig(
        **{
            **ch_base,
            "value_vocab_file": str(_vvf),
            "value_vocab_curve": True,
            "continuous_category_logits": True,
        }
    )
    vc_mdl = BasisBlendedCausalLM(vc_cfg)
    _n_vc = max(len(vc_cfg.categories), 1)
    _k_vc = int(vc_cfg.k)
    _st_vc = int(vc_mdl.category_base_id_t.min())
    with t.no_grad():
        _o_vc = vc_mdl(
            input_ids=c_ids,
            category_ids=c_cat,
            ranks=c_rank,
            rank_widths=_w,
            output_hidden_states=True,
        )
        _lz_vc = vc_mdl._category_logz(_o_vc.hidden_states[-1]).float()
    # level 1's slots carry log Z_c
    _ids_vc = (
        t.arange(_n_vc).unsqueeze(-1) * _k_vc + _st_vc + t.arange(_k_vc)
    ).reshape(-1)
    _lg_vc = _o_vc.logits.float()
    _d1_vc = float(
        (
            t.logsumexp(_lg_vc[..., _ids_vc].view(*_lg_vc.shape[:-1], _n_vc, _k_vc), -1)
            - _lz_vc
        )
        .abs()
        .max()
    )
    assert _d1_vc < 1e-4, f"slots do not carry log Z_c ({_d1_vc})"
    # and log Z_c is the ENTRY sum, not the quadrature integral
    with t.no_grad():
        _W, _lw, _vd, _ = vc_mdl._entry_tables(_lz_vc.device, vc_mdl._curve_side())
        _E = (
            vc_mdl.get_input_embeddings()
            .weight[_st_vc : _st_vc + _n_vc * _k_vc]
            .view(_n_vc, _k_vc, -1)
            .float()
        )
        _ba, _ = vc_mdl._active_masks(_lz_vc.device)
        _ac = _ba.unsqueeze(-1).float()
        _eb = (_E * _ac).sum(1) / _ac.sum(1).clamp_min(1.0)
        _D = (_E - _eb.unsqueeze(1)) * _ac
        _h = _o_vc.hidden_states[-1].reshape(-1, _E.shape[-1]).float()
        _bd = float(getattr(vc_cfg, "continuous_logit_bound", 8.0))
        _g = _bd * t.tanh(
            t.einsum("cek,nck->nce", _W, t.einsum("ckh,nh->nck", _D, _h)) / _bd
        )
        _ref_vc = (
            t.logsumexp(
                (_g + _lw.unsqueeze(0)).masked_fill(~_vd.unsqueeze(0), float("-inf")),
                -1,
            )
            + _h @ _eb.t()
        ).view(*_o_vc.hidden_states[-1].shape[:2], _n_vc)
    _d2_vc = float((_lz_vc - _ref_vc).abs().max())
    assert _d2_vc < 1e-4, f"log Z_c is not the entry sum ({_d2_vc})"
    # and it is genuinely different from what the quadrature would give
    vc_mdl.config.value_vocab_curve = False
    with t.no_grad():
        _quad_vc = vc_mdl._category_logz(_o_vc.hidden_states[-1]).float()
    vc_mdl.config.value_vocab_curve = True
    _d3_vc = float((_lz_vc - _quad_vc).abs().mean())
    assert _d3_vc > 1e-4, (
        "the entry measure and the quadrature agree exactly -- _category_logz "
        "is not switching on value_vocab_curve"
    )
    print(
        f"value_vocab_curve + continuous_category_logits: both levels use the "
        f"ENTRY measure (slots {_d1_vc:.1e}, entry sum {_d2_vc:.1e}), and it "
        f"differs from the quadrature by {_d3_vc:.1e} nats: OK"
    )

    print(
        f"value_vocab_curve: log P <= 0, g==0 recovers the base rate "
        f"({_e_vvc:.1e}), continuous_logprob suppressed, float32, gradients "
        "flow, refused without the head/file and against FiLM: OK"
    )

    print(
        f"continuous_category_logits: slots carry log Z_c ({_d1_ccl:.1e}), "
        f"P(c) = Z_c/N ({_d2_ccl:.1e}), matches an independent Simpson "
        f"integral ({_d3_ccl:.1e}), changes only the numeric logits, "
        "gradients flow, refused without the head and against FiLM: OK"
    )

    # B-SPLINE WEIGHTS. Partition of unity is a construction here, not
    # something a softmax enforces, so it is worth asserting exactly that --
    # plus the local support that distinguishes this from softmax weights.
    bs_cfg = BasisBlendedConfig(
        **{
            **tch_cfg.to_dict(),
            "decoupled_magnitude": False,
            "bspline_weights": True,
            "bspline_degree": 3,
            "interval_mixture_weights": True,
        }
    )
    bs_mdl = BasisBlendedCausalLM(bs_cfg)
    assert bs_mdl.knot_logits.shape == (max(len(bv_dm["categories"]), 1), k_dm - 3)
    _ci = t.zeros(4, dtype=t.long)
    with t.no_grad():
        bs_mdl.knot_logits.normal_(0.0, 0.4)  # non-uniform knots
    _Bs = bs_mdl._bspline_basis(_ci, t.rand(4, 500))
    assert (_Bs >= 0).all(), "B-spline basis must be non-negative"
    assert (_Bs.sum(-1) - 1).abs().max() < 1e-5, "must sum to exactly 1"
    assert int((_Bs > 1e-9).sum(-1).max()) <= 4, "support must be degree+1 spans"
    bs_w = bs_mdl(
        input_ids=c_ids,
        category_ids=c_cat,
        ranks=c_rank,
        rank_widths=t.full((Bc, Tc), 0.05),
    )["mixture_weights"]
    _n = c_cat >= 0
    assert (bs_w[_n].sum(-1) - 1).abs().max() < 1e-5
    # POLY CURVE BASIS. Not a blend: the coefficients are signed and do not
    # sum to one, so what must hold instead is that g(r) equals h . e_c(r) up
    # to a constant, that the basis is orthonormal under the uniform measure
    # on [0,1], and that log P is still a probability.
    pc_cfg = BasisBlendedConfig(
        **{
            **tch_cfg.to_dict(),
            "decoupled_magnitude": False,
            "poly_curve_basis": True,
            "interval_mixture_weights": True,
        }
    )
    pc_mdl = BasisBlendedCausalLM(pc_cfg)
    assert not hasattr(pc_mdl, "knot_logits")
    _r = t.linspace(1e-6, 1 - 1e-6, 40001)
    _Bm = pc_mdl._legendre_basis(_r, k_dm)
    _gram = (_Bm.t() @ _Bm) / len(_r)
    assert (_gram - t.eye(k_dm)).abs().max() < 1e-3, "basis must be orthonormal"
    _lp = pc_mdl(
        input_ids=c_ids,
        category_ids=c_cat,
        ranks=c_rank,
        rank_widths=t.full((Bc, Tc), 0.05),
    )["continuous_logprob"]
    assert bool((_lp[:, 1:] <= 1e-6).all()), "log P must be <= 0"
    _full = pc_mdl(
        input_ids=c_ids,
        category_ids=c_cat,
        ranks=c_rank,
        rank_widths=t.full((Bc, Tc), 2.0),
    )["continuous_logprob"]
    assert _full[0, [1, 3]].abs().max() < 1e-4, "P(r in [0,1]) must be 1"
    pc_mdl.zero_grad()
    pc_mdl(
        input_ids=c_ids,
        category_ids=c_cat,
        ranks=c_rank,
        rank_widths=t.full((Bc, Tc), 0.05),
    )["continuous_logprob"].sum().backward()
    assert pc_mdl.get_input_embeddings().weight.grad.abs().sum() > 0
    print(
        "poly_curve_basis: orthonormal to "
        f"{float((_gram - t.eye(k_dm)).abs().max()):.1e}, log P <= 0, "
        "P(r in [0,1]) = 1, no shape params, gradients reach embed_tokens: OK"
    )

    print(
        "bspline_weights: partition of unity "
        f"{float((_Bs.sum(-1) - 1).abs().max()):.1e}, support <= degree+1, "
        f"{2 * k_dm} shape params -> {k_dm - 3 - 1} free knots: OK"
    )

    # PLAIN BLEND + tied head: decoupled_magnitude is optional. With it off,
    # e_c(r) = e_hat_c + sum_i w_i e_c,i, so g(r) = sum_i w_i z_i is LINEAR in
    # the weights -- no direction normalisation, no magnitude channel.
    pb_cfg = BasisBlendedConfig(**{**tch_cfg.to_dict(), "decoupled_magnitude": False})
    pb_mdl = BasisBlendedCausalLM(pb_cfg)
    assert not hasattr(pb_mdl, "basis_log_magnitude"), (
        "no magnitude parameter should exist under the plain blend"
    )
    with t.no_grad():
        pb_mdl.get_input_embeddings().weight.normal_(0.0, 0.05)

    def pb_lp(width):
        w = t.zeros(Bc, Tc)
        w[0, 1] = w[0, 3] = width
        return pb_mdl(input_ids=c_ids, category_ids=c_cat, ranks=c_rank, rank_widths=w)[
            "continuous_logprob"
        ]

    assert pb_lp(2.0)[0, [1, 3]].abs().max() < 1e-4, "P(r in [0,1]) must be 1"
    for _w in (0.001, 0.02, 0.3, 1.0, 2.0):
        assert bool((pb_lp(_w)[0, [1, 3]] <= 0).all()), f"log P <= 0 failed at {_w}"
    pb_mdl.zero_grad()
    pb_lp(0.06).sum().backward()
    assert pb_mdl.log_alpha.grad is not None and pb_mdl.log_alpha.grad.abs().sum() > 0
    _pe = pb_mdl.get_input_embeddings().weight.grad
    assert _pe is not None and _pe.abs().sum() > 0
    print(
        "tied_continuous_head + PLAIN BLEND: no magnitude param, P(r in [0,1])=1, "
        "log P <= 0 across widths, gradients reach log_alpha/embed_tokens: OK"
    )

    # REGRESSION. log P must be <= 0 -- it is a probability. An earlier
    # version computed the numerator and Z with INDEPENDENT quadrature rules,
    # so P <= 1 held only by accuracy; training then learned a density spike
    # narrower than a quadrature panel, Z was under-estimated by up to 21
    # nats, log P reached +25.77 and the loss went unbounded below (observed:
    # eval_loss 0.7306, grad_norm 116). The rules now share a node set, so the
    # numerator's terms are a strict subset of the denominator's and P <= 1 is
    # exact. Drive the head hard and check it holds.
    with t.no_grad():
        tch_mdl.basis_log_magnitude.fill_(4.0)  # softplus ~ 4.02, vs 0.64 init
        tch_mdl.get_input_embeddings().weight.normal_(0.0, 0.5)
    for _w in (0.001, 0.02, 0.3, 1.0, 2.0):
        _lp = tch_lp(_w)[0, [1, 3]]
        assert bool((_lp <= 0).all()), (
            f"log P must be <= 0; width {_w} gave {[float(x) for x in _lp]}"
        )
    print(
        "tied_continuous_head: log P <= 0 under a hard-driven head "
        "(magnitude 4.0, embeddings sigma 0.5), widths 0.001-2.0: OK"
    )

    # ---- numeric_loss on the tied continuous head ----
    # both_mdl already emits continuous_crps AND continuous_moments, so one
    # forward feeds all three kinds. The assertion is that the knob is
    # exactly the coefficient of the statistic the model returned -- if the
    # loss ever recomputed the quantity itself, or read the arithmetic
    # mixture instead of the head's own density, these would drift.
    tl_ids = t.tensor(
        [[1, bv_dm["category_base_id"][0], bv_dm["category_base_id"][1], 2]]
    )
    tl_cat = t.tensor([[-1, 0, 1, -1]])
    tl_rank = t.tensor([[0.0, 0.27, 0.81, 0.0]])
    tl_wid = t.full((1, 4), 0.06)
    tl_kw = dict(category_ids=tl_cat, ranks=tl_rank, rank_widths=tl_wid)
    with t.no_grad():
        tl_out = both_mdl(input_ids=tl_ids, **tl_kw)
    _m_num = tl_cat[:, 1:] >= 0

    def tl_loss_at(kind, weight, train=True):
        cfg_ = {
            "custom_loss": True,
            "basis_blended_tokens": {
                "k": k_dm,
                "mixture_nll_loss": True,
                "interval_nll_loss": True,
                "interval_mixture_weights": True,
                "tied_continuous_head": True,
                "component_family": "truncnorm",
            },
        }
        if kind is not None:
            cfg_["numeric_loss"] = kind
            cfg_["numeric_loss_weight"] = weight
        lo_ = Loss(cfg=OmegaConf.create(cfg_), tkzr_cfg=tkzr_cfg, basis_vocab=bv_dm)
        lo_._is_train = train
        return float(lo_.basis_blended_token_loss(tl_out, tl_ids, **tl_kw))

    _tl_0 = tl_loss_at("was", 0.0)
    assert _tl_0 == tl_loss_at(None, 0.0), (
        "numeric_loss at weight 0 must be bit-identical to no numeric_loss"
    )
    _exp = {
        "was": float(tl_out["continuous_w1"][:, 1:][_m_num].sum()),
        "mse": float(
            ((tl_out["continuous_mean"][:, 1:] - tl_rank[:, 1:])[_m_num] ** 2).sum()
        ),
        "crps": float(tl_out["continuous_crps"][:, 1:][_m_num].sum()),
    }
    for _kind, _sum in _exp.items():
        _d = tl_loss_at(_kind, 0.3) - _tl_0
        assert abs(_d - 0.3 * _sum) < 1e-4, (
            f"numeric_loss: {_kind} added {_d:.6f}, expected {0.3 * _sum:.6f}"
        )
    # TRAIN WITH IT, EVALUATE WITHOUT IT. At eval the extra term must vanish
    # entirely, whatever its weight -- otherwise a smaller weight would score
    # a lower eval_loss mechanically and the tuner would chase that instead
    # of the better model. See Loss.custom_loss.
    for _kind in ("was", "mse", "crps"):
        _ev = tl_loss_at(_kind, 3.0, train=False)
        assert abs(_ev - _tl_0) < 1e-6, (
            f"numeric_loss: {_kind} still contributes at eval ({_ev} vs {_tl_0})"
        )
    print(
        "numeric_loss on the tied head: was/mse/crps each enter at exactly "
        "their weight, weight 0 is a no-op, and NONE of them contributes at "
        "eval: OK"
    )

    # ---- decile-baseline numeric losses (NTL over the bin vocabulary) ----
    # No basis_blended_tokens: these score the RAW (code, decile) vocabulary,
    # which is what the baseline arm predicts. With zero logits every
    # category's 4 bins are equiprobable, and every quantity below is then
    # closed form -- including the CRPS, because equal mass on 4 equal-width
    # bins IS the uniform density, so it must reproduce the same
    # CRPS(U, r) = (r^2 + (1-r)^2)/2 - 1/6 the tied head's test uses.
    q_labels = t.tensor([[0, 3, 7, 1]])  # targets: AGE Q0, VTL Q0, BOS
    q_true, q_w = 0.125, 0.25  # bin midpoint, bin width (n_bins = 4)
    bl_cfg = OmegaConf.create(
        {"custom_loss": True, "numeric_loss": "was", "numeric_loss_weight": 0.0}
    )
    bl = Loss(cfg=bl_cfg, tkzr_cfg=tkzr_cfg)
    q_out = {"logits": t.zeros(1, q_labels.shape[1], len(bl.vocab))}
    _mse = bl.quantile_token_loss(q_out, q_labels)
    _was = bl.was_token_loss(q_out, q_labels)
    _crp = bl.crps_token_loss(q_out, q_labels)
    for _nm, _v in (("mse", _mse), ("was", _was), ("crps", _crp)):
        assert _v.dim() == 0, f"{_nm} must be a scalar, got shape {tuple(_v.shape)}"
    # two numeric targets, one per category, both Q0 -- so each term is
    # exactly twice its per-position value, which also pins the
    # sum-reduction and the per-category loop.
    _mse_x = 2 * (0.5 - q_true) ** 2
    _was_x = 2 * sum(q_w * abs((i + 0.5) * q_w - q_true) for i in range(4))
    _crp_x = 2 * ((q_true**2 + (1 - q_true) ** 2) / 2 - 1 / 6)
    assert abs(float(_mse) - _mse_x) < 1e-6, f"NTL-MSE {float(_mse)} != {_mse_x}"
    assert abs(float(_was) - _was_x) < 1e-6, f"NTL-WAS {float(_was)} != {_was_x}"
    assert abs(float(_crp) - _crp_x) < 1e-6, f"bin CRPS {float(_crp)} != {_crp_x}"
    # a batch whose only target is non-numeric contributes nothing
    assert float(bl.was_token_loss(q_out, t.tensor([[0, 1, 2, 1]]))) == 0.0

    # numeric_loss_weight is exactly the coefficient, and weight 0 is
    # bit-identical to not configuring the term at all.
    _n_tok = q_labels[:, 1:].numel()
    _l0 = bl.custom_loss(q_out, q_labels)
    _lw = Loss(
        cfg=OmegaConf.create(
            {"custom_loss": True, "numeric_loss": "was", "numeric_loss_weight": 0.3}
        ),
        tkzr_cfg=tkzr_cfg,
    ).custom_loss(q_out, q_labels)
    _lnone = Loss(
        cfg=OmegaConf.create({"custom_loss": True}), tkzr_cfg=tkzr_cfg
    ).custom_loss(q_out, q_labels)
    assert abs(float(_lw - _l0) - 0.3 * _was_x / _n_tok) < 1e-6, (
        "numeric_loss_weight must scale the added term exactly"
    )
    assert float(_l0) == float(_lnone), (
        "numeric_loss at weight 0 must be bit-identical to no numeric_loss"
    )
    _bl_w = Loss(
        cfg=OmegaConf.create(
            {"custom_loss": True, "numeric_loss": "was", "numeric_loss_weight": 0.3}
        ),
        tkzr_cfg=tkzr_cfg,
    )
    _tr = float(_bl_w.custom_loss(q_out, q_labels, training=True))
    _ev = float(_bl_w.custom_loss(q_out, q_labels, training=False))
    assert abs((_tr - _ev) - 0.3 * _was_x / _n_tok) < 1e-6, (
        "the train/eval difference must be exactly the weighted term"
    )
    assert _ev == float(_lnone), (
        "at eval the decile-baseline numeric term must vanish, leaving the "
        "same loss as a config with no numeric_loss at all"
    )
    # ...and these are baseline-only: under basis_blended_tokens they would
    # index the raw vocabulary against collapsed-vocabulary logits.
    _raised_bl = False
    try:
        Loss(
            cfg=OmegaConf.create(
                {
                    "custom_loss": True,
                    "basis_blended_tokens": {"k": 3},
                    "was_token_loss": {"qt_weight": 1.0},
                }
            ),
            tkzr_cfg=tkzr_cfg,
        )
    except AssertionError:
        _raised_bl = True
    assert _raised_bl, "was_token_loss under basis_blended_tokens must be refused"
    print(
        f"baseline numeric losses: NTL-MSE {float(_mse):.6f}, NTL-WAS "
        f"{float(_was):.6f}, bin CRPS {float(_crp):.6f} all exact; weight "
        "scales exactly and 0 is a no-op; basis combination refused: OK"
    )

    print("all basis_blended.py self-tests passed")
    # breakpoint()
