#!/usr/bin/env python3

"""
basis blended tokens: wraps a base causal LM so that numeric (category, exact
rank) entries are embedded as a per-category shared vector plus a
Beta-mixture blend of k learned basis elements per category, and predicted
via a matching soft-label loss target (see loss.py's basis_blended_token_loss),
instead of a single fused bin token
"""

import dataclasses
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

_RANK_EPS = 1e-4


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
        train_importance_scale: bool = True,
        train_category_embed: bool = True,
        numerical_basis_model: bool = False,
        num_non_numeric: int = 0,
        categories: list[str] | None = None,
        category_base_id: list[int] | None = None,
        basis_lookup: dict | None = None,
        raw_to_category: list[int] | None = None,
        raw_to_collapsed: list[int] | None = None,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.base_model_type = base_model_type
        self.base_config = base_config or {}
        self.k = k
        self.train_beta_params = train_beta_params
        self.train_importance_scale = train_importance_scale
        self.train_category_embed = train_category_embed
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
        self.num_non_numeric = num_non_numeric
        self.categories = categories or []
        self.category_base_id = category_base_id or []
        self.basis_lookup = basis_lookup or {}
        self.raw_to_category = raw_to_category or []
        self.raw_to_collapsed = raw_to_collapsed or []
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        super().__init__(pad_token_id=pad_token_id, **kwargs)

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
            self.e2 = nn.Parameter(t.zeros(n_cat, inner_cfg.hidden_size))
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
            self.gaussian_head = nn.Linear(inner_cfg.hidden_size, 2 * n_cat)
        else:
            k = config.k
            idx = t.arange(k, dtype=t.float32)
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
            self.log_importance = nn.Parameter(
                t.zeros(n_cat, k), requires_grad=config.train_importance_scale
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
        self.register_buffer(
            "category_base_id_t", t.tensor(config.category_base_id or [0], dtype=t.long)
        )

        self.post_init()

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

    def _mixture_weights(
        self,
        category_ids: t.Tensor,
        ranks: t.Tensor,
        return_log_pdf: bool = False,
        return_beta_params: bool = False,
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
        # alpha/beta are unconstrained during training and can drift into
        # ranges where lgamma is poorly conditioned (very small or very
        # large); clamp defensively rather than let that surface as NaN here.
        a = self.log_alpha[cat].exp().clamp(1e-3, 1e4)  # (B,T,k)
        b = self.log_beta[cat].exp().clamp(1e-3, 1e4)  # (B,T,k)
        r = ranks.to(a.dtype).clamp(_RANK_EPS, 1 - _RANK_EPS).unsqueeze(-1)
        log_pdf = (
            (a - 1) * r.log()
            + (b - 1) * (1 - r).log()
            - (t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b))
        )
        log_importance = self.log_importance[cat]  # (B,T,k)
        logits = log_pdf + log_importance
        # last-resort sanitation: never let a NaN/inf reach the softmax,
        # whatever its cause -- this is a plain terminal function of finite
        # inputs, so any non-finite value here is a bug, not real signal.
        logits = t.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
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
            if self.config.numerical_basis_model:
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
                e2 = self.e2[cat].to(e1.dtype)  # (B,T,H), untied
                v = ranks.to(e1.dtype).unsqueeze(-1)
                blended = e1 + v * e2
            else:
                w, log_pdf, beta_a, beta_b = self._mixture_weights(
                    category_ids, ranks, return_log_pdf=True, return_beta_params=True
                )  # (B,T,k), (B,T,k), (B,T,k), (B,T,k)
                k_dim = w.shape[-1]
                basis_ids = self.category_base_id_t[
                    category_ids.clamp(min=0)
                ].unsqueeze(-1) + t.arange(k_dim, device=input_ids.device)  # (B,T,k)
                blended = (w.unsqueeze(-1) * embed(basis_ids)).sum(dim=2)
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
        need_hidden = self.config.numerical_basis_model
        outputs = self.inner_model(
            inputs_embeds=base_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or need_hidden,
            **kwargs,
        )

        gaussian_params = None
        if need_hidden:
            # cast to float32 for the same numerical-stability reason
            # logits/beta_log_pdf are cast to float32 downstream in
            # loss.py -- this head's output feeds a Gaussian NLL directly.
            last_hidden = outputs.hidden_states[-1].to(dtype=t.float32)
            gaussian_params = self.gaussian_head(last_hidden)  # (B,T,2*n_cat)

        outputs_dict = dict(outputs.items())
        if need_hidden and not output_hidden_states:
            outputs_dict.pop("hidden_states", None)

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

    print("all basis_blended.py self-tests passed")
    # breakpoint()
