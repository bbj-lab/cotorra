#!/usr/bin/env python3

"""
basis blended tokens: wraps a base causal LM so that numeric (category, exact
rank) entries are embedded as a Beta-mixture blend of k learned basis
elements per category, and predicted via a matching soft-label loss target
(see loss.py's basis_blended_token_loss), instead of a single fused bin token
"""

import dataclasses

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


def build_basis_vocab(tkzr_cfg, k: int) -> dict:
    """
    collapse cocoa's fused bin vocabulary (num_non_numeric + num_categories *
    n_bins) down to the basis blended vocabulary (num_non_numeric +
    num_categories * k); single source of truth shared by the model config
    and the loss
    """
    lookup = dict(tkzr_cfg.lookup)
    raw_vocab_size = len(lookup)
    vocab = np.array(sorted(lookup, key=lookup.get))  # index == raw token id
    n_bins = tkzr_cfg.cfg.n_bins

    q_type = np.array(
        [v.endswith(tuple(f"Q{i}" for i in range(n_bins))) for v in vocab]
    )
    if q_type.any():
        qt_cats, _ = map(
            np.array, zip(*np.char.rsplit(vocab[q_type], sep="Q", maxsplit=1))
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

    category_base_id = []
    for i, cat in enumerate(categories):
        base = num_non_numeric + i * k
        category_base_id.append(base)
        for j in range(k):
            basis_lookup[f"{cat}Q{j}"] = base + j

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


@dataclasses.dataclass
class BasisBlendedCausalLMOutput(CausalLMOutputWithPast):
    """
    CausalLMOutputWithPast plus mixture_weights as a *declared* dataclass
    field (not a dynamically-added dict key) -- ModelOutput reconstruction
    (e.g. Accelerate's bf16 `convert_outputs_to_fp32`, which rebuilds via
    `type(data)({k: v for k, v in data.items()})`) only preserves declared
    fields, silently dropping anything else.
    """

    mixture_weights: t.Tensor | None = None


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

        n_cat, k = max(len(config.categories), 1), config.k
        idx = t.arange(k, dtype=t.float32)
        log_alpha0 = (idx + 1).log().unsqueeze(0).expand(n_cat, k).clone()
        log_beta0 = (k - idx).log().unsqueeze(0).expand(n_cat, k).clone()
        self.log_alpha = nn.Parameter(
            log_alpha0, requires_grad=config.train_beta_params
        )
        self.log_beta = nn.Parameter(log_beta0, requires_grad=config.train_beta_params)
        # importance scaling a_c,i (see "Determination of mixture weights" in
        # fuzzy_token_planning.md, extended): reweights each basis element's
        # density contribution independently of its Beta shape. log_a=0 (a=1
        # uniformly) at init reproduces the plain density-ratio formula
        # exactly, so this can't change behavior until training moves it.
        self.log_importance = nn.Parameter(
            t.zeros(n_cat, k), requires_grad=config.train_importance_scale
        )
        self.register_buffer(
            "category_base_id_t", t.tensor(config.category_base_id or [0], dtype=t.long)
        )

        self.post_init()

    def _init_weights(self, module):
        # HF's from_pretrained calls this for `self` (this whole module) any
        # time even one of its direct params is missing from the checkpoint
        # (e.g. an old checkpoint saved before log_importance existed) --
        # under the low_cpu_mem_usage/meta-device load path, skipping this
        # override leaves such params as uninitialized garbage memory rather
        # than the values __init__ intends. But the call is per-*module*,
        # not per-param, and log_alpha/log_beta/log_importance are 3
        # separate params of this one module -- so a blanket overwrite here
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
        idx = t.arange(self.config.k, dtype=t.float32)
        with t.no_grad():
            if not getattr(self.log_alpha, "_is_hf_initialized", False):
                self.log_alpha.copy_(
                    (idx + 1).log().unsqueeze(0).expand(n_cat, self.config.k)
                )
            if not getattr(self.log_beta, "_is_hf_initialized", False):
                self.log_beta.copy_(
                    (self.config.k - idx).log().unsqueeze(0).expand(n_cat, self.config.k)
                )
            if not getattr(self.log_importance, "_is_hf_initialized", False):
                self.log_importance.zero_()

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

    def _mixture_weights(self, category_ids: t.Tensor, ranks: t.Tensor) -> t.Tensor:
        """log-space softmax over the k basis elements' *importance-scaled*
        Beta densities at `ranks`: w_c,i = a_c,i*f(r,i) / sum_j a_c,j*f(r,j),
        for whatever category each position belongs to; see "Determination
        of mixture weights" in fuzzy_token_planning.md. Scaling by a_c,i in
        log-space before the softmax is exactly this ratio (softmax(x)_i =
        exp(x_i)/sum_j exp(x_j) with x = log(a) + log_pdf), so a_c,i can
        reweight each component's overall influence independently of its
        Beta shape."""
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
        return t.softmax(logits, dim=-1)

    def forward(
        self,
        input_ids: t.Tensor,
        category_ids: t.Tensor | None = None,
        ranks: t.Tensor | None = None,
        position_ids: t.Tensor | None = None,
        attention_mask: t.Tensor | None = None,
        output_hidden_states: bool = False,
        labels=None,  # unused: loss is computed externally by cotorra.loss
        **kwargs,
    ) -> BasisBlendedCausalLMOutput:
        embed = self.get_input_embeddings()
        base_embeds = embed(input_ids)

        if category_ids is None:
            category_ids = t.full_like(input_ids, -1)
        numeric = category_ids >= 0

        w = None
        if numeric.any():
            if ranks is None:
                ranks = t.zeros_like(input_ids, dtype=base_embeds.dtype)
            w = self._mixture_weights(category_ids, ranks)  # (B,T,k)
            k = w.shape[-1]
            basis_ids = self.category_base_id_t[category_ids.clamp(min=0)].unsqueeze(
                -1
            ) + t.arange(k, device=input_ids.device)  # (B,T,k)
            blended = (w.unsqueeze(-1) * embed(basis_ids)).sum(dim=2)
            base_embeds = t.where(
                numeric.unsqueeze(-1), blended.to(base_embeds.dtype), base_embeds
            )

        outputs = self.inner_model(
            inputs_embeds=base_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return BasisBlendedCausalLMOutput(
            **{k: v for k, v in outputs.items()}, mixture_weights=w
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
    print("order-statistic init (incl. log_importance=0): OK")

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

    loss = out.logits.sum()
    loss.backward()
    assert mdl.log_alpha.grad is not None and mdl.log_alpha.grad.abs().sum() > 0
    assert (
        mdl.log_importance.grad is not None and mdl.log_importance.grad.abs().sum() > 0
    )
    print("gradient reaches log_alpha/log_beta/log_importance: OK")

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

    mdl2 = BasisBlendedCausalLM(BasisBlendedConfig(**cfg.to_dict()))
    for pn, p in mdl.named_parameters():
        mdl2.state_dict()[pn].copy_(p.detach())
    with tempfile.TemporaryDirectory() as d:
        mdl2.save_pretrained(d)
        reloaded = AutoModelForCausalLM.from_pretrained(d)
    assert isinstance(reloaded, BasisBlendedCausalLM)
    assert t.allclose(reloaded.log_alpha, mdl2.log_alpha)
    assert reloaded.config.categories == bv["categories"]
    print("save_pretrained / from_pretrained (via AutoModelForCausalLM): OK")

    # regression: a checkpoint saved before log_importance existed (or any
    # future param added the same way) must reload it as the intended zero
    # init, not uninitialized garbage -- see _init_weights.
    with tempfile.TemporaryDirectory() as d:
        mdl2.save_pretrained(d)
        import safetensors.torch as _st

        sd = _st.load_file(f"{d}/model.safetensors")
        del sd["log_importance"]
        _st.save_file(sd, f"{d}/model.safetensors", metadata={"format": "pt"})
        reloaded_missing = AutoModelForCausalLM.from_pretrained(d)
    assert t.isfinite(reloaded_missing.log_importance).all(), (
        "log_importance not finite after reloading a checkpoint missing it"
    )
    assert t.allclose(
        reloaded_missing.log_importance, t.zeros_like(reloaded_missing.log_importance)
    )
    assert t.allclose(reloaded_missing.log_alpha, mdl2.log_alpha)
    print("missing log_importance reloads as zero, not garbage: OK")

    frozen_cfg = BasisBlendedConfig(
        base_model_type="llama",
        base_config=cfg.base_config,
        train_beta_params=False,
        train_importance_scale=False,
        **bv,
    )
    frozen = BasisBlendedCausalLM(frozen_cfg)
    assert not frozen.log_alpha.requires_grad and not frozen.log_beta.requires_grad
    assert not frozen.log_importance.requires_grad
    print("train_beta_params=False / train_importance_scale=False freeze params: OK")

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

    print("all basis_blended.py self-tests passed")
    # breakpoint()
