#!/usr/bin/env python3

"""tests for cotorra.model: the secondary heads, CotorraConfig and CotorraForCausalLM"""

import itertools
import math

import pytest
import torch as t
from helpers import TINY_MODEL_ARGS, base_training_cfg, write_cfg
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from cotorra.loss import Loss
from cotorra.model import (
    HEADS,
    CotorraConfig,
    CotorraForCausalLM,
    DispositionHead,
    Log1pHoursHead,
    TntMixtureHead,
    TntPointHead,
    TteHead,
    ZeroInflatedLogNormalMixture,
    head_options,
    standard_normal_quadrature,
)

VOCAB = 64
NAN = float("nan")
K = 3  # mixture components, wherever a time-to-next-token head is a mixture
CLASSES = ["DSCG//expired", "DSCG//home", "DSCG//hospice"]

# each head's options, as `head_options` resolves them, weighted apart so a
# test can tell which term is which
OPTIONS = {
    "tte": {"weight": 0.5},
    "tnt": {"weight": 0.25},
    "disposition": {"weight": 0.125, "classes": CLASSES},
}

# usable targets for each head, and targets it leaves wholly unscored
TARGETS = {
    "tte": lambda n, length: t.rand(n, length) * 200,
    "tnt": lambda n, length: t.rand(n, length) * 5,
    "disposition": lambda n, length: t.randint(0, len(CLASSES), (n, length)),
}
UNSCORED = {
    "tte": lambda n, length: t.full((n, length), NAN),
    "tnt": lambda n, length: t.full((n, length), NAN),
    "disposition": lambda n, length: t.full((n, length), -100),
}

# every combination of heads, each with a time-to-next-token head built both
# ways where it carries one
COMBOS = [c for r in range(1, len(HEADS) + 1) for c in itertools.combinations(HEADS, r)]
VARIANTS = [(c, False) for c in COMBOS] + [(c, True) for c in COMBOS if "tnt" in c]


def variant_id(variant) -> str:
    names, mixture = variant
    return "+".join(names) + ("(mixture)" if mixture else "")


def backbone_cfg(**overrides) -> AutoConfig:
    """a tiny llama backbone config, built without touching the hub"""
    return AutoConfig.for_model(
        "llama", vocab_size=VOCAB, **{**TINY_MODEL_ARGS, **overrides}
    )


def options(*names, mixture=False) -> dict:
    heads = {name: dict(OPTIONS[name]) for name in names}
    if mixture:
        heads["tnt"]["mixture_components"] = K
    return heads


def build(*names, mixture=False, **backbone) -> CotorraForCausalLM:
    """
    built through `from_config` rather than by calling the class, which is what
    `Trainer.model_init` does and what puts the heads under the config's dtype
    """
    t.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        CotorraConfig(
            text_config=backbone_cfg(**backbone), heads=options(*names, mixture=mixture)
        )
    )


def targets(names, n=2, length=7, fill=TARGETS) -> dict:
    """each named head's target, under the batch key it reads"""
    return {HEADS[name].target: fill[name](n, length) for name in names}


def emit(head: Log1pHoursHead, log1p_hours: float):
    """make a point head emit a constant prediction, set through the softplus
    that keeps it non-negative: its exact inverse for a positive value, and a
    bias far enough below zero for 0 that softplus underflows to it"""
    with t.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(
            math.log(math.expm1(log1p_hours)) if log1p_hours > 0 else -50.0
        )


def is_mixture(model: CotorraForCausalLM) -> bool:
    return "tnt" in model.heads and isinstance(model.heads["tnt"], TntMixtureHead)


def ids(n=2, length=7) -> t.Tensor:
    return t.randint(3, VOCAB - 1, (n, length))


@pytest.fixture(params=VARIANTS, ids=[variant_id(v) for v in VARIANTS])
def model(request) -> CotorraForCausalLM:
    """every combination of heads: whatever holds of one head has to hold
    whichever others it is built alongside"""
    names, mixture = request.param
    return build(*names, mixture=mixture)


# ---------------------------------------------------------------- the config


def test_config_requires_a_backbone():
    """the wrapper is meaningless without something to wrap, so it says so
    rather than failing later inside `AutoModel.from_config(None)`"""
    with pytest.raises(ValueError, match="text_config"):
        CotorraConfig()


def test_config_mirrors_common_keys_off_the_backbone():
    """cotorra's own stages read `vocab_size`/`eos_token_id`/... straight off
    `model.config`, so they have to be present at the top level too"""
    bb = backbone_cfg(bos_token_id=11, eos_token_id=22, tie_word_embeddings=True)
    cfg = CotorraConfig(text_config=bb)
    assert (cfg.vocab_size, cfg.hidden_size) == (bb.vocab_size, bb.hidden_size)
    assert (cfg.bos_token_id, cfg.eos_token_id) == (11, 22)
    assert cfg.tie_word_embeddings is True
    assert cfg.use_cache == bb.use_cache


def test_config_keeps_an_explicitly_passed_value_over_the_mirror():
    cfg = CotorraConfig(text_config=backbone_cfg(eos_token_id=22), eos_token_id=33)
    assert cfg.eos_token_id == 33


def test_config_accepts_a_backbone_given_as_a_plain_dict():
    """what `from_pretrained` hands back: `text_config` arrives as json"""
    cfg = CotorraConfig(text_config={"model_type": "llama", "vocab_size": VOCAB})
    assert type(cfg.text_config).__name__ == "LlamaConfig"
    assert cfg.vocab_size == VOCAB


def test_config_serializes_its_heads_despite_having_no_default_constructor():
    """
    `has_no_defaults_at_init` is what stops `to_diff_dict` (and so
    `save_pretrained`) from calling the no-argument constructor that
    `test_config_requires_a_backbone` pins as raising
    """
    heads = options(*HEADS, mixture=True)
    cfg = CotorraConfig(text_config=backbone_cfg(), heads=heads)
    assert cfg.to_diff_dict()["heads"] == heads
    for js in (cfg.to_json_string(), cfg.to_json_string(use_diff=False)):
        assert '"mixture_components": 3' in js and '"DSCG//hospice"' in js


def test_config_refuses_a_head_it_does_not_know():
    with pytest.raises(ValueError, match="mpp"):
        CotorraConfig(text_config=backbone_cfg(), heads={"mpp": {}})


# ------------------------------------------------- reading the training config

LOOKUP = OmegaConf.create(
    {"BOS": 0, "EOS": 1, "DSCG//home": 2, "LAB//na_Q3": 3, "DSCG//expired": 4}
)


def test_head_options_reads_one_block_per_objective():
    """any combination of blocks; each weight defaults to 1.0, so a block left
    empty -- which yaml parses to None -- still asks for its head"""
    cfg = OmegaConf.create(
        {
            "tte_objective": None,
            "tnt_objective": {"weight": 0.25, "mixture_components": 8},
            "custom_loss": True,
        }
    )
    assert head_options(cfg, LOOKUP) == {
        "tte": {"weight": 1.0},
        "tnt": {"weight": 0.25, "mixture_components": 8},
    }
    assert head_options(OmegaConf.create({}), LOOKUP) == {}


def test_head_options_refuses_an_objective_naming_no_head():
    """the blocks a config written for an earlier layout would carry: left
    alone, they would train a plain backbone without a word"""
    cfg = OmegaConf.create({"mpp_objective": {"tnt_weight": 1.0}})
    with pytest.raises(ValueError, match="mpp_objective"):
        head_options(cfg, LOOKUP)


@pytest.mark.parametrize(
    "classes, expected",
    [
        (None, ["DSCG//home", "DSCG//expired"]),
        (["DSCG//expired", "DSCG//home"], ["DSCG//home", "DSCG//expired"]),
        ("DSCG//exp*", ["DSCG//expired"]),
    ],
    ids=["default", "listed", "one-pattern"],
)
def test_disposition_classes_resolve_to_tokens_in_id_order(classes, expected):
    """fnmatch patterns, `DSCG//*` by default, resolved against the vocabulary
    so the checkpoint names its classes outright"""
    block = {} if classes is None else {"classes": classes}
    cfg = OmegaConf.create({"disposition_objective": block})
    assert head_options(cfg, LOOKUP)["disposition"]["classes"] == expected


def test_disposition_classes_matching_nothing_are_refused():
    cfg = OmegaConf.create({"disposition_objective": {"classes": ["DISCH//*"]}})
    with pytest.raises(ValueError, match="matched no token"):
        head_options(cfg, LOOKUP)


# ------------------------------------------------------- dtype and structure


@pytest.mark.parametrize("variant", VARIANTS, ids=[variant_id(v) for v in VARIANTS])
def test_only_the_heads_asked_for_are_built(variant):
    """rather than built and left untrained, which would cost parameters and
    which `ddp_find_unused_parameters: false` would refuse"""
    names, mixture = variant
    mdl = build(*names, mixture=mixture)
    assert list(mdl.heads) == list(names)
    kinds = {
        "tte": TteHead,
        "tnt": TntMixtureHead if mixture else TntPointHead,
        "disposition": DispositionHead,
    }
    assert all(type(mdl.heads[name]) is kinds[name] for name in names)
    assert {k.split(".")[1] for k in mdl.state_dict() if k.startswith("heads.")} == (
        set(names)
    )


def test_a_component_count_below_one_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        AutoModelForCausalLM.from_config(
            CotorraConfig(
                text_config=backbone_cfg(), heads={"tnt": {"mixture_components": 0}}
            )
        )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_every_module_is_built_under_the_backbones_dtype(dtype):
    """
    the shipped presets name a hub model whose config carries a `dtype`
    (`Llama-3.2-1B` is bfloat16), and `_from_config` initializes under the
    *top-level* dtype and overwrites the backbone's with it. Unless the
    wrapper mirrors `dtype` up, a preset the stock path builds in bfloat16 is
    quietly built in float32 -- and calling the class directly leaves the
    heads in float32 against a bfloat16 trunk, which is a hard error on matmul
    """
    stock = AutoModelForCausalLM.from_config(backbone_cfg(dtype=dtype))
    assert next(stock.parameters()).dtype == getattr(t, dtype)
    for mixture in (False, True):
        mdl = build(*HEADS, mixture=mixture, dtype=dtype)
        assert {p.dtype for p in mdl.parameters()} == {getattr(t, dtype)}


@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
def test_the_heads_are_initialized_as_huggingface_does(mixture):
    """a zero bias and a normal weight of `initializer_range` spread, rather
    than torch's uniform default (spread ~0.1 at this width): `post_init` has
    to run after every head exists"""
    mdl = build(*HEADS, mixture=mixture)
    layers = [m for m in mdl.heads.modules() if isinstance(m, t.nn.Linear)]
    assert len(layers) == (4 if mixture else 3)
    for layer in layers:
        assert t.equal(layer.bias, t.zeros_like(layer.bias))
        assert layer.weight.std().item() == pytest.approx(
            backbone_cfg().initializer_range, rel=0.5
        )


@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
def test_state_dict_matches_the_stock_class_once_the_heads_are_dropped(mixture):
    """
    the heads sit under `heads.*` and the rest is laid out as a llama-family
    `*ForCausalLM` is, so a checkpoint can be handed to anything that only
    knows the stock architecture (sglang, which `generative-score` serves
    through, resolves `architectures[0]` against its own registry and has never
    heard of `cotorra`)
    """
    mdl, stock = (
        build(*HEADS, mixture=mixture),
        AutoModelForCausalLM.from_config(backbone_cfg()),
    )
    weights = {k: v for k, v in mdl.state_dict().items() if not k.startswith("heads.")}
    assert set(weights) == set(stock.state_dict())

    missing, unexpected = stock.load_state_dict(weights, strict=False)
    assert (list(missing), list(unexpected)) == ([], [])
    x = ids(1, 5)
    mdl.eval()
    stock.eval()
    assert t.equal(mdl(input_ids=x).logits, stock(input_ids=x).logits)


def test_word_embeddings_are_tied_when_the_backbone_asks_for_it():
    mdl = build("tte", tie_word_embeddings=True)
    assert mdl.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    assert mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr()


def test_untied_when_the_backbone_says_not_to():
    assert build("tte", tie_word_embeddings=False).all_tied_weights_keys == {}


def test_tying_follows_a_backbone_that_names_its_embedding_differently():
    """
    `model_name` is a free-form config key, and the upstream idiom hardcodes
    `model.embed_tokens.weight`; the older architectures call it `wte` and
    would raise `GPT2Model has no attribute embed_tokens` on that guess
    """
    bb = AutoConfig.for_model(
        "gpt2",
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        tie_word_embeddings=True,
    )
    mdl = AutoModelForCausalLM.from_config(
        CotorraConfig(text_config=bb, heads=options("tte"))
    )
    assert mdl.all_tied_weights_keys == {"lm_head.weight": "model.wte.weight"}
    assert mdl.lm_head.weight.data_ptr() == mdl.model.wte.weight.data_ptr()


def test_a_saved_model_reloads_through_the_auto_class(model, tmp_path):
    """
    `Extractor` and `GenerativeScorer` only ever see `mdl-<run_name>/`, which
    they open with `AutoModelForCausalLM.from_pretrained`; importing
    `cotorra.model` is what registers the pair that makes that resolve, and the
    heads come back as they were saved, disposition classes included
    """
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert type(reloaded) is CotorraForCausalLM
    assert reloaded.config.heads == model.config.heads
    assert type(reloaded.config.text_config).__name__ == "LlamaConfig"
    if "disposition" in reloaded.heads:
        assert reloaded.heads["disposition"].classes == CLASSES
    x = ids(2, 6)
    model.eval()
    reloaded.eval()
    before, after = model(input_ids=x), reloaded(input_ids=x)
    assert t.equal(before.logits, after.logits)
    for name in model.heads:
        a, b = getattr(before, f"{name}_pred"), getattr(after, f"{name}_pred")
        assert (a is None and b is None) or t.equal(a, b), name
    if is_mixture(model):
        with t.no_grad():
            hidden = model.model(input_ids=x).last_hidden_state
            dists = [m.tnt_distribution(hidden, x) for m in (model, reloaded)]
        assert t.equal(dists[0].zero_logit, dists[1].zero_logit)


# ------------------------------------------------------------------- forward


def test_forward_returns_a_prediction_from_each_head_it_carries(model):
    out = model(input_ids=ids(2, 7))
    assert out.logits.shape == (2, 7, VOCAB)
    expected = {
        "tte": (2, 7),
        "tnt": None if is_mixture(model) else (2, 7),
        "disposition": (2, 7, len(CLASSES)),
    }
    for name in HEADS:
        pred = getattr(out, f"{name}_pred")
        shape = expected[name] if name in model.heads else None
        assert (pred is None) if shape is None else (pred.shape == shape), name
        assert out.get(f"{name}_loss") is None
    assert out.loss is None


def test_forward_exposes_hidden_states_for_the_extractor(model):
    """`Extractor.extract_final` reads `.hidden_states[-1]`"""
    out = model(input_ids=ids(2, 7), output_hidden_states=True)
    assert len(out.hidden_states) == TINY_MODEL_ARGS["num_hidden_layers"] + 1
    assert out.hidden_states[-1].shape == (2, 7, TINY_MODEL_ARGS["hidden_size"])


def test_the_loss_adds_each_heads_weighted_term(model):
    x = ids(2, 7)
    full = model(input_ids=x, labels=x, **targets(model.heads))
    lm_only = model(input_ids=x, labels=x)

    expected = lm_only.loss.item() + sum(
        head.weight * getattr(full, f"{name}_loss").item()
        for name, head in model.heads.items()
    )
    assert full.loss.item() == pytest.approx(expected, rel=1e-5)


def test_each_head_is_scored_without_labels(model):
    """
    on cotorra's default `custom_loss: true`, `TrainerWithCustomLoss` pops
    `labels` before calling the model, so every head's term has to be reachable
    from the returned output for `Loss.custom_loss` to fold it in
    """
    out = model(input_ids=ids(2, 7), **targets(model.heads))
    assert out.loss is None
    for name in model.heads:
        assert t.isfinite(getattr(out, f"{name}_loss")), name


def test_each_head_reads_only_its_own_target(model):
    """blanking one head's target leaves it nothing to score, and leaves every
    other head's term exactly as it was"""
    x, full = ids(2, 6), targets(model.heads, 2, 6)
    base = model(input_ids=x, **full)
    for name, head in model.heads.items():
        out = model(input_ids=x, **full | {head.target: UNSCORED[name](2, 6)})
        assert (
            getattr(out, f"{name}_loss").item()
            == 0.0
            != getattr(base, f"{name}_loss").item()
        )
        for other in model.heads:
            if other != name:
                assert t.equal(
                    getattr(out, f"{other}_loss"), getattr(base, f"{other}_loss")
                )


@pytest.mark.parametrize("name", list(HEADS))
def test_a_target_for_a_head_the_model_lacks_is_refused(name):
    """rather than swallowed by the backbone's `**kwargs`, which would quietly
    train one head fewer than the caller expects"""
    mdl = build(*(other for other in HEADS if other != name))
    with pytest.raises(ValueError, match=f"got `{HEADS[name].target}`"):
        mdl(input_ids=ids(2, 7), **targets([name]))


def test_a_batch_with_nothing_to_score_still_reaches_every_parameter(model):
    """
    summed-then-normalized rather than masked-then-averaged, so each zero is
    differentiable: a bare `0.0` would leave its head with no gradient at all,
    which `ddp_find_unused_parameters: false` reports as an error
    """
    x = ids(2, 6)
    out = model(input_ids=x, labels=x, **targets(model.heads, 2, 6, fill=UNSCORED))
    for name in model.heads:
        assert getattr(out, f"{name}_loss").item() == 0.0
        assert getattr(out, f"{name}_loss").requires_grad
    out.loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(t.isfinite(p.grad).all() for p in model.heads.parameters())


def test_logits_to_keep_trims_what_is_returned_but_not_what_is_scored(model):
    """generation asks for the last position alone; training never trims, but
    a target still scores every position whatever comes back"""
    x, tg = ids(2, 6), targets(model.heads, 2, 6)
    whole, last = model(input_ids=x, **tg), model(input_ids=x, logits_to_keep=1, **tg)
    assert last.logits.shape == (2, 1, VOCAB)
    for name in model.heads:
        assert t.allclose(getattr(last, f"{name}_loss"), getattr(whole, f"{name}_loss"))
        if (pred := getattr(whole, f"{name}_pred")) is not None:
            assert t.equal(getattr(last, f"{name}_pred"), pred[:, -1:])


# ------------------------------------------------------- the time-to-event head


@pytest.fixture
def tte() -> CotorraForCausalLM:
    return build("tte")


def test_the_tte_target_is_the_current_positions_hours(tte):
    """
    unshifted, unlike the language-modelling loss: the prediction at position i
    is scored against position i's own target -- the hours remaining once token
    i has been read -- never against the one at i+1. The targets here fall away
    down the sequence, as remaining-hours always do, so scoring the wrong
    pairing gives a different number
    """
    emit(tte.heads["tte"], 0.0)  # a constant 0, whatever the input
    hours = t.tensor([[7.0, 3.0, 1.0, 0.0]])
    out = tte(input_ids=ids(1, 4), hours_to_end_time=hours)

    shifted = (t.log1p(hours[0, 1:]) ** 2).mean()
    unshifted = (t.log1p(hours[0, :]) ** 2).mean()
    assert out.tte_loss.item() == pytest.approx(unshifted.item(), rel=1e-5)
    assert unshifted.item() != pytest.approx(shifted.item(), rel=1e-3)


def test_a_zero_target_is_kept_rather_than_masked(tte):
    """the last token of a record has nothing left to wait for, and `log1p(0)`
    is exactly 0 -- a real target, not a missing one"""
    emit(tte.heads["tte"], 0.0)
    out = tte(input_ids=ids(2, 6), hours_to_end_time=t.zeros(2, 6))
    assert out.tte_loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "hours",
    [
        pytest.param(t.full((2, 6), NAN), id="missing-end-time"),
        pytest.param(t.full((2, 6), float("inf")), id="infinite"),
    ],
)
def test_unusable_targets_are_masked_out(tte, hours):
    x = ids(2, 6)
    out = tte(input_ids=x, labels=x, hours_to_end_time=hours)
    assert out.tte_loss.item() == 0.0
    assert t.isfinite(out.loss)


def test_only_the_usable_positions_contribute(tte):
    """a half-masked batch scores the same as the usable half on its own"""
    emit(tte.heads["tte"], 0.0)
    masked = tte(
        input_ids=ids(1, 4), hours_to_end_time=t.tensor([[3.0, NAN, 1.0, 0.0]])
    ).tte_loss
    expected = (t.log1p(t.tensor([3.0, 1.0, 0.0])) ** 2).mean()
    assert masked.item() == pytest.approx(expected.item(), rel=1e-5)


def test_a_target_past_the_end_time_counts_as_zero_hours(tte):
    """
    a token recorded after the reference end time is scored against 0 hours
    remaining rather than left out, while a missing end time (a nan) stays
    masked. A head emitting a constant 1 tells the two apart: a masked target
    adds nothing, a clipped one adds (1 - log1p(0))^2 = 1
    """
    emit(tte.heads["tte"], 1.0)
    out = tte(input_ids=ids(1, 4), hours_to_end_time=t.tensor([[2.0, -3.0, NAN, -0.5]]))
    first = (1 - math.log1p(2.0)) ** 2
    assert out.tte_loss.item() == pytest.approx((first + 1 + 1) / 3, rel=1e-5)


@pytest.mark.parametrize("bias", [-50.0, -5.0, 0.0, 5.0])
def test_the_point_heads_never_predict_negative_hours(bias):
    """even a head whose raw output is pushed far below zero, on inputs with
    wide random weights"""
    mdl = build("tte", "tnt")
    with t.no_grad():
        for head in mdl.heads.values():
            head.linear.weight.normal_(std=5.0)
            head.linear.bias.fill_(bias)
    out = mdl(input_ids=ids(4, 9))
    for pred in (out.tte_pred, out.tnt_pred):
        assert (pred >= 0).all() and (pred.expm1() >= 0).all()


def test_a_point_head_below_zero_still_learns(tte):
    """why a softplus rather than a clamp: a clamped head whose raw output fell
    below zero would get no gradient, and stay predicting 0 for good"""
    with t.no_grad():
        tte.heads["tte"].linear.bias.fill_(-5.0)
    out = tte(input_ids=ids(2, 6), hours_to_end_time=t.full((2, 6), 10.0))
    out.tte_loss.backward()
    assert tte.heads["tte"].linear.bias.grad.abs().item() > 0


@pytest.mark.parametrize("name", ["tte", "tnt"])
def test_a_point_head_learns_its_target(name):
    mdl = build(name)
    x, tg = ids(4, 8), targets([name], 4, 8)
    opt = t.optim.AdamW(mdl.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        loss = getattr(mdl(input_ids=x, **tg), f"{name}_loss")
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2


# ---------------------------------------------- the time-to-next-token heads


def test_the_tnt_target_is_the_current_positions_gap():
    """
    unshifted: position i's target already holds the hours from token i to
    token i+1, so it is scored against the prediction made at i. The trailing
    nan is a record's last token, which has no successor. The tte head is set
    to a constant 3 to show the term reads the tnt head, not its sibling
    """
    mdl = build("tte", "tnt")
    emit(mdl.heads["tnt"], 0.0)
    emit(mdl.heads["tte"], 3.0)
    out = mdl(input_ids=ids(1, 4), hours_to_next_token=t.tensor([[2.0, 0.0, 5.0, NAN]]))

    unshifted = (t.log1p(t.tensor([2.0, 0.0, 5.0])) ** 2).mean()
    shifted = (t.log1p(t.tensor([0.0, 5.0])) ** 2).mean()
    assert out.tnt_loss.item() == pytest.approx(unshifted.item(), rel=1e-5)
    assert unshifted.item() != pytest.approx(shifted.item(), rel=1e-3)


def test_a_point_head_gives_its_prediction_whatever_comes_next():
    """all a point head has to give generation -- the reason to train a mixture
    head for it -- greedy or not"""
    mdl = build("tnt").eval()
    hidden = t.randn(3, 32)
    expected = mdl.heads["tnt"].predict(hidden).expm1()
    for nxt in (t.tensor([4, 5, 6]), t.tensor([40, 50, 60])):
        for do_sample in (True, False):
            hours = mdl.sample_hours_to_next_token(hidden, nxt, do_sample=do_sample)
            assert t.equal(hours, expected)


def test_a_point_head_has_no_distribution_to_give():
    with pytest.raises(ValueError, match="point estimate"):
        build("tnt").tnt_distribution(t.zeros(1, 32), t.tensor([3]))


def test_without_a_tnt_head_there_are_no_times_to_sample():
    with pytest.raises(ValueError, match="no time-to-next-token head"):
        build("tte").sample_hours_to_next_token(t.zeros(1, 32), t.tensor([3]))


@pytest.fixture
def mixture() -> CotorraForCausalLM:
    return build("tnt", mixture=True)


def raw_params(zero_logit, weights, means, raw_scales) -> t.Tensor:
    """a `TntMixtureHead`'s raw output laid out by hand"""
    return t.tensor([zero_logit, *weights, *means, *raw_scales])


def test_the_mixture_log_prob_matches_its_definition():
    """zero-inflated: log p0 at an exact zero; otherwise log(1 - p0) plus the
    gaussian mixture's density at the log-hours"""
    t.manual_seed(0)
    params = t.randn(4, 1 + 3 * K)
    hours = t.tensor([0.0, 0.25, 3.0, 100.0])
    dist = ZeroInflatedLogNormalMixture(params)

    zero_logit, w, mu, raw = params.split([1, K, K, K], dim=-1)
    gaussians = t.distributions.MixtureSameFamily(
        t.distributions.Categorical(logits=w),
        t.distributions.Normal(mu, t.nn.functional.softplus(raw) + 1e-3),
    )
    p0 = zero_logit.squeeze(-1).sigmoid()
    expected = t.where(
        hours == 0,
        p0.log(),
        (1 - p0).log() + gaussians.log_prob(hours.clamp(min=1e-6).log()),
    )
    assert t.allclose(dist.log_prob(hours), expected, atol=1e-5)


def test_mixture_samples_follow_the_distribution():
    """a single component, so the positive draws' log-hours are N(mu, sigma)"""
    p0, mu, sigma = 0.3, math.log(2.0), 0.5
    raw_scale = math.log(math.expm1(sigma - 1e-3))  # softplus^-1
    params = raw_params(math.log(p0 / (1 - p0)), [0.0], [mu], [raw_scale])
    dist = ZeroInflatedLogNormalMixture(params.expand(20_000, -1))
    draws = dist.sample(t.Generator().manual_seed(0))

    assert (draws >= 0).all()
    assert (draws == 0).float().mean().item() == pytest.approx(p0, abs=0.02)
    log_positive = draws[draws > 0].log()
    assert log_positive.mean().item() == pytest.approx(mu, abs=0.02)
    assert log_positive.std().item() == pytest.approx(sigma, abs=0.02)


def test_sampling_is_reproducible_under_a_generator():
    dist = ZeroInflatedLogNormalMixture(t.randn(50, 1 + 3 * K))
    a = dist.sample(t.Generator().manual_seed(7))
    b = dist.sample(t.Generator().manual_seed(7))
    assert t.equal(a, b)


@pytest.mark.parametrize(
    "zero_logit, expected", [(1.0, 0.0), (-1.0, 2.0)], ids=["zero", "positive"]
)
def test_the_point_estimate_is_zero_when_likelier_else_the_geometric_mean(
    zero_logit, expected
):
    params = raw_params(
        zero_logit, [0.0, 0.0], [math.log(1.0), math.log(4.0)], [0.0, 0.0]
    )
    point = ZeroInflatedLogNormalMixture(params).point_estimate()
    assert point.item() == pytest.approx(expected, rel=1e-5)


def test_the_mixture_head_emits_the_parameters_of_k_components(mixture):
    assert mixture.heads["tnt"].proj[2].out_features == 1 + 3 * K


def test_the_next_token_is_heard_at_the_hidden_states_scale(mixture):
    """the next token's embedding is normalized before it meets the hidden
    state, so the head reads it the same whatever its raw scale; left at an
    init's ~0.02 against the backbone's normed ~1, it went all but unheard"""
    hidden = t.randn(3, 32)
    embeds = mixture.get_input_embeddings()(t.tensor([4, 9, 17]))
    head = mixture.heads["tnt"]
    quiet, loud = (head.distribution(hidden, c * embeds) for c in (1.0, 100.0))
    # not exactly equal: layer norm's eps is a few percent of the variance of
    # embeddings at init scale; unnormalized, the gap is ~1e-2
    assert t.allclose(quiet.zero_logit, loud.zero_logit, atol=1e-3)
    assert t.allclose(quiet.means, loud.means, atol=1e-3)


def test_the_mixture_distribution_conditions_on_the_next_token(mixture):
    """position i's gap depends on which token i+1 is, and on nothing later"""
    x = ids(1, 6)
    hidden = mixture.model(input_ids=x).last_hidden_state[:, :-1]
    base = mixture.tnt_distribution(hidden, x[:, 1:])

    swapped = x.clone()
    swapped[0, 3] = (x[0, 3] + 1 - 3) % (VOCAB - 4) + 3  # token 3 changes
    hidden2 = mixture.model(input_ids=swapped).last_hidden_state[:, :-1]
    other = mixture.tnt_distribution(hidden2, swapped[:, 1:])
    # position 2 conditions on token 3, so it moves; positions before it don't
    assert not t.allclose(other.zero_logit[0, 2], base.zero_logit[0, 2])
    assert t.equal(other.zero_logit[0, :2], base.zero_logit[0, :2])


def test_the_mixture_loss_is_the_mean_nll_over_usable_targets(mixture):
    x = ids(2, 6)
    hours = t.tensor([[0.0, 0.5, NAN, 2.0, 0.0, NAN], [1.0, -1.0, 0.0, 0.0, 3.0, 4.0]])
    out = mixture(input_ids=x, hours_to_next_token=hours)

    hidden = mixture.model(input_ids=x).last_hidden_state
    lp = mixture.tnt_distribution(hidden[:, :-1], x[:, 1:]).log_prob(
        hours[:, :-1].nan_to_num(0.0).clamp(min=0)
    )
    usable = t.isfinite(hours[:, :-1]) & (hours[:, :-1] >= 0)
    assert out.tnt_loss.item() == pytest.approx(-lp[usable].mean().item(), rel=1e-5)


def test_the_last_position_is_left_out_having_no_next_token(mixture):
    """in a packed chunk the true next token lies past the chunk's end"""
    x = ids(1, 5)
    a = mixture(input_ids=x, hours_to_next_token=t.tensor([[1.0, 0, 2, 0, 5.0]]))
    b = mixture(input_ids=x, hours_to_next_token=t.tensor([[1.0, 0, 2, 0, 99.0]]))
    assert t.equal(a.tnt_loss, b.tnt_loss)


def test_scoring_needs_the_input_ids_the_head_conditions_on(mixture):
    embeds = mixture.get_input_embeddings()(ids(1, 4))
    with pytest.raises(ValueError, match="needs `input_ids`"):
        mixture(inputs_embeds=embeds, hours_to_next_token=t.rand(1, 4))


def test_the_mixture_head_learns_gaps_that_depend_on_the_next_token():
    """
    tokens 20-29 play the quantile tokens that complete an event, arriving at
    once; 3-19 the events, ~2 hours apart. The sequence is random, so only the
    next token tells the two apart -- what a point head can't see
    """
    mdl = build("tnt", mixture=True)
    events, quantiles = t.arange(3, 20), t.arange(20, 30)

    def batch(n=16, length=12):
        is_q = t.rand(n, length) < 0.5
        x = t.where(
            is_q,
            quantiles[t.randint(0, 10, (n, length))],
            events[t.randint(0, 17, (n, length))],
        )
        to_next = t.where(
            is_q[:, 1:], 0.0, (0.3 * t.randn(n, length - 1) + math.log(2.0)).exp()
        )
        return x, t.cat([to_next, t.full((n, 1), NAN)], dim=1)

    opt = t.optim.AdamW(mdl.parameters(), lr=1e-2)
    for _ in range(100):
        x, hours = batch()
        loss = mdl(input_ids=x, hours_to_next_token=hours).tnt_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    mdl.eval()
    with t.no_grad():
        hidden = mdl.model(input_ids=batch(n=1)[0]).last_hidden_state[:, -1]
        to_quantile = mdl.tnt_distribution(hidden, quantiles[:1])
        to_event = mdl.tnt_distribution(hidden, events[:1])
    assert to_quantile.zero_logit.sigmoid().item() > 0.95
    assert to_event.zero_logit.sigmoid().item() < 0.05
    assert to_event.point_estimate().item() == pytest.approx(2.0, rel=0.25)


# ------------------------------- the time to the next token, given the history

CAP = ZeroInflatedLogNormalMixture.max_log_hours
# log1p of the capped hours, the most a mean of log1p-hours can be, rounded to
# float32 as the clamp that keeps it there is
MOST_LOG1P_HOURS = t.tensor(math.log1p(math.exp(CAP))).item()


def softplus_inverse(scale: float) -> float:
    """the raw parameter that makes a component's scale, softplus(raw) + 1e-3"""
    x = scale - 1e-3
    return x + math.log(-math.expm1(-x))  # log(expm1(x)), without overflowing


def test_the_quadrature_integrates_polynomials_against_a_standard_normal():
    """exactly, up to degree 2n - 1: the moments 1, 0, 1, 0, 3, 0, 15"""
    nodes, weights = standard_normal_quadrature(4)
    moments = [(weights * nodes**k).sum() for k in range(7)]
    assert moments == pytest.approx([1, 0, 1, 0, 3, 0, 15], abs=1e-12)


@pytest.mark.parametrize("scale", [0.05, 0.5, 1.0, 2.0, 3.0])
def test_the_mean_log1p_hours_matches_dense_integration(scale):
    """
    against the trapezoid rule over a fine grid, in float64, for single
    components below, around and beyond the cap, whose kink plain quadrature
    resolves to only ~1e-2; within 2e-5 of it, for components as wide as 3
    log-hours
    """
    means = [-8.0, -2.0, 0.0, 1.0, 4.0, 8.0, 11.0, 13.0, CAP, 15.0, 20.0]
    params = t.stack(
        [raw_params(-50.0, [0.0], [m], [softplus_inverse(scale)]) for m in means]
    )
    got = ZeroInflatedLogNormalMixture(params).mean_log1p_hours()
    z = t.linspace(-12.0, 12.0, 480_001, dtype=t.float64)
    log_hours = (t.tensor(means, dtype=t.float64)[:, None] + scale * z).clamp(max=CAP)
    density = (-(z**2) / 2).exp() / math.sqrt(2 * math.pi)
    expected = t.trapezoid(t.nn.functional.softplus(log_hours) * density, z, dim=-1)
    assert got.dtype == t.float32
    assert t.allclose(got.double(), expected, rtol=0, atol=2e-5)


# components chosen to stress the mean: a mixture with its zero part, the same
# mixture nearly all zero part, and one component so wide and so high that a
# third of its draws reach the cap
COMPONENTS = ([0.3, -0.2, 0.5], [-2.0, 0.5, 3.0], [-1.0, 0.0, 0.5])
MEAN_CASES = {
    "mixture": raw_params(-0.7, *COMPONENTS),
    "mostly-zero": raw_params(4.6, *COMPONENTS),
    "capped": raw_params(-5.0, [0.0], [12.0], [softplus_inverse(4.0)]),
}


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["float32", "bf16"])
@pytest.mark.parametrize("case", list(MEAN_CASES))
def test_the_mean_log1p_hours_is_what_draws_average_to(case, dtype):
    """
    the mean of log1p over a million draws from `sample`, cap and zero part
    included, to within four standard errors -- from bfloat16 parameters too,
    which the distribution takes up to float32
    """
    params = MEAN_CASES[case].to(dtype=dtype)
    mean = ZeroInflatedLogNormalMixture(params[None]).mean_log1p_hours()
    draws = (
        ZeroInflatedLogNormalMixture(params.expand(1_000_000, -1))
        .sample(t.Generator().manual_seed(0))
        .log1p()
    )
    if case == "mostly-zero":
        assert (draws == 0).float().mean().item() > 0.98
    if case == "capped":
        assert (draws >= MOST_LOG1P_HOURS - 1e-5).float().mean().item() > 0.3
    assert mean.dtype == t.float32 and mean.shape == (1,)
    error = draws.std().item() / math.sqrt(len(draws))
    assert mean.item() == pytest.approx(draws.mean().item(), abs=4 * error)


def test_the_mean_log1p_hours_stays_within_bounds_whatever_the_parameters():
    """between 0 and log1p of the cap, finite, for parameters far past any a
    trained head would give -- components wide enough to straddle both 0 and
    the cap, and raw values of 1e4 in either direction. Each component's mean is
    clamped there, so the mixture's strays by no more than rounding"""
    t.manual_seed(0)
    raw = t.cat(
        [
            50 * t.randn(10_000, 1 + 3 * K),
            t.full((1, 1 + 3 * K), 1e4),
            t.full((1, 1 + 3 * K), -1e4),
        ]
    )
    mean = ZeroInflatedLogNormalMixture(raw).mean_log1p_hours()
    assert t.isfinite(mean).all() and (mean >= 0).all()
    assert (mean <= MOST_LOG1P_HOURS * (1 + 2 * t.finfo(t.float32).eps)).all()


def naive_mean_log1p_hours(model: CotorraForCausalLM, hidden: t.Tensor) -> t.Tensor:
    """a mixture head's mean given the history, the long way round: its
    distribution given each token of the vocabulary in turn, through the head's
    own forward, weighted by the probability of that token coming next"""
    head, embed = model.heads["tnt"], model.get_input_embeddings()
    probs = model.lm_head(hidden).float().softmax(dim=-1)
    total = t.zeros(hidden.shape[:-1])
    for token in range(probs.shape[-1]):
        given = head.distribution(hidden, embed(t.full(hidden.shape[:-1], token)))
        total += probs[..., token] * given.mean_log1p_hours()
    return total


@pytest.fixture
def wide_mixture() -> CotorraForCausalLM:
    """initialized wide enough that the gap the head gives depends a good deal on
    the token that comes next, so averaging over the wrong tokens would show,
    and with the head's biases and norm drawn at random rather than left at
    their init, since no trained head's are"""
    mdl = build("tnt", mixture=True, initializer_range=0.5).eval()
    with t.no_grad():
        for p in mdl.heads["tnt"].parameters():
            if p.dim() == 1:
                p.normal_(std=0.5)
    return mdl


@pytest.mark.parametrize(
    "per_chunk, chunks",
    [(1, [1] * 14), (3, [3, 3, 3, 3, 2]), (None, [14])],
    ids=["one-at-a-time", "uneven", "all-at-once"],
)
def test_the_history_only_mean_averages_over_the_next_token(
    wide_mixture, per_chunk, chunks, monkeypatch
):
    """
    the mixture head's mean given each next token, averaged over the language
    model's probability of each, as the naive loop over the vocabulary has it,
    however the positions are chunked: one at a time, in chunks the positions
    don't divide into, and all at once
    """
    if per_chunk is not None:  # the rule: elements per position, at the widest
        n_nodes = len(ZeroInflatedLogNormalMixture.quadrature[0])
        width = max(TINY_MODEL_ARGS["hidden_size"], K * n_nodes)
        monkeypatch.setattr(TntMixtureHead, "chunk_elements", per_chunk * VOCAB * width)
    seen = []

    def lm_head(hidden):
        seen.append(len(hidden))
        return wide_mixture.lm_head(hidden)

    head, embed = wide_mixture.heads["tnt"], wide_mixture.get_input_embeddings()
    with t.no_grad():
        hidden = wide_mixture.model(input_ids=ids(2, 7)).last_hidden_state
        got = head.mean_log1p_hours(hidden, lm_head, embed)
        expected = naive_mean_log1p_hours(wide_mixture, hidden)
        given_each = head.distribution(
            hidden[..., None, :].expand(-1, -1, VOCAB, -1),
            embed(t.arange(VOCAB)).expand(2, 7, -1, -1),
        ).mean_log1p_hours()
    assert seen == chunks
    assert got.shape == (2, 7) and got.dtype == t.float32
    assert t.allclose(got, expected, rtol=1e-5, atol=1e-6)
    # the next token matters, so this is no average of near-equal values
    assert (given_each.amax(dim=-1) - given_each.amin(dim=-1)).max() > 1.0


def test_a_point_heads_history_only_hours_are_its_prediction():
    mdl = build("tnt", initializer_range=0.5).eval()
    with t.no_grad():
        out = mdl(input_ids=ids(2, 7), output_hidden_states=True)
        hours = mdl.predict_hours_to_next_token(out.hidden_states[-1])
    assert t.equal(hours, out.tnt_pred.float().expm1())


def test_a_mixture_heads_history_only_hours_are_its_mean_over_the_next_token(
    wide_mixture,
):
    with t.no_grad():
        hidden = wide_mixture.model(input_ids=ids(2, 7)).last_hidden_state
        hours = wide_mixture.predict_hours_to_next_token(hidden)
        expected = naive_mean_log1p_hours(wide_mixture, hidden).expm1()
    assert t.allclose(hours, expected, rtol=1e-5)


@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
@pytest.mark.parametrize(
    "shape", [(32,), (5, 32), (2, 3, 32), (2, 2, 2, 32), (0, 32)], ids=str
)
def test_history_only_hours_take_any_leading_shape(mixture, shape):
    mdl = build("tnt", mixture=mixture, initializer_range=0.5).eval()
    hidden = t.randn(*shape)
    with t.no_grad():
        hours = mdl.predict_hours_to_next_token(hidden)
        flat = mdl.predict_hours_to_next_token(hidden.reshape(-1, shape[-1]))
    assert hours.shape == shape[:-1]
    assert t.allclose(hours.reshape(-1), flat, rtol=1e-5)


@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
def test_history_only_hours_are_float32_from_a_bfloat16_model(mixture):
    """the head's output taken up to float32 before `expm1`, which in bfloat16
    rounds hours to 3 significant digits"""
    mdl = build("tnt", mixture=mixture, initializer_range=0.5, dtype="bfloat16")
    with t.no_grad():
        hidden = mdl.eval().model(input_ids=ids(2, 5)).last_hidden_state
        hours = mdl.predict_hours_to_next_token(hidden)
        if mixture:  # bfloat16 heads, given each next token in turn
            expected = naive_mean_log1p_hours(mdl, hidden).expm1()
        else:
            expected = mdl.heads["tnt"].predict(hidden).float().expm1()
    assert hidden.dtype == t.bfloat16
    assert hours.dtype == t.float32 and t.isfinite(hours).all()
    if mixture:
        assert t.allclose(hours, expected, rtol=1e-3)
    else:
        assert t.equal(hours, expected)


@pytest.mark.parametrize("bias", [-50.0, 50.0])
@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
def test_history_only_hours_stay_finite_and_non_negative_at_extremes(mixture, bias):
    """wide random head weights and biases pushed far either way: a point head's
    log1p-hours run to ~60 here, 1e26 hours, and a mixture's raw parameters to
    the thousands"""
    mdl = build("tnt", mixture=mixture).eval()
    head = mdl.heads["tnt"]
    with t.no_grad():
        for p in head.parameters():
            if p.dim() > 1:
                p.normal_(std=5.0 if mixture else 1.0)
        (head.proj[-1] if mixture else head.linear).bias.fill_(bias)
        hours = mdl.predict_hours_to_next_token(
            mdl.model(input_ids=ids(4, 9)).last_hidden_state
        )
    assert t.isfinite(hours).all() and (hours >= 0).all()
    if mixture:  # a draw is capped at a million hours, and so is their mean
        assert (hours <= 1e6 * (1 + 1e-6)).all()


def test_without_a_tnt_head_there_is_no_time_to_next_token_to_predict():
    with pytest.raises(ValueError, match="carries no time-to-next-token head"):
        build("tte", "disposition").predict_hours_to_next_token(t.zeros(1, 32))


# ------------------------------------------------------- the disposition head


@pytest.fixture
def disposition() -> CotorraForCausalLM:
    return build("disposition")


def test_disposition_logits_span_the_classes(disposition):
    out = disposition(input_ids=ids(2, 7))
    assert disposition.heads["disposition"].classes == CLASSES
    assert out.disposition_pred.shape == (2, 7, len(CLASSES))


def test_the_disposition_loss_is_cross_entropy_over_the_scored_positions(disposition):
    """unshifted, like the time heads: position i is scored against its own
    target, and the -100s -- from the disposition token on, and records ending
    in none of the classes -- leave the mean"""
    x = ids(2, 5)
    target = t.tensor([[2, 2, 2, -100, -100], [0, 0, 0, 0, -100]])
    out = disposition(input_ids=x, disposition=target)

    logits = out.disposition_pred.float()
    scored = target != -100
    expected = t.nn.functional.cross_entropy(logits[scored], target[scored])
    assert out.disposition_loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_the_disposition_head_learns_how_records_end(disposition):
    """each record's tokens come from a range its disposition picks, so every
    position has what it needs to tell -- and the head has to learn to"""

    def batch(n=16, length=10):
        cls = t.randint(0, len(CLASSES), (n, 1))
        x = 10 + 15 * cls + t.randint(0, 15, (n, length))
        return x, cls.expand(n, length)

    opt = t.optim.AdamW(disposition.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        x, target = batch()
        loss = disposition(input_ids=x, disposition=target).disposition_loss
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()

    x, target = batch()
    with t.no_grad():
        pred = disposition(input_ids=x).disposition_pred.argmax(dim=-1)
    assert loss.item() < first / 4
    assert (pred == target).float().mean().item() > 0.9


# --------------------------------------------------------------- integration

# one variant of each head on its own, and all three at once
TRAINER_VARIANTS = [
    (("tte",), False),
    (("tnt",), False),
    (("tnt",), True),
    (("disposition",), False),
    (tuple(HEADS), True),
]


def rows(n_vocab: int, n_classes: int, seq_len: int = 8, n: int = 2) -> list[dict]:
    """rows as `Loader.get_train_data` yields them with every objective
    configured; the collator picks out the ones its config asks for"""
    return [
        {
            "input_ids": t.randint(3, n_vocab - 1, (seq_len,)),
            "s_elapsed": t.arange(seq_len, dtype=t.float32) * 300,
            "hours_to_end_time": t.rand(seq_len) * 100,
            "hours_to_next_token": t.rand(seq_len),
            "disposition": t.randint(0, n_classes, (seq_len,)).index_fill(
                0, t.tensor([seq_len - 1]), -100
            ),
        }
        for _ in range(n)
    ]


@pytest.fixture(params=TRAINER_VARIANTS, ids=[variant_id(v) for v in TRAINER_VARIANTS])
def configured(request, built_trainer, monkeypatch):
    """`built_trainer` with the variant's objectives in its config, and what
    `Trainer.__init__` reads off them derived again"""
    names, mixture = request.param
    for name in names:
        block = {"weight": OPTIONS[name]["weight"]}
        if name == "tnt" and mixture:
            block["mixture_components"] = K
        monkeypatch.setitem(built_trainer.cfg, f"{name}_objective", block)
    lookup = built_trainer.tkzr_cfg.lookup
    monkeypatch.setattr(
        built_trainer, "head_options", head_options(built_trainer.cfg, lookup)
    )
    loss = Loss(built_trainer.cfg, built_trainer.tkzr_cfg).custom_loss
    monkeypatch.setattr(built_trainer, "loss", loss)
    monkeypatch.setattr(built_trainer.trainer, "compute_loss_func", loss)
    return built_trainer


def n_classes(trainer) -> int:
    return len(trainer.head_options.get("disposition", {}).get("classes", [None]))


def test_model_init_builds_a_plain_backbone_without_objectives(built_trainer):
    assert built_trainer.head_options == {}
    assert not isinstance(built_trainer.model_init(), CotorraForCausalLM)


def test_model_init_builds_the_heads_the_objectives_ask_for(configured):
    """`Trainer.model_init` is the only place the pipeline builds a model, so
    the blocks have to reach it, each head weighted as its block says"""
    mdl = configured.model_init()
    tkzr = configured.tkzr_cfg
    assert type(mdl) is CotorraForCausalLM
    assert mdl.config.heads == configured.head_options
    assert all(mdl.heads[name].weight == OPTIONS[name]["weight"] for name in mdl.heads)
    if "disposition" in mdl.heads:
        assert mdl.heads["disposition"].classes == sorted(
            (k for k in tkzr.lookup if k.startswith("DSCG//")), key=tkzr.lookup.get
        )
    assert mdl.config.vocab_size == len(tkzr.lookup)
    assert (mdl.config.bos_token_id, mdl.config.eos_token_id) == (
        tkzr.lookup.BOS,
        tkzr.lookup.EOS,
    )


def test_empty_blocks_still_build_their_heads(built_trainer, monkeypatch):
    """a block left empty in the yaml parses to None; every weight defaults to
    1.0 and the disposition classes to every `DSCG//*` token"""
    for name in HEADS:
        monkeypatch.setitem(built_trainer.cfg, f"{name}_objective", None)
    opts = head_options(built_trainer.cfg, built_trainer.tkzr_cfg.lookup)
    monkeypatch.setattr(built_trainer, "head_options", opts)
    mdl = built_trainer.model_init()
    assert list(mdl.heads) == list(HEADS)
    assert all(head.weight == 1.0 for head in mdl.heads.values())
    assert isinstance(mdl.heads["tnt"], TntPointHead)
    assert all(c.startswith("DSCG//") for c in mdl.heads["disposition"].classes)


def test_an_objective_naming_no_head_is_refused(processed, tmp_path):
    """an `mpp_objective` left over from an earlier layout would otherwise
    train a plain backbone without a word"""
    from cotorra.trainer import Trainer

    cfg = base_training_cfg(mpp_objective={"tnt_weight": 1.0})
    with pytest.raises(ValueError, match="mpp_objective"):
        Trainer(
            training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
            processed_data_home=processed,
            output_home=tmp_path,
        )


def test_the_model_consumes_what_the_trainers_collator_produces(configured):
    """
    the contract between `Trainer.collate_fn` and the model: the batch carries
    float `position_ids` (time-based rope) and each configured head's target,
    and nothing else is needed
    """
    n_vocab = len(configured.tkzr_cfg.lookup)
    collated = configured.collate_fn(rows(n_vocab, n_classes(configured), 4))
    assert set(collated) == {"input_ids", "labels", "position_ids"} | {
        HEADS[name].target for name in configured.head_options
    }

    out = configured.model_init()(**collated)
    assert out.logits.shape == (2, 4, n_vocab)
    assert t.isfinite(out.loss)
    assert all(
        t.isfinite(getattr(out, f"{name}_loss")) for name in configured.head_options
    )


def test_the_custom_loss_path_reaches_every_head(configured):
    """
    `TrainerWithCustomLoss.compute_loss` pops `labels`, so the model's own
    weighted sum is never formed and `Loss.custom_loss` owns the whole
    objective; a head whose term it dropped would take no gradient at all --
    and DDP, configured with `ddp_find_unused_parameters: false`, would refuse
    the step
    """
    assert configured.trainer.compute_loss_func is not None  # `custom_loss: true`
    mdl = configured.model_init()
    n_vocab = len(configured.tkzr_cfg.lookup)
    batch = configured.collate_fn(rows(n_vocab, n_classes(configured)))
    configured.trainer.compute_loss(mdl, batch).backward()

    for name, head in mdl.heads.items():
        grads = [p.grad for p in head.parameters()]
        assert all(g is not None and t.isfinite(g).all() for g in grads), name
        assert max(g.abs().max().item() for g in grads) > 0, name
    assert all(p.grad is not None for p in mdl.parameters() if p.requires_grad)


@pytest.mark.slow
def test_a_real_training_step_updates_every_head(configured, tmp_path):
    """the model driven by cotorra's own `TrainerWithCustomLoss`, on both the
    stock-loss and `custom_loss` paths"""
    from transformers import TrainingArguments

    from cotorra.trainer import TrainerWithCustomLoss

    n_vocab = len(configured.tkzr_cfg.lookup)
    dataset = rows(n_vocab, n_classes(configured), configured.cfg.max_seq_len, n=8)

    for compute_loss_func in (None, configured.loss):
        t.manual_seed(0)
        model = configured.model_init()
        before = {
            name: [p.detach().cpu().clone() for p in head.parameters()]
            for name, head in model.heads.items()
        }
        trainer = TrainerWithCustomLoss(
            model=model,
            data_collator=configured.collate_fn,
            compute_loss_func=compute_loss_func,
            train_dataset=dataset,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                **{**configured.cfg.training_args, "learning_rate": 1e-2},
            ),
        )
        trainer.train()
        for name, params in before.items():
            moved = max(
                (now.detach().cpu() - then).abs().max().item()
                for now, then in zip(model.heads[name].parameters(), params)
            )
            assert moved > 0, f"{name} head never updated with {compute_loss_func=}"
