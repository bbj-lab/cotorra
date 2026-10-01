#!/usr/bin/env python3

"""tests for cotorra.model's TteAwareForCausalLM and MppForCausalLM and their configs"""

import pytest
import torch as t
from helpers import TINY_MODEL_ARGS, base_training_cfg, write_cfg
from transformers import AutoConfig, AutoModelForCausalLM

from cotorra.model import MppConfig, MppForCausalLM, TteAwareConfig, TteAwareForCausalLM

VOCAB = 64


def backbone_cfg(**overrides) -> AutoConfig:
    """a tiny llama backbone config, built without touching the hub"""
    return AutoConfig.for_model(
        "llama", vocab_size=VOCAB, **{**TINY_MODEL_ARGS, **overrides}
    )


def tte_model(**cfg_overrides) -> TteAwareForCausalLM:
    """
    built through `from_config` rather than by calling the class, which is what
    `Trainer.model_init` does and what puts the heads under the config's dtype
    """
    return AutoModelForCausalLM.from_config(
        TteAwareConfig(text_config=backbone_cfg(), **cfg_overrides)
    )


def mpp_model(**cfg_overrides) -> MppForCausalLM:
    """`tte_model`'s counterpart for the marked-point-process variant"""
    return AutoModelForCausalLM.from_config(
        MppConfig(text_config=backbone_cfg(), **cfg_overrides)
    )


@pytest.fixture(params=[TteAwareConfig, MppConfig], ids=["tte_aware", "mpp"])
def model(request) -> TteAwareForCausalLM:
    """either model: an `mpp` one carries the time-to-event head too, and every
    test of that head has to hold for it unchanged"""
    t.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        request.param(text_config=backbone_cfg(), tte_weight=0.5)
    )


# ---------------------------------------------------------------- the config


def test_config_requires_a_backbone():
    """the wrapper is meaningless without something to wrap, so it says so
    rather than failing later inside `AutoModel.from_config(None)`"""
    with pytest.raises(ValueError, match="text_config"):
        TteAwareConfig()


def test_config_mirrors_common_keys_off_the_backbone():
    """cotorra's own stages read `vocab_size`/`eos_token_id`/... straight off
    `model.config`, so they have to be present at the top level too"""
    bb = backbone_cfg(bos_token_id=11, eos_token_id=22, tie_word_embeddings=True)
    cfg = TteAwareConfig(text_config=bb)
    assert (cfg.vocab_size, cfg.hidden_size) == (bb.vocab_size, bb.hidden_size)
    assert (cfg.bos_token_id, cfg.eos_token_id) == (11, 22)
    assert cfg.tie_word_embeddings is True
    assert cfg.use_cache == bb.use_cache


def test_config_keeps_an_explicitly_passed_value_over_the_mirror():
    cfg = TteAwareConfig(text_config=backbone_cfg(eos_token_id=22), eos_token_id=33)
    assert cfg.eos_token_id == 33


def test_config_accepts_a_backbone_given_as_a_plain_dict():
    """what `from_pretrained` hands back: `text_config` arrives as json"""
    cfg = TteAwareConfig(text_config={"model_type": "llama", "vocab_size": VOCAB})
    assert type(cfg.text_config).__name__ == "LlamaConfig"
    assert cfg.vocab_size == VOCAB


def test_config_serializes_despite_having_no_default_constructor():
    """
    `has_no_defaults_at_init` is what stops `to_diff_dict` (and so
    `save_pretrained`) from calling the no-argument constructor that
    `test_config_requires_a_backbone` pins as raising
    """
    cfg = TteAwareConfig(text_config=backbone_cfg(), tte_weight=0.25)
    assert cfg.to_diff_dict()["tte_weight"] == 0.25
    assert '"tte_weight": 0.25' in cfg.to_json_string()
    assert '"tte_weight": 0.25' in cfg.to_json_string(use_diff=False)


def test_tte_weight_defaults_to_one():
    assert TteAwareConfig(text_config=backbone_cfg()).tte_weight == 1.0


# ------------------------------------------------------- dtype and structure


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
    bb = backbone_cfg(dtype=dtype)
    stock = AutoModelForCausalLM.from_config(bb)
    mdl = AutoModelForCausalLM.from_config(TteAwareConfig(text_config=bb))
    expected = next(stock.parameters()).dtype
    assert expected == getattr(t, dtype)
    assert next(mdl.model.parameters()).dtype == expected
    assert mdl.lm_head.weight.dtype == expected
    assert mdl.tte_head.weight.dtype == expected


def test_state_dict_matches_the_stock_class_once_the_tte_head_is_dropped():
    """
    the head is laid out as a llama-family `*ForCausalLM` is, so a checkpoint
    can be handed to anything that only knows the stock architecture (sglang,
    which `generative-score` serves through, resolves `architectures[0]`
    against its own registry and has never heard of `tte_aware`)
    """
    bb = backbone_cfg()
    mdl, stock = tte_model(), AutoModelForCausalLM.from_config(bb)
    weights = {k: v for k, v in mdl.state_dict().items() if k != "tte_head.weight"}
    weights.pop("tte_head.bias")
    assert set(weights) == set(stock.state_dict())

    missing, unexpected = stock.load_state_dict(weights, strict=False)
    assert (list(missing), list(unexpected)) == ([], [])
    ids = t.randint(3, VOCAB - 1, (1, 5))
    mdl.eval()
    stock.eval()
    assert t.equal(mdl(input_ids=ids).logits, stock(input_ids=ids).logits)


def test_word_embeddings_are_tied_when_the_backbone_asks_for_it():
    mdl = AutoModelForCausalLM.from_config(
        TteAwareConfig(text_config=backbone_cfg(tie_word_embeddings=True))
    )
    assert mdl.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    assert mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr()


def test_untied_when_the_backbone_says_not_to():
    mdl = AutoModelForCausalLM.from_config(
        TteAwareConfig(text_config=backbone_cfg(tie_word_embeddings=False))
    )
    assert mdl.all_tied_weights_keys == {}


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
    mdl = AutoModelForCausalLM.from_config(TteAwareConfig(text_config=bb))
    assert mdl.all_tied_weights_keys == {"lm_head.weight": "model.wte.weight"}
    assert mdl.lm_head.weight.data_ptr() == mdl.model.wte.weight.data_ptr()


def test_a_saved_model_reloads_through_the_auto_class(tmp_path):
    """
    `Extractor` and `GenerativeScorer` only ever see `mdl-<run_name>/`, which
    they open with `AutoModelForCausalLM.from_pretrained`; importing
    `cotorra.model` is what registers the pair that makes that resolve
    """
    mdl = tte_model(tte_weight=0.25)
    mdl.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert type(reloaded) is TteAwareForCausalLM
    assert reloaded.config.tte_weight == 0.25
    assert type(reloaded.config.text_config).__name__ == "LlamaConfig"
    ids = t.randint(3, VOCAB - 1, (2, 6))
    mdl.eval()
    reloaded.eval()
    assert t.equal(reloaded(input_ids=ids).logits, mdl(input_ids=ids).logits)
    assert t.equal(reloaded(input_ids=ids).tte_pred, mdl(input_ids=ids).tte_pred)


# ------------------------------------------------------------------- forward


def test_forward_returns_both_heads(model):
    ids = t.randint(3, VOCAB - 1, (2, 7))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 7, VOCAB)
    assert out.tte_pred.shape == (2, 7)
    assert (out.loss, out.tte_loss) == (None, None)


def test_forward_exposes_hidden_states_for_the_extractor(model):
    """`Extractor.extract_final` reads `.hidden_states[-1]`"""
    out = model(input_ids=t.randint(3, VOCAB - 1, (2, 7)), output_hidden_states=True)
    assert len(out.hidden_states) == TINY_MODEL_ARGS["num_hidden_layers"] + 1
    assert out.hidden_states[-1].shape == (2, 7, TINY_MODEL_ARGS["hidden_size"])


def test_loss_is_the_language_modelling_term_plus_the_weighted_tte_term(model):
    ids = t.randint(3, VOCAB - 1, (2, 7))
    hours = t.rand(2, 7) * 200
    both = model(input_ids=ids, labels=ids, hours_to_end_time=hours)
    lm_only = model(input_ids=ids, labels=ids)

    assert lm_only.tte_loss is None
    assert both.loss.item() == pytest.approx(
        lm_only.loss.item() + 0.5 * both.tte_loss.item(), rel=1e-5
    )


def test_the_tte_term_is_computed_without_labels(model):
    """
    on cotorra's default `custom_loss: true`, `TrainerWithCustomLoss` pops
    `labels` before calling the model, so the tte term has to be reachable
    from the returned output for `Loss.custom_loss` to fold it in
    """
    out = model(
        input_ids=t.randint(3, VOCAB - 1, (2, 7)), hours_to_end_time=t.rand(2, 7) * 50
    )
    assert out.loss is None
    assert out.get("tte_loss") is not None and t.isfinite(out.tte_loss)


# -------------------------------------------------------- the tte loss itself


def test_the_tte_target_is_the_current_positions_hours(model):
    """
    unshifted, unlike the language-modelling loss: the prediction at position i
    is scored against position i's own target -- the hours remaining once token
    i has been read -- never against the one at i+1. The targets here fall away
    down the sequence, as remaining-hours always do, so scoring the wrong
    pairing gives a different number
    """
    with t.no_grad():  # a head that emits a constant 0, whatever the input
        model.tte_head.weight.zero_()
        model.tte_head.bias.zero_()
    hours = t.tensor([[7.0, 3.0, 1.0, 0.0]])
    out = model(input_ids=t.randint(3, VOCAB - 1, (1, 4)), hours_to_end_time=hours)

    shifted = (t.log1p(hours[0, 1:]) ** 2).mean()
    unshifted = (t.log1p(hours[0, :]) ** 2).mean()
    assert out.tte_loss.item() == pytest.approx(unshifted.item(), rel=1e-5)
    assert unshifted.item() != pytest.approx(shifted.item(), rel=1e-3)


def test_a_zero_target_is_kept_rather_than_masked(model):
    """the last token of a record has nothing left to wait for, and `log1p(0)`
    is exactly 0 -- a real target, not a missing one"""
    with t.no_grad():
        model.tte_head.weight.zero_()
        model.tte_head.bias.zero_()
    out = model(
        input_ids=t.randint(3, VOCAB - 1, (2, 6)), hours_to_end_time=t.zeros(2, 6)
    )
    assert out.tte_loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "hours",
    [
        pytest.param(t.full((2, 6), float("nan")), id="missing-end-time"),
        pytest.param(t.full((2, 6), float("inf")), id="infinite"),
        pytest.param(t.full((2, 6), -3.0), id="recorded-after-the-end-time"),
    ],
)
def test_unusable_targets_are_masked_out(model, hours):
    ids = t.randint(3, VOCAB - 1, (2, 6))
    out = model(input_ids=ids, labels=ids, hours_to_end_time=hours)
    assert out.tte_loss.item() == 0.0
    assert t.isfinite(out.loss)


def test_a_fully_masked_batch_still_carries_a_gradient(model):
    """
    summed-then-normalized rather than masked-then-averaged, so the zero is
    differentiable: a bare `0.0` would leave `tte_head` with no gradient at
    all, which `ddp_find_unused_parameters: false` reports as an error
    """
    ids = t.randint(3, VOCAB - 1, (2, 6))
    out = model(
        input_ids=ids, labels=ids, hours_to_end_time=t.full((2, 6), float("nan"))
    )
    assert out.tte_loss.requires_grad
    out.loss.backward()
    assert model.tte_head.weight.grad is not None
    assert t.isfinite(model.tte_head.weight.grad).all()


def test_only_the_usable_positions_contribute(model):
    """a half-masked batch scores the same as the usable half on its own"""
    with t.no_grad():
        model.tte_head.weight.zero_()
        model.tte_head.bias.zero_()
    ids = t.randint(3, VOCAB - 1, (1, 4))
    nan = float("nan")
    # every position is scored; blank one, which must also leave the denominator
    masked = model(
        input_ids=ids, hours_to_end_time=t.tensor([[3.0, nan, 1.0, 0.0]])
    ).tte_loss
    expected = (t.log1p(t.tensor([3.0, 1.0, 0.0])) ** 2).mean()
    assert masked.item() == pytest.approx(expected.item(), rel=1e-5)


def test_the_head_learns_the_target():
    t.manual_seed(0)
    mdl = tte_model()
    ids = t.randint(3, VOCAB - 1, (4, 8))
    hours = t.rand(4, 8) * 200
    opt = t.optim.AdamW(mdl.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        loss = mdl(input_ids=ids, hours_to_end_time=hours).tte_loss
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2


# --------------------------------------------------------------- integration


def test_model_init_wraps_the_backbone_when_the_objective_is_configured(
    built_trainer, monkeypatch
):
    """`Trainer.model_init` is the only place the pipeline builds a model, so
    the block has to reach it -- it used to build a plain backbone that
    silently swallowed the `hours_to_end_time` batch column"""
    assert not isinstance(built_trainer.model_init(), TteAwareForCausalLM)

    monkeypatch.setitem(built_trainer.cfg, "tte_aware_objective", {"tte_weight": 0.25})
    mdl = built_trainer.model_init()
    tkzr = built_trainer.tkzr_cfg
    assert type(mdl) is TteAwareForCausalLM
    assert mdl.config.tte_weight == 0.25
    assert mdl.config.vocab_size == len(tkzr.lookup)
    assert (mdl.config.bos_token_id, mdl.config.eos_token_id) == (
        tkzr.lookup.BOS,
        tkzr.lookup.EOS,
    )


def test_the_custom_loss_path_still_reaches_the_tte_head(built_trainer, monkeypatch):
    """
    the pairing that used to drop the objective on the floor:
    `TrainerWithCustomLoss.compute_loss` pops `labels`, so the model's own
    `loss + tte_weight * tte_loss` is never formed and `Loss.custom_loss` owns
    the whole objective. With the term missing there, `tte_head` took no
    gradient at all -- and DDP, configured with
    `ddp_find_unused_parameters: false`, would refuse the step
    """
    monkeypatch.setitem(built_trainer.cfg, "tte_aware_objective", {"tte_weight": 0.5})
    assert built_trainer.trainer.compute_loss_func is not None  # `custom_loss: true`

    mdl = built_trainer.model_init()
    n_vocab, seq_len = len(built_trainer.tkzr_cfg.lookup), 8
    batch = built_trainer.collate_fn(
        [
            {
                "input_ids": t.randint(3, n_vocab - 1, (seq_len,)),
                "s_elapsed": t.arange(seq_len, dtype=t.float32) * 300,
                "hours_to_end_time": t.rand(seq_len) * 100,
            }
            for _ in range(2)
        ]
    )
    loss = built_trainer.trainer.compute_loss(mdl, batch)
    loss.backward()

    grad = mdl.tte_head.weight.grad
    assert grad is not None and t.isfinite(grad).all()
    assert grad.abs().max().item() > 0
    assert all(p.grad is not None for p in mdl.parameters() if p.requires_grad)


def test_it_consumes_what_the_trainers_collator_produces(built_trainer, monkeypatch):
    """
    the contract between `Trainer.collate_fn` and this model: with both
    optional blocks configured the batch carries float `position_ids` (time
    based rope) alongside `hours_to_end_time`, and nothing else is needed
    """
    monkeypatch.setitem(built_trainer.cfg, "tte_aware_objective", {"tte_weight": 0.5})
    batch = [
        {
            "input_ids": t.arange(4),
            "s_elapsed": t.tensor([0.0, 300.0, 600.0, 900.0]),
            "hours_to_end_time": t.tensor([3.0, 2.0, 1.0, 0.0]),
        },
        {
            "input_ids": t.arange(4, 8),
            "s_elapsed": t.tensor([0.0, 150.0, 300.0, 450.0]),
            "hours_to_end_time": t.tensor([1.5, 1.0, 0.5, 0.0]),
        },
    ]
    collated = built_trainer.collate_fn(batch)
    assert set(collated) == {"input_ids", "labels", "position_ids", "hours_to_end_time"}

    tkzr = built_trainer.tkzr_cfg
    mdl = AutoModelForCausalLM.from_config(
        TteAwareConfig(
            text_config=AutoConfig.for_model(
                "llama", vocab_size=len(tkzr.lookup), **TINY_MODEL_ARGS
            )
        )
    )
    out = mdl(**collated)
    assert out.logits.shape == (2, 4, len(tkzr.lookup))
    assert out.tte_pred.shape == (2, 4)
    assert t.isfinite(out.loss) and t.isfinite(out.tte_loss)


@pytest.mark.slow
def test_a_real_training_step_updates_both_heads(built_trainer, monkeypatch, tmp_path):
    """the model driven by cotorra's own `TrainerWithCustomLoss`, on both the
    stock-loss and `custom_loss` paths"""
    from transformers import TrainingArguments

    from cotorra.trainer import TrainerWithCustomLoss

    def custom_loss(outputs, labels, **kwargs):
        logits = outputs.get("logits")
        loss = t.nn.CrossEntropyLoss()(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
        )
        return loss + 0.5 * outputs.get("tte_loss")

    monkeypatch.setitem(built_trainer.cfg, "tte_aware_objective", {"tte_weight": 0.5})
    n_vocab = len(built_trainer.tkzr_cfg.lookup)
    seq_len = built_trainer.cfg.max_seq_len
    dataset = [
        {
            "input_ids": t.randint(3, n_vocab - 1, (seq_len,)),
            "s_elapsed": t.arange(seq_len, dtype=t.float32) * 300,
            "hours_to_end_time": t.rand(seq_len) * 100,
        }
        for _ in range(8)
    ]

    for compute_loss_func in (None, custom_loss):
        t.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            TteAwareConfig(
                text_config=AutoConfig.for_model(
                    "llama", vocab_size=n_vocab, **TINY_MODEL_ARGS
                ),
                tte_weight=0.5,
            )
        )
        before = model.tte_head.weight.detach().cpu().clone()
        trainer = TrainerWithCustomLoss(
            model=model,
            data_collator=built_trainer.collate_fn,
            compute_loss_func=compute_loss_func,
            train_dataset=dataset,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                **{**built_trainer.cfg.training_args, "learning_rate": 1e-2},
            ),
        )
        trainer.train()
        moved = (model.tte_head.weight.detach().cpu() - before).abs().max().item()
        assert moved > 0, f"tte_head never updated with {compute_loss_func=}"


# ------------------------------------------------------------- the mpp model

NAN = float("nan")


@pytest.fixture(params=[None, 0.5], ids=["ttnt_only", "with_tte"])
def mpp(request) -> MppForCausalLM:
    """an `mpp` model without and with its optional time-to-event head; every
    test of the time-to-next-token head has to hold for both"""
    t.manual_seed(0)
    return mpp_model(tte_weight=request.param, ttnt_weight=0.25)


@pytest.fixture
def ttnt_only() -> MppForCausalLM:
    t.manual_seed(0)
    return mpp_model(ttnt_weight=0.25)


@pytest.fixture
def mpp_with_tte() -> MppForCausalLM:
    t.manual_seed(0)
    return mpp_model(tte_weight=0.5, ttnt_weight=0.25)


def test_mpp_config_is_a_tte_aware_config_under_its_own_model_type():
    """everything the wrapper does for its backbone carries over; the model
    type -- what the auto classes resolve a checkpoint by -- differs"""
    cfg = MppConfig(text_config=backbone_cfg(bos_token_id=11, eos_token_id=22))
    assert isinstance(cfg, TteAwareConfig)
    assert cfg.model_type == "mpp"
    assert (cfg.vocab_size, cfg.bos_token_id, cfg.eos_token_id) == (VOCAB, 11, 22)
    with pytest.raises(ValueError, match="`mpp` config"):
        MppConfig()


def test_an_mpp_config_leaves_the_tte_head_out_unless_asked():
    cfg = MppConfig(text_config=backbone_cfg())
    assert (cfg.tte_weight, cfg.ttnt_weight) == (None, 1.0)


@pytest.mark.parametrize("tte_weight", [None, 0.5], ids=["ttnt_only", "with_tte"])
def test_mpp_config_serializes_both_weights(tte_weight):
    """an unset `tte_weight` included, so a reloaded model gets the same heads"""
    cfg = MppConfig(text_config=backbone_cfg(), tte_weight=tte_weight, ttnt_weight=0.25)
    written = "null" if tte_weight is None else "0.5"
    for js in (cfg.to_json_string(), cfg.to_json_string(use_diff=False)):
        assert f'"tte_weight": {written}' in js
        assert '"ttnt_weight": 0.25' in js


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_the_mpp_heads_are_built_under_the_backbones_dtype(dtype):
    mdl = AutoModelForCausalLM.from_config(
        MppConfig(text_config=backbone_cfg(dtype=dtype), tte_weight=1.0)
    )
    for module in (mdl.model, mdl.lm_head, mdl.tte_head, mdl.ttnt_head):
        assert next(module.parameters()).dtype == getattr(t, dtype)


def test_without_a_tte_weight_the_tte_head_is_left_out(ttnt_only):
    """rather than built and left untrained, which would cost parameters and
    which `ddp_find_unused_parameters: false` would refuse"""
    assert ttnt_only.tte_head is None
    assert not any(k.startswith("tte_head.") for k in ttnt_only.state_dict())


def test_the_scalar_heads_are_initialized_as_huggingface_does(mpp):
    """a zero bias and a normal weight of `initializer_range` spread, rather
    than torch's uniform default (spread ~0.1 at this width): `post_init` has
    to run after every head exists"""
    for head in (h for h in (mpp.tte_head, mpp.ttnt_head) if h is not None):
        assert t.equal(head.bias, t.zeros_like(head.bias))
        assert head.weight.std().item() == pytest.approx(
            backbone_cfg().initializer_range, rel=0.5
        )


def test_mpp_state_dict_matches_the_stock_class_once_the_scalar_heads_are_dropped(mpp):
    stock = AutoModelForCausalLM.from_config(backbone_cfg())
    weights = {
        k: v
        for k, v in mpp.state_dict().items()
        if not k.startswith(("tte_head.", "ttnt_head."))
    }
    assert set(weights) == set(stock.state_dict())

    missing, unexpected = stock.load_state_dict(weights, strict=False)
    assert (list(missing), list(unexpected)) == ([], [])
    ids = t.randint(3, VOCAB - 1, (1, 5))
    mpp.eval()
    stock.eval()
    assert t.equal(mpp(input_ids=ids).logits, stock(input_ids=ids).logits)


def test_mpp_word_embeddings_are_tied_when_the_backbone_asks_for_it():
    mdl = AutoModelForCausalLM.from_config(
        MppConfig(text_config=backbone_cfg(tie_word_embeddings=True))
    )
    assert mdl.all_tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    assert mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr()


def test_a_saved_mpp_model_reloads_through_the_auto_class(mpp, tmp_path):
    """to the `mpp` class, carrying the heads it was saved with"""
    mpp.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert type(reloaded) is MppForCausalLM
    assert reloaded.config.tte_weight == mpp.config.tte_weight
    assert reloaded.config.ttnt_weight == 0.25
    assert (reloaded.tte_head is None) == (mpp.tte_head is None)
    ids = t.randint(3, VOCAB - 1, (2, 6))
    mpp.eval()
    reloaded.eval()
    before, after = mpp(input_ids=ids), reloaded(input_ids=ids)
    for field in ("logits", "tte_pred", "ttnt_pred"):
        a, b = getattr(before, field), getattr(after, field)
        assert (a is None and b is None) or t.equal(a, b), field


def test_mpp_forward_returns_each_head_it_carries(mpp):
    out = mpp(input_ids=t.randint(3, VOCAB - 1, (2, 7)))
    assert out.logits.shape == (2, 7, VOCAB)
    assert out.ttnt_pred.shape == (2, 7)
    if mpp.tte_head is None:
        assert out.tte_pred is None
    else:
        assert out.tte_pred.shape == (2, 7)
    assert (out.loss, out.tte_loss, out.ttnt_loss) == (None, None, None)


def test_mpp_loss_is_the_language_modelling_term_plus_the_weighted_ttnt_term(ttnt_only):
    ids = t.randint(3, VOCAB - 1, (2, 7))
    full = ttnt_only(input_ids=ids, labels=ids, hours_to_next_token=t.rand(2, 7) * 5)
    lm_only = ttnt_only(input_ids=ids, labels=ids)

    assert (full.tte_loss, lm_only.ttnt_loss) == (None, None)
    assert full.loss.item() == pytest.approx(
        lm_only.loss.item() + 0.25 * full.ttnt_loss.item(), rel=1e-5
    )


def test_with_a_tte_head_the_loss_adds_both_weighted_terms(mpp_with_tte):
    ids = t.randint(3, VOCAB - 1, (2, 7))
    full = mpp_with_tte(
        input_ids=ids,
        labels=ids,
        hours_to_end_time=t.rand(2, 7) * 200,
        hours_to_next_token=t.rand(2, 7) * 5,
    )
    lm_only = mpp_with_tte(input_ids=ids, labels=ids)

    assert full.loss.item() == pytest.approx(
        lm_only.loss.item() + 0.5 * full.tte_loss.item() + 0.25 * full.ttnt_loss.item(),
        rel=1e-5,
    )


def test_a_tte_target_without_a_tte_head_is_refused(ttnt_only):
    """rather than dropped, which would quietly train one head fewer than the
    caller expects"""
    with pytest.raises(ValueError, match="without a time-to-event head"):
        ttnt_only(
            input_ids=t.randint(3, VOCAB - 1, (2, 7)),
            hours_to_end_time=t.rand(2, 7) * 200,
        )


def test_the_ttnt_term_is_computed_without_labels(mpp):
    """reachable from the returned output, as the tte term is, for
    `Loss.custom_loss` to fold in"""
    out = mpp(
        input_ids=t.randint(3, VOCAB - 1, (2, 7)), hours_to_next_token=t.rand(2, 7)
    )
    assert (out.loss, out.tte_loss) == (None, None)
    assert out.get("ttnt_loss") is not None and t.isfinite(out.ttnt_loss)


def test_the_ttnt_target_is_the_current_positions_gap(mpp_with_tte):
    """
    unshifted: position i's target already holds the hours from token i to
    token i+1, so it is scored against the prediction made at i. The trailing
    nan is a record's last token, which has no successor. The tte head is set
    to a constant 3 to show the term reads `ttnt_head`, not its sibling
    """
    with t.no_grad():
        mpp_with_tte.ttnt_head.weight.zero_()
        mpp_with_tte.ttnt_head.bias.zero_()
        mpp_with_tte.tte_head.weight.zero_()
        mpp_with_tte.tte_head.bias.fill_(3.0)
    hours = t.tensor([[2.0, 0.0, 5.0, NAN]])
    out = mpp_with_tte(
        input_ids=t.randint(3, VOCAB - 1, (1, 4)), hours_to_next_token=hours
    )

    unshifted = (t.log1p(t.tensor([2.0, 0.0, 5.0])) ** 2).mean()
    shifted = (t.log1p(t.tensor([0.0, 5.0])) ** 2).mean()
    assert out.ttnt_loss.item() == pytest.approx(unshifted.item(), rel=1e-5)
    assert unshifted.item() != pytest.approx(shifted.item(), rel=1e-3)


def test_each_time_head_reads_only_its_own_target(mpp_with_tte):
    ids = t.randint(3, VOCAB - 1, (2, 6))
    to_end, to_next = t.rand(2, 6) * 100, t.rand(2, 6) * 5
    both = mpp_with_tte(
        input_ids=ids, hours_to_end_time=to_end, hours_to_next_token=to_next
    )
    blanked = mpp_with_tte(
        input_ids=ids, hours_to_end_time=to_end, hours_to_next_token=t.full((2, 6), NAN)
    )
    assert t.equal(blanked.tte_loss, both.tte_loss)
    assert blanked.ttnt_loss.item() == 0.0 < both.ttnt_loss.item()


def test_a_fully_masked_ttnt_batch_still_carries_a_gradient(mpp):
    """a batch of one-token records has no gap to score anywhere; every
    parameter still needs a gradient for `ddp_find_unused_parameters: false`"""
    ids = t.randint(3, VOCAB - 1, (2, 6))
    targets = {"hours_to_next_token": t.full((2, 6), NAN)}
    if mpp.tte_head is not None:
        targets["hours_to_end_time"] = t.rand(2, 6) * 100
    out = mpp(input_ids=ids, labels=ids, **targets)
    assert out.ttnt_loss.item() == 0.0 and out.ttnt_loss.requires_grad
    out.loss.backward()
    assert all(p.grad is not None for p in mpp.parameters() if p.requires_grad)
    assert t.isfinite(mpp.ttnt_head.weight.grad).all()


def test_the_ttnt_head_learns_the_target():
    t.manual_seed(0)
    mdl = mpp_model()
    ids = t.randint(3, VOCAB - 1, (4, 8))
    hours = t.rand(4, 8) * 5
    opt = t.optim.AdamW(mdl.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        loss = mdl(input_ids=ids, hours_to_next_token=hours).ttnt_loss
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2


# --------------------------------------------------------- mpp integration


def mpp_batch(n_vocab: int, seq_len: int = 8, n: int = 2) -> list[dict]:
    """rows as `Loader.get_train_data` yields them with both time objectives
    configured; the collator picks out the ones its config asks for"""
    return [
        {
            "input_ids": t.randint(3, n_vocab - 1, (seq_len,)),
            "s_elapsed": t.arange(seq_len, dtype=t.float32) * 300,
            "hours_to_end_time": t.rand(seq_len) * 100,
            "hours_to_next_token": t.rand(seq_len),
        }
        for _ in range(n)
    ]


@pytest.fixture(params=[False, True], ids=["ttnt_only", "with_tte"])
def mpp_trainer(request, built_trainer, monkeypatch):
    """`built_trainer` configured for an `mpp` model, without and with the
    time-to-event objective alongside"""
    monkeypatch.setitem(built_trainer.cfg, "mpp_objective", {"ttnt_weight": 0.25})
    if request.param:
        monkeypatch.setitem(
            built_trainer.cfg, "tte_aware_objective", {"tte_weight": 0.5}
        )
    return built_trainer


def test_model_init_builds_an_mpp_model_from_its_objective(mpp_trainer):
    """with a time-to-event head exactly when `tte_aware_objective` is set
    too, weighted as that block says"""
    mdl = mpp_trainer.model_init()
    tkzr = mpp_trainer.tkzr_cfg
    tte = "tte_aware_objective" in mpp_trainer.cfg
    assert type(mdl) is MppForCausalLM
    assert (mdl.tte_head is not None) == tte
    assert mdl.config.tte_weight == (0.5 if tte else None)
    assert mdl.config.ttnt_weight == 0.25
    assert mdl.config.vocab_size == len(tkzr.lookup)
    assert (mdl.config.bos_token_id, mdl.config.eos_token_id) == (
        tkzr.lookup.BOS,
        tkzr.lookup.EOS,
    )


@pytest.mark.parametrize("tte", [False, True], ids=["ttnt_only", "with_tte"])
def test_empty_blocks_still_build_their_heads(built_trainer, monkeypatch, tte):
    """a block left empty in the yaml parses to None; every weight defaults to
    1.0"""
    monkeypatch.setitem(built_trainer.cfg, "mpp_objective", None)
    if tte:
        monkeypatch.setitem(built_trainer.cfg, "tte_aware_objective", None)
    mdl = built_trainer.model_init()
    assert type(mdl) is MppForCausalLM
    assert mdl.config.tte_weight == (1.0 if tte else None)
    assert mdl.config.ttnt_weight == 1.0


def test_a_tte_weight_under_mpp_objective_is_refused(processed, tmp_path):
    """that weight belongs under `tte_aware_objective`; left where it is, it
    would build a time-to-event head that nothing loads a target for"""
    from cotorra.trainer import Trainer

    cfg = base_training_cfg(mpp_objective={"tte_weight": 1.0, "ttnt_weight": 1.0})
    with pytest.raises(ValueError, match="add a `tte_aware_objective` block"):
        Trainer(
            training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
            processed_data_home=processed,
            output_home=tmp_path,
        )


def test_an_mpp_model_consumes_what_the_trainers_collator_produces(mpp_trainer):
    tte = "tte_aware_objective" in mpp_trainer.cfg
    n_vocab = len(mpp_trainer.tkzr_cfg.lookup)
    collated = mpp_trainer.collate_fn(mpp_batch(n_vocab, seq_len=4))
    assert set(collated) == {
        "input_ids",
        "labels",
        "position_ids",
        "hours_to_next_token",
    } | ({"hours_to_end_time"} if tte else set())

    out = mpp_trainer.model_init()(**collated)
    assert out.ttnt_pred.shape == (2, 4)
    terms = [out.loss, out.ttnt_loss] + ([out.tte_loss] if tte else [])
    assert all(t.isfinite(x) for x in terms)


def test_the_custom_loss_path_reaches_every_head(mpp_trainer):
    """`Loss.custom_loss` owns the whole objective on this path, so it is the
    one that has to fold in each time term"""
    assert mpp_trainer.trainer.compute_loss_func is not None  # `custom_loss: true`

    mdl = mpp_trainer.model_init()
    batch = mpp_trainer.collate_fn(mpp_batch(len(mpp_trainer.tkzr_cfg.lookup)))
    mpp_trainer.trainer.compute_loss(mdl, batch).backward()

    for head in (h for h in (mdl.tte_head, mdl.ttnt_head) if h is not None):
        assert t.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().max().item() > 0
    assert all(p.grad is not None for p in mdl.parameters() if p.requires_grad)


@pytest.mark.slow
def test_a_real_training_step_updates_every_head(mpp_trainer, tmp_path):
    """on both the stock-loss and cotorra's own `custom_loss` paths"""
    from transformers import TrainingArguments

    from cotorra.trainer import TrainerWithCustomLoss

    n_vocab = len(mpp_trainer.tkzr_cfg.lookup)
    dataset = mpp_batch(n_vocab, seq_len=mpp_trainer.cfg.max_seq_len, n=8)

    for compute_loss_func in (None, mpp_trainer.loss):
        t.manual_seed(0)
        model = mpp_trainer.model_init()
        before = {
            name: head.weight.detach().cpu().clone()
            for name, head in (
                ("tte_head", model.tte_head),
                ("ttnt_head", model.ttnt_head),
            )
            if head is not None
        }
        trainer = TrainerWithCustomLoss(
            model=model,
            data_collator=mpp_trainer.collate_fn,
            compute_loss_func=compute_loss_func,
            train_dataset=dataset,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                **{**mpp_trainer.cfg.training_args, "learning_rate": 1e-2},
            ),
        )
        trainer.train()
        for name, weight in before.items():
            moved = (getattr(model, name).weight.detach().cpu() - weight).abs().max()
            assert moved.item() > 0, f"{name} never updated with {compute_loss_func=}"
