#!/usr/bin/env python3

"""tests for cotorra.model.TteAwareForCausalLM and its config"""

import pytest
import torch as t
from helpers import TINY_MODEL_ARGS
from transformers import AutoConfig, AutoModelForCausalLM

from cotorra.model import TteAwareConfig, TteAwareForCausalLM

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


@pytest.fixture
def model() -> TteAwareForCausalLM:
    t.manual_seed(0)
    return tte_model(tte_weight=0.5)


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

    assert isinstance(reloaded, TteAwareForCausalLM)
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


def test_the_tte_target_is_the_next_positions_hours(model):
    """
    shifted exactly like the language-modelling loss: the prediction at
    position i is scored against the target at i+1, never against position i's
    own. The targets here fall away down the sequence, as remaining-hours
    always do, so scoring the wrong pairing gives a different number
    """
    with t.no_grad():  # a head that emits a constant 0, whatever the input
        model.tte_head.weight.zero_()
        model.tte_head.bias.zero_()
    hours = t.tensor([[7.0, 3.0, 1.0, 0.0]])
    out = model(input_ids=t.randint(3, VOCAB - 1, (1, 4)), hours_to_end_time=hours)

    shifted = (t.log1p(hours[0, 1:]) ** 2).mean()
    unshifted = (t.log1p(hours[0, :]) ** 2).mean()
    assert out.tte_loss.item() == pytest.approx(shifted.item(), rel=1e-5)
    assert shifted.item() != pytest.approx(unshifted.item(), rel=1e-3)


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
    # targets at positions 1..3 are what get scored; blank the middle one
    masked = model(
        input_ids=ids, hours_to_end_time=t.tensor([[0.0, 1.0, nan, 3.0]])
    ).tte_loss
    expected = (t.log1p(t.tensor([1.0, 3.0])) ** 2).mean()
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
    assert isinstance(mdl, TteAwareForCausalLM)
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
