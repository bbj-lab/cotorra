#!/usr/bin/env python3

"""tests for cotorra.generator: autoregressive (time, token) generation"""

import math

import polars as pl
import pytest
import torch as t
from helpers import TINY_MODEL_ARGS, base_training_cfg, write_cfg
from transformers import AutoConfig, AutoModelForCausalLM

from cotorra.generator import Trajectory, generate
from cotorra.model import (
    CotorraConfig,
    CotorraForCausalLM,
    ZeroInflatedLogNormalMixture,
)

VOCAB, EOS, SEC_PER_POS_ID = 64, 2, 300


def tnt_model(mixture_components=None, vocab=VOCAB, eos=EOS) -> CotorraForCausalLM:
    """a tiny model with a time-to-next-token head; the wide init makes attention
    -- and so the positions the generated times feed back through -- matter,
    which the default 0.02 leaves near-uniform"""
    t.manual_seed(0)
    bb = AutoConfig.for_model(
        "llama",
        vocab_size=vocab,
        eos_token_id=eos,
        initializer_range=0.5,
        **TINY_MODEL_ARGS,
    )
    tnt = (
        {} if mixture_components is None else {"mixture_components": mixture_components}
    )
    cfg = CotorraConfig(text_config=bb, heads={"tnt": tnt})
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.fixture(params=[None, 3], ids=["point_head", "mixture_head"])
def model(request) -> CotorraForCausalLM:
    return tnt_model(request.param)


@pytest.fixture
def model64(model) -> CotorraForCausalLM:
    """
    for the equivalence tests: a batch computes a row in a different order than
    the row alone (or a cached step than a full pass), and float32's ~1e-7 of
    rounding, fed back through a wide random model's positions, can flip a
    later token -- noise, not a bug, which float64 keeps out of the comparison
    """
    return model.double()


class Forced(t.nn.Module):
    """an lm head whose logits always favor `token`"""

    def __init__(self, head, token):
        super().__init__()
        self.head, self.token = head, token

    def forward(self, hidden_states):
        logits = self.head(hidden_states).clone()
        logits[..., self.token] += 1e4
        return logits


PROMPTS = [t.tensor([1, 5, 7]), t.tensor([1, 9, 9, 4, 11]), t.tensor([1, 3])]
PROMPT_SECS = [
    t.tensor([0.0, 0.0, 900.0]),
    t.tensor([0.0, 60.0, 60.0, 3600.0, 7200.0]),
    t.tensor([0.0, 1800.0]),
]


def gaps_at(model, hidden, next_ids) -> t.Tensor:
    """the hours greedy decoding takes from the time head, recomputed"""
    return model.sample_hours_to_next_token(hidden, next_ids, do_sample=False)


def test_greedy_generation_matches_a_full_pass_over_the_result(model64):
    """
    the cached, one-token-at-a-time loop has to agree with reading the whole
    finished sequence in one go -- with each generated token at the rope
    position its generated time gives it -- on every token and every gap
    """
    prompt, prompt_secs = PROMPTS[1], PROMPT_SECS[1]
    (traj,) = generate(
        model64,
        [prompt],
        [prompt_secs],
        sec_per_pos_id=SEC_PER_POS_ID,
        max_new_tokens=10,
        do_sample=False,
    )
    assert len(traj.tokens) > 1

    ids = t.cat([prompt, traj.tokens])[None]
    secs = t.cat([prompt_secs.double(), traj.s_elapsed])[None]
    positions = (secs / SEC_PER_POS_ID + t.arange(ids.shape[-1])).float()
    with t.inference_mode():
        hidden = model64.model(input_ids=ids, position_ids=positions).last_hidden_state
        n0, n = len(prompt), len(traj.tokens)
        before = hidden[0, n0 - 1 : n0 - 1 + n]  # what each new token followed
        assert t.equal(model64.lm_head(before).argmax(-1), traj.tokens)
        gaps = gaps_at(model64, before, traj.tokens)
    assert t.allclose(
        gaps.double(), secs[0, n0:].diff(prepend=secs[0, n0 - 1 : n0]) / 3600, rtol=1e-5
    )


def test_left_padded_prompts_generate_as_they_would_alone(model64):
    kwargs = dict(sec_per_pos_id=SEC_PER_POS_ID, max_new_tokens=8, do_sample=False)
    together = generate(model64, PROMPTS, PROMPT_SECS, **kwargs)
    for prompt, secs, traj in zip(PROMPTS, PROMPT_SECS, together):
        (alone,) = generate(model64, [prompt], [secs], **kwargs)
        assert t.equal(traj.tokens, alone.tokens)
        assert t.allclose(traj.s_elapsed, alone.s_elapsed, rtol=1e-6)


def test_sampled_times_start_after_the_prompt_and_never_go_back(model):
    trajs = generate(
        model,
        PROMPTS,
        PROMPT_SECS,
        sec_per_pos_id=SEC_PER_POS_ID,
        max_new_tokens=20,
        generator=t.Generator().manual_seed(0),
    )
    for secs, traj in zip(PROMPT_SECS, trajs):
        assert isinstance(traj, Trajectory)
        assert len(traj.tokens) == len(traj.s_elapsed) > 0
        assert ((traj.tokens >= 0) & (traj.tokens < VOCAB)).all()
        assert (traj.s_elapsed.diff(prepend=secs[-1:].double()) >= 0).all()


def test_sampling_is_reproducible_under_a_generator(model):
    kwargs = dict(sec_per_pos_id=SEC_PER_POS_ID, max_new_tokens=12)
    a, b = (
        generate(
            model,
            PROMPTS,
            PROMPT_SECS,
            generator=t.Generator().manual_seed(3),
            **kwargs,
        )
        for _ in range(2)
    )
    for x, y in zip(a, b):
        assert t.equal(x.tokens, y.tokens)
        assert t.equal(x.s_elapsed, y.s_elapsed)


def test_a_row_stops_at_its_first_eos_and_keeps_it(model, monkeypatch):
    monkeypatch.setattr(model, "lm_head", Forced(model.lm_head, EOS))
    trajs = generate(
        model, PROMPTS, PROMPT_SECS, sec_per_pos_id=SEC_PER_POS_ID, max_new_tokens=5
    )
    assert all(traj.tokens.tolist() == [EOS] for traj in trajs)


def test_without_eos_a_row_runs_to_max_new_tokens(model, monkeypatch):
    monkeypatch.setattr(model, "lm_head", Forced(model.lm_head, 7))
    trajs = generate(
        model, PROMPTS, PROMPT_SECS, sec_per_pos_id=SEC_PER_POS_ID, max_new_tokens=6
    )
    assert all(traj.tokens.tolist() == [7] * 6 for traj in trajs)


def test_the_horizon_drops_the_first_token_past_it(monkeypatch):
    """a point head forced to a 2-hour gap: +2h and +4h fit in a 5-hour
    horizon, and the token at +6h -- the first outside it -- is dropped"""
    model = tnt_model()
    monkeypatch.setattr(model, "lm_head", Forced(model.lm_head, 7))
    with t.no_grad():  # softplus(log 2) = log1p(2): two hours
        model.heads["tnt"].linear.weight.zero_()
        model.heads["tnt"].linear.bias.fill_(math.log(2.0))
    (traj,) = generate(
        model,
        [PROMPTS[0]],
        [PROMPT_SECS[0]],
        sec_per_pos_id=SEC_PER_POS_ID,
        max_new_tokens=50,
        max_hours=5,
    )
    end = PROMPT_SECS[0][-1].item()
    assert traj.tokens.tolist() == [7, 7]
    assert traj.s_elapsed.tolist() == pytest.approx([end + 7200, end + 14400])


def test_same_timestamp_draws_keep_the_clock_still():
    """a mixture head sure every next token shares this one's timestamp"""
    model = tnt_model(mixture_components=3)
    with t.no_grad():
        model.heads["tnt"].proj[2].weight[0].zero_()
        model.heads["tnt"].proj[2].bias[0] = 30.0  # the zero-gap logit
    trajs = generate(
        model, PROMPTS, PROMPT_SECS, sec_per_pos_id=SEC_PER_POS_ID, max_new_tokens=10
    )
    for secs, traj in zip(PROMPT_SECS, trajs):
        assert (traj.s_elapsed == secs[-1].double()).all()


def test_each_gap_is_drawn_given_the_token_just_sampled(monkeypatch):
    """
    the mixture head's whole point: a stand-in distribution makes the gap to an
    even token exactly 0 and to an odd one an hour, so the generated times have
    to follow the generated tokens step by step
    """
    model = tnt_model(mixture_components=1)

    def by_parity(hidden_states, next_input_ids, do_sample=True, generator=None):
        odd = (next_input_ids % 2).to(dtype=t.float64)
        params = t.stack(
            [
                30.0 - 60.0 * odd,  # zero-gap logit: sure for even, never for odd
                t.zeros_like(odd),  # the one component's weight
                t.zeros_like(odd),  # its mean log-hours: an hour
                t.full_like(odd, -30.0),  # its (softplus) scale: ~1e-3
            ],
            dim=-1,
        )
        return ZeroInflatedLogNormalMixture(params).sample(generator)

    monkeypatch.setattr(model, "sample_hours_to_next_token", by_parity)
    trajs = generate(
        model,
        PROMPTS,
        PROMPT_SECS,
        sec_per_pos_id=SEC_PER_POS_ID,
        max_new_tokens=30,
        generator=t.Generator().manual_seed(0),
    )
    for secs, traj in zip(PROMPT_SECS, trajs):
        hours = traj.s_elapsed.diff(prepend=secs[-1:].double()) / 3600
        odd = traj.tokens % 2 == 1
        assert odd.any() and (~odd).any()
        assert (hours[~odd] == 0).all()
        assert hours[odd].tolist() == pytest.approx([1.0] * int(odd.sum()), rel=0.01)


def test_without_sec_per_pos_id_the_prompts_times_go_unread(model):
    """a model trained without time-based rope positions tokens by index, so
    its generated tokens can't depend on when the prompt's events happened"""
    shifted = [s + 3600 * 24 * t.arange(len(s)) for s in PROMPT_SECS]
    kwargs = dict(max_new_tokens=8, do_sample=False)
    plain = generate(model, PROMPTS, PROMPT_SECS, sec_per_pos_id=None, **kwargs)
    moved = generate(model, PROMPTS, shifted, sec_per_pos_id=None, **kwargs)
    timed = generate(model, PROMPTS, shifted, sec_per_pos_id=SEC_PER_POS_ID, **kwargs)
    assert all(t.equal(a.tokens, b.tokens) for a, b in zip(plain, moved))
    assert not all(t.equal(a.tokens, b.tokens) for a, b in zip(plain, timed))


def test_only_a_model_with_a_tnt_head_has_times_to_give():
    bb = AutoConfig.for_model("llama", vocab_size=VOCAB, **TINY_MODEL_ARGS)
    for mdl in (
        AutoModelForCausalLM.from_config(bb),
        AutoModelForCausalLM.from_config(
            CotorraConfig(
                text_config=bb, heads={"tte": {}, "disposition": {"classes": ["X"]}}
            )
        ),
    ):
        with pytest.raises(TypeError, match="time-to-next-token head"):
            generate(mdl, PROMPTS, PROMPT_SECS, sec_per_pos_id=None, max_new_tokens=2)


@pytest.mark.parametrize(
    "prompts, max_new_tokens",
    [([t.tensor([], dtype=t.long)], 4), (PROMPTS[:1], 0)],
    ids=["empty-prompt", "no-new-tokens"],
)
def test_degenerate_requests_are_refused(model, prompts, max_new_tokens):
    with pytest.raises(ValueError, match="non-empty prompts"):
        generate(
            model,
            prompts,
            [t.zeros(len(p)) for p in prompts],
            sec_per_pos_id=None,
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize("components", [None, 3], ids=["point_head", "mixture_head"])
def test_it_continues_the_inference_tables_prompts(
    processed, tokenizer_cfg, components
):
    """the prompts `cocoa winnow` writes -- `tokens_past` with `s_elapsed_past`,
    in integer seconds -- go in as they are"""
    model = tnt_model(
        components, vocab=len(tokenizer_cfg.lookup), eos=tokenizer_cfg.lookup.EOS
    )
    df = pl.read_parquet(processed / "held_out_for_inference.parquet", n_rows=6)
    prompts = [t.tensor(x) for x in df["tokens_past"]]
    secs = [t.tensor(x) for x in df["s_elapsed_past"]]
    trajs = generate(
        model,
        prompts,
        secs,
        sec_per_pos_id=SEC_PER_POS_ID,
        max_new_tokens=16,
        max_hours=24,
        generator=t.Generator().manual_seed(0),
    )
    assert len(trajs) == len(prompts)
    for s, traj in zip(secs, trajs):
        assert ((traj.tokens >= 0) & (traj.tokens < len(tokenizer_cfg.lookup))).all()
        since_end = traj.s_elapsed - s[-1].item()
        assert ((since_end >= 0) & (since_end <= 24 * 3600)).all()


@pytest.mark.slow
def test_a_trained_mixture_model_generates_from_its_saved_checkpoint(
    processed, tmp_path
):
    """train -> save -> reload -> generate, with the rope spacing the training
    config used"""
    from cotorra.trainer import Trainer

    cfg = base_training_cfg(tnt_objective={"mixture_components": 3})
    trainer = Trainer(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=tmp_path / "out",
    )
    trainer.train()
    model = AutoModelForCausalLM.from_pretrained(
        tmp_path / "out" / f"mdl-{trainer.run_name}"
    ).eval()
    assert model.config.heads["tnt"]["mixture_components"] == 3

    df = pl.read_parquet(processed / "held_out_for_inference.parquet", n_rows=4)
    secs = [t.tensor(x) for x in df["s_elapsed_past"]]
    trajs = generate(
        model,
        [t.tensor(x) for x in df["tokens_past"]],
        secs,
        sec_per_pos_id=trainer.cfg.time_based_rope.sec_per_pos_id,
        max_new_tokens=16,
        max_hours=48,
        generator=t.Generator().manual_seed(0),
    )
    for s, traj in zip(secs, trajs):
        assert len(traj.tokens) > 0
        since_end = traj.s_elapsed - s[-1].item()
        assert (since_end.diff(prepend=since_end.new_zeros(1)) >= 0).all()
        assert (since_end <= 48 * 3600).all()
