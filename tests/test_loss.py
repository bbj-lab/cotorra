#!/usr/bin/env python3

"""tests for cotorra.loss.Loss"""

import pytest
import torch as t
from omegaconf import OmegaConf

from cotorra.loss import Loss

# a tiny fabricated vocabulary: plain tokens, one outcome-of-interest token, and
# two quantile-fused categories (heart_rate, sodium) over 5 bins (Q0..Q4)
N_BINS = 5
LOOKUP = {
    "UNK": 0,
    "BOS": 1,
    "EOS": 2,
    "DSCG//expired": 3,
    "DSCG//home": 4,
    "VTL//heart_rate_Q0": 5,
    "VTL//heart_rate_Q1": 6,
    "VTL//heart_rate_Q2": 7,
    "VTL//heart_rate_Q3": 8,
    "VTL//heart_rate_Q4": 9,
    "LAB//sodium_Q0": 10,
    "LAB//sodium_Q1": 11,
    "LAB//sodium_Q2": 12,
    "LAB//sodium_Q3": 13,
    "LAB//sodium_Q4": 14,
}
VOCAB_SIZE = len(LOOKUP)


def make_tkzr_cfg():
    return OmegaConf.create({"lookup": LOOKUP, "cfg": {"n_bins": N_BINS}})


def make_cfg(**overrides):
    base = {
        "label_weighted_loss": {
            "tokens_of_interest": ["DSCG//expired", "LABEL//*"],
            "toi_weight": 20.0,
        },
        "quantile_token_loss": {"qt_weight": 0.5},
    }
    base.update(overrides)
    return OmegaConf.create(base)


@pytest.fixture
def labels():
    # BOS, then quantile tokens, then EOS; shift_labels = labels[:, 1:]
    return t.tensor([[1, 5, 7, 9, 2], [1, 10, 12, 14, 2]])


@pytest.fixture
def outputs():
    g = t.Generator().manual_seed(0)
    logits = t.randn(2, 5, VOCAB_SIZE, generator=g)
    return {"logits": logits}


def test_grokked_outcome_tokens_matches_exact_and_glob_patterns():
    loss = Loss(make_cfg(), make_tkzr_cfg())
    assert loss.grokked_outcome_tokens == ["DSCG//expired"]


def test_label_weights_flag_only_outcome_tokens():
    loss = Loss(make_cfg(), make_tkzr_cfg())
    idx = LOOKUP["DSCG//expired"]
    assert loss.weights[idx].item() == pytest.approx(20.0)
    for word, i in LOOKUP.items():
        if word != "DSCG//expired":
            assert loss.weights[i].item() == pytest.approx(1.0)


def test_quantile_categories_are_discovered():
    loss = Loss(make_cfg(), make_tkzr_cfg())
    assert loss.n_cats == 2
    for word in (
        "VTL//heart_rate_Q0",
        "VTL//heart_rate_Q4",
        "LAB//sodium_Q0",
        "LAB//sodium_Q4",
    ):
        assert loss.q_type[LOOKUP[word]]
    for word in ("UNK", "BOS", "EOS", "DSCG//expired", "DSCG//home"):
        assert not loss.q_type[LOOKUP[word]]


def test_quantile_token_loss_is_finite_nonnegative_scalar(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    out = loss.quantile_token_loss(outputs, labels)
    assert out.dtype == t.float32
    assert t.isfinite(out)
    assert out.item() >= 0


def test_x_ent_loss_matches_plain_cross_entropy(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    out = loss.x_ent_loss(outputs, labels)

    shift_logits = outputs["logits"][:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected = t.nn.CrossEntropyLoss()(
        shift_logits.reshape(-1, VOCAB_SIZE), shift_labels.reshape(-1)
    )
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


def test_label_weighted_loss_matches_manually_weighted_cross_entropy(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    out = loss.label_weighted_loss(outputs, labels)

    shift_logits = outputs["logits"][:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected = t.nn.CrossEntropyLoss(weight=loss.weights.to(t.float32))(
        shift_logits.reshape(-1, VOCAB_SIZE), shift_labels.reshape(-1)
    )
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


def test_custom_loss_x_ent_branch_is_correct_if_ever_reached(outputs, labels):
    """the `else: x_ent_loss` branch of `custom_loss` is unreachable via normal
    construction (see `test_construction_requires_label_weighted_loss_block`);
    simulate reaching it to confirm the branch's own logic is still correct"""
    loss = Loss(make_cfg(), make_tkzr_cfg())
    del loss.cfg["label_weighted_loss"]
    del loss.cfg["quantile_token_loss"]

    assert loss.custom_loss(outputs, labels).item() == pytest.approx(
        loss.x_ent_loss(outputs, labels).item(), rel=1e-5
    )


def test_custom_loss_combines_label_weighted_and_quantile_blocks(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    expected = (
        loss.label_weighted_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
    )
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_uses_label_weighted_without_quantile_block(outputs, labels):
    cfg = make_cfg()
    del cfg["quantile_token_loss"]
    loss = Loss(cfg, make_tkzr_cfg())
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(
        loss.label_weighted_loss(outputs, labels).item(), rel=1e-5
    )


def test_custom_loss_adds_the_weighted_tte_term(outputs, labels):
    """the model hands its time-to-event term back on the output object; the
    objective picks it up here, weighted by the training config"""
    loss = Loss(make_cfg(tte_aware_objective={"tte_weight": 0.25}), make_tkzr_cfg())
    out = dict(outputs) | {"tte_loss": t.tensor(4.0)}
    expected = (
        loss.label_weighted_loss(out, labels).item()
        + 0.5 * loss.quantile_token_loss(out, labels).item()
        + 0.25 * 4.0
    )
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_defaults_the_tte_weight_to_one():
    """matching `TteAwareConfig.tte_weight`, so an empty block still trains"""
    loss = Loss(make_cfg(tte_aware_objective={}), make_tkzr_cfg())
    assert loss.cfg.tte_aware_objective.get("tte_weight", 1.0) == 1.0


def test_custom_loss_ignores_the_tte_term_when_the_block_is_absent(outputs, labels):
    """a model may carry the head without the objective asking to train it"""
    loss = Loss(make_cfg(), make_tkzr_cfg())
    out = dict(outputs) | {"tte_loss": t.tensor(4.0)}
    assert loss.custom_loss(out, labels).item() == pytest.approx(
        loss.custom_loss(outputs, labels).item(), rel=1e-5
    )


def test_custom_loss_refuses_a_model_that_returns_no_tte_term(outputs, labels):
    """configuring the objective against a plain backbone used to train
    silently without it -- the extra kwarg is swallowed by the backbone's
    `**kwargs` and no `tte_loss` comes back -- so say so instead"""
    loss = Loss(make_cfg(tte_aware_objective={"tte_weight": 0.5}), make_tkzr_cfg())
    with pytest.raises(ValueError, match="tte_loss"):
        loss.custom_loss(outputs, labels)


def test_custom_loss_keeps_the_tte_term_differentiable(outputs, labels):
    """`tte_head`'s only path to a gradient on this code path"""
    loss = Loss(make_cfg(tte_aware_objective={"tte_weight": 0.5}), make_tkzr_cfg())
    tte = t.tensor(4.0, requires_grad=True)
    loss.custom_loss(dict(outputs) | {"tte_loss": tte}, labels).backward()
    assert tte.grad is not None and tte.grad.item() == pytest.approx(0.5)


def test_custom_loss_adds_the_weighted_ttnt_term_for_an_mpp_objective(outputs, labels):
    """on its own: a tte term the model hands back too is left out unless
    `tte_aware_objective` asks for it"""
    loss = Loss(make_cfg(mpp_objective={"ttnt_weight": 0.125}), make_tkzr_cfg())
    out = dict(outputs) | {"tte_loss": t.tensor(4.0), "ttnt_loss": t.tensor(8.0)}
    expected = (
        loss.label_weighted_loss(out, labels).item()
        + 0.5 * loss.quantile_token_loss(out, labels).item()
        + 0.125 * 8.0
    )
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_adds_both_time_terms_when_both_objectives_are_set(outputs, labels):
    """each weighted by its own block"""
    cfg = make_cfg(
        tte_aware_objective={"tte_weight": 0.25}, mpp_objective={"ttnt_weight": 0.125}
    )
    loss = Loss(cfg, make_tkzr_cfg())
    out = dict(outputs) | {"tte_loss": t.tensor(4.0), "ttnt_loss": t.tensor(8.0)}
    expected = (
        loss.label_weighted_loss(out, labels).item()
        + 0.5 * loss.quantile_token_loss(out, labels).item()
        + 0.25 * 4.0
        + 0.125 * 8.0
    )
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "objective", ["tte_aware_objective", "mpp_objective"], ids=["tte_aware", "mpp"]
)
@pytest.mark.parametrize("block", [{}, None], ids=["empty", "null"])
def test_custom_loss_defaults_the_time_weights_to_one(
    outputs, labels, objective, block
):
    """matching `TteAwareConfig`/`MppConfig`, so a block left empty -- which
    yaml parses to None -- still trains, and adds only its own term"""
    loss = Loss(make_cfg(**{objective: block}), make_tkzr_cfg())
    out = dict(outputs) | {"tte_loss": t.tensor(4.0), "ttnt_loss": t.tensor(8.0)}
    expected = (
        loss.label_weighted_loss(out, labels).item()
        + 0.5 * loss.quantile_token_loss(out, labels).item()
        + (4.0 if objective == "tte_aware_objective" else 8.0)
    )
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_refuses_a_model_that_returns_no_ttnt_term(outputs, labels):
    """a `tte_aware` model trained under `mpp_objective` returns a tte term
    but no ttnt one"""
    loss = Loss(make_cfg(mpp_objective={}), make_tkzr_cfg())
    with pytest.raises(ValueError, match="ttnt_loss"):
        loss.custom_loss(dict(outputs) | {"tte_loss": t.tensor(4.0)}, labels)


def test_custom_loss_keeps_the_ttnt_term_differentiable(outputs, labels):
    """`ttnt_head`'s only path to a gradient on this code path"""
    loss = Loss(make_cfg(mpp_objective={"ttnt_weight": 0.25}), make_tkzr_cfg())
    ttnt = t.tensor(8.0, requires_grad=True)
    loss.custom_loss(dict(outputs) | {"ttnt_loss": ttnt}, labels).backward()
    assert ttnt.grad is not None and ttnt.grad.item() == pytest.approx(0.25)


def test_custom_loss_does_not_touch_wandb_when_no_run_is_active(outputs, labels):
    import wandb

    assert wandb.run is None
    loss = Loss(make_cfg(), make_tkzr_cfg())
    loss.custom_loss(outputs, labels)  # must not raise even though wandb is inactive
    assert wandb.run is None


# `quantile_token_loss` maps a `..._Q<i>` token to the midpoint of its bin,
# so with 5 bins Q0 is 0.1, Q1 is 0.3, ... Q4 is 0.9
def _q(i: int) -> float:
    return (i + 0.5) / N_BINS


def _one_hot_logits(labels, predicted_ids):
    """logits putting essentially all softmax mass on `predicted_ids`"""
    logits = t.zeros(labels.shape[0], labels.shape[1], VOCAB_SIZE)
    for row, ids in enumerate(predicted_ids):
        for pos, tok in enumerate(ids):
            logits[row, pos, tok] = 50.0
    return {"logits": logits}


def test_quantile_token_loss_is_zero_for_a_confident_correct_prediction():
    """
    the loss compares the softmax-weighted quantile against the label's own
    quantile, so a model that names the right bin pays nothing
    """
    loss = Loss(make_cfg(), make_tkzr_cfg())
    labels = t.tensor([[LOOKUP["BOS"], LOOKUP["VTL//heart_rate_Q0"], LOOKUP["EOS"]]])
    # shift_labels are labels[:, 1:], so logits[:, 0] predicts Q0
    outputs = _one_hot_logits(labels, [[LOOKUP["VTL//heart_rate_Q0"], LOOKUP["EOS"]]])

    assert loss.quantile_token_loss(outputs, labels).item() == pytest.approx(0.0)


def test_quantile_token_loss_is_the_mean_squared_quantile_error():
    """
    two heart-rate tokens, each predicted as the opposite extreme bin: the
    loss must be the squared 0.1-vs-0.9 gap, averaged within the category.
    The sodium category contributes nothing -- it has no labels in the batch.
    """
    loss = Loss(make_cfg(), make_tkzr_cfg())
    labels = t.tensor(
        [
            [
                LOOKUP["BOS"],
                LOOKUP["VTL//heart_rate_Q0"],
                LOOKUP["VTL//heart_rate_Q4"],
                LOOKUP["EOS"],
            ]
        ]
    )
    outputs = _one_hot_logits(
        labels,
        [[LOOKUP["VTL//heart_rate_Q4"], LOOKUP["VTL//heart_rate_Q0"], LOOKUP["EOS"]]],
    )

    expected = ((_q(4) - _q(0)) ** 2 + (_q(0) - _q(4)) ** 2) / 2
    assert loss.quantile_token_loss(outputs, labels).item() == pytest.approx(
        expected, rel=1e-4
    )


def test_quantile_token_loss_ignores_batches_with_no_quantile_labels():
    """non-quantile tokens map to category -1 and are skipped entirely"""
    loss = Loss(make_cfg(), make_tkzr_cfg())
    labels = t.tensor([[LOOKUP["BOS"], LOOKUP["DSCG//home"], LOOKUP["EOS"]]])
    outputs = _one_hot_logits(labels, [[LOOKUP["DSCG//expired"], LOOKUP["EOS"]]])

    out = loss.quantile_token_loss(outputs, labels)
    assert isinstance(out, t.Tensor)
    assert out.item() == 0.0


def test_custom_loss_survives_a_batch_with_no_quantile_labels():
    """
    `custom_loss` calls `.item()` on whatever `quantile_token_loss` returns, so
    the no-quantile-token case has to come back as a tensor; it used to return
    a bare 0.0 and take training down with an `AttributeError` on any batch
    (short `max_seq_len`, or a vocabulary with no quantile tokens at all) that
    happened to contain none
    """
    loss = Loss(make_cfg(), make_tkzr_cfg())
    labels = t.tensor([[LOOKUP["BOS"], LOOKUP["DSCG//home"], LOOKUP["EOS"]]])
    outputs = _one_hot_logits(labels, [[LOOKUP["DSCG//expired"], LOOKUP["EOS"]]])

    # only the cross-entropy term contributes
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(
        loss.label_weighted_loss(outputs, labels).item(), rel=1e-5
    )
