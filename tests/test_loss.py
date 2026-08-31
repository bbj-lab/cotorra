#!/usr/bin/env python3

"""tests for cotorra.loss.Loss"""

import pytest
import torch as t
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

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


def test_construction_requires_label_weighted_loss_block(outputs, labels):
    """
    `Loss.__init__` unconditionally reads
    `cfg.label_weighted_loss.tokens_of_interest` to compute
    `grokked_outcome_tokens`, before the `"label_weighted_loss" in self.cfg`
    presence check that gates everything else -- so despite the docs'
    "toggled purely by presence of the block" convention, the block cannot
    actually be omitted even if a user only wants `quantile_token_loss` (or no
    custom loss component at all beyond plain cross entropy)
    """
    cfg = make_cfg()
    del cfg["label_weighted_loss"]
    with pytest.raises(ConfigAttributeError):
        Loss(cfg, make_tkzr_cfg())


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


def test_custom_loss_does_not_touch_wandb_when_no_run_is_active(outputs, labels):
    import wandb

    assert wandb.run is None
    loss = Loss(make_cfg(), make_tkzr_cfg())
    loss.custom_loss(outputs, labels)  # must not raise even though wandb is inactive
    assert wandb.run is None
