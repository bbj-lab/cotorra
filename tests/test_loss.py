#!/usr/bin/env python3

"""tests for cotorra.loss.Loss"""

import pytest
import torch as t
from omegaconf import OmegaConf

from cotorra.loss import Loss
from cotorra.model import HEADS

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
    base = {"quantile_token_loss": {"qt_weight": 0.5}}
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


def test_quantile_categories_are_discovered():
    loss = Loss(make_cfg(), make_tkzr_cfg())
    assert loss.n_cats == 2
    for word in (
        "VTL//heart_rate_Q0",
        "VTL//heart_rate_Q4",
        "LAB//sodium_Q0",
        "LAB//sodium_Q4",
    ):
        assert loss.label_to_cat[LOOKUP[word]] >= 0
    for word in ("UNK", "BOS", "EOS", "DSCG//expired", "DSCG//home"):
        assert loss.label_to_cat[LOOKUP[word]] == -1
    # each category's row of the lookup table lists its bins in order
    hr = loss.label_to_cat[LOOKUP["VTL//heart_rate_Q0"]]
    assert loss.qt_table[hr].tolist() == [
        LOOKUP[f"VTL//heart_rate_Q{i}"] for i in range(N_BINS)
    ]


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


def test_custom_loss_is_cross_entropy_alone_without_other_blocks(outputs, labels):
    loss = Loss(OmegaConf.create({}), make_tkzr_cfg())
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(
        loss.x_ent_loss(outputs, labels).item(), rel=1e-5
    )


def test_custom_loss_combines_cross_entropy_and_quantile_blocks(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    expected = (
        loss.x_ent_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
    )
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(expected, rel=1e-5)


# what each secondary head's term comes back from the model as
HEAD_TERMS = {
    "tte_loss": t.tensor(4.0),
    "tnt_loss": t.tensor(8.0),
    "disposition_loss": t.tensor(16.0),
}


def base_terms(loss, outputs, labels) -> float:
    """the next-token objective `make_cfg` configures, without any head's term"""
    return (
        loss.x_ent_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
    )


@pytest.mark.parametrize("name", list(HEADS))
def test_custom_loss_adds_a_heads_weighted_term(outputs, labels, name):
    """the model hands each head's term back on the output object; the
    objective picks up only the ones the training config asks for, each
    weighted by its own block"""
    loss = Loss(make_cfg(**{f"{name}_objective": {"weight": 0.25}}), make_tkzr_cfg())
    out = dict(outputs) | HEAD_TERMS
    expected = base_terms(loss, out, labels) + 0.25 * HEAD_TERMS[f"{name}_loss"].item()
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_adds_every_configured_heads_term(outputs, labels):
    weights = {"tte": 0.25, "tnt": 0.125, "disposition": 0.5}
    cfg = make_cfg(**{f"{n}_objective": {"weight": w} for n, w in weights.items()})
    loss = Loss(cfg, make_tkzr_cfg())
    out = dict(outputs) | HEAD_TERMS
    expected = base_terms(loss, out, labels) + sum(
        w * HEAD_TERMS[f"{n}_loss"].item() for n, w in weights.items()
    )
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


def test_custom_loss_ignores_head_terms_their_blocks_do_not_ask_for(outputs, labels):
    """a model may carry a head without the objective asking to train it"""
    loss = Loss(make_cfg(), make_tkzr_cfg())
    assert loss.custom_loss(dict(outputs) | HEAD_TERMS, labels).item() == (
        pytest.approx(loss.custom_loss(outputs, labels).item(), rel=1e-5)
    )


@pytest.mark.parametrize("name", list(HEADS))
@pytest.mark.parametrize("block", [{}, None], ids=["empty", "null"])
def test_custom_loss_defaults_each_weight_to_one(outputs, labels, name, block):
    """matching the heads' own default, so a block left empty -- which yaml
    parses to None -- still trains"""
    loss = Loss(make_cfg(**{f"{name}_objective": block}), make_tkzr_cfg())
    out = dict(outputs) | HEAD_TERMS
    expected = base_terms(loss, out, labels) + HEAD_TERMS[f"{name}_loss"].item()
    assert loss.custom_loss(out, labels).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("name", list(HEADS))
def test_custom_loss_refuses_a_model_that_returns_no_head_term(outputs, labels, name):
    """configuring an objective against a model without its head used to train
    silently without it -- the extra batch column is swallowed by the
    backbone's `**kwargs` and no term comes back -- so say so instead"""
    loss = Loss(make_cfg(**{f"{name}_objective": {}}), make_tkzr_cfg())
    others = {k: v for k, v in HEAD_TERMS.items() if k != f"{name}_loss"}
    with pytest.raises(ValueError, match=f"`{name}_loss`"):
        loss.custom_loss(dict(outputs) | others, labels)


@pytest.mark.parametrize("name", list(HEADS))
def test_custom_loss_keeps_each_head_term_differentiable(outputs, labels, name):
    """each head's only path to a gradient on this code path"""
    loss = Loss(make_cfg(**{f"{name}_objective": {"weight": 0.25}}), make_tkzr_cfg())
    term = t.tensor(4.0, requires_grad=True)
    loss.custom_loss(dict(outputs) | {f"{name}_loss": term}, labels).backward()
    assert term.grad is not None and term.grad.item() == pytest.approx(0.25)


def test_an_objective_naming_no_head_is_refused():
    with pytest.raises(ValueError, match="mpp_objective"):
        Loss(make_cfg(mpp_objective={}), make_tkzr_cfg())


# ------------------------------------------- label_weighted_loss (deprecated)


def weighted_cfg(**overrides):
    """the deprecated block, as configs written before `balanced_toi_loss` have it"""
    return make_cfg(
        label_weighted_loss={
            "tokens_of_interest": ["DSCG//expired", "LABEL//*"],
            "toi_weight": 20.0,
        },
        **overrides,
    )


def test_grokked_outcome_tokens_matches_exact_and_glob_patterns():
    loss = Loss(weighted_cfg(), make_tkzr_cfg())
    assert loss.grokked_outcome_tokens == ["DSCG//expired"]


def test_label_weights_flag_only_outcome_tokens():
    loss = Loss(weighted_cfg(), make_tkzr_cfg())
    idx = LOOKUP["DSCG//expired"]
    assert loss.weights[idx].item() == pytest.approx(20.0)
    for word, i in LOOKUP.items():
        if word != "DSCG//expired":
            assert loss.weights[i].item() == pytest.approx(1.0)


def test_label_weighted_loss_matches_manually_weighted_cross_entropy(outputs, labels):
    loss = Loss(weighted_cfg(), make_tkzr_cfg())
    out = loss.label_weighted_loss(outputs, labels)

    shift_logits = outputs["logits"][:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected = t.nn.CrossEntropyLoss(weight=loss.weights.to(t.float32))(
        shift_logits.reshape(-1, VOCAB_SIZE), shift_labels.reshape(-1)
    )
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


def test_custom_loss_uses_label_weighting_in_place_of_plain_cross_entropy(
    outputs, labels
):
    """as it did before the block was deprecated"""
    loss = Loss(weighted_cfg(), make_tkzr_cfg())
    expected = (
        loss.label_weighted_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
    )
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(expected, rel=1e-5)


# ------------------------------------------------------- balanced_toi_loss

DSCG = [LOOKUP["DSCG//expired"], LOOKUP["DSCG//home"]]
# one count per vocabulary token: a batch with labels in exactly these
# proportions makes its mean loss the expected loss under them
COUNTS = [6, 5, 4, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2]


def balanced_cfg(**block):
    """`balanced_toi_loss` on the discharge tokens"""
    return make_cfg(
        balanced_toi_loss={"tokens_of_interest": ["DSCG//*"], "bce_weight": 0.5} | block
    )


def test_tokens_of_interest_resolve_exact_names_and_glob_patterns():
    """`LABEL//*` matches nothing in this vocabulary and is no error so long as
    some pattern matches"""
    cfg = make_cfg(
        balanced_toi_loss={"tokens_of_interest": ["DSCG//expired", "LABEL//*"]}
    )
    flag = Loss(cfg, make_tkzr_cfg()).balanced_toi_flag
    assert flag.nonzero().flatten().tolist() == [LOOKUP["DSCG//expired"]]


def expected_loss_gradient(loss_fn) -> t.Tensor:
    """the gradient of `loss_fn`'s expected value under the label distribution
    `COUNTS` gives, taken where the logits are that distribution's log"""
    q = t.tensor(COUNTS, dtype=t.float32) / sum(COUNTS)
    labels = t.tensor([[1] + [y for y, n in enumerate(COUNTS) for _ in range(n)]])
    log_q = q.log().requires_grad_()
    loss_fn({"logits": log_q.expand(1, labels.shape[1], -1)}, labels).backward()
    return log_q.grad


def test_balanced_toi_loss_is_the_bce_of_the_probability_on_the_tokens_of_interest():
    g = t.Generator().manual_seed(0)
    logits = t.randn(2, 5, VOCAB_SIZE, generator=g)
    labels = t.tensor([[1, 3, 7, 4, 2], [1, 10, 3, 3, 2]])
    p_toi = logits[:, :-1].softmax(-1)[..., DSCG].sum(-1)
    is_toi = t.isin(labels[:, 1:], t.tensor(DSCG)).float()
    expected = t.nn.functional.binary_cross_entropy(p_toi, is_toi)

    loss = Loss(balanced_cfg(), make_tkzr_cfg())
    assert loss.balanced_toi_loss({"logits": logits}, labels).item() == pytest.approx(
        expected.item(), rel=1e-5
    )


def test_cross_entropy_plus_balanced_toi_loss_is_minimized_by_the_true_distribution():
    """what makes the term safe to add: at the true probabilities the expected
    loss is flat, so training has no reason to make any token likelier"""
    loss = Loss(balanced_cfg(), make_tkzr_cfg())
    grad = expected_loss_gradient(
        lambda o, y: loss.x_ent_loss(o, y) + 0.5 * loss.balanced_toi_loss(o, y)
    )
    assert grad.abs().max().item() < 1e-6


def test_label_weighting_is_not_minimized_there():
    """why `label_weighted_loss` is deprecated: at the true probabilities, the
    weighted loss still pushes the weighted token's logit up"""
    loss = Loss(weighted_cfg(), make_tkzr_cfg())
    grad = expected_loss_gradient(loss.label_weighted_loss)
    assert grad[LOOKUP["DSCG//expired"]].item() < -1e-3


def test_custom_loss_adds_the_weighted_balanced_toi_term(outputs, labels):
    loss = Loss(balanced_cfg(), make_tkzr_cfg())
    expected = (
        loss.x_ent_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
        + 0.5 * loss.balanced_toi_loss(outputs, labels).item()
    )
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(expected, rel=1e-5)


def test_the_bce_weight_defaults_to_one(outputs, labels):
    cfg = balanced_cfg()
    del cfg.balanced_toi_loss["bce_weight"]
    loss = Loss(cfg, make_tkzr_cfg())
    expected = (
        loss.x_ent_loss(outputs, labels).item()
        + 0.5 * loss.quantile_token_loss(outputs, labels).item()
        + loss.balanced_toi_loss(outputs, labels).item()
    )
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "block",
    [{"tokens_of_interest": ["NOPE//*"]}, {"tokens_of_interest": ["*"]}, None],
    ids=["matches-nothing", "matches-everything", "empty-block"],
)
def test_a_set_matching_none_or_all_of_the_vocabulary_is_refused(block):
    """either way the event is certain, and the term has nothing to learn"""
    with pytest.raises(ValueError, match="balanced_toi_loss.tokens_of_interest"):
        Loss(make_cfg(balanced_toi_loss=block), make_tkzr_cfg())


def test_balanced_toi_loss_stays_finite_however_sure_the_model_is():
    """sure of a discharge where none comes, and sure there's none where one
    does: large, but finite, with finite gradients"""
    logits = t.zeros(1, 3, VOCAB_SIZE)
    logits[0, 0, DSCG] = 1e4
    logits[0, 1, 7] = 1e4
    logits.requires_grad_()
    loss = Loss(balanced_cfg(), make_tkzr_cfg()).balanced_toi_loss(
        {"logits": logits}, t.tensor([[1, 7, 3]])
    )
    loss.backward()
    assert t.isfinite(loss) and loss.item() > 1e3
    assert t.isfinite(logits.grad).all()


def test_custom_loss_records_each_term_unweighted_and_detached(outputs, labels):
    """what `TrainerWithCustomLoss` averages for its logs: every term the
    objective sums, under the name it gets logged by, off the graph"""
    loss = Loss(make_cfg(tte_objective={"weight": 0.25}), make_tkzr_cfg())
    logits = outputs["logits"].clone().requires_grad_()
    out = {"logits": logits} | HEAD_TERMS
    terms = dict()
    total = loss.custom_loss(out, labels, terms=terms)
    assert set(terms) == {"x_ent_loss", "quantile_token_loss", "tte_loss"}
    assert not any(term.requires_grad for term in terms.values())
    assert terms["x_ent_loss"].item() == pytest.approx(
        loss.x_ent_loss(out, labels).item(), rel=1e-5
    )
    assert terms["tte_loss"].item() == HEAD_TERMS["tte_loss"].item()
    assert total.item() == pytest.approx(
        terms["x_ent_loss"].item()
        + 0.5 * terms["quantile_token_loss"].item()
        + 0.25 * terms["tte_loss"].item(),
        rel=1e-5,
    )


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
    `custom_loss` calls `.detach()` on whatever `quantile_token_loss` returns, so
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
        loss.x_ent_loss(outputs, labels).item(), rel=1e-5
    )


def _quantile_reference(loss, outputs, labels):
    """the loss written out position by position"""
    logits, errs = outputs["logits"], []
    for row in range(labels.shape[0]):
        for pos in range(labels.shape[1] - 1):
            label = labels[row, pos + 1].item()
            if (c := loss.label_to_cat[label].item()) < 0:
                continue
            ids = [i for i in loss.qt_table[c].tolist() if i >= 0]
            vals = [loss.label_to_q[i].item() for i in ids]
            p = t.softmax(logits[row, pos, ids], dim=-1)
            errs.append(((p * t.tensor(vals)).sum() - loss.label_to_q[label]) ** 2)
    return t.stack(errs).mean().item()


def test_quantile_token_loss_matches_a_position_by_position_reference(outputs, labels):
    loss = Loss(make_cfg(), make_tkzr_cfg())
    assert loss.quantile_token_loss(outputs, labels).item() == pytest.approx(
        _quantile_reference(loss, outputs, labels), rel=1e-5
    )


def test_quantile_token_loss_averages_over_tokens_not_categories():
    """
    three heart-rate tokens named right and one sodium token named at the far
    end: each token counts once, so the sodium token's error is diluted by the
    heart-rate tokens rather than standing as a category on its own
    """
    loss = Loss(make_cfg(), make_tkzr_cfg())
    hr0, na0 = LOOKUP["VTL//heart_rate_Q0"], LOOKUP["LAB//sodium_Q0"]
    labels = t.tensor([[LOOKUP["BOS"], hr0, hr0, hr0, na0, LOOKUP["EOS"]]])
    outputs = _one_hot_logits(
        labels, [[hr0, hr0, hr0, LOOKUP["LAB//sodium_Q4"], LOOKUP["EOS"]]]
    )

    assert loss.quantile_token_loss(outputs, labels).item() == pytest.approx(
        (_q(4) - _q(0)) ** 2 / 4, rel=1e-4
    )


def test_quantile_token_loss_handles_unfused_bins():
    """
    unfused, cocoa emits a bare `Q<i>` after the code; those bins form one
    category, and mass the model puts outside them doesn't count
    """
    lookup = {
        "UNK": 0,
        "BOS": 1,
        "EOS": 2,
        "LAB//sodium": 3,
        "VTL//heart_rate": 4,
        **{f"Q{i}": 5 + i for i in range(N_BINS)},
    }
    tkzr_cfg = OmegaConf.create({"lookup": lookup, "cfg": {"n_bins": N_BINS}})
    loss = Loss(make_cfg(), tkzr_cfg)
    assert loss.n_cats == 1

    labels = t.tensor([[1, 3, 5 + 1, 4, 5 + 3, 2]])
    logits = t.zeros(1, labels.shape[1], len(lookup))
    logits[..., lookup["LAB//sodium"]] = 50.0  # outside the bins: ignored
    # uniform over the bins, so the model expects the middle one, 0.5
    expected = ((0.5 - _q(1)) ** 2 + (0.5 - _q(3)) ** 2) / 2
    assert loss.quantile_token_loss({"logits": logits}, labels).item() == (
        pytest.approx(expected, rel=1e-5)
    )


def test_quantile_token_loss_skips_bins_missing_from_the_vocabulary():
    """
    tied breaks skip bins, so a code can lack some; the softmax spans only the
    bins it has, and the masked-out ones leave the gradient finite
    """
    lookup = {
        "BOS": 0,
        "EOS": 1,
        "LAB//sodium_Q0": 2,
        "LAB//sodium_Q2": 3,
        "LAB//sodium_Q4": 4,
    }
    tkzr_cfg = OmegaConf.create({"lookup": lookup, "cfg": {"n_bins": N_BINS}})
    loss = Loss(make_cfg(), tkzr_cfg)
    assert loss.qt_table.tolist() == [[2, -1, 3, -1, 4]]

    labels = t.tensor([[0, 4, 1]])
    logits = t.zeros(1, 3, len(lookup), requires_grad=True)
    out = loss.quantile_token_loss({"logits": logits}, labels)
    # uniform over Q0, Q2, Q4 expects (0.1 + 0.5 + 0.9) / 3 = 0.5
    assert out.item() == pytest.approx((0.5 - _q(4)) ** 2, rel=1e-5)
    out.backward()
    assert t.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_quantile_token_loss_with_no_quantile_tokens_in_the_vocabulary():
    lookup = {"UNK": 0, "BOS": 1, "EOS": 2, "DSCG//home": 3}
    tkzr_cfg = OmegaConf.create({"lookup": lookup, "cfg": {"n_bins": N_BINS}})
    loss = Loss(make_cfg(), tkzr_cfg)
    assert loss.n_cats == 0

    labels = t.tensor([[1, 3, 2]])
    outputs = {"logits": t.randn(1, 3, len(lookup))}
    out = loss.quantile_token_loss(outputs, labels)
    assert isinstance(out, t.Tensor) and out.item() == 0.0
    assert loss.custom_loss(outputs, labels).item() == pytest.approx(
        loss.x_ent_loss(outputs, labels).item(), rel=1e-5
    )
