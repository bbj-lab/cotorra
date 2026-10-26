#!/usr/bin/env python3

"""tests for cotorra.util"""

import datasets as ds
import numpy as np
import pytest

from cotorra.util import (
    batched,
    batched_iter,
    best_fit_packed_iter,
    bootstrap_aggregate_ci,
    bootstrap_aggregate_pval,
    bootstrap_ci,
    bootstrap_pval,
    pr_auc_score,
)


def test_batched_keeps_the_remainder():
    assert list(batched("ABCDEFG", 3)) == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_iter_drops_the_remainder():
    dset = ds.Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7]]})
    chunks = list(batched_iter(dset, seq_len=3))
    assert chunks == [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]


def test_batched_iter_accumulates_across_examples():
    dset = ds.Dataset.from_dict({"input_ids": [[1, 2], [3, 4], [5, 6]]})
    chunks = list(batched_iter(dset, seq_len=4))
    assert chunks == [{"input_ids": [1, 2, 3, 4]}]


def test_batched_iter_carries_multiple_columns_in_lockstep():
    dset = ds.Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]], "s_elapsed": [[0.0, 1.0, 2.0, 3.0]]}
    )
    chunks = list(batched_iter(dset, seq_len=2))
    assert chunks == [
        {"input_ids": [1, 2], "s_elapsed": [0.0, 1.0]},
        {"input_ids": [3, 4], "s_elapsed": [2.0, 3.0]},
    ]


def test_batched_handles_a_batch_larger_than_the_iterable():
    assert list(batched("AB", 5)) == [("A", "B")]


def test_batched_iter_yields_nothing_when_no_full_chunk_fits():
    dset = ds.Dataset.from_dict({"input_ids": [[1, 2], [3]]})
    assert list(batched_iter(dset, seq_len=4)) == []


# ------------------------------------------------- best-fit decreasing packing

SEQ_LEN = 8


def records(*lengths: int) -> ds.Dataset:
    """records of the given lengths, each token naming its record and offset
    (`100 * record + offset`), with a second column in step with the first"""
    ids = [[100 * i + j for j in range(n)] for i, n in enumerate(lengths)]
    return ds.Dataset.from_dict(
        {"input_ids": ids, "negated": [[-x for x in r] for r in ids]}
    )


def chunks(*lengths: int) -> list[list[int]]:
    """what packing should place whole: each record cut every `SEQ_LEN` tokens"""
    ids = [[100 * i + j for j in range(n)] for i, n in enumerate(lengths)]
    return sorted(r[s : s + SEQ_LEN] for r in ids for s in range(0, len(r), SEQ_LEN))


def packed(dset: ds.Dataset, seed: int = 0) -> list[dict]:
    return list(best_fit_packed_iter(dset, SEQ_LEN, np.random.default_rng(seed)))


def pieces(row: dict) -> list[list[int]]:
    """a packed row's tokens, split where `doc_pos` restarts, padding dropped"""
    out = []
    for token, pos, pad in zip(row["input_ids"], row["doc_pos"], row["pad_mask"]):
        if pad:
            break
        if pos == 0:
            out.append([])
        out[-1].append(token)
    return out


def best_fit_decreasing(lengths: list[int]) -> list[int]:
    """how full a plain best-fit decreasing leaves each row: every chunk, longest
    first, into the open row with the least room that still fits it"""
    sizes = sorted((len(c) for c in chunks(*lengths)), reverse=True)
    room = []
    for n in sizes:
        if fits := [i for i, r in enumerate(room) if r >= n]:
            room[min(fits, key=room.__getitem__)] -= n
        else:
            room.append(SEQ_LEN - n)
    return sorted(SEQ_LEN - r for r in room)


LENGTHS = (3, 5, 8, 19, 2, 1, 6)


def test_best_fit_packing_places_every_chunk_whole_and_once():
    """a record longer than a row is cut into chunks of at most a row, a
    shorter one kept whole, and each lands in one row: every token exactly once,
    nothing dropped, as concatenation drops its last partial row"""
    assert sorted(p for row in packed(records(*LENGTHS)) for p in pieces(row)) == (
        chunks(*LENGTHS)
    )


def test_each_packed_row_is_padded_out_and_its_padding_flagged():
    rows = packed(records(*LENGTHS))
    assert sum(r["pad_mask"].count(1) for r in rows) > 0
    for row in rows:
        assert set(row) == {"input_ids", "negated", "pad_mask", "doc_pos"}
        assert all(len(v) == SEQ_LEN for v in row.values())
        n = row["pad_mask"].count(0)
        assert row["pad_mask"] == [0] * n + [1] * (SEQ_LEN - n)
        assert row["input_ids"][n:] == [0] * (SEQ_LEN - n)
        # a segment of its own, its positions restarting at 0
        assert row["doc_pos"][n:] == list(range(SEQ_LEN - n))


def test_each_pieces_positions_restart_at_zero_a_split_records_too():
    """`doc_pos` counts from 0 in every piece, the continuation of a record cut
    across rows included, whose tokens sit at offsets 8, 16 in their record"""
    continued = 0
    for row in packed(records(*LENGTHS)):
        start = 0
        for piece in pieces(row):
            assert row["doc_pos"][start : start + len(piece)] == list(range(len(piece)))
            continued += piece[0] % 100 > 0
            start += len(piece)
    assert continued == 2  # record 3's two continuations


def test_every_column_is_packed_in_step_with_the_tokens():
    for row in packed(records(*LENGTHS)):
        assert row["negated"] == [-x for x in row["input_ids"]]


@pytest.mark.parametrize("seed", range(5))
def test_rows_fill_as_a_plain_best_fit_decreasing_fills_them(seed):
    lengths = np.random.default_rng(seed).integers(1, 3 * SEQ_LEN, size=40).tolist()
    rows = packed(records(*lengths))
    assert sorted(r["pad_mask"].count(0) for r in rows) == best_fit_decreasing(lengths)


def test_a_shuffled_dataset_packs_each_record_at_its_own_length():
    """the loader hands over its split repeated an epoch a copy, and shuffled;
    each record's length has to be read through the shuffle's indices, or its
    tokens would be cut to another record's length"""
    dset = records(*LENGTHS).repeat(2).shuffle(seed=0)
    assert sorted(p for row in packed(dset) for p in pieces(row)) == sorted(
        2 * chunks(*LENGTHS)
    )


def from_records(row: dict) -> list[int]:
    """the record each of a packed row's pieces comes from"""
    return [piece[0] // 100 for piece in pieces(row)]


def test_each_epoch_is_packed_apart_so_no_row_holds_a_record_twice():
    """packed in one pass, a record's copies, all of one length, sit side by
    side among the chunks sorted by length, and best fit puts them in the same
    rows, where a later copy could attend to an earlier one and copy it"""
    lengths = range(1, SEQ_LEN)
    together = packed(records(*lengths).repeat(3).shuffle(seed=0))
    assert any(len(set(r := from_records(row))) < len(r) for row in together)
    rng = np.random.default_rng(0)
    apart = list(best_fit_packed_iter(records(*lengths), SEQ_LEN, rng, n_epochs=3))
    assert all(len(set(r := from_records(row))) == len(r) for row in apart)
    assert sorted(p for row in apart for p in pieces(row)) == sorted(
        3 * chunks(*lengths)
    )


def test_packing_is_reproducible_by_its_seed_and_shuffled_by_it():
    dset = records(*range(1, 20))
    assert packed(dset, seed=1) == packed(dset, seed=1)
    assert packed(dset, seed=1) != packed(dset, seed=2)


def test_an_empty_record_takes_no_room():
    assert [pieces(row) for row in packed(records(0, 3, 0))] == [[[100, 101, 102]]]


def test_pr_auc_score_is_near_perfect_for_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
    assert pr_auc_score(y_true, y_score) == pytest.approx(1.0, abs=1e-6)


def test_pr_auc_score_is_low_for_inverted_scores():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.99, 0.95, 0.9, 0.1, 0.05, 0.01])
    assert pr_auc_score(y_true, y_score) < 0.5


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


def test_bootstrap_ci_returns_ordered_intervals_for_requested_metrics(rng):
    n = 200
    y_true = (rng.random(n) > 0.5).astype(int)
    y_score = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)

    out = bootstrap_ci(
        y_true, y_score, n_samples=200, rng=rng, n_jobs=1, metrics=("roc_auc", "brier")
    )
    assert set(out.keys()) == {"roc_auc", "brier"}
    for k, (lo, hi) in out.items():
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0


def test_bootstrap_ci_stratified_matches_unstratified_keys(rng):
    n = 300
    y_true = (rng.random(n) > 0.95).astype(int)  # rare positive class
    y_score = rng.random(n)

    out = bootstrap_ci(
        y_true,
        y_score,
        n_samples=100,
        rng=rng,
        n_jobs=1,
        metrics=("roc_auc",),
        stratified=True,
    )
    assert list(out.keys()) == ["roc_auc"]
    assert out["roc_auc"][0] <= out["roc_auc"][1]


def test_bootstrap_aggregate_ci_allows_different_length_labels(rng):
    y_trues = [(rng.random(50) > 0.5).astype(int), (rng.random(80) > 0.5).astype(int)]
    y_scores = [np.clip(yt + rng.normal(0, 0.3, size=len(yt)), 0, 1) for yt in y_trues]

    out = bootstrap_aggregate_ci(
        y_trues, y_scores, n_samples=100, rng=rng, n_jobs=1, metrics=("avg_roc_auc",)
    )
    assert list(out.keys()) == ["avg_roc_auc"]
    lo, hi = out["avg_roc_auc"]
    assert lo <= hi


def test_bootstrap_pval_two_sided_is_large_for_identical_scores(rng):
    n = 200
    y_true = (rng.random(n) > 0.5).astype(int)
    y_score = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)

    out = bootstrap_pval(
        y_true,
        y_score,
        y_score,
        n_samples=200,
        rng=rng,
        n_jobs=1,
        metrics=("roc_auc", "pr_auc", "brier"),
    )
    assert set(out.keys()) == {"roc_auc", "pr_auc", "brier"}
    for p in out.values():
        assert 0.0 <= p <= 1.0
        assert p > 0.5


def test_bootstrap_pval_one_sided_prefers_the_better_scores(rng):
    n = 300
    y_true = (rng.random(n) > 0.5).astype(int)
    y_score_good = np.clip(y_true + rng.normal(0, 0.1, size=n), 0, 1)
    y_score_bad = rng.random(n)

    out = bootstrap_pval(
        y_true,
        y_score_bad,
        y_score_good,
        n_samples=200,
        rng=rng,
        n_jobs=1,
        alternative="one-sided",
        metrics=("roc_auc",),
        paired=True,
    )
    assert out["roc_auc"] < 0.5


def test_bootstrap_aggregate_pval_two_sided_is_large_for_identical_scores(rng):
    y_trues = [(rng.random(60) > 0.5).astype(int), (rng.random(40) > 0.5).astype(int)]
    y_scores = [np.clip(yt + rng.normal(0, 0.3, size=len(yt)), 0, 1) for yt in y_trues]

    out = bootstrap_aggregate_pval(
        y_trues,
        y_scores,
        y_scores,
        n_samples=100,
        rng=rng,
        n_jobs=1,
        metrics=("avg_roc_auc", "avg_brier"),
    )
    for p in out.values():
        assert p > 0.5


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("alternative", ["one-sided", "two-sided"])
def test_every_p_value_is_a_python_float(rng, alternative, paired):
    """`bootstrap_pval`'s two-sided branch called `.item()` on the observed
    difference inside its mean rather than on the mean, and so returned
    np.float64 where every other branch returns the float both promise"""
    n = 60
    y_true = (rng.random(n) > 0.5).astype(int)
    y_score0 = rng.random(n)
    y_score1 = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)
    common = dict(
        n_samples=50, rng=rng, n_jobs=1, alternative=alternative, paired=paired
    )
    single = bootstrap_pval(
        y_true, y_score0, y_score1, metrics=("roc_auc", "brier"), **common
    )
    aggregate = bootstrap_aggregate_pval(
        [y_true], [y_score0], [y_score1], metrics=("avg_roc_auc",), **common
    )
    for p in (*single.values(), *aggregate.values()):
        assert type(p) is float


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    y_true = (np.random.default_rng(1).random(120) > 0.5).astype(int)
    y_score = np.clip(y_true + np.random.default_rng(2).normal(0, 0.3, 120), 0, 1)
    kwargs = dict(n_samples=100, n_jobs=1, metrics=("roc_auc",))

    first = bootstrap_ci(y_true, y_score, rng=np.random.default_rng(7), **kwargs)
    second = bootstrap_ci(y_true, y_score, rng=np.random.default_rng(7), **kwargs)
    np.testing.assert_array_equal(first["roc_auc"], second["roc_auc"])


def test_bootstrap_ci_brackets_the_point_estimate(rng):
    from sklearn.metrics import roc_auc_score

    n = 300
    y_true = (rng.random(n) > 0.5).astype(int)
    y_score = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)

    lo, hi = bootstrap_ci(
        y_true, y_score, n_samples=400, rng=rng, n_jobs=1, metrics=("roc_auc",)
    )["roc_auc"]
    assert lo <= roc_auc_score(y_true, y_score) <= hi


def _labels_and_scores(seed: int = 0):
    """a label, two scores of it -- one useful, one not -- and a second, shorter
    label scored usefully, for the aggregate functions"""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(80) > 0.5).astype(int)
    useful = np.clip(y_true + rng.normal(0, 0.4, size=80), 0, 1)
    other = (rng.random(50) > 0.4).astype(int)
    other_score = np.clip(other + rng.normal(0, 0.4, size=50), 0, 1)
    return y_true, rng.random(80), useful, other, other_score


BOOTSTRAPS = {
    "bootstrap_ci": (
        bootstrap_ci,
        ("roc_auc", "pr_auc", "brier"),
        lambda y, noise, useful, o, o_score: (y, useful),
    ),
    "bootstrap_aggregate_ci": (
        bootstrap_aggregate_ci,
        ("avg_roc_auc", "avg_pr_auc", "avg_brier"),
        lambda y, noise, useful, o, o_score: ([y, o], [useful, o_score]),
    ),
    "bootstrap_pval": (
        bootstrap_pval,
        ("roc_auc", "pr_auc", "brier"),
        lambda y, noise, useful, o, o_score: (y, noise, useful),
    ),
    "bootstrap_aggregate_pval": (
        bootstrap_aggregate_pval,
        ("avg_roc_auc", "avg_pr_auc", "avg_brier"),
        lambda y, noise, useful, o, o_score: ([y, o], [noise, o], [useful, o_score]),
    ),
}


@pytest.mark.parametrize("name", list(BOOTSTRAPS))
def test_each_metric_comes_out_alike_alone_and_among_the_others(name):
    """
    every function computes each of its metrics, by default all of them, and
    the resamples don't depend on which are asked for, so a metric computed
    alone matches its value computed alongside the rest
    """
    fn, metrics, args = BOOTSTRAPS[name]
    data = args(*_labels_and_scores())
    common = dict(n_samples=40, n_jobs=1)
    together = fn(*data, rng=np.random.default_rng(3), **common)
    assert list(together) == list(metrics)
    for metric in metrics:
        alone = fn(*data, rng=np.random.default_rng(3), metrics=(metric,), **common)
        assert list(alone) == [metric]
        np.testing.assert_array_equal(alone[metric], together[metric])
