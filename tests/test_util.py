#!/usr/bin/env python3

"""tests for cotorra.util"""

import datasets as ds
import numpy as np
import pytest

from cotorra.util import (
    batched,
    batched_iter,
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
