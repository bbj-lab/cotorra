#!/usr/bin/env python3

"""
utility functions
"""

import collections
import itertools
import typing
import warnings

import datasets as ds
import joblib as jl
import numpy as np
from sklearn import metrics as skl_mets

Generator: typing.TypeAlias = np.random._generator.Generator


def batched(iterable, n) -> collections.abc.Iterator:
    """
    `itertools.batched` introduced in Python 3.12
    cf. https://docs.python.org/3/library/itertools.html#itertools.batched
    batched('ABCDEFG', 3) → ABC DEF G
    """
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def batched_iter(dset: ds.Dataset, seq_len: int) -> collections.abc.Iterator:
    """
    batched iteration on a huggingface dataset;
    as opposed to `batched`, the remainder here is dropped
    """
    dq = {k: collections.deque() for k in dset.column_names}
    for eg in iter(dset):
        for k in dq:
            dq[k].extend(list(eg[k]))
        while len(dq[list(dq.keys())[0]]) >= seq_len:
            yield {k: [dq[k].popleft() for _ in range(seq_len)] for k in dq}


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray):
    precs, recs, *_ = skl_mets.precision_recall_curve(
        y_true, np.round(y_score, decimals=4), drop_intermediate=True
    )
    return skl_mets.auc(recs, precs)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_samples: int = 10_000,
    alpha: float = 0.05,
    rng: Generator = np.random.default_rng(seed=42),
    metrics: collections.abc.Sequence[typing.Literal["roc_auc", "pr_auc", "brier"]] = (
        "roc_auc",
        "pr_auc",
        "brier",
    ),
    n_jobs: int = -1,
    stratified: bool = False,
) -> dict[str, np.ndarray]:
    """
    Calculates a bootstrapped percentile interval for objectives `objs` as
    described in §13.3 of Efron & Tibshirani's "An Introduction to the Bootstrap"
    (Chapman & Hall, Boca Raton, 1993), ignoring variance due to model-fitting
    (i.e. this measures variability in the test-set alone);
    """

    def get_scores_i(rng_i: Generator) -> dict[str, float]:
        warnings.filterwarnings("ignore")
        if stratified:
            pos = np.flatnonzero(y_true.astype(int))
            neg = np.flatnonzero(1 - (y_true).astype(int))
            samp_pos = rng_i.choice(pos, size=len(pos), replace=True)
            samp_neg = rng_i.choice(neg, size=len(neg), replace=True)
            samp_i = rng_i.permutation(np.concatenate([samp_pos, samp_neg]))
        else:
            samp_i = rng_i.choice(len(y_true), size=len(y_true), replace=True)
        yti, ysi = y_true[samp_i], y_score[samp_i]
        ret = dict()
        if "roc_auc" in metrics:
            ret["roc_auc"] = skl_mets.roc_auc_score(yti, ysi)
        if "pr_auc" in metrics:
            ret["pr_auc"] = pr_auc_score(yti, ysi)
        if "brier" in metrics:
            ret["brier"] = skl_mets.brier_score_loss(yti, ysi)
        return ret

    with jl.Parallel(n_jobs=n_jobs) as par:
        scores = par(jl.delayed(get_scores_i)(rng_i) for rng_i in rng.spawn(n_samples))

    return {
        m: np.nanquantile([s[m] for s in scores], q=[alpha / 2, 1 - (alpha / 2)])
        for m in metrics
    }


def bootstrap_aggregate_ci(
    y_trues: tuple[np.ndarray],
    y_scores: tuple[np.ndarray],
    *,
    n_samples: int = 10_000,
    alpha: float = 0.05,
    rng: Generator = np.random.default_rng(seed=42),
    metrics: collections.abc.Sequence[
        typing.Literal["avg_roc_auc", "avg_pr_auc", "avg_brier"]
    ] = ("avg_roc_auc", "avg_pr_auc", "avg_brier"),
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    """
    Like `bootstrap_ci` but for the average performance over multiple labels;
    allow pairs of (y_true, y_score) to have different lengths
    """

    def get_scores_i(rng_i: Generator) -> dict[str, float]:
        warnings.filterwarnings("ignore")
        resamples = [
            (yt[samp], ys[samp])
            for yt, ys in zip(y_trues, y_scores)
            if (samp := rng_i.choice(len(yt), size=len(yt), replace=True)) is not None
        ]
        ret = dict()
        if "avg_roc_auc" in metrics:
            ret["avg_roc_auc"] = np.mean(
                [skl_mets.roc_auc_score(yt, ys) for yt, ys in resamples]
            )
        if "avg_pr_auc" in metrics:
            ret["avg_pr_auc"] = np.mean([pr_auc_score(yt, ys) for yt, ys in resamples])
        if "avg_brier" in metrics:
            ret["avg_brier"] = np.mean(
                [skl_mets.brier_score_loss(yt, ys) for yt, ys in resamples]
            )
        return ret

    with jl.Parallel(n_jobs=n_jobs) as par:
        scores = par(jl.delayed(get_scores_i)(rng_i) for rng_i in rng.spawn(n_samples))

    return {
        m: np.nanquantile([s[m] for s in scores], q=[alpha / 2, 1 - (alpha / 2)])
        for m in metrics
    }
