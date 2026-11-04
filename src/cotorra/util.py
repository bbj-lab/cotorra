#!/usr/bin/env python3

"""
utility functions
"""

import collections.abc
import itertools
import typing
import warnings

import datasets as ds
import joblib as jl
import numpy as np
from sklearn import metrics as skl_mets

Generator: typing.TypeAlias = np.random._generator.Generator


def batched(iterable: collections.abc.Iterable, n: int) -> collections.abc.Iterator:
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


def best_fit_packed_iter(
    dset: ds.Dataset, seq_len: int, rng: Generator, n_epochs: int = 1
) -> collections.abc.Iterator:
    """
    Best-fit Decreasing packing (Ding et al., ICML 2024, "Fewer Truncations
    Improve Language Modeling"): a record is split only when it is longer
    than `seq_len`, into chunks of at most `seq_len`; every chunk is then
    placed whole into the open sequence whose remaining space is the smallest
    that still fits it, largest chunks first. Leftover space is padded with
    zeros and flagged in `pad_mask`, which the collator keeps from being any
    loss's target. Each of `n_epochs` passes over `dset` is packed apart, its
    records taken in an order of its own drawn from `rng`, which also breaks
    ties between equal-length chunks: packed together, a record's copies, all
    of one length, would tend to share sequences, where a later copy could
    attend to an earlier one. The sequences, every pass's together, and the
    records within each are shuffled with `rng` so neither "longest first"
    nor packing order is systematic.
    """
    import pyarrow.compute as pc

    # through the arrow formatter, not `dset.data`: the latter is the table in
    # its stored order and ignores a shuffle's indices mapping, which would pair
    # each row's fetched tokens with another row's length
    arrow = dset.with_format("arrow")
    lengths = pc.list_value_length(arrow["input_ids"]).to_numpy()

    # bins by remaining capacity: free[r] holds the ids of open bins with r left;
    # a segment tree over r finds the smallest r >= c with an open bin
    size = 1
    while size < seq_len + 1:
        size *= 2

    def pack(order: np.ndarray) -> list[list[tuple[int, int, int]]]:
        """one pass's bins, each a list of (row, start, length) chunks, its
        records taken in `order`"""
        # stable-sorted by decreasing length, so ties keep `order`
        chunks = [
            (int(i), s, min(seq_len, int(lengths[i]) - s))
            for i in order
            for s in range(0, int(lengths[i]), seq_len)
        ]
        chunks.sort(key=lambda c: -c[2])
        tree = np.zeros(2 * size, np.int64)
        free = collections.defaultdict(list)
        bins = []

        def mark(r, d):
            i = r + size
            while i:
                tree[i] += d
                i //= 2

        def smallest_ge(c):
            if tree[1] == 0:
                return -1
            stack = [(1, 0, size - 1)]
            while stack:
                i, lo, hi = stack.pop()
                if hi < c or tree[i] == 0:
                    continue
                if lo == hi:
                    return lo
                mid = (lo + hi) // 2
                stack.append((2 * i + 1, mid + 1, hi))  # right after left
                stack.append((2 * i, lo, mid))
            return -1

        for row, start, n in chunks:
            r = smallest_ge(n) if n < seq_len else -1
            if r < 0:
                b, r = len(bins), seq_len
                bins.append([])
            else:
                b = free[r].pop()
                mark(r, -1)
            bins[b].append((row, start, n))
            if r - n > 0:
                free[r - n].append(b)
                mark(r - n, 1)
        return bins

    bins = [b for _ in range(n_epochs) for b in pack(rng.permutation(len(lengths)))]
    for b in rng.permutation(len(bins)):
        pieces = [bins[b][j] for j in rng.permutation(len(bins[b]))]
        out = {k: [] for k in dset.column_names}
        for row, start, n in pieces:
            # sliced before it's made python, so that a long record isn't made
            # whole once for each of its chunks
            record = arrow[row]
            for k in out:
                out[k].extend(record[k][0].values.slice(start, n).to_pylist())
        n_pad = seq_len - len(out["input_ids"])
        for k in out:
            out[k].extend([0] * n_pad)
        out["pad_mask"] = [0] * (seq_len - n_pad) + [1] * n_pad
        # position within each piece, restarting at 0 for every piece (a long
        # record's continuation chunk too) and for the padding: `doc_masking`
        # passes these as position_ids, and transformers takes every restart as
        # the start of a new sequence that attends only to itself
        out["doc_pos"] = [i for _, _, n in pieces for i in range(n)] + list(
            range(n_pad)
        )
        yield out


def record_chunks_iter(
    dset: ds.Dataset, seq_len: int, rng: Generator | None = None, n_epochs: int = 1
) -> collections.abc.Iterator:
    """
    Each record in a row of its own, for `packing: pad`: a record is split only
    when it is longer than `seq_len`, into chunks of at most `seq_len`, and each
    chunk makes a row, unpadded, at its own length -- the collator pads each
    batch to its longest row. Given `rng`, each of `n_epochs` passes over `dset`
    takes its records in an order of its own drawn from it, a record's chunks in
    turn. Without, every pass's chunks come longest first, ties in their
    records' order, so that batches taken in turn each hold rows of about one
    length, the longest batch first. A chunk of a single token is dropped: it
    has no next token to predict, so a batch of such rows alone would have no
    target at all, which hf's own loss averages to a nan. Only a record's last
    chunk can be that short, and its one token, the record's last, has no time
    to the next token or disposition to come either.
    """
    import pyarrow.compute as pc

    # through the arrow formatter, as `best_fit_packed_iter` reads it, so that a
    # shuffled dataset's lengths follow its indices mapping
    arrow = dset.with_format("arrow")
    lengths = pc.list_value_length(arrow["input_ids"]).to_numpy()
    orders = [
        np.arange(len(lengths)) if rng is None else rng.permutation(len(lengths))
        for _ in range(n_epochs)
    ]
    chunks = [
        (int(i), s, min(seq_len, int(lengths[i]) - s))
        for order in orders
        for i in order
        for s in range(0, int(lengths[i]), seq_len)
        if int(lengths[i]) - s > 1
    ]
    if rng is None:
        chunks.sort(key=lambda c: -c[2])  # stable, so ties keep their order
    for row, start, n in chunks:
        # sliced before it's made python, so that a long record isn't made whole
        # once for each of its chunks
        record = arrow[row]
        yield {
            k: record[k][0].values.slice(start, n).to_pylist()
            for k in dset.column_names
        }


def fill_to_budget(
    rows: collections.abc.Iterable[int],
    lengths: np.ndarray,
    budget: int,
    shards: int = 1,
) -> list[list[int]]:
    """`rows`, in the order given, cut into batches of as many rows as fit
    `budget` tokens once each is padded to its batch's longest: k rows, the
    longest of them n tokens, take k × n. Split across `shards` devices, as
    `nn.DataParallel` splits a batch's rows, ceil(k / shards) to a device, the
    budget is each device's: a batch takes ceil(k / shards) × n of it"""
    batches, batch, longest = [], [], 0
    for r in rows:
        n = int(lengths[r])
        # ceil((k + 1) / shards) rows to the busiest device, with this one
        per_shard = (len(batch) + shards) // shards
        if batch and per_shard * max(longest, n) > budget:
            batches.append(batch)
            batch, longest = [], 0
        batch.append(int(r))
        longest = max(longest, n)
    if batch:
        batches.append(batch)
    return batches


def token_budget_batches(
    lengths: collections.abc.Sequence[int] | np.ndarray,
    budget: int,
    rng: Generator | None = None,
    n_passes: int = 1,
    shards: int = 1,
) -> list[list[int]]:
    """
    `packing: pad`'s rows, by index, in batches of as many rows as fit `budget`
    tokens once each is padded to its batch's longest, so that every batch holds
    about as many tokens. Given `shards`, the gpus `nn.DataParallel` splits each
    batch's rows across, `budget` is each gpu's, and a batch holds as many rows
    as keep every gpu's share within it (`fill_to_budget`), where a budget of
    them all together would let the gpu given the larger share of an uneven
    split -- two of three long rows, say -- run past its own.

    A step's loss is a mean over its batch's tokens, and every step weighs alike
    (clipping rescales each step's gradient whenever its norm passes
    `max_grad_norm`, so no rescaling of a step's loss changes that), so a batch
    of a fixed number of rows would weigh each of a row's tokens by about 1 /
    the row's length: each record about alike, the short ones' tokens far above
    the long ones'. A record's length is its outcome -- a short stay is a short
    record -- so that tilts what the model learns toward the records that end
    soon.

    Given `rng`, for training: `lengths` holds `n_passes` passes over the
    records in turn, of as many rows each (`record_chunks_iter`'s), batched pass
    by pass, so that a record's copies never share a batch, each pass's batches
    in an order of their own. A pass's rows are sorted by length, ties in an
    order drawn from `rng`, and cut in turn, shortest first, as fairseq buckets
    its batches: rows of about one length to a batch, padded little. Where the
    lengths run densely, a batch closes only when its next row, about as long as
    its own, won't fit, so that every batch but the pass's last holds all but
    under a row of its budget, and the last, part filled, holds its longest
    rows, a few of which fill most of a budget. Sorted over the whole pass
    rather than by megabatches, as hf's length-grouped sampler sorts: among a
    few hundred rows, the short ones are few and far apart in length, so that a
    batch of them, padded to its longest, would hold a fraction of the budget's
    tokens, each of them -- short records', each ending soon -- weighing above
    the rest. Where they're sparse all the same (records of a few dozen tokens),
    some of that remains. The batch holding the first pass's longest row goes
    first, so that a budget too large for memory fails at the first step.

    Without `rng`, for evaluation: the rows in their own order (the tuning
    split's come longest first), cut in turn, whatever `n_passes`.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(lengths) % n_passes:
        raise ValueError(
            f"{len(lengths)} rows don't make {n_passes} passes of as many rows each"
        )
    if len(lengths) and lengths.max() > budget:
        raise ValueError(
            f"a row of {lengths.max()} tokens doesn't fit a batch of {budget}"
        )
    if rng is None:
        return fill_to_budget(range(len(lengths)), lengths, budget, shards)
    per_pass = len(lengths) // n_passes
    batches = []
    for p in range(n_passes):
        rows = rng.permutation(np.arange(p * per_pass, (p + 1) * per_pass))
        # shortest first, ties in their shuffled order
        rows = rows[np.argsort(lengths[rows], kind="stable")]
        cut = fill_to_budget(rows, lengths, budget, shards)
        batches += [cut[i] for i in rng.permutation(len(cut))]
        if p == 0 and batches:
            first = max(range(len(batches)), key=lambda i: lengths[batches[i]].max())
            batches.insert(0, batches.pop(first))
    return batches


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    y_trues: list[np.ndarray],
    y_scores: list[np.ndarray],
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


def bootstrap_pval(
    y_true: np.ndarray,
    y_score0: np.ndarray,
    y_score1: np.ndarray,
    *,
    n_samples: int = 10_000,
    rng: Generator = np.random.default_rng(seed=42),
    alternative: typing.Literal["one-sided", "two-sided"] = "two-sided",
    metrics: typing.Tuple[typing.Literal["roc_auc", "pr_auc", "brier"], ...] = (
        "roc_auc",
        "pr_auc",
        "brier",
    ),
    n_jobs: int = -1,
    paired: bool = False,
) -> dict[str, float]:
    """
    Performs a bootstrapped test for the null hypothesis that `y_score0` &
    `y_score1` are equally good predictions of y_true (in terms of `objs`), as
    outlined in Algorithm 16.1 of Efron & Tibshirani's "An Introduction to the
    Bootstrap" (Chapman & Hall, Boca Raton, 1993), ignoring variance due to
    model-fitting (i.e. a 'liberal' bootstrap for variability in the test-set
    alone); one-sided alternative corresponds to`y_score1` being better than
    `y_score0`. When `paired`, both scores are assumed to be evaluated on the
    same subjects: subjects are resampled as units (preserving the
    within-subject correlation between the two models) and the null is enforced
    by swapping the two models' scores within each pair; otherwise the two
    score populations are pooled and resampled as independent groups.
    """

    def get_diffs(
        yt0: np.ndarray, ys0: np.ndarray, yt1: np.ndarray, ys1: np.ndarray
    ) -> dict[str, float]:
        diffs = dict()
        if "roc_auc" in metrics:
            diffs["roc_auc"] = skl_mets.roc_auc_score(
                yt1, ys1
            ) - skl_mets.roc_auc_score(yt0, ys0)
        if "pr_auc" in metrics:
            diffs["pr_auc"] = pr_auc_score(yt1, ys1) - pr_auc_score(yt0, ys0)
        if "brier" in metrics:  # higher brier is worse
            diffs["brier"] = -1 * (
                skl_mets.brier_score_loss(yt1, ys1)
                - skl_mets.brier_score_loss(yt0, ys0)
            )
        return diffs

    diff_obs = get_diffs(y_true, y_score0, y_true, y_score1)

    n = len(y_true)
    y_trues = np.concatenate([y_true, y_true])
    y_scores = np.concatenate([y_score0, y_score1])

    def get_diffs_i(rng_i: Generator) -> dict[str, float]:
        if paired:
            # resample subjects as units, then enforce H0 by swapping the two
            # models' scores within each pair
            idx = rng_i.choice(n, size=n, replace=True)
            flip = rng_i.integers(0, 2, size=n).astype(bool)
            yt = y_true[idx]
            return get_diffs(
                yt,
                np.where(flip, y_score1[idx], y_score0[idx]),
                yt,
                np.where(flip, y_score0[idx], y_score1[idx]),
            )
        else:
            # pool all 2n score observations and draw two independent groups
            samp0_i = rng_i.choice(len(y_trues), size=n, replace=True)
            samp1_i = rng_i.choice(len(y_trues), size=n, replace=True)
            return get_diffs(
                y_trues[samp0_i], y_scores[samp0_i], y_trues[samp1_i], y_scores[samp1_i]
            )

    with jl.Parallel(n_jobs=n_jobs) as par:
        diffs = par(jl.delayed(get_diffs_i)(rng_i) for rng_i in rng.spawn(n_samples))

    if alternative == "one-sided":
        return {
            ob: np.mean([d[ob] > diff_obs[ob] for d in diffs]).item() for ob in metrics
        }
    else:  # two-sided
        return {
            ob: np.mean([np.abs(d[ob]) > np.abs(diff_obs[ob]) for d in diffs]).item()
            for ob in metrics
        }


def bootstrap_aggregate_pval(
    y_trues: list[np.ndarray],
    y_score0s: list[np.ndarray],
    y_score1s: list[np.ndarray],
    *,
    n_samples: int = 10_000,
    rng: Generator = np.random.default_rng(seed=42),
    alternative: typing.Literal["one-sided", "two-sided"] = "two-sided",
    metrics: typing.Tuple[
        typing.Literal["avg_roc_auc", "avg_pr_auc", "avg_brier"], ...
    ] = ("avg_roc_auc", "avg_pr_auc", "avg_brier"),
    n_jobs: int = -1,
    paired: bool = False,
) -> dict[str, float]:
    """
    Like `bootstrap_pval` but for the average performance over multiple labels
    (cf. `bootstrap_aggregate_ci` vs. `bootstrap_ci`); tests the null that
    `y_score0s` & `y_score1s` are equally good predictions of `y_trues` in terms
    of the label-averaged `objs`, following Algorithm 16.1 of Efron &
    Tibshirani's "An Introduction to the Bootstrap" (Chapman & Hall, Boca Raton,
    1993). Each label is resampled independently and pairs of (y_true, y_score)
    may have different lengths; the one-sided alternative corresponds to
    `y_score1s` being better than `y_score0s`. When `paired`, the two scores for
    each label are assumed to be evaluated on the same subjects (resampled as
    units, null enforced by swapping the two models within each pair); otherwise
    each label's two score populations are pooled and resampled as independent
    groups.
    """

    def get_diffs(resamples: list[tuple]) -> dict[str, float]:
        # `resamples`: per-label tuples of (yt0, ys0, yt1, ys1)
        diffs = dict()
        if "avg_roc_auc" in metrics:
            diffs["avg_roc_auc"] = np.mean(
                [
                    skl_mets.roc_auc_score(yt1, ys1) - skl_mets.roc_auc_score(yt0, ys0)
                    for yt0, ys0, yt1, ys1 in resamples
                ]
            )
        if "avg_pr_auc" in metrics:
            diffs["avg_pr_auc"] = np.mean(
                [
                    pr_auc_score(yt1, ys1) - pr_auc_score(yt0, ys0)
                    for yt0, ys0, yt1, ys1 in resamples
                ]
            )
        if "avg_brier" in metrics:  # higher brier is worse
            diffs["avg_brier"] = np.mean(
                [
                    -1
                    * (
                        skl_mets.brier_score_loss(yt1, ys1)
                        - skl_mets.brier_score_loss(yt0, ys0)
                    )
                    for yt0, ys0, yt1, ys1 in resamples
                ]
            )
        return diffs

    diff_obs = get_diffs(
        [(yt, s0, yt, s1) for yt, s0, s1 in zip(y_trues, y_score0s, y_score1s)]
    )

    # per-label pooled (y_true, y_score) for the unpaired null
    pooled = [
        (np.concatenate([yt, yt]), np.concatenate([s0, s1]))
        for yt, s0, s1 in zip(y_trues, y_score0s, y_score1s)
    ]

    def get_diffs_i(rng_i: Generator) -> dict[str, float]:
        warnings.filterwarnings("ignore")
        if paired:
            resamples = []
            for yt, s0, s1 in zip(y_trues, y_score0s, y_score1s):
                n_l = len(yt)
                idx = rng_i.choice(n_l, size=n_l, replace=True)
                flip = rng_i.integers(0, 2, size=n_l).astype(bool)
                yti = yt[idx]
                resamples.append(
                    (
                        yti,
                        np.where(flip, s1[idx], s0[idx]),
                        yti,
                        np.where(flip, s0[idx], s1[idx]),
                    )
                )
        else:
            resamples = []
            for yt_pool, ys_pool in pooled:
                n_l = len(yt_pool) // 2
                samp0_i = rng_i.choice(len(yt_pool), size=n_l, replace=True)
                samp1_i = rng_i.choice(len(yt_pool), size=n_l, replace=True)
                resamples.append(
                    (
                        yt_pool[samp0_i],
                        ys_pool[samp0_i],
                        yt_pool[samp1_i],
                        ys_pool[samp1_i],
                    )
                )
        return get_diffs(resamples)

    with jl.Parallel(n_jobs=n_jobs) as par:
        diffs = par(jl.delayed(get_diffs_i)(rng_i) for rng_i in rng.spawn(n_samples))

    if alternative == "one-sided":
        return {
            ob: np.mean([d[ob] > diff_obs[ob] for d in diffs]).item() for ob in metrics
        }
    else:  # two-sided
        return {
            ob: np.mean([np.abs(d[ob]) > np.abs(diff_obs[ob]) for d in diffs]).item()
            for ob in metrics
        }
