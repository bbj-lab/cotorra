#!/usr/bin/env python3

"""tests for cotorra.baselines"""

import collections
import logging
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
import xgboost as xgb
from helpers import LIBOMP_HINT, base_scoring_cfg, write_cfg
from omegaconf import OmegaConf
from sklearn import pipeline
from sklearn.metrics import roc_auc_score

from cotorra.baselines import (
    BagOfTokensScorer,
    BaselineEstimator,
    Featurization,
    bag_of_tokens,
)

SPLITS = ("train", "tuning", "held_out")

ESTIMATORS = [e.value for e in BaselineEstimator]

FEATURIZATIONS = [f.value for f in Featurization]

# XGBoost's `min_child_weight=5` is a sum of hessians that the synthetic train
# split's few dozen rows can't give both sides of any split, so it never splits
# there and scores every row alike (see tests/test_scorer_rep_based.py); only
# the AUC assertion is lifted
NO_AUC_ON_TINY_TRAIN_SPLIT = {"XGBoost"}


class Capture(logging.Handler):
    """collects a `Logger`'s messages, which don't propagate to `caplog`"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def single_token_cfg_path(target_token, tmp_path_factory):
    """a scoring config narrowed to the one token the synthetic data supports"""
    return write_cfg(
        tmp_path_factory.mktemp("single-token-cfg") / "scoring.yaml",
        base_scoring_cfg(score={"target_tokens": [target_token]}),
    )


@pytest.fixture(scope="module")
def past_vocabulary(processed) -> list[int]:
    """each token that some split's past holds, once and in order"""
    pasts = pl.concat(
        pl.read_parquet(
            processed / f"{split}_for_inference.parquet", columns=["tokens_past"]
        )
        for split in SPLITS
    )["tokens_past"]
    return sorted(set(pasts.explode().drop_nulls().to_list()))


@pytest.fixture(scope="module")
def marker(past_vocabulary, tokenizer_cfg, target_token) -> int:
    """a token in no split's past, nor the outcome, for a test to plant in some"""
    absent = (
        set(tokenizer_cfg.lookup.values())
        - set(past_vocabulary)
        - {tokenizer_cfg.lookup[target_token]}
    )
    return min(absent)


def plant(home, split: str, shared: list[int], marker: int, rows: pl.Expr) -> None:
    """
    give each row of `split` in `home` the same past, `shared`, with `marker`
    added for its `rows`, so that the marker alone tells their bags apart: the
    synthetic pasts give the outcomes away on their own (a patient's race and
    its first stay's discharge cycle with the same index), and would score well
    with no help from the marker
    """
    path = home / f"{split}_for_inference.parquet"
    df = pl.read_parquet(path)
    dtype = df.schema["tokens_past"]
    df.with_columns(
        tokens_past=pl.when(rows)
        .then(pl.lit(shared + [marker], dtype=dtype))
        .otherwise(pl.lit(shared, dtype=dtype))
    ).write_parquet(path)


@pytest.fixture(scope="module")
def marked(processed, target_token, past_vocabulary, marker, tmp_path_factory):
    """
    a copy of `processed` in which every row shares a past, but for the marker
    in the past of each row, in every split, that goes on to `target_token`
    """
    home = tmp_path_factory.mktemp("bag-of-tokens-marked")
    shutil.copytree(processed, home, dirs_exist_ok=True)
    for split in SPLITS:
        plant(home, split, past_vocabulary, marker, pl.col(f"{target_token}_future"))
    return home


def held_out(home, target_token) -> tuple[np.ndarray, np.ndarray]:
    """which held-out rows are scored, and which of them go on to the outcome"""
    df = pl.read_parquet(home / "held_out_for_inference.parquet")
    valid = ~df[f"{target_token}_past"].to_numpy()
    return valid, df[f"{target_token}_future"].to_numpy()


def held_out_auc(scorer, home, target_token) -> float:
    valid, y_true = held_out(home, target_token)
    scores = scorer.score_label(target_token=target_token)
    return roc_auc_score(y_true[valid], scores[valid])


# ------------------------------------------------------------------ the bags


def test_bag_of_tokens_gives_each_list_a_row_of_its_own():
    """an empty or missing list keeps its row, so that the rows stay in step
    with the labels they're fit against"""
    tokens = pl.Series([[3, 1, 3], [], None, [0]], dtype=pl.List(pl.UInt32))
    counts = bag_of_tokens(tokens, n_tokens=5)
    assert isinstance(counts, sp.csr_matrix)
    assert counts.dtype == np.float32
    # each (row, token) pair stored once, its ones summed, as scipy builds it
    # from 1.13.1 on (1.13.0 left them apart, hence pyproject's floor): reading
    # the matrix densely, as below, sums them anyway, but LightGBM reads one of
    # a pair's entries alone, and XGBoost aborts
    assert counts.nnz == 3
    np.testing.assert_array_equal(
        counts.toarray(), [[0, 1, 0, 2, 0], [0] * 5, [0] * 5, [1, 0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        bag_of_tokens(tokens, n_tokens=5, one_hot=True).toarray(),
        [[0, 1, 0, 1, 0], [0] * 5, [0] * 5, [1, 0, 0, 0, 0]],
    )


def test_counts_are_of_the_tokens_before_the_threshold(processed, tokenizer_cfg):
    """`tokens_past` alone: the inference tables hold each whole timeline too
    (`tokens`), whose future holds the very outcomes being predicted"""
    scorer = BagOfTokensScorer(processed_data_home=processed)
    for split in SPLITS:
        df = pl.read_parquet(processed / f"{split}_for_inference.parquet")
        expected = np.zeros((df.height, len(tokenizer_cfg.lookup)))
        for row, tokens in enumerate(df["tokens_past"].to_list()):
            for token, count in collections.Counter(tokens).items():
                expected[row, token] = count
        np.testing.assert_array_equal(scorer.features[split].toarray(), expected)
        assert scorer.features[split].nnz == np.count_nonzero(expected)
        # which a bag of the whole timeline wouldn't match
        assert (df["tokens"].list.len() > df["tokens_past"].list.len()).any()


def test_one_hot_marks_which_tokens_come_before_the_threshold(processed):
    counts = BagOfTokensScorer(processed_data_home=processed)
    one_hot = BagOfTokensScorer(processed_data_home=processed, featurization="one-hot")
    for split in SPLITS:
        assert counts.features[split].max() > 1  # so that the two differ
        np.testing.assert_array_equal(
            one_hot.features[split].toarray(), counts.features[split].toarray() > 0
        )


# ---------------------------------------------------------------- the scores


@pytest.mark.parametrize("featurization", FEATURIZATIONS)
@pytest.mark.parametrize("estimator_type", ESTIMATORS)
def test_score_label_finds_a_token_that_foretells_the_outcome(
    marked, target_token, estimator_type, featurization, boosters_usable
):
    if not boosters_usable.get(estimator_type, True):
        pytest.skip(LIBOMP_HINT)
    scorer = BagOfTokensScorer(
        processed_data_home=marked,
        featurization=featurization,
        estimator_type=estimator_type,
    )
    scores = scorer.score_label(target_token=target_token)

    valid, y_true = held_out(marked, target_token)
    assert scores.shape == valid.shape
    assert np.isnan(scores[~valid]).all()
    assert ((scores[valid] >= 0) & (scores[valid] <= 1)).all()
    if estimator_type not in NO_AUC_ON_TINY_TRAIN_SPLIT:
        assert roc_auc_score(y_true[valid], scores[valid]) > 0.9


# what `estimator` builds for each estimator type, and what the boosters are
# handed to evaluate on: lightgbm 4.7 deprecates `eval_set` for `eval_X` and
# `eval_y`, which xgboost's `fit` doesn't take
EXPECTED_ESTIMATOR = {
    "lightGBM": ["LGBMClassifier"],
    "logistic": ["MaxAbsScaler", "LogisticRegression"],
    "XGBoost": ["XGBClassifier"],
}
EVAL_KWARGS = {"lightGBM": {"eval_X", "eval_y", "eval_metric"}, "XGBoost": {"eval_set"}}


@pytest.fixture
def fit_spy(monkeypatch) -> dict:
    """
    stubs out `fit`/`predict_proba` on every estimator `score_label` can build,
    recording which it chose and what it was fit on; nothing is trained, so the
    boosters' arms are reached without risking the libomp segfault
    """
    seen: dict = {}

    def fit(self, X, y, **kwargs):
        seen["chosen"] = (
            [type(step).__name__ for _, step in self.steps]
            if isinstance(self, pipeline.Pipeline)
            else [type(self).__name__]
        )
        seen.update(X=X, y=y, fit_kwargs=kwargs)
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (X.shape[0], 1))

    for cls in (pipeline.Pipeline, lgb.LGBMClassifier, xgb.XGBClassifier):
        monkeypatch.setattr(cls, "fit", fit)
        monkeypatch.setattr(cls, "predict_proba", predict_proba)
    return seen


@pytest.fixture(scope="module")
def partly_past(processed, target_token, tmp_path_factory):
    """
    a copy of `processed` in which a third of each split's rows, whatever their
    outcome, already hold `target_token` in their past, as no synthetic stay
    does: each ends, and so expires if it does, after the threshold
    """
    home = tmp_path_factory.mktemp("bag-of-tokens-partly-past")
    shutil.copytree(processed, home, dirs_exist_ok=True)
    for split in SPLITS:
        path = home / f"{split}_for_inference.parquet"
        pl.read_parquet(path).with_columns(
            pl.col(f"{target_token}_past") | (pl.int_range(pl.len()) % 3 == 0)
        ).write_parquet(path)
    return home


@pytest.mark.parametrize("estimator_type", ESTIMATORS)
def test_each_estimator_fits_and_scores_the_rows_whose_past_lacks_the_outcome(
    partly_past, target_token, estimator_type, fit_spy
):
    """fit on the train split's bags, the boosters evaluating on the tuning
    split's, and the rest of the held-out rows left a nan"""
    scorer = BagOfTokensScorer(
        processed_data_home=partly_past, estimator_type=estimator_type
    )
    scores = scorer.score_label(target_token=target_token)

    tables = {
        split: pl.read_parquet(partly_past / f"{split}_for_inference.parquet")
        for split in SPLITS
    }
    valid = {s: ~df[f"{target_token}_past"].to_numpy() for s, df in tables.items()}
    assert all(0 < v.sum() < len(v) for v in valid.values())  # each a choice
    assert fit_spy["chosen"] == EXPECTED_ESTIMATOR[estimator_type]
    assert set(fit_spy["fit_kwargs"]) == EVAL_KWARGS.get(estimator_type, set())
    assert sp.issparse(fit_spy["X"])
    np.testing.assert_array_equal(
        fit_spy["X"].toarray(), scorer.features["train"][valid["train"]].toarray()
    )
    np.testing.assert_array_equal(
        fit_spy["y"],
        tables["train"][f"{target_token}_future"].to_numpy()[valid["train"]],
    )
    if estimator_type in EVAL_KWARGS:
        kwargs = fit_spy["fit_kwargs"]
        evaluated = kwargs["eval_X"] if "eval_X" in kwargs else kwargs["eval_set"][0][0]
        assert evaluated.shape[0] == valid["tuning"].sum()
    np.testing.assert_array_equal(np.isnan(scores), ~valid["held_out"])


def test_members_and_their_values_are_taken_alike(processed):
    """where `RepBasedScorer`, handed an `EstimatorType` member, fits lightGBM"""
    scorer = BagOfTokensScorer(
        processed_data_home=processed,
        featurization=Featurization.one_hot,
        estimator_type=BaselineEstimator.logistic,
    )
    assert scorer.featurization is Featurization.one_hot
    assert scorer.estimator_type is BaselineEstimator.logistic
    assert scorer.output_home.name == "scores-bag-of-tokens-one-hot-logistic.parquet"


@pytest.mark.parametrize(
    "kwargs", [{"featurization": "tf-idf"}, {"estimator_type": "k-NN"}]
)
def test_an_unknown_featurization_or_estimator_is_refused(processed, kwargs):
    """rather than fit as something else, as `RepBasedScorer` does lightGBM"""
    with pytest.raises(ValueError, match="is not a valid"):
        BagOfTokensScorer(processed_data_home=processed, **kwargs)


def test_a_scorer_without_its_data_is_refused_by_name():
    with pytest.raises(
        TypeError, match=r"^BagOfTokensScorer needs `processed_data_home`$"
    ):
        BagOfTokensScorer()


def test_target_tokens_give_a_score_column_per_matching_token_in_vocab_order(
    processed, tokenizer_cfg, tmp_path, fit_spy
):
    """fnmatch patterns and plain names alike, resolved against the vocabulary
    in its order, each column named for the featurization"""
    patterns = ["XFR-IN//icu", "DSCG//exp*", "RESP//imv"]
    cfg = base_scoring_cfg(score={"target_tokens": patterns})
    scorer = BagOfTokensScorer(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", cfg),
        processed_data_home=processed,
        featurization="one-hot",
    )
    names = sorted(
        ["DSCG//expired", "RESP//imv", "XFR-IN//icu"], key=tokenizer_cfg.lookup.get
    )
    assert scorer.grokked_outcome_tokens == names
    assert list(scorer.score()) == [f"{name}_one-hot_score" for name in names]


# ---------------------------------------------------------------- the output


def test_save_all_writes_the_held_out_table_with_a_score_column(
    marked, target_token, single_token_cfg_path
):
    scorer = BagOfTokensScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=marked,
        featurization="one-hot",
        estimator_type="logistic",
    )
    assert scorer.output_home == (
        marked.resolve() / "scores-bag-of-tokens-one-hot-logistic.parquet"
    )
    scorer.save_all()

    held_out_table = pl.read_parquet(marked / "held_out_for_inference.parquet")
    scores = pl.read_parquet(scorer.output_home)
    column = f"{target_token}_one-hot_score"
    assert scores.columns == held_out_table.columns + [column]
    assert scores.height == held_out_table.height
    np.testing.assert_array_equal(
        scores[column].to_numpy(), scorer.score_label(target_token=target_token)
    )


def test_scores_go_to_the_output_home_when_one_is_given(
    marked, single_token_cfg_path, tmp_path
):
    out = tmp_path / "scores"
    out.mkdir()
    scorer = BagOfTokensScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=marked,
        output_home=out,
        estimator_type="logistic",
    )
    assert scorer.output_home == (
        out.resolve() / "scores-bag-of-tokens-counts-logistic.parquet"
    )
    beside = {p: p.stat().st_mtime_ns for p in marked.glob("scores-*")}
    scorer.save_all()
    assert scorer.output_home.is_file()
    # nothing written beside the inference tables, where the scores go by default
    assert {p: p.stat().st_mtime_ns for p in marked.glob("scores-*")} == beside


@pytest.mark.parametrize("featurization", FEATURIZATIONS)
def test_save_all_verbose_summarizes_the_scores_it_wrote(
    marked, single_token_cfg_path, featurization
):
    """under the featurization's name, which `summarize_preds` has to be told:
    left to look for the other scorers' methods, it would log no intervals"""
    scorer = BagOfTokensScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=marked,
        featurization=featurization,
        estimator_type="logistic",
    )
    scorer.logger.addHandler(capture := Capture())
    scorer.save_all(verbose=True)
    assert f"method='{featurization}'" in capture.messages
    assert any(m.startswith("roc_auc") for m in capture.messages)


# -------------------------------------------------------------------- transfer


@pytest.fixture
def transfer_homes(processed, target_token, past_vocabulary, marker, tmp_path):
    """
    two copies of `processed` whose rows share a past but for the marker,
    planted by outcome in opposite ways: `there`, fit on, in the train and
    tuning pasts of the rows that go on to `target_token`, and the held-out
    pasts of those that don't; `here`, scored, in the held-out pasts of the rows
    that go on to it, and the train and tuning pasts of those that don't. Fit on
    `there`'s train split, the marker foretells the outcome on `here`'s held-out
    split; fit on `here`'s, or scoring `there`'s held-out split, the reverse
    """
    homes = {name: tmp_path / name for name in ("here", "there")}
    for home in homes.values():
        shutil.copytree(processed, home)
    future = pl.col(f"{target_token}_future")
    for split in SPLITS:
        here, there = (future, ~future) if split == "held_out" else (~future, future)
        plant(homes["here"], split, past_vocabulary, marker, here)
        plant(homes["there"], split, past_vocabulary, marker, there)
    return homes


def test_a_training_home_lends_the_estimator_its_bags_and_labels(
    transfer_homes, target_token, single_token_cfg_path
):
    """
    `training_home`, as `RepBasedScorer` takes it: the train and tuning tables
    come from it, the held-out one from `processed_data_home`, where the scores
    are written too
    """
    here, there = transfer_homes["here"], transfer_homes["there"]
    transferred = BagOfTokensScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=here,
        training_home=there,
        estimator_type="logistic",
    )
    for split in SPLITS:
        home = here if split == "held_out" else there
        table = pl.read_parquet(home / f"{split}_for_inference.parquet")
        assert transferred.labels[split].collect().equals(table)
        bag = transferred.features[split]
        np.testing.assert_array_equal(
            bag.toarray(), bag_of_tokens(table["tokens_past"], bag.shape[1]).toarray()
        )
    assert held_out_auc(transferred, here, target_token) > 0.9
    assert transferred.output_home.parent == here.resolve()
    at_home = BagOfTokensScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=here,
        estimator_type="logistic",
    )
    assert held_out_auc(at_home, here, target_token) < 0.1


def test_a_training_home_with_another_vocabulary_is_refused(processed, tmp_path):
    """a column stands for a token id, which its vocabulary gives another token"""
    there = tmp_path / "there"
    shutil.copytree(processed, there)
    tkzr = OmegaConf.load(there / "tokenizer.yaml")
    first, second = list(tkzr.lookup)[1:3]
    tkzr.lookup[first], tkzr.lookup[second] = tkzr.lookup[second], tkzr.lookup[first]
    OmegaConf.save(tkzr, there / "tokenizer.yaml")
    with pytest.raises(ValueError, match="tokenize differently"):
        BagOfTokensScorer(processed_data_home=processed, training_home=there)
