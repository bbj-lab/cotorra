#!/usr/bin/env python3

"""tests for cotorra.baselines"""

import collections
import datetime
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
    TsfreshScorer,
    bag_of_tokens,
    past_meds,
    tsfresh_points,
)

SPLITS = ("train", "tuning", "held_out")

ESTIMATORS = [e.value for e in BaselineEstimator]

FEATURIZATIONS = [f.value for f in Featurization]

# each baseline, and what it's built with here beyond what a test passes: the
# tsfresh baseline extracts in this process, sparing the suite a pool of workers
BASELINES = {
    "bag-of-tokens": (BagOfTokensScorer, {}),
    "tsfresh": (TsfreshScorer, {"n_jobs": 0}),
}


def baseline(name: str, **kwargs):
    cls, defaults = BASELINES[name]
    return cls(**defaults, **kwargs)


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


# what `estimator` builds for each estimator type, the logistic regression's
# preparation of the features being each baseline's own, and what the boosters
# are handed to evaluate on: lightgbm 4.7 deprecates `eval_set` for `eval_X`
# and `eval_y`, which xgboost's `fit` doesn't take
EXPECTED_ESTIMATOR = (
    {
        ("bag-of-tokens", "logistic"): ["MaxAbsScaler", "LogisticRegression"],
        ("tsfresh", "logistic"): [
            "SimpleImputer",
            "MaxAbsScaler",
            "LogisticRegression",
        ],
    }
    | {(b, "lightGBM"): ["LGBMClassifier"] for b in BASELINES}
    | {(b, "XGBoost"): ["XGBClassifier"] for b in BASELINES}
)
EVAL_KWARGS = {"lightGBM": {"eval_X", "eval_y", "eval_metric"}, "XGBoost": {"eval_set"}}


def dense(features) -> np.ndarray:
    return features.toarray() if sp.issparse(features) else np.asarray(features)


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
@pytest.mark.parametrize("name", BASELINES)
def test_each_estimator_fits_and_scores_the_rows_whose_past_lacks_the_outcome(
    partly_past, target_token, name, estimator_type, fit_spy
):
    """fit on the train split's features (a sparse bag, or tsfresh's dense
    float32 ones), the boosters evaluating on the tuning split's, and the rest of
    the held-out rows left a nan"""
    scorer = baseline(
        name, processed_data_home=partly_past, estimator_type=estimator_type
    )
    scores = scorer.score_label(target_token=target_token)

    tables = {
        split: pl.read_parquet(partly_past / f"{split}_for_inference.parquet")
        for split in SPLITS
    }
    valid = {s: ~df[f"{target_token}_past"].to_numpy() for s, df in tables.items()}
    assert all(0 < v.sum() < len(v) for v in valid.values())  # each a choice
    assert fit_spy["chosen"] == EXPECTED_ESTIMATOR[name, estimator_type]
    assert set(fit_spy["fit_kwargs"]) == EVAL_KWARGS.get(estimator_type, set())
    if name == "bag-of-tokens":
        assert sp.issparse(fit_spy["X"])
    else:
        assert isinstance(fit_spy["X"], np.ndarray)
        assert fit_spy["X"].dtype == np.float32
    np.testing.assert_array_equal(
        dense(fit_spy["X"]), dense(scorer.features["train"][valid["train"]])
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
    "name, kwargs",
    [
        ("bag-of-tokens", {"featurization": "tf-idf"}),
        ("bag-of-tokens", {"estimator_type": "k-NN"}),
        ("tsfresh", {"estimator_type": "k-NN"}),
    ],
)
def test_an_unknown_featurization_or_estimator_is_refused(processed, name, kwargs):
    """rather than fit as something else, as `RepBasedScorer` does lightGBM"""
    with pytest.raises(ValueError, match="is not a valid"):
        baseline(name, processed_data_home=processed, **kwargs)


@pytest.mark.parametrize("name", BASELINES)
def test_a_scorer_without_its_data_is_refused_by_name(name):
    with pytest.raises(
        TypeError, match=rf"^{BASELINES[name][0].__name__} needs `processed_data_home`$"
    ):
        baseline(name)


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


# --------------------------------------------------------------------- tsfresh

# tsfresh's minimal features, each measured code's
MINIMAL = {
    "absolute_maximum",
    "length",
    "maximum",
    "mean",
    "median",
    "minimum",
    "root_mean_square",
    "standard_deviation",
    "sum_values",
    "variance",
}

T0 = datetime.datetime(2024, 1, 1)
HOUR = datetime.timedelta(hours=1)


def test_past_meds_are_a_subjects_rows_up_to_its_last_past_token(processed):
    """
    each split's, row by row: no later than the last token of its past, and so
    before the first of its future, after every row its past was tokenized from
    """
    meds = pl.read_parquet(processed / "meds.parquet")
    for split in SPLITS:
        table = pl.read_parquet(processed / f"{split}_for_inference.parquet")
        past = past_meds(table.lazy(), meds.lazy()).collect()
        expected, futures = [], 0
        for row, (subject, times, n_past) in enumerate(
            table.select("subject_id", "times", pl.col("tokens_past").list.len()).rows()
        ):
            rows = meds.filter(
                pl.col("subject_id") == subject, pl.col("time") <= times[n_past - 1]
            )
            expected += [(row, *r) for r in rows.select("time", "code").rows()]
            if n_past < len(times):
                futures += 1
                assert (rows["time"] < times[n_past]).all()
        assert sorted(past.select("id", "time", "code").rows()) == sorted(expected)
        assert futures  # so that the check of the futures checked something


def test_an_empty_past_takes_no_rows():
    """where `list.get(-1)` would cut it at its timeline's last token, so taking
    every row; cocoa never writes one"""
    table = pl.LazyFrame(
        {
            "subject_id": ["a", "b"],
            "times": [[T0, T0 + HOUR]] * 2,
            "tokens_past": [[], [7]],
        },
        schema_overrides={
            "times": pl.List(pl.Datetime("us")),
            "tokens_past": pl.List(pl.UInt32),
        },
    )
    meds = pl.LazyFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "time": [T0, T0 + HOUR] * 2,
            "code": ["X"] * 4,
            "numeric_value": [None] * 4,
            "text_value": [None] * 4,
        },
        schema_overrides={
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "text_value": pl.String,
        },
    )
    assert past_meds(table, meds).collect().select("id", "time").rows() == [(1, T0)]


def test_tsfresh_points_are_a_measured_codes_values_and_a_one_for_any_other_row():
    """a measured code's rows without a finite value dropped; any other code's
    rows each a 1, whatever value they hold, under the code fused with the row's
    text value as cocoa's tokenizer fuses them"""
    past = pl.LazyFrame(
        {
            "id": [0, 0, 0, 0, 0, 1, 1, 1],
            "time": [T0] * 8,
            "code": ["LAB//na"] * 4 + ["ASMT//cam", "ASMT//cam", "RESP//imv", "LAB//k"],
            "numeric_value": [140.0, None, np.nan, np.inf, None, None, None, 4.0],
            "text_value": [None] * 4 + ["yes", "no", None, None],
        },
        schema_overrides={"time": pl.Datetime("us"), "numeric_value": pl.Float32},
    )
    points = tsfresh_points(past, measured={"LAB//na"}).collect()
    assert sorted(points.select("id", "kind", "value").rows()) == [
        (0, "ASMT//cam_yes", 1.0),
        (0, "LAB//na", 140.0),
        (1, "ASMT//cam_no", 1.0),
        (1, "LAB//k", 1.0),
        (1, "RESP//imv", 1.0),
    ]


@pytest.fixture(scope="module")
def tsfresh_scorer(processed, single_token_cfg_path, tmp_path_factory):
    """the synthetic data's tsfresh baseline, extracted once for the tests that
    read its features; it writes its scores beside the data, not into it"""
    return baseline(
        "tsfresh",
        scoring_cfg=single_token_cfg_path,
        processed_data_home=processed,
        output_home=tmp_path_factory.mktemp("tsfresh-scores"),
        estimator_type="logistic",
    )


def per_row(points: pl.DataFrame, n_rows: int, **aggs: pl.Expr) -> dict:
    """each of `aggs` over each row's `points`, a nan for a row without any"""
    out = {name: np.full(n_rows, np.nan) for name in aggs}
    for row, *values in points.group_by("id").agg(**aggs).rows():
        for name, value in zip(aggs, values):
            out[name][row] = value
    return out


def test_tsfresh_features_summarize_each_code_in_each_past(processed, tsfresh_scorer):
    """
    the minimal features of each code with a value in the train split's pasts,
    and only the count of every other kind, as computed here from each row's
    past meds rows; missing where a past lacks the kind
    """
    meds = pl.scan_parquet(processed / "meds.parquet")
    pasts = {
        s: past_meds(pl.scan_parquet(processed / f"{s}_for_inference.parquet"), meds)
        .collect()
        .with_columns(
            kind=pl.concat_str("code", "text_value", separator="_", ignore_nulls=True)
        )
        for s in SPLITS
    }
    measured = set(pasts["train"].filter(pl.col("numeric_value").is_finite())["code"])
    names = tsfresh_scorer.feature_names
    features = collections.defaultdict(set)
    for name in names:
        kind, feature = name.rsplit("__", 1)
        features[kind].add(feature)
    assert {kind for kind, f in features.items() if f == MINIMAL} == measured
    assert all(f == {"length"} for kind, f in features.items() if kind not in measured)
    assert any(kind not in measured for kind in features)  # some events
    # the kinds of the train split's pasts, and not those only another's reach
    # (the synthetic tuning and held-out splits' markers)
    events = pasts["train"].filter(~pl.col("code").is_in(list(measured)))["kind"]
    assert set(features) == measured | set(events)
    assert {kind for s in ("tuning", "held_out") for kind in pasts[s]["kind"]} - set(
        features
    )

    # the code measured most often, and the most frequent kind with a text value
    code = (
        pasts["train"]
        .filter(pl.col("code").is_in(list(measured)))["code"]
        .value_counts(sort=True)
        .item(0, "code")
    )
    event = (
        pasts["train"]
        .filter(pl.col("text_value").is_not_null())["kind"]
        .value_counts(sort=True)
        .item(0, "kind")
    )
    missing = 0
    for split in SPLITS:
        X = tsfresh_scorer.features[split]
        values = pasts[split].filter(
            pl.col("code") == code, pl.col("numeric_value").is_finite()
        )
        expected = per_row(
            values,
            X.shape[0],
            mean=pl.col("numeric_value").cast(pl.Float64).mean(),
            maximum=pl.col("numeric_value").max(),
            minimum=pl.col("numeric_value").min(),
            length=pl.len(),
        ) | per_row(
            pasts[split].filter(pl.col("kind") == event), X.shape[0], count=pl.len()
        )
        for feature in ("mean", "maximum", "minimum", "length"):
            np.testing.assert_allclose(
                X[:, names.index(f"{code}__{feature}")], expected[feature], rtol=1e-6
            )
        np.testing.assert_array_equal(
            X[:, names.index(f"{event}__length")], expected["count"]
        )
        missing += np.isnan(expected["mean"]).sum() + np.isnan(expected["count"]).sum()
    assert missing  # so that a past without the kind, missing it, is checked too


def write_meds(home, marked: dict[str, pl.Expr], marker: str) -> None:
    """
    replace `home`'s meds.parquet with one giving every subject the same two
    rows at its timeline's first time (a measurement, `VTL//shared`, and an
    event, `RESP//shared`), but for a marker beside them in the rows that
    `marked` picks out of each split's table: a measurement of 1, where the
    others' is 0 (`marker="measured"`), or an event the others lack (`"event"`).
    As with `plant`, the marker is then all that tells one past from another,
    the synthetic meds giving the outcomes away on their own
    """
    schema = pl.read_parquet_schema(home / "meds.parquet")
    none = pl.lit(None, dtype=pl.Float64)
    frames = []
    for split, picked in marked.items():
        first = pl.read_parquet(home / f"{split}_for_inference.parquet").select(
            "subject_id", time=pl.col("times").list.first(), picked=picked
        )
        frames += [
            first.with_columns(code=pl.lit("VTL//shared"), numeric_value=pl.lit(1.0)),
            first.with_columns(code=pl.lit("RESP//shared"), numeric_value=none),
            first.with_columns(
                code=pl.lit("LAB-RES//marker"),
                numeric_value=pl.col("picked").cast(pl.Float64),
            )
            if marker == "measured"
            else first.filter("picked").with_columns(
                code=pl.lit("RESP//marker"), numeric_value=none
            ),
        ]
    pl.concat(
        frame.select(
            "subject_id",
            "time",
            "code",
            "numeric_value",
            text_value=pl.lit(None, dtype=pl.String),
        )
        for frame in frames
    ).cast(schema).write_parquet(home / "meds.parquet")


@pytest.fixture(scope="module", params=["measured", "event"])
def meds_marked(request, processed, target_token, tmp_path_factory):
    """a copy of `processed` whose meds give every row the same past, but for a
    marker in those of the rows that go on to `target_token`: a measurement of
    1 rather than 0 (`measured`), or an event the others lack (`event`)"""
    home = tmp_path_factory.mktemp(f"tsfresh-marked-{request.param}")
    shutil.copytree(processed, home, dirs_exist_ok=True)
    future = pl.col(f"{target_token}_future")
    write_meds(home, {split: future for split in SPLITS}, marker=request.param)
    return home


@pytest.mark.parametrize("estimator_type", ESTIMATORS)
def test_tsfresh_finds_a_code_that_foretells_the_outcome(
    meds_marked, target_token, estimator_type, boosters_usable
):
    """by a measured code's value, or an event's count"""
    if not boosters_usable.get(estimator_type, True):
        pytest.skip(LIBOMP_HINT)
    scorer = baseline(
        "tsfresh", processed_data_home=meds_marked, estimator_type=estimator_type
    )
    scores = scorer.score_label(target_token=target_token)

    valid, y_true = held_out(meds_marked, target_token)
    assert scores.shape == valid.shape
    assert np.isnan(scores[~valid]).all()
    assert ((scores[valid] >= 0) & (scores[valid] <= 1)).all()
    if estimator_type not in NO_AUC_ON_TINY_TRAIN_SPLIT:
        assert roc_auc_score(y_true[valid], scores[valid]) > 0.9


def test_tsfresh_save_all_writes_and_summarizes_its_scores(
    processed, tsfresh_scorer, target_token
):
    tsfresh_scorer.logger.addHandler(capture := Capture())
    try:
        tsfresh_scorer.save_all(verbose=True)
    finally:
        tsfresh_scorer.logger.removeHandler(capture)

    assert tsfresh_scorer.output_home.name == "scores-tsfresh-logistic.parquet"
    held_out_table = pl.read_parquet(processed / "held_out_for_inference.parquet")
    scores = pl.read_parquet(tsfresh_scorer.output_home)
    assert scores.columns == held_out_table.columns + [f"{target_token}_tsfresh_score"]
    assert scores.height == held_out_table.height
    assert "method='tsfresh'" in capture.messages
    assert any(m.startswith("roc_auc") for m in capture.messages)


@pytest.fixture
def meds_transfer_homes(processed, target_token, tmp_path):
    """
    `transfer_homes` with meds: two copies of `processed` whose rows share a past
    but for the measured marker, 1 by outcome in opposite ways: `there`, fit on,
    for the train and tuning rows that go on to `target_token` and the held-out
    rows that don't; `here`, scored, for the held-out rows that go on to it and
    the train and tuning rows that don't
    """
    homes = {name: tmp_path / name for name in ("here", "there")}
    for home in homes.values():
        shutil.copytree(processed, home)
    future = pl.col(f"{target_token}_future")
    for name, held_out_marked in (("here", future), ("there", ~future)):
        write_meds(
            homes[name],
            {
                split: held_out_marked if split == "held_out" else ~held_out_marked
                for split in SPLITS
            },
            marker="measured",
        )
    return homes


def test_a_training_home_lends_tsfresh_its_meds_and_labels(
    meds_transfer_homes, target_token, single_token_cfg_path
):
    """the train and tuning features from its meds, the held-out ones from
    `processed_data_home`'s; fit on `here`'s, or scoring `there`'s held-out
    split, the marker would foretell the reverse"""
    here, there = meds_transfer_homes["here"], meds_transfer_homes["there"]
    transferred = baseline(
        "tsfresh",
        scoring_cfg=single_token_cfg_path,
        processed_data_home=here,
        training_home=there,
        estimator_type="logistic",
    )
    assert held_out_auc(transferred, here, target_token) > 0.9
    assert transferred.output_home.parent == here.resolve()
    at_home = baseline(
        "tsfresh",
        scoring_cfg=single_token_cfg_path,
        processed_data_home=here,
        estimator_type="logistic",
    )
    assert held_out_auc(at_home, here, target_token) < 0.1


def test_tsfresh_without_meds_is_refused_saying_where_it_looked(processed, tmp_path):
    home = tmp_path / "no-meds"
    shutil.copytree(processed, home)
    (home / "meds.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=r"no-meds/meds\.parquet"):
        baseline("tsfresh", processed_data_home=home)


def test_which_codes_are_measured_is_learned_on_the_train_split(processed, tmp_path):
    """as cocoa learns its bins there: a code without a value in the train
    split's pasts is counted, though the other splits' rows give it values"""
    home = tmp_path / "valued-elsewhere"
    shutil.copytree(processed, home)
    meds = pl.scan_parquet(home / "meds.parquet")
    pasts = {
        s: past_meds(
            pl.scan_parquet(home / f"{s}_for_inference.parquet"), meds
        ).collect()
        for s in SPLITS
    }
    measured = set(pasts["train"].filter(pl.col("numeric_value").is_finite())["code"])
    counted = pasts["train"].filter(
        ~pl.col("code").is_in(list(measured)), pl.col("text_value").is_null()
    )["code"]
    code = sorted(set(counted) & set(pasts["held_out"]["code"]))[0]
    others = pl.concat(
        pl.read_parquet(home / f"{s}_for_inference.parquet", columns=["subject_id"])
        for s in ("tuning", "held_out")
    )["subject_id"]
    pl.read_parquet(home / "meds.parquet").with_columns(
        numeric_value=pl.when(
            pl.col("code") == code, pl.col("subject_id").is_in(others.to_list())
        )
        .then(pl.lit(1.0, dtype=pl.Float32))
        .otherwise("numeric_value")
    ).write_parquet(home / "meds.parquet")

    names = baseline("tsfresh", processed_data_home=home).feature_names
    assert [name for name in names if name.startswith(f"{code}__")] == [
        f"{code}__length"
    ]
