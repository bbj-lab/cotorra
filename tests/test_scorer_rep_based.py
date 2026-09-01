#!/usr/bin/env python3

"""tests for cotorra.scorer_rep_based.RepBasedScorer"""

import pathlib
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from helpers import LIBOMP_HINT, base_scoring_cfg, write_cfg
from sklearn import linear_model, neighbors, pipeline
from sklearn.metrics import roc_auc_score

from cotorra.scorer_rep_based import EstimatorType, RepBasedScorer

N_FEATURES = 8

# noise draws averaged over in `test_score_label_is_uninformative_on_noise_features`
N_NOISE_DRAWS = 10

ESTIMATORS = [e.value for e in EstimatorType]

# XGBoost's `min_child_weight=5` is a sum-of-hessians threshold -- roughly 20
# rows per leaf for a balanced logistic objective -- and not a row count like
# lightGBM's `min_data_in_leaf=5`. The synthetic training split is a few dozen
# rows, too few for either child of any split to clear it, so no tree ever
# splits and every row gets the same score. Not a defect in the shipped
# hyperparameters (the same fit recovers the signal from ~60 rows up), so only
# the AUC assertion is lifted; the structural ones still apply.
NO_AUC_ON_TINY_TRAIN_SPLIT = {"XGBoost"}


@pytest.fixture(scope="module")
def scorer_data(processed, fake_model_home, target_token, tmp_path_factory):
    """
    a copy of the synthetic processed dir (so we don't write into the shared
    session fixture) with fabricated but genuinely predictive `features`
    columns for `fake_model_home`, keyed to `target_token`'s future label
    """
    dest = tmp_path_factory.mktemp("rep-based-processed")
    shutil.copytree(processed, dest, dirs_exist_ok=True)

    rng = np.random.default_rng(0)
    for split in ("train", "tuning", "held_out"):
        df = pl.read_parquet(dest / f"{split}_for_inference.parquet")
        y = df[f"{target_token}_future"].to_numpy().astype(float)
        feats = y[:, None] * 5.0 + rng.normal(0, 0.5, size=(len(y), N_FEATURES))
        pl.DataFrame({"features": feats.tolist()}).write_parquet(
            dest / f"features-{split}-{fake_model_home.name}.parquet"
        )
    return dest


@pytest.fixture(scope="module")
def single_token_cfg_path(target_token, tmp_path_factory) -> pathlib.Path:
    """a scoring config narrowed to the one token the synthetic data supports"""
    return write_cfg(
        tmp_path_factory.mktemp("single-token-cfg") / "scoring.yaml",
        base_scoring_cfg(score={"target_tokens": [target_token]}),
    )


def make_scorer(scorer_data, fake_model_home, estimator_type="logistic", cfg=None):
    return RepBasedScorer(
        scoring_cfg=cfg,
        processed_data_home=scorer_data,
        model_home=fake_model_home,
        estimator_type=estimator_type,
    )


def test_missing_extracted_features_raises_a_compute_error(processed, fake_model_home):
    """
    the constructor's `except FileNotFoundError` clause is dead code: polars
    raises its own `ComputeError` (not the builtin `FileNotFoundError`) when a
    glob pattern matches no files, so the intended "please run `cotorra
    extract` first" message is never actually shown
    """
    with pytest.raises(pl.exceptions.ComputeError):
        RepBasedScorer(processed_data_home=processed, model_home=fake_model_home)


def test_grokked_outcome_tokens_are_discovered_from_scoring_config(
    scorer_data, fake_model_home, target_token
):
    scorer = make_scorer(scorer_data, fake_model_home)
    assert target_token in scorer.grokked_outcome_tokens


def test_grokked_outcome_tokens_honor_a_narrowed_config(
    scorer_data, fake_model_home, target_token, single_token_cfg_path
):
    scorer = make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path)
    assert scorer.grokked_outcome_tokens == [target_token]


@pytest.mark.parametrize("estimator_type", ESTIMATORS)
def test_score_label_recovers_signal_from_predictive_features(
    scorer_data, fake_model_home, target_token, estimator_type, boosters_usable
):
    if not boosters_usable.get(estimator_type, True):
        pytest.skip(LIBOMP_HINT)
    scorer = make_scorer(scorer_data, fake_model_home, estimator_type)
    scores = scorer.score_label(target_token=target_token)

    held_out = pl.read_parquet(scorer_data / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)

    assert scores.shape == (held_out.height,)
    assert np.isnan(scores[~valid]).all()
    assert np.isfinite(scores[valid]).all()
    assert ((scores[valid] >= 0) & (scores[valid] <= 1)).all()
    if estimator_type not in NO_AUC_ON_TINY_TRAIN_SPLIT:
        assert roc_auc_score(y_true[valid], scores[valid]) > 0.8


def test_score_label_is_uninformative_on_noise_features(
    processed, fake_model_home, target_token, tmp_path_factory
):
    """
    the AUC above must come from the features, not from the harness.

    Averaged over several noise draws rather than asserted on one: the
    held-out split is only a dozen rows, so a single draw's AUC is a wide
    random variable, and a fixed seed does not pin it either -- cocoa's
    pipeline makes no row-order guarantee, so the same seed pairs the same
    features with a different permutation of the labels from session to
    session. A single-draw threshold of 0.8 failed roughly one run in ten.
    """
    dest = tmp_path_factory.mktemp("rep-based-noise")
    shutil.copytree(processed, dest, dirs_exist_ok=True)
    held_out = pl.read_parquet(dest / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)

    aucs = []
    for seed in range(N_NOISE_DRAWS):
        rng = np.random.default_rng(seed)
        for split in ("train", "tuning", "held_out"):
            n = pl.read_parquet(dest / f"{split}_for_inference.parquet").height
            pl.DataFrame(
                {"features": rng.normal(0, 1, size=(n, N_FEATURES)).tolist()}
            ).write_parquet(dest / f"features-{split}-{fake_model_home.name}.parquet")
        scores = make_scorer(dest, fake_model_home, "logistic").score_label(
            target_token=target_token
        )
        aucs.append(roc_auc_score(y_true[valid], scores[valid]))

    assert np.mean(aucs) < 0.75, aucs


def test_score_covers_every_grokked_token(
    scorer_data, fake_model_home, target_token, single_token_cfg_path
):
    scorer = make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path)
    assert set(scorer.score()) == {f"{target_token}_rep_score"}


def test_lightgbm_is_the_default_estimator(
    scorer_data, fake_model_home, target_token, boosters_usable
):
    """the estimator every other test deliberately steps around"""
    if not boosters_usable["lightGBM"]:
        pytest.skip(LIBOMP_HINT)
    scorer = RepBasedScorer(processed_data_home=scorer_data, model_home=fake_model_home)
    assert scorer.estimator_type == "lightGBM"
    scores = scorer.score_label(target_token=target_token)
    held_out = pl.read_parquet(scorer_data / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)
    assert roc_auc_score(y_true[valid], scores[valid]) > 0.8


def test_score_label_scores_every_held_out_row_for_each_kept_label(
    scorer_data, fake_model_home
):
    height = pl.read_parquet(scorer_data / "held_out_for_inference.parquet").height
    res = make_scorer(scorer_data, fake_model_home).score()
    for column, scores in res.items():
        assert scores.shape == (height,), column


def test_save_all_writes_a_scores_parquet_with_rep_score_column(
    scorer_data, fake_model_home, target_token, single_token_cfg_path
):
    scorer = make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path)
    scorer.save_all()

    assert scorer.output_home.name == f"scores-rep-based-{fake_model_home.name}.parquet"
    df = pl.read_parquet(scorer.output_home)
    held_out = pl.read_parquet(scorer_data / "held_out_for_inference.parquet")
    assert f"{target_token}_rep_score" in df.columns
    assert df.height == held_out.height


def test_save_all_verbose_summarizes_without_raising(
    scorer_data, fake_model_home, single_token_cfg_path
):
    """`summarize_preds` runs a bootstrap over the written scores"""
    make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path).save_all(
        verbose=True
    )


def test_output_home_defaults_to_the_processed_data_home(
    scorer_data, fake_model_home, single_token_cfg_path
):
    scorer = make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path)
    assert scorer.output_home.parent == scorer_data


# what `score_label`'s `match` should build for each `--estimator` value; the
# `-z` variants arrive wrapped in a `Pipeline` behind a `StandardScaler`
EXPECTED_ESTIMATOR = {
    "k-NN": ["KNeighborsClassifier"],
    "lightGBM": ["LGBMClassifier"],
    "logistic": ["LogisticRegression"],
    "logistic-z": ["StandardScaler", "LogisticRegression"],
    "logistic-CV": ["LogisticRegressionCV"],
    "logistic-CV-z": ["StandardScaler", "LogisticRegressionCV"],
    "XGBoost": ["XGBClassifier"],
}

# only the two boosting estimators are handed a tuning-split `eval_set`
EVAL_SET_ESTIMATORS = {"lightGBM", "XGBoost"}


@pytest.fixture
def fit_spy(monkeypatch) -> dict:
    """
    stubs out `fit`/`predict_proba` on every estimator `score_label` can
    build, recording which one it chose and what it was fit with. Nothing is
    actually trained, so both boosting arms are reached without risking the
    libomp segfault.
    """
    seen: dict = {}

    def fit(self, X, y, **kwargs):
        seen["chosen"] = (
            [type(step).__name__ for _, step in self.steps]
            if isinstance(self, pipeline.Pipeline)
            else [type(self).__name__]
        )
        seen["fit_kwargs"] = sorted(kwargs)
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))

    for cls in (
        neighbors.KNeighborsClassifier,
        linear_model.LogisticRegression,
        linear_model.LogisticRegressionCV,
        pipeline.Pipeline,
        lgb.LGBMClassifier,
        xgb.XGBClassifier,
    ):
        monkeypatch.setattr(cls, "fit", fit)
        monkeypatch.setattr(cls, "predict_proba", predict_proba)
    return seen


@pytest.mark.parametrize("estimator_type", sorted(EXPECTED_ESTIMATOR))
def test_each_estimator_value_selects_its_own_estimator(
    scorer_data, fake_model_home, target_token, estimator_type, fit_spy
):
    """
    `score_label`'s `match` ends in a catch-all `_` that quietly builds
    LightGBM, so a renamed `EstimatorType` value or a typo'd arm would look
    like a working `--estimator` rather than an error; check each value really
    reaches its own branch
    """
    scorer = make_scorer(scorer_data, fake_model_home, estimator_type)
    scorer.score_label(target_token=target_token)

    assert fit_spy["chosen"] == EXPECTED_ESTIMATOR[estimator_type]
    assert ("eval_set" in fit_spy["fit_kwargs"]) == (
        estimator_type in EVAL_SET_ESTIMATORS
    )


def test_an_estimator_type_member_silently_falls_back_to_lightgbm(
    scorer_data, fake_model_home, target_token, fit_spy
):
    """
    `score_label` dispatches on `str(self.estimator_type).lower()`, and for a
    `(str, enum.Enum)` member that is `"estimatortype.logistic"`, not
    `"logistic"` -- so handing the constructor an `EstimatorType` rather than
    its `.value` lands in the catch-all and trains LightGBM instead, without
    even the `eval_set` a real `--estimator lightGBM` run would get. `cli.py`
    passes `.value` and is unaffected; pinned because the constructor's own
    type hint invites the enum.
    """
    scorer = make_scorer(scorer_data, fake_model_home, EstimatorType.logistic)
    scorer.score_label(target_token=target_token)

    assert fit_spy["chosen"] == ["LGBMClassifier"]
    assert fit_spy["fit_kwargs"] == []
