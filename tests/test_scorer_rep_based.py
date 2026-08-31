#!/usr/bin/env python3

"""tests for cotorra.scorer_rep_based.RepBasedScorer"""

import pathlib
import shutil

import numpy as np
import polars as pl
import pytest
from helpers import LIBOMP_HINT, base_scoring_cfg, write_cfg
from sklearn.metrics import roc_auc_score

from cotorra.scorer_rep_based import EstimatorType, RepBasedScorer

N_FEATURES = 8

ESTIMATORS = [
    pytest.param(
        e.value,
        marks=pytest.mark.xfail(
            strict=True,
            raises=TypeError,
            reason="score_label passes eval_metric to XGBClassifier.fit(), which "
            "modern xgboost moved to the constructor; --estimator XGBoost is broken",
        ),
    )
    if e is EstimatorType.xgboost
    else pytest.param(e.value)
    for e in EstimatorType
]


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
    scorer_data, fake_model_home, target_token, estimator_type, lightgbm_usable
):
    if estimator_type == EstimatorType.lightgbm.value and not lightgbm_usable:
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
    assert roc_auc_score(y_true[valid], scores[valid]) > 0.8


def test_score_label_is_uninformative_on_noise_features(
    processed, fake_model_home, target_token, tmp_path_factory
):
    """the AUC above must come from the features, not from the harness"""
    dest = tmp_path_factory.mktemp("rep-based-noise")
    shutil.copytree(processed, dest, dirs_exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "tuning", "held_out"):
        n = pl.read_parquet(dest / f"{split}_for_inference.parquet").height
        pl.DataFrame(
            {"features": rng.normal(0, 1, size=(n, N_FEATURES)).tolist()}
        ).write_parquet(dest / f"features-{split}-{fake_model_home.name}.parquet")

    scores = make_scorer(dest, fake_model_home, "logistic").score_label(
        target_token=target_token
    )
    held_out = pl.read_parquet(dest / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)
    assert roc_auc_score(y_true[valid], scores[valid]) < 0.8


def test_score_covers_every_grokked_token(
    scorer_data, fake_model_home, target_token, single_token_cfg_path
):
    scorer = make_scorer(scorer_data, fake_model_home, cfg=single_token_cfg_path)
    assert set(scorer.score()) == {f"{target_token}_rep_score"}


def test_lightgbm_is_the_default_estimator(
    scorer_data, fake_model_home, target_token, lightgbm_usable
):
    """the estimator every other test deliberately steps around"""
    if not lightgbm_usable:
        pytest.skip(LIBOMP_HINT)
    scorer = RepBasedScorer(processed_data_home=scorer_data, model_home=fake_model_home)
    assert scorer.estimator_type == "lightGBM"
    scores = scorer.score_label(target_token=target_token)
    held_out = pl.read_parquet(scorer_data / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)
    assert roc_auc_score(y_true[valid], scores[valid]) > 0.8


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="score() fits an estimator for every grokked token, including ones "
    "with no valid rows (or a single class) in a split; such labels should be "
    "skipped with a warning rather than aborting the whole run",
)
def test_score_over_all_default_target_tokens_handles_degenerate_labels(
    scorer_data, fake_model_home
):
    make_scorer(scorer_data, fake_model_home).score()


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
