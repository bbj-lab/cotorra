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

# only the two boosting estimators are handed the tuning split to evaluate on:
# lightgbm 4.7 deprecates `eval_set` for `eval_X` and `eval_y`, which xgboost's
# `fit` doesn't take
EVAL_KWARGS = {"lightGBM": {"eval_X", "eval_y"}, "XGBoost": {"eval_set"}}


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
    assert set(fit_spy["fit_kwargs"]) & {"eval_set", "eval_X", "eval_y"} == (
        EVAL_KWARGS.get(estimator_type, set())
    )
    # xgboost's `fit` no longer takes `eval_metric`; it raises a TypeError
    assert ("eval_metric" in fit_spy["fit_kwargs"]) == (estimator_type == "lightGBM")


def test_an_estimator_type_member_silently_falls_back_to_lightgbm(
    scorer_data, fake_model_home, target_token, fit_spy
):
    """
    `score_label` dispatches on `str(self.estimator_type).lower()`, and for a
    `(str, enum.Enum)` member that is `"estimatortype.logistic"`, not
    `"logistic"` -- so handing the constructor an `EstimatorType` rather than
    its `.value` lands in the catch-all and trains LightGBM instead, without
    even the `eval_X`/`eval_y` a real `--estimator lightGBM` run would get. `cli.py`
    passes `.value` and is unaffected; pinned because the constructor's own
    type hint invites the enum.
    """
    scorer = make_scorer(scorer_data, fake_model_home, EstimatorType.logistic)
    scorer.score_label(target_token=target_token)

    assert fit_spy["chosen"] == ["LGBMClassifier"]
    assert fit_spy["fit_kwargs"] == []


def test_a_scorer_without_its_directories_is_refused_by_name(tmp_path):
    """both default to None, a holdover from when the config named them, and
    leaving one out used to fail inside `pathlib` without saying which"""
    with pytest.raises(TypeError, match=r"^RepBasedScorer needs `model_home`$"):
        RepBasedScorer(processed_data_home=tmp_path)
    with pytest.raises(
        TypeError,
        match=r"^RepBasedScorer needs `processed_data_home` and `model_home`$",
    ):
        RepBasedScorer()


# ------------------------------------------- transfer, outputs, and outcomes


def write_features(home, model_home, split: str, features: np.ndarray) -> None:
    pl.DataFrame({"features": features.tolist()}).write_parquet(
        home / f"features-{split}-{model_home.name}.parquet"
    )


def keyed(labels: np.ndarray, sign: float, seed: int) -> np.ndarray:
    """features that tell `labels` apart, or with a negative sign do the reverse"""
    noise = np.random.default_rng(seed).normal(0, 0.5, size=(len(labels), N_FEATURES))
    return sign * 5.0 * labels[:, None] + noise


@pytest.fixture
def transfer_homes(processed, fake_model_home, target_token, tmp_path):
    """
    two copies of `processed` for the two kinds of transfer to tell apart:
    `here`, scored, whose held-out features predict the target and whose train
    and tuning features predict the reverse; and `there`, trained on, whose
    train labels are flipped, and whose train and tuning features predict
    those flipped labels. Fit on `there` alone -- its features with its own
    labels -- an estimator scores `here`'s held-out split well; fit on `here`'s
    features or labels, or on a mix, it scores it backwards
    """
    homes = {name: tmp_path / name for name in ("here", "there")}
    for home in homes.values():
        shutil.copytree(processed, home)
    there_train = homes["there"] / "train_for_inference.parquet"
    flipped = pl.read_parquet(there_train).with_columns(
        ~pl.col(f"{target_token}_future")
    )
    flipped.write_parquet(there_train)
    for split in ("train", "tuning", "held_out"):
        labels = {
            name: pl.read_parquet(home / f"{split}_for_inference.parquet")[
                f"{target_token}_future"
            ]
            .to_numpy()
            .astype(float)
            for name, home in homes.items()
        }
        write_features(
            homes["here"],
            fake_model_home,
            split,
            keyed(labels["here"], 1.0 if split == "held_out" else -1.0, seed=0),
        )
        write_features(
            homes["there"], fake_model_home, split, keyed(labels["there"], 1.0, seed=1)
        )
    return homes


def held_out_auc(scorer, home, target_token) -> float:
    held_out = pl.read_parquet(home / "held_out_for_inference.parquet")
    valid = ~held_out[f"{target_token}_past"].to_numpy().astype(bool)
    y_true = held_out[f"{target_token}_future"].to_numpy().astype(int)
    scores = scorer.score_label(target_token=target_token)
    return roc_auc_score(y_true[valid], scores[valid])


def test_a_training_home_lends_the_estimator_its_features_and_labels(
    transfer_homes, fake_model_home, target_token, single_token_cfg_path
):
    """
    `--training-home`, the second kind of transfer: the train and tuning
    features and labels come from it, the held-out ones from
    `processed_data_home`, where the scores are written too
    """
    here, there = transfer_homes["here"], transfer_homes["there"]
    transferred = RepBasedScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=here,
        model_home=fake_model_home,
        training_home=there,
        estimator_type="logistic",
    )
    for split in ("train", "tuning", "held_out"):
        home = here if split == "held_out" else there
        written = pl.read_parquet(
            home / f"features-{split}-{fake_model_home.name}.parquet"
        )
        np.testing.assert_array_equal(
            transferred.features[split], np.vstack(written["features"].to_list())
        )
        assert (
            transferred.labels[split]
            .collect()
            .equals(pl.read_parquet(home / f"{split}_for_inference.parquet"))
        )
    assert held_out_auc(transferred, here, target_token) > 0.9
    assert transferred.output_home.parent == here.resolve()
    at_home = make_scorer(here, fake_model_home, cfg=single_token_cfg_path)
    assert held_out_auc(at_home, here, target_token) < 0.1


def test_scores_go_to_the_output_home_when_one_is_given(
    scorer_data, fake_model_home, single_token_cfg_path, target_token, tmp_path
):
    out = tmp_path / "scores"
    out.mkdir()
    scorer = RepBasedScorer(
        scoring_cfg=single_token_cfg_path,
        processed_data_home=scorer_data,
        model_home=fake_model_home,
        output_home=out,
        estimator_type="logistic",
    )
    assert scorer.output_home == (
        out.resolve() / f"scores-rep-based-{fake_model_home.name}.parquet"
    )
    beside = {p: p.stat().st_mtime_ns for p in scorer_data.glob("scores-*")}
    scorer.save_all()
    held_out = pl.read_parquet(scorer_data / "held_out_for_inference.parquet")
    scores = pl.read_parquet(scorer.output_home)
    assert scores.columns == held_out.columns + [f"{target_token}_rep_score"]
    # nothing written beside the inference tables, where the scores go by default
    assert {p: p.stat().st_mtime_ns for p in scorer_data.glob("scores-*")} == beside


def test_target_tokens_give_a_score_column_per_matching_token_in_vocab_order(
    scorer_data, fake_model_home, tokenizer_cfg, tmp_path, fit_spy
):
    """fnmatch patterns and plain names alike, resolved against the vocabulary
    in its order, whatever order they came in"""
    patterns = ["XFR-IN//icu", "DSCG//exp*", "RESP//imv"]
    cfg = base_scoring_cfg(score={"target_tokens": patterns})
    scorer = make_scorer(
        scorer_data, fake_model_home, cfg=write_cfg(tmp_path / "scoring.yaml", cfg)
    )
    names = sorted(
        ["DSCG//expired", "RESP//imv", "XFR-IN//icu"], key=tokenizer_cfg.lookup.get
    )
    assert scorer.grokked_outcome_tokens == names
    assert list(scorer.score()) == [f"{name}_rep_score" for name in names]
