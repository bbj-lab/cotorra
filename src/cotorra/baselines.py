#!/usr/bin/env python3

"""
make baseline predictions on held-out data, to set a model's scores against
"""

import enum
import fnmatch
import pathlib
import typing

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
import sklearn as skl
import tqdm
import xgboost as xgb
from omegaconf import OmegaConf

from cotorra.configurable import Configurable


class Featurization(str, enum.Enum):
    counts = "counts"
    one_hot = "one-hot"


class BaselineEstimator(str, enum.Enum):
    lightgbm = "lightGBM"
    logistic = "logistic"
    xgboost = "XGBoost"


def bag_of_tokens(
    tokens: pl.Series, n_tokens: int, one_hot: bool = False
) -> sp.csr_matrix:
    """
    a sparse matrix with a row for each of `tokens`' lists and a column for each
    of the vocabulary's `n_tokens` ids, counting how many times each list holds
    each token or, `one_hot`, marking with a 1 whether it holds it at all; an
    empty or missing list makes a row of zeros
    """
    flat = (
        tokens.to_frame("token")
        .with_row_index("row")
        .explode("token")
        .drop_nulls("token")  # polars 1 explodes an empty list to a null
    )
    bag = sp.csr_matrix(
        (
            np.ones(flat.height, dtype=np.float32),
            (flat["row"].to_numpy(), flat["token"].to_numpy()),
        ),
        shape=(len(tokens), n_tokens),
    )  # summing the ones a (row, token) pair is given into one stored count
    if one_hot:
        bag.data[:] = 1
    return bag


class BagOfTokensScorer(Configurable):
    """
    fits an estimator for each outcome on a bag of the tokens in each timeline's
    past (`tokens_past`, cut at the threshold): how many times each token of the
    vocabulary appears there (`counts`), or whether it does at all (`one-hot`),
    keeping none of their order or timing
    """

    default_file = "scoring.yaml"

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        training_home: pathlib.Path | str | None = None,
        featurization: Featurization | typing.Literal["counts", "one-hot"] = "counts",
        estimator_type: BaselineEstimator
        | typing.Literal["lightGBM", "logistic", "XGBoost"] = "lightGBM",
        **kwargs,
    ) -> None:
        self.require(processed_data_home=processed_data_home)
        super().__init__(scoring_cfg, **kwargs)
        # a member or its value alike, and nothing else
        self.featurization = Featurization(featurization)
        self.estimator_type = BaselineEstimator(estimator_type)
        self.processed_data_home = (
            pathlib.Path(processed_data_home).expanduser().resolve()
        )
        self.training_home = (
            pathlib.Path(training_home).expanduser().resolve()
            if training_home is not None
            else self.processed_data_home
        )
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        ) / "scores-bag-of-tokens-{}-{}.parquet".format(
            self.featurization.value, self.estimator_type.value
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        # a column stands for a token id, which another vocabulary gives another
        # token, so an estimator fit on one vocabulary's bags misreads another's
        if self.training_home != self.processed_data_home and (
            OmegaConf.load(self.training_home / "tokenizer.yaml").lookup
            != self.tkzr_cfg.lookup
        ):
            raise ValueError(
                f"{self.training_home} and {self.processed_data_home} tokenize "
                "differently; tokenize both with one tokenizer (cocoa's "
                "`--tokenizer-home`) to fit on the one and score the other"
            )

        self.splits = ("train", "tuning", "held_out")
        self.labels = {
            s: pl.scan_parquet(
                (self.processed_data_home if s == "held_out" else self.training_home)
                / f"{s}_for_inference.parquet"
            )
            for s in self.splits
        }
        # `tokens_past` alone: these tables also hold each whole timeline
        # (`tokens`), whose future holds the very outcomes to be predicted
        n_tokens = max(self.tkzr_cfg.lookup.values()) + 1
        self.features = {
            s: bag_of_tokens(
                self.labels[s].select("tokens_past").collect().to_series(),
                n_tokens,
                one_hot=self.featurization is Featurization.one_hot,
            )
            for s in self.splits
        }

        self.grokked_outcome_tokens = [
            x
            for x in self.tkzr_cfg.lookup.keys()
            if any(fnmatch.fnmatch(x, p) for p in self.cfg.score.target_tokens)
        ]
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )

    def estimator(self) -> skl.base.BaseEstimator:
        """an unfitted `estimator_type`, the boosters set as `RepBasedScorer`'s are"""
        match self.estimator_type:
            case BaselineEstimator.lightgbm:
                return lgb.LGBMClassifier(
                    min_data_in_leaf=5, num_leaves=64, n_estimators=250, n_jobs=-1
                )
            case BaselineEstimator.logistic:
                # each token scaled by its largest value over the training rows,
                # so that the penalty weighs every token alike: counts run into
                # the hundreds, and fit unscaled, mimic's held-out RESP//imv
                # scored an AUC of 0.51 (scaled, 0.70; z-scored, which inflates
                # rare tokens, 0.51). Indicators already top out at 1, so one-hot
                # features are fit as they are
                return skl.pipeline.make_pipeline(
                    skl.preprocessing.MaxAbsScaler(),
                    skl.linear_model.LogisticRegression(max_iter=10_000),
                )
            case BaselineEstimator.xgboost:
                return xgb.XGBClassifier(
                    min_child_weight=5, max_leaves=64, n_estimators=250, n_jobs=-1
                )

    def score_label(self, target_token: str = "DSCG//expired") -> np.ndarray:
        """
        the probability of `target_token` in each held-out row's future, by an
        estimator fit on the train split's rows; like them, the rows scored are
        those whose past doesn't already hold it, and the others get a nan
        """
        cols = (~pl.col(f"{target_token}_past"), f"{target_token}_future")
        train_valid, train_label = (
            self.labels["train"].select(*cols).collect().to_numpy().T
        )
        tuning_valid, tuning_label = (
            self.labels["tuning"].select(*cols).collect().to_numpy().T
        )
        held_out_valid = (
            self.labels["held_out"].select(cols[0]).collect().to_numpy().ravel()
        )

        # the tuning split for the boosters to evaluate on, as `RepBasedScorer`
        # hands it to them (lightgbm 4.7 deprecates `eval_set` for `eval_X` and
        # `eval_y`, which xgboost's `fit` doesn't take)
        eval_X = self.features["tuning"][tuning_valid]
        eval_y = tuning_label[tuning_valid]
        match self.estimator_type:
            case BaselineEstimator.lightgbm:
                fit_kwargs = dict(eval_X=eval_X, eval_y=eval_y, eval_metric="auc")
            case BaselineEstimator.xgboost:
                fit_kwargs = dict(eval_set=[(eval_X, eval_y)])
            case _:
                fit_kwargs = dict()

        mdl = self.estimator()
        mdl.fit(
            X=self.features["train"][train_valid],
            y=train_label[train_valid],
            **fit_kwargs,
        )

        scores = np.full(held_out_valid.shape, np.nan)
        scores[held_out_valid] = mdl.predict_proba(
            X=self.features["held_out"][held_out_valid]
        )[:, 1]

        return scores

    def score(self) -> dict[str, np.ndarray]:
        """`score_label`'s scores for each outcome, keyed by their column's name"""
        return {
            f"{tt}_{self.featurization.value}_score": self.score_label(target_token=tt)
            for tt in tqdm.tqdm(self.grokked_outcome_tokens, position=0)
        }

    def save_all(self, verbose: bool = False) -> None:
        """
        writes the held-out inference table to `output_home` with a score column
        for each outcome, `<token>_<featurization>_score`; `verbose` logs
        bootstrapped intervals for each
        """
        (
            df_res := self.labels["held_out"].with_columns(pl.from_dict(self.score()))
        ).sink_parquet(self.output_home)

        if verbose:
            self.logger.summarize_preds(
                df_res, self.grokked_outcome_tokens, methods=(self.featurization.value,)
            )


if __name__ == "__main__":
    self = BagOfTokensScorer(processed_data_home="./processed/mimic")
    self.save_all(verbose=True)
    # breakpoint()
