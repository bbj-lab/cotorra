#!/usr/bin/env python3

"""
make representation-based predictions on held-out data
"""

import enum
import fnmatch
import pathlib
import typing

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn as skl
import tqdm
import xgboost as xgb
from omegaconf import OmegaConf

from cotorra.configurable import Configurable

# the estimators `score_label` hands a tuning-split `eval_set`; they are the
# only ones for which the tuning split has to be fittable too
EVAL_SET_ESTIMATORS = ("lightgbm", "xgboost")


class EstimatorType(str, enum.Enum):
    knn = "k-NN"
    lightgbm = "lightGBM"
    logistic = "logistic"
    logistic_z = "logistic-z"
    logistic_cv = "logistic-CV"
    logistic_cv_z = "logistic-CV-z"
    xgboost = "XGBoost"


class RepBasedScorer(Configurable):
    default_file = "scoring.yaml"

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        model_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        estimator_type: typing.Literal[
            "k-NN",
            "lightGBM",
            "logistic",
            "logistic-z",
            "logistic-CV",
            "logistic-CV-z",
            "XGBoost",
        ] = "lightGBM",
        **kwargs,
    ):
        super().__init__(scoring_cfg, **kwargs)
        self.processed_data_home, self.model_home = map(
            lambda x: pathlib.Path(x).expanduser().resolve(),
            (processed_data_home, model_home),
        )
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        ) / f"scores-rep-based-{self.model_home.name}.parquet"
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

        self.splits = ("train", "tuning", "held_out")
        self.estimator_type = estimator_type

        try:
            self.features = {
                s: np.vstack(
                    pl.scan_parquet(
                        self.processed_data_home
                        / f"features-{s}*-{self.model_home.name}.parquet"
                    )
                    .select("features")
                    .collect()
                    .to_series()
                    .to_list()
                )
                for s in self.splits
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Expected extracted features at: "
                f"{self.processed_data_home / 'features-<split>-<model_name>.parquet'},"
                " but not found."
                " Please run `cotorra extract` first."
            ) from e

        self.labels = {
            s: pl.scan_parquet(self.processed_data_home / f"{s}_for_inference.parquet")
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

    def score_label(self, target_token="DSCG//expired"):
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

        match str(self.estimator_type).lower():
            case "logistic" | "lr" | "logistic-regression":
                self.logger.info("Using logistic regression classifier")
                mdl = skl.linear_model.LogisticRegression(max_iter=10_000)
            case "logistic-z" | "lr-z" | "logistic-regression-z":
                self.logger.info(
                    "Using logistic regression classifier on z-scored features"
                )
                mdl = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegression(max_iter=10_000),
                )
            case "logistic-cv" | "lr-cv":
                self.logger.info(
                    "Using logistic regression classifier with cross-validation"
                )
                mdl = skl.linear_model.LogisticRegressionCV(
                    n_jobs=-1,
                    scoring="roc_auc",
                    max_iter=10_000,
                    use_legacy_attributes=False,
                    l1_ratios=(0,),
                )
            case "logistic-cv-z" | "lr-cv-z":
                self.logger.info(
                    "Using logistic regression classifier with cross-validation "
                    "on z-scored features"
                )
                mdl = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegressionCV(
                        n_jobs=-1,
                        scoring="roc_auc",
                        max_iter=10_000,
                        use_legacy_attributes=False,
                        l1_ratios=(0,),  # suppresses a warning
                    ),
                )
            case "k-nn" | "knn" | "k_nn":
                self.logger.info("Using k-nn classifier")
                mdl = skl.neighbors.KNeighborsClassifier(
                    n_neighbors=max(25, int(0.2 * sum(train_valid))), n_jobs=-1
                )
            case "xgboost":
                self.logger.info("Using XGBoost classifier")
                mdl = xgb.XGBClassifier(
                    min_child_weight=5,
                    max_leaves=64,
                    n_estimators=250,
                    n_jobs=-1,
                    # xgboost >= 2.0 takes `eval_metric` here rather than on
                    # `fit`, where lightGBM still wants it
                    eval_metric="auc",
                )
            case _:
                self.logger.info("Using (default) lightGBM classifier")
                mdl = lgb.LGBMClassifier(
                    min_data_in_leaf=5, num_leaves=64, n_estimators=250, n_jobs=-1
                )

        fit_kwargs = dict()
        if (estimator := str(self.estimator_type).lower()) in EVAL_SET_ESTIMATORS:
            fit_kwargs["eval_set"] = [
                (self.features["tuning"][tuning_valid], tuning_label[tuning_valid])
            ]
            if estimator == "lightgbm":
                fit_kwargs["eval_metric"] = "auc"

        mdl.fit(
            X=self.features["train"][train_valid],
            y=train_label[train_valid],
            **fit_kwargs,
        )

        scores = np.nan * np.ones_like(held_out_valid)
        scores[held_out_valid] = mdl.predict_proba(
            X=self.features["held_out"][held_out_valid]
        )[:, 1]

        return scores

    def unfittable_reason(self, target_token: str) -> str | None:
        """
        why `target_token` cannot be fit, or `None` if it can. The winnowed
        inference tables routinely hold labels no estimator can be trained
        on -- a token that never made it into the tables at all, or one whose
        rows not already past the threshold are all a single class -- so
        `score` checks before fitting rather than letting one bad label abort
        the whole run and lose the scores for every other one.
        """
        splits = ("train", "tuning")
        if str(self.estimator_type).lower() not in EVAL_SET_ESTIMATORS:
            splits = ("train",)

        for split in splits:
            cols = self.labels[split].collect_schema().names()
            if missing := [
                c
                for c in (f"{target_token}_past", f"{target_token}_future")
                if c not in cols
            ]:
                return f"{split} is missing {', '.join(missing)}"

            valid, label = (
                self.labels[split]
                .select(~pl.col(f"{target_token}_past"), f"{target_token}_future")
                .collect()
                .to_numpy()
                .T
            )
            n_classes = len(np.unique(label[valid]))
            if n_classes < 2:
                return (
                    f"{split} has {int(valid.sum())} row(s) not already past the "
                    f"threshold, covering {n_classes} class(es)"
                )

        return None

    def score(self):
        res = dict()
        for tt in tqdm.tqdm(self.grokked_outcome_tokens, position=0):
            if (reason := self.unfittable_reason(tt)) is not None:
                self.logger.warning(f"Skipping {tt}: {reason}")
                continue
            res[f"{tt}_rep_score"] = self.score_label(target_token=tt)

        return res

    def save_all(self, verbose: bool = False):
        (
            df_res := self.labels["held_out"].with_columns(pl.from_dict(self.score()))
        ).sink_parquet(self.output_home)

        if verbose:
            self.logger.summarize_preds(df_res, self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = RepBasedScorer()
    self.save_all(verbose=True)
    # breakpoint()
