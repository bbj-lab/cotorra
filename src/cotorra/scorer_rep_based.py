#!/usr/bin/env python3

"""
make representation-based predictions on held-out data
"""

import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl
import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression

from cotorra.logger import Logger


class RepBasedScorer:
    def __init__(self, main_cfg: pathlib.Path | str = None, ridge: bool = False, **kwargs):
        parsed = OmegaConf.load(
            pathlib.Path(main_cfg if main_cfg is not None else "./config/main.yaml")
            .expanduser()
            .resolve()
        )
        self.cfg = OmegaConf.merge(
            parsed, OmegaConf.create({k: v for k, v in kwargs.items() if v is not None})
        )
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.logger = Logger()
        self.ridge = ridge

        self.splits = ("train", "tuning", "held_out")

        self.features = {
            s: np.vstack(
                pl.scan_parquet(self.processed_data_home / f"features-{s}.parquet")
                .select("features")
                .collect()
                .to_series()
                .to_list()
            )
            for s in self.splits
        }

        self.labels = {
            s: pl.scan_parquet(self.processed_data_home / f"{s}_for_inference.parquet")
            for s in self.splits
        }

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

        if self.ridge:
            bst = LogisticRegression(penalty="l2", C=10.0, max_iter=1000)
            bst.fit(
                X=self.features["train"][train_valid],
                y=train_label[train_valid],
            )
            scores = np.nan * np.ones_like(held_out_valid, dtype=float)
            scores[held_out_valid] = bst.predict_proba(
                X=self.features["held_out"][held_out_valid]
            )[:, 1]
        else:
            bst = lgb.LGBMClassifier(min_data_in_leaf=5, num_leaves=64)
            bst.fit(
                X=self.features["train"][train_valid],
                y=train_label[train_valid],
                eval_set=[
                    (self.features["tuning"][tuning_valid], tuning_label[tuning_valid])
                ],
                eval_metric="auc",
            )
            scores = np.nan * np.ones_like(held_out_valid)
            scores[held_out_valid] = bst.predict_proba(
                X=self.features["held_out"][held_out_valid]
            )[:, 1]

        return scores

    def score(self):
        suffix = "ridge_score" if self.ridge else "rep_score"
        res = dict()
        for tt in tqdm.tqdm(self.cfg.score.target_tokens, position=0):
            res[f"{tt}_{suffix}"] = self.score_label(target_token=tt)

        return res

    def save_all(self, verbose: bool = False):
        scorer_tag = "ridge" if self.ridge else "rep-based"
        (
            df_res := self.labels["held_out"].with_columns(pl.from_dict(self.score()))
        ).sink_parquet(
            self.processed_data_home
            / f"scores-{scorer_tag}-{self.cfg.wandb.run_name}.parquet"
        )

        if verbose:
            self.logger.summarize_preds(df_res, self.cfg.score.target_tokens)


if __name__ == "__main__":
    self = RepBasedScorer()
    self.save_all(verbose=True)
    # breakpoint()
