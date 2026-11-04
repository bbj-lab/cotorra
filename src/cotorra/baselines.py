#!/usr/bin/env python3

"""
make baseline predictions on held-out data, to set a model's scores against
"""

import collections.abc
import enum
import fnmatch
import logging
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


class Baseline(str, enum.Enum):
    counts = "counts"
    one_hot = "one-hot"
    tsfresh = "tsfresh"


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


def past_meds(table: pl.LazyFrame, meds: pl.LazyFrame) -> pl.LazyFrame:
    """
    the rows of `meds` (cocoa's meds.parquet) that come up to the threshold of
    each of `table`'s timelines (an inference table's), keyed by its row (`id`):
    those no later than the last token of its `tokens_past`. cocoa tokenizes each
    row at its time, so where the threshold is a time (cocoa's `duration_s` or
    `after_start_time_s`, or one sampled for each timeline), these are the rows
    its past was tokenized from; under `first_occurrence`, they take in any
    others sharing the triggering token's time too
    """
    cuts = table.select(
        "subject_id",
        # its head, then the last of that, so that an empty past (which cocoa
        # never writes) cuts at nothing, where `list.get(-1)` would take it all
        cut=pl.col("times").list.head(pl.col("tokens_past").list.len()).list.last(),
    ).with_row_index("id")
    return (
        meds.join(cuts, on="subject_id")
        .filter(pl.col("time") <= pl.col("cut"))
        .drop("subject_id", "cut")
    )


def tsfresh_points(
    past: pl.LazyFrame, measured: collections.abc.Collection[str]
) -> pl.LazyFrame:
    """
    `past`'s rows (as `past_meds` gives them) in tsfresh's long format: `id`,
    `time`, `kind` and `value`. Each finite value of a `measured` code is a
    point under the code, its rows without one dropped; any other code's rows
    are each a 1 under the code joined to the row's text value, as cocoa's
    tokenizer fuses them (`ASMT//cam_total` with `yes` is `ASMT//cam_total_yes`),
    so that their count stands for them
    """
    is_measured = pl.col("code").is_in(list(measured))
    return pl.concat(
        [
            past.filter(is_measured, pl.col("numeric_value").is_finite()).select(
                "id",
                "time",
                kind="code",
                value=pl.col("numeric_value").cast(pl.Float64),
            ),
            past.filter(~is_measured).select(
                "id",
                "time",
                kind=pl.concat_str(
                    "code", "text_value", separator="_", ignore_nulls=True
                ),
                value=pl.lit(1.0),
            ),
        ]
    )


def tsfresh_features(
    points: pl.DataFrame,
    n_rows: int,
    event_kinds: collections.abc.Iterable[str],
    n_jobs: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    tsfresh's minimal features (`MinimalFCParameters`) of each kind of `points`
    (as `tsfresh_points` gives them, their `id`s running over `n_rows` rows),
    but for the `event_kinds`, whose points are all 1s and so get only their
    count (`length`): a float32 matrix with a row for each id, a nan where its
    row has no points of a kind, and the names of its columns,
    `<kind>__<feature>`. `n_jobs` is tsfresh's, the processes it extracts in: 0
    extracts here, and its default uses half the CPUs
    """
    # imported here, as it takes a second to import (half a minute the first time)
    from tsfresh import defaults, extract_features
    from tsfresh.feature_extraction import MinimalFCParameters

    # tsfresh builds the minimal set by paring down its full one, warning as it
    # does, without its optional matrixprofile extra, that matrix_profile (no
    # minimal feature) is disabled
    settings_log = logging.getLogger("tsfresh.feature_extraction.settings")
    level = settings_log.level
    settings_log.setLevel(logging.ERROR)
    try:
        minimal = MinimalFCParameters()
    finally:
        settings_log.setLevel(level)

    frame = extract_features(
        points.with_columns(pl.col("id").cast(pl.Int64)).to_pandas(),
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=minimal,
        kind_to_fc_parameters={kind: {"length": None} for kind in event_kinds},
        n_jobs=defaults.N_PROCESSES if n_jobs is None else n_jobs,
    ).reindex(range(n_rows))
    # a copy, as pandas 3 (copy-on-write) can hand out a read-only one
    features = frame.to_numpy(dtype=np.float32, copy=True)
    features[~np.isfinite(features)] = np.nan  # each estimator reads a nan as missing
    return features, frame.columns.to_list()


class BaselineScorer(Configurable):
    """
    fits an estimator for each outcome on features of each timeline's past, as
    `RepBasedScorer` fits one on a model's, and from the same config (whose
    `score.target_tokens` name the outcomes): on the train split's rows whose
    past doesn't already hold the outcome, scoring the held-out split's alike. A
    baseline sets `features`, a matrix for each split with a row for each row of
    its inference table, and prepares them for `logistic`; `name` names the
    file of scores, and `method` their columns
    """

    default_file = "scoring.yaml"
    splits = ("train", "tuning", "held_out")

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        training_home: pathlib.Path | str | None = None,
        estimator_type: BaselineEstimator
        | typing.Literal["lightGBM", "logistic", "XGBoost"] = "lightGBM",
        *,
        name: str,
        method: str,
        **kwargs,
    ) -> None:
        self.require(processed_data_home=processed_data_home)
        super().__init__(scoring_cfg, **kwargs)
        # a member or its value alike, and nothing else
        self.estimator_type = BaselineEstimator(estimator_type)
        self.method = method
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
        ) / f"scores-{name}-{self.estimator_type.value}.parquet"
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.features = {}  # the baseline's own, a matrix for each split
        self.labels = {
            s: pl.scan_parquet(self.home(s) / f"{s}_for_inference.parquet")
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

    def home(self, split: str) -> pathlib.Path:
        """where `split` is read: the held-out split in `processed_data_home`,
        the others in `training_home`"""
        return self.processed_data_home if split == "held_out" else self.training_home

    def logistic(self) -> skl.pipeline.Pipeline:
        """logistic regression, on the features as the baseline prepares them"""
        raise NotImplementedError

    def estimator(self) -> skl.base.BaseEstimator:
        """an unfitted `estimator_type`, the boosters set as `RepBasedScorer`'s are"""
        match self.estimator_type:
            case BaselineEstimator.lightgbm:
                return lgb.LGBMClassifier(
                    min_data_in_leaf=5, num_leaves=64, n_estimators=250, n_jobs=-1
                )
            case BaselineEstimator.logistic:
                return self.logistic()
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
            f"{tt}_{self.method}_score": self.score_label(target_token=tt)
            for tt in tqdm.tqdm(self.grokked_outcome_tokens, position=0)
        }

    def save_all(self, verbose: bool = False) -> None:
        """
        writes the held-out inference table to `output_home` with a score column
        for each outcome, `<token>_<method>_score`; `verbose` logs bootstrapped
        intervals for each
        """
        (
            df_res := self.labels["held_out"].with_columns(pl.from_dict(self.score()))
        ).sink_parquet(self.output_home)

        if verbose:
            self.logger.summarize_preds(
                df_res, self.grokked_outcome_tokens, methods=(self.method,)
            )


class BagOfTokensScorer(BaselineScorer):
    """
    fits an estimator for each outcome on a bag of the tokens in each timeline's
    past (`tokens_past`, cut at the threshold): how many times each token of the
    vocabulary appears there (`counts`), or whether it does at all (`one-hot`),
    keeping none of their order or timing
    """

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
        # a member or its value alike, and nothing else
        self.featurization = Featurization(featurization)
        super().__init__(
            scoring_cfg,
            processed_data_home,
            output_home,
            training_home,
            estimator_type,
            name=f"bag-of-tokens-{self.featurization.value}",
            method=self.featurization.value,
            **kwargs,
        )
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

    def logistic(self) -> skl.pipeline.Pipeline:
        # each token scaled by its largest value over the training rows, so that
        # the penalty weighs every token alike: counts run into the hundreds,
        # and fit unscaled, mimic's held-out RESP//imv scored an AUC of 0.51
        # (scaled, 0.70; z-scored, which inflates rare tokens, 0.51). Indicators
        # already top out at 1, so one-hot features are fit as they are
        return skl.pipeline.make_pipeline(
            skl.preprocessing.MaxAbsScaler(),
            skl.linear_model.LogisticRegression(max_iter=10_000),
        )


class TsfreshScorer(BaselineScorer):
    """
    fits an estimator for each outcome on tsfresh's minimal features of each
    timeline's past, read off its subject's rows of cocoa's meds.parquet up to
    its threshold (see `past_meds`): ten statistics of the values of each code
    measured in the train split's pasts (their count, sum, mean, median,
    standard deviation, variance, root mean square, maximum, largest magnitude
    and minimum), and how many times each other code comes up (see
    `tsfresh_points`), all missing for a past without the code. With `n_jobs`
    above 0, tsfresh extracts them in processes that, where Python spawns them
    (macOS, and Linux from 3.14), import the calling script again, which so has
    to keep its work under `if __name__ == "__main__":`
    """

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        training_home: pathlib.Path | str | None = None,
        estimator_type: BaselineEstimator
        | typing.Literal["lightGBM", "logistic", "XGBoost"] = "lightGBM",
        n_jobs: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            scoring_cfg,
            processed_data_home,
            output_home,
            training_home,
            estimator_type,
            name="tsfresh",
            method="tsfresh",
            **kwargs,
        )
        for split in self.splits:
            if not (meds := self.home(split) / "meds.parquet").is_file():
                raise FileNotFoundError(
                    f"{type(self).__name__} reads the meds.parquet that cocoa's "
                    f"collator writes, and there is none at {meds}"
                )
        pasts = {
            s: past_meds(
                self.labels[s], pl.scan_parquet(self.home(s) / "meds.parquet")
            ).collect()
            for s in self.splits
        }
        # which codes are measured, and so what kinds of points there are, is
        # learned on the train split, as cocoa learns its bins there
        measured = set(
            pasts["train"].filter(pl.col("numeric_value").is_finite())["code"]
        )
        points = {
            s: tsfresh_points(past.lazy(), measured).collect()
            for s, past in pasts.items()
        }
        kinds = set(points["train"]["kind"])
        rows = {
            s: self.labels[s].select(pl.len()).collect().item() for s in self.splits
        }
        offsets = dict(zip(self.splits, np.cumsum([0, *rows.values()]).tolist()))
        # every split at once, its ids offset to follow the split before's
        features, self.feature_names = tsfresh_features(
            pl.concat(
                points[s]
                .filter(pl.col("kind").is_in(list(kinds)))
                .with_columns(pl.col("id").cast(pl.Int64) + offsets[s])
                for s in self.splits
            ),
            n_rows=sum(rows.values()),
            event_kinds=kinds - measured,
            n_jobs=n_jobs,
        )
        self.features = {
            s: features[offsets[s] : offsets[s] + rows[s]] for s in self.splits
        }
        self.logger.info(
            f"tsfresh: {len(self.feature_names)} features of {len(kinds)} kinds "
            f"({len(measured & kinds)} measured codes) from the train split's pasts"
        )

    def logistic(self) -> skl.pipeline.Pipeline:
        # a missing feature (a code that a past lacks) filled with 0 and marked
        # missing, then each scaled by its largest magnitude over the training
        # rows, as `BagOfTokensScorer`'s counts are: on the `both` data's
        # held-out split, this scored DSCG//expired, XFR-IN//icu and RESP//imv
        # at AUCs of 0.89, 0.80 and 0.64, where a median filled in and z-scored
        # scored 0.81, 0.65 and 0.56
        return skl.pipeline.make_pipeline(
            skl.impute.SimpleImputer(
                strategy="constant",
                fill_value=0,
                add_indicator=True,
                keep_empty_features=True,
            ),
            skl.preprocessing.MaxAbsScaler(),
            skl.linear_model.LogisticRegression(max_iter=10_000),
        )


if __name__ == "__main__":
    self = BagOfTokensScorer(processed_data_home="./processed/mimic")
    self.save_all(verbose=True)
    self = TsfreshScorer(processed_data_home="./processed/demo")
    self.save_all(verbose=True)
    # breakpoint()
