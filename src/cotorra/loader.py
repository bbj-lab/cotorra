#!/usr/bin/env python3

"""
load data and prepare for training / evaluation
"""

import collections.abc
import pathlib

import datasets as ds
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from cotorra.configurable import Configurable
from cotorra.model import HEADS, DispositionHead, TteHead, head_options
from cotorra.util import batched_iter, best_fit_packed_iter

# the positions of a record that each head with a `target_weight` scores -- the
# ones its loss doesn't mask -- as a polars expression over the record's targets
SCORED = {
    TteHead.name: pl.element().is_finite(),
    DispositionHead.name: pl.element() != -100,
}


def disposition_targets(
    input_ids: collections.abc.Sequence[int], class_of: dict[int, int]
) -> np.ndarray:
    """
    each token's target for a disposition head: the class of its record's last
    disposition token -- `class_of` maps each such token's id to its class -- for
    every token before it, and -100, unscored, from it on, the disposition no
    longer being to come. A record with none of those tokens goes unscored
    """
    ids = np.asarray(input_ids)
    hits = np.flatnonzero(np.isin(ids, list(class_of)))
    target = np.full(len(ids), -100, dtype=np.int64)
    if len(hits):
        target[: hits[-1]] = class_of[int(ids[hits[-1]])]
    return target


def label_dispositions(
    batch: collections.abc.Mapping[str, list], class_of: dict[int, int]
) -> dict:
    """`disposition_targets` for a batch of records, as `datasets.map` takes it"""
    return {
        "disposition": [disposition_targets(x, class_of) for x in batch["input_ids"]]
    }


def record_weights(counts: np.ndarray | collections.abc.Sequence[int]) -> np.ndarray:
    """
    each record's weight under a per-record objective, given how many of its
    tokens the head scores: `mean / n` for a record with n of them, where `mean`
    is n's mean over the records with any. A record's scored tokens then weigh
    `mean` between them however many there are, and the weights average 1 over
    every record's scored tokens; a record with none weighs nothing
    """
    counts = np.asarray(counts, dtype=np.float64)
    scored = counts > 0
    mean = counts[scored].mean() if scored.any() else 0.0
    return np.divide(mean, counts, out=np.zeros_like(counts), where=scored)


def weigh_records(
    targets: list[list], indices: list[int], column: str, weights: np.ndarray
) -> dict:
    """each record's weight, from `weights`, on every one of its tokens, for a
    batch of records as `datasets.map` hands over a head's target column and
    their indices"""
    return {
        column: [
            np.full(len(x), weights[i], dtype=np.float32)
            for x, i in zip(targets, indices)
        ]
    }


class Loader(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str | DictConfig | None = None,
        processed_data_home: pathlib.Path | None = None,
    ) -> None:
        self.require(processed_data_home=processed_data_home)
        super().__init__(training_cfg)
        self.rng = np.random.default_rng(42)
        self.processed_data_home = processed_data_home
        self.tokenizer_info = OmegaConf.load(
            self.processed_data_home / "tokenizer.yaml"
        )
        self.splits: tuple = ("train", "tuning", "held_out")
        self.heads = head_options(self.cfg, self.tokenizer_info.lookup)

        tt_all = self.processed_data_home / "tokens_times.parquet"
        assert tt_all.is_file(), FileNotFoundError(
            f"Expected token and time data at {tt_all}, but not found."
        )

        tt_split = {
            s: self.processed_data_home / f"{s}_tokens_times.parquet"
            for s in self.splits
        }
        if (
            not all(s.is_file() for s in tt_split.values())
            or any(
                tt_all.stat().st_mtime > s.stat().st_mtime for s in tt_split.values()
            )
            or any(
                "hours_to_next_token" not in pl.read_parquet_schema(s)
                for s in tt_split.values()
            )
        ):  # pull out training and tuning sets if not already done,
            # if tokens have been updated, or if the caches predate a derived column
            self.subject_splits = pl.scan_parquet(
                self.processed_data_home / "subject_splits.parquet"
            )
            self.tokens_times = pl.scan_parquet(tt_all).with_columns(
                s_elapsed=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                ),
                # the target of a time-to-next-token head; a record's last
                # token has no successor and gets a nan
                hours_to_next_token=pl.col("times").list.eval(
                    (pl.element().shift(-1) - pl.element())
                    .dt.total_seconds()
                    .truediv(3600)
                    .fill_null(float("nan"))
                ),
            )
            to_split = self.tokens_times.join(self.subject_splits, on="subject_id")
            for s in self.splits:
                to_split.filter(pl.col("split") == s).drop("split").sink_parquet(
                    tt_split[s]
                )

        dataset = ds.load_dataset(
            "parquet", data_files={s: str(tt_split[s]) for s in self.splits}
        ).rename_column("tokens", "input_ids")
        if "disposition" in self.heads:
            # labeled per record, before packing splits records across chunks;
            # computed here rather than cached with the splits, since the classes
            # come from the config
            lookup = self.tokenizer_info.lookup
            classes = self.heads["disposition"]["classes"]
            dataset = dataset.map(
                label_dispositions,
                batched=True,
                fn_kwargs={"class_of": {lookup[c]: i for i, c in enumerate(classes)}},
            )
        self.dataset = dataset.select_columns(
            ["input_ids"]
            + (["s_elapsed"] if "time_based_rope" in self.cfg else [])
            + [HEADS[name].target for name in self.heads]
        )
        for name, options in self.heads.items():
            head = HEADS[name]
            if head.target_weight is None or not options.get("per_record", False):
                continue
            # weighted per record before packing too, so that a record's every
            # fragment carries its whole record's weight; each split by its own
            # records, so that its loss -- the tuning split's `eval_loss`
            # included -- averages over them alike. Counted off the dataset
            # rather than the split caches, which lack the disposition target
            for s in self.splits:
                counts = (
                    pl.Series(self.dataset[s].with_format("arrow")[head.target])
                    .list.eval(SCORED[name])
                    .list.sum()
                    .to_numpy()
                )
                self.dataset[s] = self.dataset[s].map(
                    weigh_records,
                    input_columns=head.target,
                    with_indices=True,
                    batched=True,
                    fn_kwargs={
                        "column": head.target_weight,
                        "weights": record_weights(counts),
                    },
                )

        self.inference_files = {
            s: str(f)
            for s in self.splits
            if (f := self.processed_data_home / f"{s}_for_inference.parquet").is_file()
        }

        # only the model's context is kept, never a head's target (as the end
        # times' `hours_to_end_time_past`): `Extractor` writes each batch it reads
        # back out beside the features, and the heads predict from the checkpoint
        # alone. Where every inference table carries a `subject_id`, it leads the
        # columns kept, to name the rows of the features written from them
        ids = all(
            "subject_id" in pl.read_parquet_schema(f)
            for f in self.inference_files.values()
        )
        self.for_inference = (
            (
                ds.load_dataset("parquet", data_files=self.inference_files)
                .rename_column("tokens_past", "input_ids")
                .select_columns(
                    (["subject_id"] if ids else [])
                    + ["input_ids"]
                    + (["s_elapsed_past"] if "time_based_rope" in self.cfg else [])
                )
            )
            if self.inference_files
            else None
        )

    def get_train_data(self) -> ds.Dataset:
        # packing: "concat" (default) concatenates the shuffled records and cuts
        # every max_seq_len tokens; "bfd" packs them whole by Best-fit
        # Decreasing, splitting only records longer than max_seq_len. See
        # util.best_fit_packed_iter
        packing = self.cfg.get("packing", "concat")
        if packing not in ("concat", "bfd"):
            raise ValueError(f"`packing` is `concat` or `bfd`; got {packing!r}")
        if packing == "bfd":
            # the split as it is: the packer shuffles it, and packs each epoch
            # apart, so that no row holds a record twice
            return ds.Dataset.from_generator(
                best_fit_packed_iter,
                gen_kwargs={
                    "dset": self.dataset[self.splits[0]],
                    "seq_len": self.cfg.max_seq_len,
                    "rng": self.rng,
                    "n_epochs": self.cfg.n_epochs,
                },
            ).with_format("torch")
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset[self.splits[0]]
                .repeat(self.cfg.n_epochs)
                .shuffle(generator=self.rng),
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")

    def get_tuning_data(self) -> ds.Dataset:
        if self.cfg.get("doc_masking", False):
            # evaluate in the training format: BFD-packed, and (through the
            # collator) each record masked from the others.
            # Concatenated, a doc_masking model is scored on context it never
            # trains on, which inflates eval_loss and misleads both the Optuna
            # objective and checkpoint selection. Its own generator, so the
            # packing does not depend on what the training data consumed.
            return ds.Dataset.from_generator(
                best_fit_packed_iter,
                gen_kwargs={
                    "dset": self.dataset[self.splits[1]],
                    "seq_len": self.cfg.max_seq_len,
                    "rng": np.random.default_rng(42),
                },
            ).with_format("torch")
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset[self.splits[1]],
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer()
    self = Loader(cfg=trainer.cfg, processed_data_home=trainer.processed_data_home)
    # breakpoint()
