#!/usr/bin/env python3

"""
load data and prepare for training / evaluation
"""

import pathlib

import datasets as ds
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from cotorra.configurable import Configurable
from cotorra.model import HEADS, head_options
from cotorra.util import batched_iter


def disposition_targets(input_ids, class_of: dict[int, int]) -> np.ndarray:
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


def label_dispositions(batch: dict, class_of: dict[int, int]) -> dict:
    """`disposition_targets` for a batch of records, as `datasets.map` takes it"""
    return {
        "disposition": [disposition_targets(x, class_of) for x in batch["input_ids"]]
    }


class Loader(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str | DictConfig = None,
        processed_data_home: pathlib.Path = None,
    ):
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

        self.inference_files = {
            s: str(f)
            for s in self.splits
            if (f := self.processed_data_home / f"{s}_for_inference.parquet").is_file()
        }

        self.for_inference = (
            (
                ds.load_dataset("parquet", data_files=self.inference_files)
                .rename_column("tokens_past", "input_ids")
                .select_columns(
                    ["input_ids"]
                    + (["s_elapsed_past"] if "time_based_rope" in self.cfg else [])
                    + (["hours_to_end_time_past"] if "tte" in self.heads else [])
                )
            )
            if self.inference_files
            else None
        )

    def get_train_data(self):
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset[self.splits[0]]
                .repeat(self.cfg.n_epochs)
                .shuffle(generator=self.rng),
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")

    def get_tuning_data(self):
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
