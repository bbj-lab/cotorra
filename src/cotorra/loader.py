#!/usr/bin/env python3

"""
load data and prepare for training / evaluation
"""

import pathlib

import datasets as ds
import numpy as np
import polars as pl
from omegaconf import OmegaConf

from cotorra.configurable import Configurable
from cotorra.util import batched_iter


class Loader(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path = None,
    ):
        super().__init__(training_cfg)
        self.rng = np.random.default_rng(42)
        self.processed_data_home = processed_data_home
        self.tokenizer_info = OmegaConf.load(
            self.processed_data_home / "tokenizer.yaml"
        )
        self.splits: tuple = ("train", "tuning", "held_out")

        # tokens_times_file: override to e.g. "tokens_times_no_label.parquet"
        # (see cocoa's Tokenizer.build_no_label_frame / exclude_tokens) to
        # train on a timeline with certain tokens (e.g. LABEL//*) removed,
        # without touching the default label-inclusive path. The per-split
        # cache filenames are derived from this same stem so switching
        # between source files can never silently reuse a stale/mismatched
        # cache from a different one.
        tt_filename = self.cfg.get("tokens_times_file", "tokens_times.parquet")
        tt_stem = pathlib.Path(tt_filename).stem
        tt_all = self.processed_data_home / tt_filename
        assert tt_all.is_file(), FileNotFoundError(
            f"Expected token and time data at {tt_all}, but not found."
        )

        tt_split = {
            s: self.processed_data_home / f"{s}_{tt_stem}.parquet"
            for s in self.splits
        }
        if not all(s.is_file() for s in tt_split.values()) or any(
            tt_all.stat().st_mtime > s.stat().st_mtime for s in tt_split.values()
        ):  # pull out training and tuning sets if not already done
            # or if tokens have been updated
            self.subject_splits = pl.scan_parquet(
                self.processed_data_home / "subject_splits.parquet"
            )
            self.tokens_times = pl.scan_parquet(tt_all).with_columns(
                s_elapsed=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                )
            )
            to_split = self.tokens_times.join(self.subject_splits, on="subject_id")
            for s in self.splits:
                to_split.filter(pl.col("split") == s).drop("split").sink_parquet(
                    tt_split[s]
                )

        self.dataset = (
            ds.load_dataset(
                "parquet", data_files={s: str(tt_split[s]) for s in self.splits}
            )
            .rename_column("tokens", "input_ids")
            .select_columns(
                ["input_ids"]
                + (["s_elapsed"] if "time_based_rope" in self.cfg else [])
                + (
                    [self.cfg.basis_blended_tokens.get("rank_column", "exact_ranks")]
                    if "basis_blended_tokens" in self.cfg
                    else []
                )
            )
        )

        self.inference_files = {
            s: str(f)
            for s in self.splits
            if (f := self.processed_data_home / f"{s}_for_inference.parquet").is_file()
        }

        # past_suffix: "" (default) reads tokens_past/s_elapsed_past/*_past as
        # always; "_no_label" reads tokens_past_no_label/etc instead (see
        # cocoa's Winnower.add_outcome_flags) -- the label-inclusive columns
        # are always present regardless, so this is purely an opt-in choice
        # of which past-context a given experiment trains/extracts on.
        past_suffix = self.cfg.get("past_suffix", "")
        self.for_inference = (
            (
                ds.load_dataset("parquet", data_files=self.inference_files)
                .rename_column(f"tokens_past{past_suffix}", "input_ids")
                .select_columns(
                    ["input_ids"]
                    + (
                        [f"s_elapsed_past{past_suffix}"]
                        if "time_based_rope" in self.cfg
                        else []
                    )
                    + (
                        [
                            self.cfg.basis_blended_tokens.get(
                                "rank_column", "exact_ranks"
                            )
                            + f"_past{past_suffix}"
                        ]
                        if "basis_blended_tokens" in self.cfg
                        else []
                    )
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
