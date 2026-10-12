#!/usr/bin/env python3

"""
logging
"""

import collections.abc
import logging

import numpy as np
import polars as pl
import torch as t
from rich.console import Console
from rich.logging import RichHandler
from transformers import PreTrainedModel

from cotorra.util import bootstrap_ci

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(500)


class Logger(logging.Logger):
    def __init__(self, name: str = __package__) -> None:
        super().__init__(name=name)
        self.setLevel(logging.INFO)
        self.handlers.clear()

        formatter = logging.Formatter(
            fmt="🦜 [%(asctime)s] %(message)s", datefmt="%H:%M:%S%Z"
        )
        ch = RichHandler(
            show_path=False, show_time=False, console=Console(width=200, soft_wrap=True)
        )
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False

    def summarize_trained_model(
        self,
        model: "PreTrainedModel",
        bos_token_id: int,
        reverse: collections.abc.Mapping[int, str],
        n_samp: int = 3,
        max_len: int = 2048,
    ) -> None:
        for i in range(n_samp):
            sample = model.generate(
                t.tensor([[bos_token_id]], dtype=t.int32).to(model.device),
                max_length=max_len,
                do_sample=True,
                top_k=len(reverse),
            )
            self.info(
                "Sample {}: {}".format(
                    i + 1, np.vectorize(reverse.get)(sample.cpu().numpy()).tolist()
                )
            )

    def summarize_preds(
        self, df_res: pl.LazyFrame, target_tokens: collections.abc.Iterable[str]
    ) -> None:
        cols = df_res.collect_schema().names()
        for tt in target_tokens:
            self.info("=" * 42)
            self.info(f"{tt=}")
            for method in ("mc", "scope", "reach", "rep"):
                if f"{tt}_{method}_score" in cols:
                    self.info(f"{method=}")
                    y_qual, y_true, y_score = (
                        df_res.select(
                            ~pl.col(f"{tt}_past"),
                            f"{tt}_future",
                            f"{tt}_{method}_score",
                        )
                        .collect()
                        .to_numpy()
                        .T
                    )
                    if (nf := np.sum(~np.isfinite(y_score[y_qual.astype(bool)]))) > 0:
                        self.warning(f"{nf} non-finite {method} estimates for {tt}.")
                    if v_rare := y_true[y_qual.astype(bool)].mean().item() < 0.001:
                        self.warning("using stratified sampling due to rarity")
                    for k, v in bootstrap_ci(
                        y_true[y_qual.astype(bool)],
                        np.nan_to_num(y_score)[y_qual.astype(bool)],
                        stratified=v_rare,
                    ).items():
                        self.info("{k:<7}: {v}".format(k=k, v=v.round(3)))


if __name__ == "__main__":
    self = Logger()
    self.info("Testing...")
    # breakpoint()
