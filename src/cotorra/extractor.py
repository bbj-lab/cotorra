#!/usr/bin/env python3

"""
extract representations up to the thresholds created by the cocoa winnower
"""

import collections.abc
import math
import pathlib

import numpy as np
import torch as t
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

import cotorra.model  # noqa: F401 -- registers `cotorra` with the auto classes
from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.model import CotorraForCausalLM

# the secondary heads whose predictions can join the features, by their names in
# `HEADS`: what to call each, should the model lack it, and the columns it adds,
# read off the last hidden states wherever the features are and kept in float32
# -- the time heads' predictions in hours, the disposition head's probability of
# each of its `classes`, in their order
HEAD_COLUMNS = {
    "tte": (
        "time-to-event",
        lambda model, hidden: {
            "time_to_event": model.heads["tte"].predict(hidden).float().expm1()
        },
    ),
    "tnt": (
        "time-to-next-token",
        lambda model, hidden: {
            "time_to_next_token": model.predict_hours_to_next_token(hidden)
        },
    ),
    "disposition": (
        "discharge-disposition",
        lambda model, hidden: dict(
            zip(
                [f"{c}_prob" for c in model.heads["disposition"].classes],
                model.heads["disposition"]
                .predict(hidden)
                .float()
                .softmax(dim=-1)
                .unbind(dim=-1),
            )
        ),
    ),
}


class Extractor(Configurable):
    """load a model and extract representations from it"""

    default_file = "extraction.yaml"

    def __init__(
        self,
        extraction_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        model_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(extraction_cfg, **kwargs)
        self.processed_data_home, self.model_home = map(
            lambda x: pathlib.Path(x).expanduser().resolve(),
            (processed_data_home, model_home),
        )
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.loader = Loader(self.cfg, self.processed_data_home)
        self.device = (
            "cuda"
            if t.cuda.is_available()
            else "mps"
            if t.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_home)
        self.model.to(self.device).eval()
        if not isinstance(self.model.config.pad_token_id, int):
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.ds = None

    def collate_fn(self, batch):
        ml = t.tensor(self.cfg.get("extract", {}).get("max_len", 4096))
        input_ids = pad_sequence(
            [x[:ml] for x in batch["input_ids"]],
            batch_first=True,
            # EOS, whatever the model's own pad id (gemma's is 0), since the first
            # EOS is where each row's features are read
            padding_value=self.model.config.eos_token_id,
        ).to(self.model.device)
        if "time_based_rope" in self.cfg:
            p_ids = (
                pad_sequence(
                    [x[:ml] for x in batch["s_elapsed_past"]],
                    batch_first=True,
                    padding_value=self.model.config.pad_token_id,
                ).to(self.model.device)
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
        else:
            p_ids = None
        return {"input_ids": input_ids, "position_ids": p_ids}

    def _check_heads(self, heads: collections.abc.Iterable[str]) -> list[str]:
        """`heads`, names in `HEAD_COLUMNS` (or just one), without repeats, once
        the model is known to carry each"""
        heads = list(dict.fromkeys([heads] if isinstance(heads, str) else heads))
        if unknown := [name for name in heads if name not in HEAD_COLUMNS]:
            raise ValueError(
                f"no secondary head is named {unknown}; the heads are "
                f"{list(HEAD_COLUMNS)}"
            )
        carried = self.model.heads if isinstance(self.model, CotorraForCausalLM) else {}
        for name in heads:
            if name not in carried:
                raise ValueError(
                    f"this model carries no {HEAD_COLUMNS[name][0]} head; train it "
                    f"with a `{name}_objective` to have one"
                )
        return heads

    def extract_final(
        self, batch, all_times: bool = False, heads: collections.abc.Iterable[str] = ()
    ):
        """the last hidden state at each row's last real token -- or with
        `all_times` at every position up to it -- as `features`, plus the columns
        `HEAD_COLUMNS` gives for each of `heads` from the same positions"""
        heads = self._check_heads(heads)
        collated = self.collate_fn(batch)
        first_eos = t.where(
            (hits := (collated["input_ids"] == self.model.config.eos_token_id)).any(
                dim=-1
            ),
            hits.long().argmax(dim=-1)
            - 1,  # -1 to get the last token before break point
            collated["input_ids"].shape[-1] - 1,
        )
        columns = dict()
        with t.inference_mode():
            features = self.model(**collated, output_hidden_states=True).hidden_states[
                -1
            ]  # last hidden layer
            if heads:  # read where the features are, from the model's own dtype
                positions = (
                    features[
                        t.arange(features.shape[1], device=features.device)
                        <= first_eos[:, None]
                    ]
                    if all_times
                    else features[t.arange(len(first_eos)), first_eos]
                )
                for name in heads:
                    _, read = HEAD_COLUMNS[name]
                    for column, values in read(self.model, positions).items():
                        columns[column] = values.float().cpu().numpy()
        if all_times:
            features = features.half().cpu().numpy()
            collated = np.full(
                shape=(features.shape[0], self.cfg.max_seq_len, features.shape[-1]),
                fill_value=np.nan,
            )
            lengths = first_eos.cpu().numpy()[:, None]
            out_mask = np.arange(collated.shape[1]) <= lengths
            feat_mask = np.arange(features.shape[1]) <= lengths
            collated[out_mask] = features[feat_mask]
            batch["features"] = collated
            for column, values in columns.items():  # laid out as the features are
                filled = np.full(out_mask.shape, np.nan, dtype=np.float32)
                filled[out_mask] = values
                batch[column] = filled
        else:
            batch["features"] = (
                features[t.arange(len(first_eos)), first_eos].half().cpu().numpy()
            )
            batch.update(columns)
        return batch

    def extract(
        self, all_times: bool = False, heads: collections.abc.Iterable[str] = ()
    ):
        """write each split's features to `output_home`, with the columns of each
        of `heads`; a head the model lacks is refused before a table is written"""
        heads = self._check_heads(heads)
        a = "-all" if all_times else ""
        shard_size = self.cfg.get("extract", {}).get("shard_size", None)
        ds = self.loader.for_inference.with_format("torch")
        for split, dset in ds.items():
            n = math.ceil(len(dset) / shard_size) if shard_size else 1
            for i in range(n):
                index = f"-{i:05d}-of-{n:05d}" if n > 1 else ""
                dset.shard(num_shards=n, index=i).map(
                    lambda batch: self.extract_final(
                        batch, all_times=all_times, heads=heads
                    ),
                    batched=True,
                    batch_size=self.cfg.get("extract", {}).get("batch_size", 8),
                    load_from_cache_file=False,  # disable caching
                ).to_parquet(
                    self.output_home
                    / f"features{a}-{split}{index}-{self.model_home.name}.parquet"
                )


if __name__ == "__main__":
    self = Extractor()
    self.extract()

    # batch_eg = self.loader.dataset.with_format("torch")["training"].batch(8)[0]
    # collated_eg = self.collate_fn(batch_eg)
    # fin_rep = self.extract_final(batch_eg)

    # breakpoint()
