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


def token_nll_bits(
    logits: t.Tensor, input_ids: t.Tensor, n_tokens: collections.abc.Sequence[int]
) -> list[np.ndarray]:
    """
    the negative log2-likelihood, in bits, of each of a row's `n_tokens` tokens
    given those before it, from `logits`, the model's next-token logits over the
    padded (and maybe truncated) `input_ids`: a float32 array as long as the row,
    zero at its first token, the BOS every timeline opens with, which carries no
    information, and NaN past the tokens it was cut to, which the model never
    saw. Worked out in float32 whatever the model's dtype, a row at a time, so
    that no intermediate outgrows one row's logits
    """
    kept = [min(n, input_ids.shape[-1]) for n in n_tokens]
    nats = t.full(input_ids.shape, t.nan, dtype=t.float32, device=logits.device)
    nats[:, 0] = 0.0
    for i, n in enumerate(kept):
        if n > 1:
            nats[i, 1:n] = t.nn.functional.cross_entropy(
                logits[i, : n - 1].float(), input_ids[i, 1:n], reduction="none"
            )
    rows = (nats / math.log(2)).cpu().numpy()
    bits = [np.full(n, np.nan, dtype=np.float32) for n in n_tokens]
    for row, b, n in zip(rows, bits, kept):
        b[:n] = row[:n]
    return bits


class Extractor(Configurable):
    """load a model and extract representations from it"""

    default_file = "extraction.yaml"

    def __init__(
        self,
        extraction_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        model_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        self.require(processed_data_home=processed_data_home, model_home=model_home)
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

    def collate_fn(self, batch: dict) -> dict[str, t.Tensor | None]:
        ml = t.tensor((self.cfg.get("extract") or {}).get("max_len", 4096))
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
            # as `Trainer.collate_fn` passes it: without a mask, a checkpoint
            # saved with `use_cache: false` would start a new sequence wherever
            # these ids don't step by exactly 1, as at every gap between
            # timestamps, and attend only within one
            mask = t.ones_like(input_ids)
        else:
            p_ids = mask = None
        return {"input_ids": input_ids, "position_ids": p_ids, "attention_mask": mask}

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
        self,
        batch: dict,
        all_times: bool = False,
        heads: collections.abc.Iterable[str] = (),
        token_nll: bool = False,
    ) -> dict:
        """the last hidden state at each row's last real token -- or with
        `all_times` at every position up to it -- as `features`, plus the columns
        `HEAD_COLUMNS` gives for each of `heads` from the same positions, and with
        `token_nll` a `token_nll_bits` list as long as each row's tokens, whatever
        `all_times`: each one's negative log2-likelihood given those before it"""
        heads = self._check_heads(heads)
        n_tokens = [len(x) for x in batch["input_ids"]]
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
            outputs = self.model(**collated, output_hidden_states=True)
            features = outputs.hidden_states[-1]  # last hidden layer
            nll = (
                token_nll_bits(outputs.logits, collated["input_ids"], n_tokens)
                if token_nll
                else None
            )
            del outputs  # every layer's hidden states and the logits, not held on to
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
        if nll is not None:
            batch["token_nll_bits"] = nll
        return batch

    def extract(
        self,
        all_times: bool = False,
        heads: collections.abc.Iterable[str] = (),
        token_nll: bool = False,
    ) -> None:
        """write each split's features to `output_home`, with the columns of each
        of `heads` and with `token_nll` each token's negative log2-likelihood; a
        head the model lacks is refused before a table is written"""
        heads = self._check_heads(heads)
        if self.loader.for_inference is None:
            raise FileNotFoundError(
                f"found no inference tables (`<split>_for_inference.parquet`, for "
                f"any of {list(self.loader.splits)}) in {self.processed_data_home} "
                "to extract from"
            )
        a = "-all" if all_times else ""
        extract_cfg = self.cfg.get("extract") or {}  # None when left empty
        shard_size = extract_cfg.get("shard_size", None)
        ds = self.loader.for_inference.with_format("torch")
        for split, dset in ds.items():
            n = math.ceil(len(dset) / shard_size) if shard_size else 1
            for i in range(n):
                index = f"-{i:05d}-of-{n:05d}" if n > 1 else ""
                dset.shard(num_shards=n, index=i).map(
                    lambda batch: self.extract_final(
                        batch, all_times=all_times, heads=heads, token_nll=token_nll
                    ),
                    batched=True,
                    batch_size=extract_cfg.get("batch_size", 8),
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
