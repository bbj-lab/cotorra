#!/usr/bin/env python3

"""
extract representations up to the thresholds created by the cocoa winnower
"""

import math
import pathlib

import numpy as np
import torch as t
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

from cotorra.basis_blended import BasisBlendedConfig  # import registers Auto* class
from cotorra.configurable import Configurable
from cotorra.loader import Loader


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
        self.loader = Loader(extraction_cfg, self.processed_data_home)
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

        # basis-blended-ness (k, category assignment, etc.) is read from the
        # loaded checkpoint's own config, not from extraction.yaml -- it's
        # fixed at train time and must be loaded faithfully, not re-derived.
        if isinstance(self.model.config, BasisBlendedConfig):
            self._raw_to_category_t = t.tensor(
                self.model.config.raw_to_category, dtype=t.long
            ).to(self.device)
            self._raw_to_collapsed_t = t.tensor(
                self.model.config.raw_to_collapsed, dtype=t.long
            ).to(self.device)
            # pad_sequence below pads *raw* cocoa token ids (batch["input_ids"]
            # is still raw-vocab space at this point), so the pad value must
            # also be raw-space -- self.model.config.pad_token_id is already
            # in collapsed space and would index raw_to_collapsed_t wrongly.
            self._raw_pad_value = self.tkzr_cfg.lookup["EOS"]
        else:
            self._raw_to_category_t = None
            self._raw_pad_value = self.model.config.pad_token_id
        self.ds = None

    def collate_fn(self, batch):
        ml = t.tensor(self.cfg.get("extract", {}).get("max_len", 4096))
        input_ids = pad_sequence(
            [x[:ml] for x in batch["input_ids"]],
            batch_first=True,
            padding_value=self._raw_pad_value,
        ).to(self.model.device)

        past_suffix = self.cfg.get("past_suffix", "")
        extra = {}
        if self._raw_to_category_t is not None:
            extra["category_ids"] = self._raw_to_category_t[input_ids]
            rank_column = (
                self.cfg.basis_blended_tokens.get("rank_column", "exact_ranks")
                + f"_past{past_suffix}"
            )
            extra["ranks"] = pad_sequence(
                [
                    t.as_tensor(x[:ml], dtype=t.float32)
                    for x in batch[rank_column]
                ],
                batch_first=True,
                padding_value=0.0,
            ).to(self.model.device)
            input_ids = self._raw_to_collapsed_t[input_ids]

        if "time_based_rope" in self.cfg:
            p_ids = (
                pad_sequence(
                    [x[:ml] for x in batch[f"s_elapsed_past{past_suffix}"]],
                    batch_first=True,
                    padding_value=self.model.config.pad_token_id,
                ).to(self.model.device)
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
        else:
            p_ids = None
        return {"input_ids": input_ids, "position_ids": p_ids, **extra}

    def extract_final(self, batch, all_times: bool = False):
        collated = self.collate_fn(batch)
        first_eos = t.where(
            (hits := (collated["input_ids"] == self.model.config.eos_token_id)).any(
                dim=-1
            ),
            hits.long().argmax(dim=-1)
            - 1,  # -1 to get the last token before break point
            collated["input_ids"].shape[-1] - 1,
        )
        with t.inference_mode():
            hidden_states = self.model(
                **collated, output_hidden_states=True
            ).hidden_states
            # k=1 (default) is the last hidden layer (unchanged behavior);
            # k=2 the second-to-last, etc. -- see extract()'s hidden_state_k
            # docstring for why more intermediate layers might separate
            # models better than the final one alone.
            k = self.cfg.get("extract", {}).get("hidden_state_k", 1)
            features = hidden_states[-k]
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
        else:
            batch["features"] = (
                features[t.arange(len(first_eos)), first_eos].half().cpu().numpy()
            )
        return batch

    def extract(self, all_times: bool = False):
        """
        extract.hidden_state_k (default 1): which hidden layer to pull
        features from, counting from the end -- k=1 is the last layer
        (output.hidden_states[-1], the original/default behavior), k=2 the
        second-to-last, and so on. HF's output_hidden_states=True returns
        num_hidden_layers+1 tensors (index 0 is the embedding output, not a
        transformer layer), so k must be in [1, num_hidden_layers+1].
        Motivation: the final layer's representation is shaped by next-
        token-prediction specifically and may compress away distinctions
        between models that an earlier, less task-specialized layer still
        carries -- worth comparing empirically against k=1 for rep-based
        scoring, where the goal is separating models/outcomes rather than
        predicting the immediate next token.

        k=1's output filename is unchanged from before this option existed
        (features-<split>-<model_name>.parquet), so every existing
        features/scores file and RepBasedScorer's own glob
        (features-<split>*-<model_name>.parquet) keep working untouched.
        k>1 gets a distinct -hsk<k> suffix on the model-name portion of the
        filename specifically so it can NEVER silently collide with (or
        get glob-matched alongside) the k=1 files for the same model --
        learned the hard way this session that an output filename which
        doesn't encode what produced it (rep-based-score's estimator type)
        makes it trivial to silently overwrite/co-mingle incompatible
        results under the same name.
        """
        a = "-all" if all_times else ""
        shard_size = self.cfg.get("extract", {}).get("shard_size", None)
        k = self.cfg.get("extract", {}).get("hidden_state_k", 1)
        max_k = self.model.config.num_hidden_layers + 1
        assert 1 <= k <= max_k, (
            f"extract.hidden_state_k={k} out of range -- must be in [1, {max_k}]"
            f" (num_hidden_layers={self.model.config.num_hidden_layers} + 1 for"
            " the embedding output HF includes at hidden_states[0])"
        )
        model_tag = self.model_home.name if k == 1 else f"{self.model_home.name}-hsk{k}"
        ds = self.loader.for_inference.with_format("torch")
        for split, dset in ds.items():
            n = math.ceil(len(dset) / shard_size) if shard_size else 1
            for i in range(n):
                index = f"-{i:05d}-of-{n:05d}" if n > 1 else ""
                dset.shard(num_shards=n, index=i).map(
                    lambda batch: self.extract_final(batch, all_times=all_times),
                    batched=True,
                    batch_size=self.cfg.get("extract", {}).get("batch_size", 8),
                    load_from_cache_file=False,  # disable caching
                ).to_parquet(
                    self.output_home / f"features{a}-{split}{index}-{model_tag}.parquet"
                )


if __name__ == "__main__":
    self = Extractor()
    self.extract()

    # batch_eg = self.loader.dataset.with_format("torch")["training"].batch(8)[0]
    # collated_eg = self.collate_fn(batch_eg)
    # fin_rep = self.extract_final(batch_eg)

    # breakpoint()
