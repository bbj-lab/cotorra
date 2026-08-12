#!/usr/bin/env python3

"""
train a model
"""

import os
import pathlib

import torch as t
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer as t_Trainer

from cotorra.basis_blended import (
    BasisBlendedCausalLM,
    BasisBlendedConfig,
    build_basis_vocab,
)
from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.loss import Loss


class TrainerWithCustomLoss(t_Trainer):
    def __init__(self, compute_loss_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.compute_loss_func is not None:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            extra = {k: inputs[k] for k in ("category_ids", "ranks") if k in inputs}
            loss = self.compute_loss_func(outputs, labels, **extra)
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)


class Trainer(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(training_cfg, **kwargs)

        self.processed_data_home, self.output_home = map(
            lambda p: pathlib.Path(p).expanduser().resolve(),
            [processed_data_home, output_home],
        )

        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

        # single source of truth for the collapsed basis vocabulary -- built once
        # here and reused by both model_init (below) and Loss, so the model's
        # embedding table and the loss's target category/basis-id assignment can
        # never disagree (see fuzzy_token_planning.md, "The collapsed basis
        # vocabulary")
        self.basis_vocab = (
            build_basis_vocab(self.tkzr_cfg, self.cfg.basis_blended_tokens.k)
            if "basis_blended_tokens" in self.cfg
            else None
        )
        if self.basis_vocab is not None:
            self._raw_to_category_t = t.tensor(
                self.basis_vocab["raw_to_category"], dtype=t.long
            )
            self._raw_to_collapsed_t = t.tensor(
                self.basis_vocab["raw_to_collapsed"], dtype=t.long
            )

        self.loss = (
            Loss(self.cfg, self.tkzr_cfg, basis_vocab=self.basis_vocab).custom_loss
            if self.cfg.custom_loss
            else None
        )
        self.run_name = self.cfg.get("run_name", self.cfg.wandb.get("run_name", ""))
        self.loader = Loader(training_cfg, self.processed_data_home)

        self.trainer = TrainerWithCustomLoss(
            model_init=self.model_init,
            data_collator=self.collate_fn,
            compute_loss_func=self.loss,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_home), **self.cfg.training_args
            ),
        )
        self.model = self.trainer.model

        os.environ["WANDB_PROJECT"] = self.cfg.get("wandb", {}).get(
            "project", "cotorra"
        )
        os.environ["WANDB_NAME"] = self.cfg.get("wandb", {}).get("run_name", "cotorra")

    def model_init(self):
        if self.basis_vocab is not None:
            inner_cfg = AutoConfig.from_pretrained(
                self.cfg.model.model_name,
                vocab_size=self.basis_vocab["vocab_size"],
                bos_token_id=self.basis_vocab["bos_token_id"],
                eos_token_id=self.basis_vocab["eos_token_id"],
                tie_word_embeddings=True,  # required -- see fuzzy_token_planning.md
                **self.cfg.model.model_args,
            )
            basis_cfg = BasisBlendedConfig(
                base_model_type=inner_cfg.model_type,
                base_config=inner_cfg.to_dict(),
                train_beta_params=self.cfg.basis_blended_tokens.get(
                    "train_beta_params", True
                ),
                train_importance_scale=self.cfg.basis_blended_tokens.get(
                    "train_importance_scale", True
                ),
                **self.basis_vocab,
            )
            mdl = BasisBlendedCausalLM(basis_cfg)
        else:
            conf_param = dict(
                vocab_size=len(self.tkzr_cfg.lookup),
                bos_token_id=self.tkzr_cfg.lookup.BOS,
                eos_token_id=self.tkzr_cfg.lookup.EOS,
            )
            config = AutoConfig.from_pretrained(
                self.cfg.model.model_name, **conf_param, **self.cfg.model.model_args
            )
            mdl = AutoModelForCausalLM.from_config(config)
        self.logger.info(
            "Loaded model {name} with {num} params ({dtype}).".format(
                name=self.cfg.model.model_name,
                num=sum(p.numel() for p in mdl.parameters()),
                dtype=next(mdl.parameters()).dtype,
            )
        )

        return mdl

    def collate_fn(self, batch):
        input_ids = t.stack([x["input_ids"] for x in batch])
        labels = input_ids
        extra = {}

        if self.basis_vocab is not None:
            # must read category_ids/ranks off the *raw* cocoa token ids before
            # input_ids gets remapped into collapsed-vocab space below
            extra["category_ids"] = self._raw_to_category_t[input_ids]
            extra["ranks"] = t.stack([x["exact_ranks"] for x in batch]).to(t.float32)
            input_ids = self._raw_to_collapsed_t[input_ids]
            labels = input_ids

        if "time_based_rope" in self.cfg:
            p_ids = (
                t.stack([x["s_elapsed"] for x in batch])
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
            extra["position_ids"] = p_ids

        return {"input_ids": input_ids, "labels": labels, **extra}

    @property
    def bos_token_id(self) -> int:
        return (
            self.basis_vocab["bos_token_id"]
            if self.basis_vocab is not None
            else self.tkzr_cfg.lookup["BOS"]
        )

    @property
    def reverse_lookup(self) -> dict:
        lookup = (
            self.basis_vocab["basis_lookup"]
            if self.basis_vocab is not None
            else self.tkzr_cfg.lookup
        )
        return {v: k for k, v in lookup.items()}

    def train(self, resume_from_checkpoint: bool = False, verbose: bool = False):
        if resume_from_checkpoint:
            try:
                self.trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                self.logger.warning(f"Encountered {e} on resume from checkpoint.")
                self.trainer.train()
        else:
            self.trainer.train()

        self.trainer.model.save_pretrained(self.output_home / f"mdl-{self.run_name}")

        with open(self.output_home / f"mdl-{self.run_name}-training.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if verbose:
            self.logger.summarize_trained_model(
                model=self.trainer.model,
                bos_token_id=self.bos_token_id,
                reverse=self.reverse_lookup,
            )


if __name__ == "__main__":
    self = Trainer(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    self.train(verbose=True)
    # breakpoint()
