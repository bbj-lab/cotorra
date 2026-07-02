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

from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.loss import Loss
from cotorra.packing import admission_relative_position_ids, mask_cross_admission_labels


class TrainerWithCustomLoss(t_Trainer):
    def __init__(self, compute_loss_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.compute_loss_func is not None:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = self.compute_loss_func(outputs, labels)
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
        self.loss = (
            Loss(self.cfg, self.tkzr_cfg).custom_loss if self.cfg.custom_loss else None
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
        collated = {"input_ids": input_ids, "labels": input_ids}
        token_idx = t.arange(input_ids.shape[-1], device=input_ids.device)
        if "admission_ids" in batch[0]:
            admission_ids = t.stack([x["admission_ids"] for x in batch])
            collated["labels"] = mask_cross_admission_labels(input_ids, admission_ids)
            token_idx = admission_relative_position_ids(admission_ids)
            collated["position_ids"] = token_idx
        if "time_based_rope" in self.cfg:
            p_ids = (
                t.stack([x["s_elapsed"] for x in batch])
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += token_idx.to(dtype=p_ids.dtype)
            collated["position_ids"] = p_ids
        return collated

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
                bos_token_id=self.tkzr_cfg.lookup["BOS"],
                reverse={v: k for k, v in self.tkzr_cfg.lookup.items()},
            )


if __name__ == "__main__":
    self = Trainer(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    self.train(verbose=True)
    # breakpoint()
