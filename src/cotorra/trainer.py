#!/usr/bin/env python3

"""
train a model
"""

import collections.abc
import os
import pathlib
import typing
import warnings

import torch as t
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainingArguments,
)
from transformers import Trainer as t_Trainer
from transformers.trainer_pt_utils import find_batch_size, nested_gather
from transformers.trainer_utils import EvalLoopOutput, TrainOutput

from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.loss import Loss
from cotorra.model import HEADS, CotorraConfig, head_options


class TrainerWithCustomLoss(t_Trainer):
    def __init__(
        self,
        compute_loss_func: collections.abc.Callable[..., t.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        # the terms a custom loss records, summed over the rows scored since they
        # were last reported, training's apart from evaluation's: `log` reports
        # training's means beside hf's `loss`, and `evaluation_loop` reports
        # evaluation's, over the whole eval set, beside its `eval_loss`
        self.term_sums, self.term_rows = dict(), dict()
        for mode in ("train", "eval"):
            self.reset_terms(mode)

    def reset_terms(self, mode: str) -> None:
        self.term_sums[mode] = collections.defaultdict(float)
        self.term_rows[mode] = 0

    def term_means(self, mode: str, prefix: str = "") -> dict[str, float]:
        """the means of `mode`'s terms since they were last reported, over every
        process, as hf gathers its own losses; reporting them resets them"""
        if not (rows := self.term_rows[mode]):
            return dict()
        names = list(self.term_sums[mode])
        totals = t.stack(
            [
                t.as_tensor(v, dtype=t.float32).to(self.args.device)
                for v in (*self.term_sums[mode].values(), rows)
            ]
        )
        totals = nested_gather(totals, self.args.parallel_mode)
        totals = totals.view(-1, len(names) + 1).sum(dim=0)
        self.reset_terms(mode)
        return {
            f"{prefix}{name}": (total / totals[-1]).item()
            for name, total in zip(names, totals[:-1])
        }

    def compute_loss(
        self,
        model: t.nn.Module,
        inputs: dict[str, t.Tensor | typing.Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> t.Tensor | tuple[t.Tensor, typing.Any]:
        if self.compute_loss_func is not None:
            inputs = dict(inputs)
            labels = inputs.pop("labels", None)
            if getattr(self, "doc_masking", False):
                # masking silently does not happen with a KV cache, so refuse
                # rather than train unmasked
                assert not getattr(model.config, "use_cache", False), (
                    "doc_masking batch reached a model with use_cache=True"
                )
            outputs = model(**inputs)
            # a `compute_loss_func` fills `terms` as `Loss.custom_loss` does, or
            # takes it in a `**kwargs` and leaves it empty
            terms = dict()
            loss = self.compute_loss_func(outputs, labels, terms=terms)
            # weighted by the batch's rows, as hf weights `eval_loss`, counted off
            # the batch itself, which needn't carry `labels`
            mode = "train" if model.training else "eval"
            rows = find_batch_size(inputs) or 0
            for name, term in terms.items():
                self.term_sums[mode][name] += term * rows
            self.term_rows[mode] += rows
            if model.training:
                # with a custom loss set, hf's `training_step` leaves averaging
                # over gradient accumulation to it -- which a per-batch mean
                # doesn't do -- so the accumulated micro-batches would sum; this
                # is the division hf's stock path makes (set only inside its
                # training loop, hence the default)
                loss = loss / getattr(self, "current_gradient_accumulation_steps", 1)
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

    def train(self, *args, **kwargs) -> TrainOutput:
        # a search's trials share this trainer, so one trial's last steps would
        # otherwise open the next one's first log
        self.reset_terms("train")
        return super().train(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # hf's `loss` averages the steps since its last log, as these do
        if "loss" in logs:
            logs = logs | self.term_means("train")
        super().log(logs, start_time)

    def evaluation_loop(
        self,
        dataloader: t.utils.data.DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # whatever was scored outside this evaluation stays out of its means
        self.reset_terms("eval")
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        output.metrics.update(self.term_means("eval", prefix=f"{metric_key_prefix}_"))
        return output


class Trainer(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        self.require(processed_data_home=processed_data_home, output_home=output_home)
        super().__init__(training_cfg, **kwargs)
        if "label_weighted_loss" in self.cfg:
            # a FutureWarning, which python shows by default, rather than a
            # DeprecationWarning, which it hides unless raised from `__main__`
            warnings.warn(
                "`label_weighted_loss` is deprecated: weighting the cross-entropy by "
                "its target makes the model predict those tokens more often. Use "
                "`balanced_toi_loss` instead, which takes the same "
                "`tokens_of_interest` and weights its term by `bce_weight`",
                FutureWarning,
                stacklevel=2,
            )
        self.processed_data_home, self.output_home = map(
            lambda p: pathlib.Path(p).expanduser().resolve(),
            [processed_data_home, output_home],
        )

        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        # one `<name>_objective` block per secondary head, in any combination
        self.head_options = head_options(self.cfg, self.tkzr_cfg.lookup)
        self.loss = (
            Loss(self.cfg, self.tkzr_cfg).custom_loss
            if self.cfg.get("custom_loss", False)
            else None
        )
        # optional, and None when left empty
        wandb_cfg = self.cfg.get("wandb") or {}
        self.run_name = self.cfg.get("run_name", wandb_cfg.get("run_name", ""))
        # the merged config, not the file it came from, so overrides reach it too
        if self.cfg.get("doc_masking", False):
            # without these, the flattened batch would silently train with
            # plain causal attention across every packed record
            assert self.cfg.get("packing", "concat") == "bfd", (
                "doc_masking needs packing: bfd"
            )
            assert self.cfg.get("attn_implementation") == "flash_attention_2", (
                "doc_masking needs attn_implementation: flash_attention_2"
            )
            assert not self.head_options and "time_based_rope" not in self.cfg, (
                "doc_masking does not yet support secondary heads or time_based_rope"
            )
        self.loader = Loader(self.cfg, self.processed_data_home)

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
        self.trainer.doc_masking = bool(self.cfg.get("doc_masking", False))

        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "cotorra")
        os.environ["WANDB_NAME"] = wandb_cfg.get("run_name", "cotorra")

    def model_init(self) -> PreTrainedModel:
        conf_param = dict(
            vocab_size=len(self.tkzr_cfg.lookup),
            bos_token_id=self.tkzr_cfg.lookup.BOS,
            eos_token_id=self.tkzr_cfg.lookup.EOS,
        )
        config = AutoConfig.from_pretrained(
            self.cfg.model.model_name, **conf_param, **self.cfg.model.model_args
        )
        if self.cfg.get("doc_masking", False):
            # with a KV cache, transformers skips flash-attention's varlen path
            # and the packed records attend to each other. HF applies
            # training_args.use_cache only to the model it builds in __init__,
            # not to the one model_init rebuilds in train()
            config.use_cache = False
        if self.head_options:
            config = CotorraConfig(text_config=config, heads=self.head_options)
        # attn_implementation: e.g. flash_attention_2, which doc_masking needs;
        # unset keeps transformers' default (sdpa)
        attn = self.cfg.get("attn_implementation", None)
        mdl = AutoModelForCausalLM.from_config(
            config, **({"attn_implementation": attn} if attn else {})
        )
        self.logger.info(
            "Loaded model {name} with {num} params ({dtype}).".format(
                name="{}{}".format(
                    self.cfg.model.model_name,
                    f" (with {', '.join(self.head_options)} heads)"
                    if self.head_options
                    else "",
                ),
                num=sum(p.numel() for p in mdl.parameters()),
                dtype=next(mdl.parameters()).dtype,
            )
        )

        return mdl

    def collate_fn(self, batch: list[dict[str, t.Tensor]]) -> dict[str, t.Tensor]:
        input_ids = t.stack([x["input_ids"] for x in batch])
        f_set = {"input_ids": input_ids, "labels": input_ids}
        if "time_based_rope" in self.cfg:
            p_ids = (
                t.stack([x["s_elapsed"] for x in batch])
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
            f_set["position_ids"] = p_ids
        for name, options in self.head_options.items():
            target, weights = HEADS[name].target, HEADS[name].target_weight
            f_set[target] = t.stack([x[target] for x in batch])
            if weights is not None and options.get("per_record", False):
                # the weights the loader gives each record, beside its target
                f_set[weights] = t.stack([x[weights] for x in batch])
        if "pad_mask" in batch[0]:
            # padding left by `packing: bfd` (util.best_fit_packed_iter): its
            # positions are never targets, which CrossEntropyLoss skips at -100
            pad = t.stack([x["pad_mask"] for x in batch]).bool()
            f_set["labels"] = f_set["labels"].masked_fill(pad, -100)
        if self.cfg.get("doc_masking", False) and "doc_pos" in batch[0]:
            # cross-record attention masking for `packing: bfd` batches (train
            # and eval): flatten to one row whose position_ids restart at every
            # packed piece, which transformers' flash_attention_2 turns into
            # varlen segments (it only does so for a batch of ONE row). A
            # piece's first token is never a target -- it would be predicted
            # from the previous, unrelated piece.
            pos = t.stack([x["doc_pos"] for x in batch])
            return {
                "input_ids": input_ids.reshape(1, -1),
                "labels": f_set["labels"].masked_fill(pos == 0, -100).reshape(1, -1),
                "position_ids": pos.reshape(1, -1),
            }
        return f_set

    def train(
        self, resume_from_checkpoint: bool = False, verbose: bool = False
    ) -> None:
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
