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
from transformers.utils.generic import is_flash_attention_requested

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
                # rather than train unmasked; asked of the model itself, which
                # DDP and DataParallel wrap
                assert not getattr(
                    self.accelerator.unwrap_model(model).config, "use_cache", False
                ), "doc_masking batch reached a model with use_cache=True"
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
                # doesn't do -- so the accumulated micro-batches would sum. This
                # weights each micro-batch alike, where hf's stock loss weights
                # each scored token alike: the same while every micro-batch scores
                # as many tokens, as concatenated rows do and `packing: bfd`'s
                # needn't (set only inside its training loop, hence the default)
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

    def _report_to_hp_search(
        self, trial: typing.Any, step: int, metrics: dict[str, float]
    ) -> None:
        # Optuna's pruner compares trials at equal reported steps. Report the
        # eval number (1, 2, ... = 10%, 20%, ... of this run) instead of the
        # global step, so runs with different epochs are compared at equal
        # fractions of their own schedule, not at equal step counts.
        every = getattr(self.state, "eval_steps", None)
        if every:
            step = step // every
        super()._report_to_hp_search(trial, step, metrics)


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
        packing = self.cfg.get("packing", "concat")
        if packing == "bfd" and self.cfg.training_args.get(
            "remove_unused_columns", True
        ):
            # hf's default, which drops every column the model's `forward` doesn't
            # name: padding that loses its `pad_mask` trains as targets, silently
            raise ValueError(
                "`packing: bfd` needs `training_args.remove_unused_columns: false`, "
                "or hf drops the `pad_mask` and `doc_pos` columns that keep its "
                "padding out of the loss"
            )
        if self.cfg.get("doc_masking", False):
            # each record attends only to itself through position ids restarting
            # at every packed piece, which bfd's `doc_pos` provides; time-based
            # position ids would take their place
            if packing != "bfd":
                raise ValueError("`doc_masking` needs `packing: bfd`")
            if self.head_options or "time_based_rope" in self.cfg:
                raise ValueError(
                    "`doc_masking` does not yet support secondary heads or "
                    "`time_based_rope`"
                )
        if self.cfg.training_args.get("label_smoothing_factor", 0):
            # hf smooths only its own loss: a custom loss replaces it, and for a
            # model with heads hf drops their terms and, the class missing from
            # the causal-lm names it shifts labels for, scores each token
            # against itself
            if self.loss is not None:
                warnings.warn(
                    "`training_args.label_smoothing_factor` has no effect with "
                    "`custom_loss: true`, whose loss replaces the one hf smooths",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.head_options:
                raise ValueError(
                    "`training_args.label_smoothing_factor` with a secondary head "
                    "and `custom_loss: false` would train the heads not at all and "
                    "score each token against itself; leave it at 0"
                )
        # the merged config, not the file it came from, so overrides reach it too
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
        if (
            self.trainer.doc_masking
            and self.trainer.args.n_gpu > 1
            and is_flash_attention_requested(config=self.model.config)
        ):
            # the batch, flattened into one row for flash attention, would go to
            # a single replica whole, and hf's stock loss be scaled for them all
            raise ValueError(
                "`doc_masking` with flash attention flattens each batch into a single "
                "row, which `nn.DataParallel` can't split across gpus; launch a "
                "process per gpu instead (`torchrun`, or `accelerate launch`)"
            )

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
            # transformers reads restarting position ids as separate sequences
            # only without a KV cache, which the model otherwise builds even in
            # training, and the packed records would attend to each other. HF
            # applies training_args.use_cache only to the model it builds in
            # __init__, not to the one model_init rebuilds in train()
            config.use_cache = False
        if self.head_options:
            config = CotorraConfig(text_config=config, heads=self.head_options)
        # attn_implementation: e.g. eager or, with flash-attn installed,
        # flash_attention_2; unset keeps transformers' default (sdpa)
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
            # every token attends to all those before it: given neither a mask nor
            # a KV cache (which gradient checkpointing turns off), transformers
            # starts a new packed sequence wherever these ids don't step by exactly
            # 1, as at every gap between timestamps. Flash attention drops a mask
            # of ones, though, and still reads a one-row batch's ids that way
            f_set["attention_mask"] = t.ones_like(input_ids)
        for name, options in self.head_options.items():
            target, weights = HEADS[name].target, HEADS[name].target_weight
            f_set[target] = t.stack([x[target] for x in batch])
            if weights is not None and options.get("per_record", False):
                # the weights the loader gives each record, beside its target
                f_set[weights] = t.stack([x[weights] for x in batch])
        if "pad_mask" in batch[0]:
            # padding left by `packing: bfd` (util.best_fit_packed_iter) is never
            # a target, neither the lm's (-100, which every loss skips) nor a
            # head's (nan, or -100 for a class). Nothing else needs masking: a
            # piece ends short of its row only where its record ends (a split
            # record's other chunks each fill a row), and a record's last token
            # has no time to the next, so a mixture time-to-next-token head never
            # conditions a gap on another record's token
            pad = t.stack([x["pad_mask"] for x in batch]).bool()
            f_set["labels"] = f_set["labels"].masked_fill(pad, -100)
            for name in self.head_options:
                target = HEADS[name].target
                f_set[target] = f_set[target].masked_fill(
                    pad, t.nan if f_set[target].is_floating_point() else -100
                )
        if self.cfg.get("doc_masking", False):
            # each packed piece attends only to itself, by position ids restarting
            # at every one, which transformers -- given no attention mask or KV
            # cache -- reads as separate sequences: a block-diagonal causal mask
            # per row for sdpa and eager attention, and for flash attention,
            # whose varlen path takes only a batch of one row, once the batch is
            # flattened into one -- read off the model, so that one built with it
            # through `model.model_args` counts too. A piece's first token is
            # never a target: it would be predicted from the previous, unrelated
            # piece
            pos = t.stack([x["doc_pos"] for x in batch])
            f_set["labels"] = f_set["labels"].masked_fill(pos == 0, -100)
            f_set["position_ids"] = pos
            if is_flash_attention_requested(config=self.trainer.model.config):
                f_set = {k: v.reshape(1, -1) for k, v in f_set.items()}
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
