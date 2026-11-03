#!/usr/bin/env python3

"""
train a model
"""

import collections.abc
import functools
import os
import pathlib
import typing
import warnings

import datasets as ds
import numpy as np
import torch as t
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainingArguments,
)
from transformers import Trainer as t_Trainer
from transformers.trainer_pt_utils import find_batch_size, nested_gather
from transformers.trainer_utils import EvalLoopOutput, TrainOutput, seed_worker
from transformers.utils.generic import is_flash_attention_requested

from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.loss import Loss
from cotorra.model import HEADS, CotorraConfig, head_options
from cotorra.util import token_budget_batches


class TokenBudgetBatches(t.utils.data.Sampler):
    """
    `packing: pad`'s batches for a `DataLoader`'s `batch_sampler`, each as many
    rows as fit `budget` tokens once padded (`util.token_budget_batches`), so
    that every step trains on about as many tokens. For training, `seed`ed:
    drawn from a generator of its own rather than torch's global one, so that
    every trial of a search draws the same batches, and redrawn for each epoch
    hf's loop sets, so that a run resumed mid-epoch picks up the batches it left
    off in. Unseeded, for evaluation: the rows in their own order
    """

    def __init__(
        self,
        lengths: collections.abc.Sequence[int],
        budget: int,
        seed: int | None = None,
        n_passes: int = 1,
    ) -> None:
        super().__init__()
        self.lengths, self.budget = np.asarray(lengths), budget
        self.seed, self.n_passes = seed, n_passes
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = None if self.seed is None else np.random.default_rng([self.seed, epoch])
        self.batches = token_budget_batches(
            self.lengths, self.budget, rng, n_passes=self.n_passes
        )

    def __iter__(self) -> collections.abc.Iterator[list[int]]:
        return (list(batch) for batch in self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def row_lengths(dataset: ds.Dataset) -> list[int]:
    """each row's tokens, read off the arrow table rather than row by row"""
    import pyarrow.compute as pc

    return (
        pc.list_value_length(dataset.with_format("arrow")["input_ids"])
        .to_numpy()
        .tolist()
    )


class TrainerWithCustomLoss(t_Trainer):
    def __init__(
        self,
        compute_loss_func: collections.abc.Callable[..., t.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        # the terms a custom loss records, summed over the batches scored since
        # they were last reported, weighted by their scoring positions for eval
        # and by rows for training's logs
        self.term_sums, self.term_weights, self.term_denominators = (
            dict(),
            dict(),
            dict(),
        )
        for mode in ("train", "eval"):
            self.reset_terms(mode)

    def reset_terms(self, mode: str) -> None:
        self.term_sums[mode] = collections.defaultdict(float)
        self.term_weights[mode] = 0
        self.term_denominators[mode] = collections.defaultdict(float)

    def tally(
        self,
        mode: str,
        terms: dict[str, t.Tensor],
        inputs: dict[str, t.Tensor | typing.Any],
        labels: t.Tensor | None,
        denominators: dict[str, t.Tensor] | None = None,
    ) -> None:
        """adds a batch's `terms` and denominators to `mode`'s sums. By default,
        terms are weighted by rows, or under `packing: pad` by the tokens the lm
        scores; a custom objective can supply each term's own denominator. Left on
        the device, so that no batch waits on the host"""
        if getattr(self, "pad_rows", False) and labels is not None:
            weight = (labels[..., 1:] != -100).sum()
        else:
            weight = find_batch_size(inputs) or 0
        for name, term in terms.items():
            denominator = (
                weight if denominators is None else denominators.get(name, weight)
            )
            self.term_sums[mode][name] += term * denominator
            self.term_denominators[mode][name] += denominator
        self.term_weights[mode] += weight

    def term_means(self, mode: str, prefix: str = "") -> dict[str, float]:
        """the means of `mode`'s terms since they were last reported, over every
        process, as hf gathers its own losses; reporting them resets them"""
        if not self.term_sums[mode]:
            return dict()
        names = list(self.term_sums[mode])
        totals = t.stack(
            [
                t.as_tensor(v, dtype=t.float32).to(self.args.device)
                for v in (
                    *self.term_sums[mode].values(),
                    *(self.term_denominators[mode][name] for name in names),
                )
            ]
        )
        totals = nested_gather(totals, self.args.parallel_mode)
        totals = totals.view(-1, 2 * len(names)).sum(dim=0)
        self.reset_terms(mode)
        return {
            f"{prefix}{name}": (
                total / denominator if denominator else total.new_zeros(())
            ).item()
            for name, total, denominator in zip(
                names, totals[: len(names)], totals[len(names) :]
            )
        }

    def evaluation_term_denominators(
        self,
        terms: dict[str, t.Tensor],
        inputs: dict[str, t.Tensor | typing.Any],
        labels: t.Tensor | None,
    ) -> dict[str, t.Tensor]:
        """the positions each configured loss term averages over"""
        loss = getattr(self.compute_loss_func, "__self__", None)
        if not isinstance(loss, Loss) or labels is None:
            return {}

        shifted = labels[..., 1:]
        scored = shifted != -100
        count = scored.sum()
        denominators = {
            name: count
            for name in (
                "x_ent_loss",
                "balanced_toi_loss",
            )
            if name in terms
        }
        if "label_weighted_loss" in terms:
            class_weights = loss.weights.to(labels.device)[shifted.clamp(min=0)]
            denominators["label_weighted_loss"] = (class_weights * scored).sum()
        if "quantile_token_loss" in terms:
            categories = loss.label_to_cat.to(labels.device)[shifted.clamp(min=0)]
            denominators["quantile_token_loss"] = (scored & (categories >= 0)).sum()

        for name in loss.heads:
            term_name = f"{name}_loss"
            target = inputs.get(HEADS[name].target)
            if term_name not in terms or target is None:
                continue
            target = target.to(labels.device)
            if name == "disposition":
                valid = target != -100
            elif name == "tte":
                valid = t.isfinite(target.clamp(min=0))
            else:
                valid = t.isfinite(target) & (target >= 0)
            denominators[term_name] = valid.sum()
        return denominators

    def custom_loss_mean(self, means: dict[str, float]) -> float:
        """the configured custom objective, combining its dataset-level means"""
        loss = getattr(self.compute_loss_func, "__self__", None)
        if not isinstance(loss, Loss):
            raise TypeError("custom_loss_mean requires cotorra.loss.Loss")
        base = (
            "label_weighted_loss"
            if "label_weighted_loss" in loss.cfg
            else "x_ent_loss"
        )
        total = means.get(base, 0.0)
        if "balanced_toi_loss" in loss.cfg:
            total += (
                loss.cfg.balanced_toi_loss.get("bce_weight", 1.0)
                * means.get("balanced_toi_loss", 0.0)
            )
        if "quantile_token_loss" in loss.cfg:
            total += (
                loss.cfg.quantile_token_loss.qt_weight
                * means.get("quantile_token_loss", 0.0)
            )
        for name, options in loss.heads.items():
            total += options["weight"] * means.get(f"{name}_loss", 0.0)
        return total

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
            mode = "train" if model.training else "eval"
            loss_owner = getattr(self.compute_loss_func, "__self__", None)
            if (
                mode == "eval"
                and getattr(self, "pad_rows", False)
                and not isinstance(loss_owner, Loss)
            ):
                # `eval_loss` itself, by tokens (see `evaluation_loop`)
                terms["loss"] = loss.detach()
            denominators = (
                self.evaluation_term_denominators(terms, inputs, labels)
                if mode == "eval"
                else None
            )
            self.tally(mode, terms, inputs, labels, denominators)
            if model.training:
                # with a custom loss set, hf's `training_step` leaves averaging
                # over gradient accumulation to it -- which a per-batch mean
                # doesn't do -- so the accumulated micro-batches would sum. This
                # weights each micro-batch alike, where hf's stock loss weights
                # each scored token alike: the same while every micro-batch scores
                # as many tokens, as concatenated rows do, `packing: pad`'s, filled
                # to a budget of tokens, about do, and `bfd`'s needn't (set only
                # inside its training loop, hence the default)
                loss = loss / getattr(self, "current_gradient_accumulation_steps", 1)
            return (loss, outputs) if return_outputs else loss
        labels = inputs.get("labels")  # which hf's own loss may pop
        result = super().compute_loss(model, inputs, return_outputs, **kwargs)
        if not model.training and getattr(self, "pad_rows", False):
            # hf's own loss, a mean over its batch's tokens, weighted by them too
            # for `eval_loss` (see `evaluation_loop`); a replica's each, under
            # `nn.DataParallel`, averaged as hf averages them
            loss = result[0] if return_outputs else result
            self.tally("eval", {"loss": loss.detach().mean()}, inputs, labels)
        return result

    def token_budget(self, is_training: bool) -> int:
        """
        the tokens each of `packing: pad`'s batches holds at most, padding
        included: `tokens_per_batch`, for training and evaluation alike, or
        `per_device_train_batch_size` (`per_device_eval_batch_size`) rows of
        `max_seq_len`, the most a batch of any other packing holds. Read off the
        arguments as they are when a dataloader is built, so that a search's
        trials follow a batch size it tunes. Per device, as hf's batch sizes are,
        so that `nn.DataParallel`, splitting each batch's rows across the gpus,
        gets a budget for each. A budget of fewer than about 4 rows of
        `max_seq_len` lets a row just over half of it take a batch alone, half
        empty, its tokens weighing up to twice the rest
        (`util.token_budget_batches`); of 4 or more, a batch of rows of about one
        length fills to some 80% of it at least
        """
        rows = (
            self.args.per_device_train_batch_size
            if is_training
            else self.args.per_device_eval_batch_size
        )
        budget = self.tokens_per_batch or rows * self.max_seq_len
        return budget * max(1, self.args.n_gpu)

    def _get_dataloader(
        self,
        dataset: t.utils.data.Dataset,
        description: str,
        batch_size: int,
        sampler_fn: collections.abc.Callable | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> t.utils.data.DataLoader:
        # `packing: pad`'s rows, batched by a budget of tokens rather than by a
        # count of rows (`TokenBudgetBatches`), which hf's own dataloader can't
        # do: it batches `batch_size` rows at a time, whatever sampler picks them.
        # Otherwise as hf builds it (both 5.5.4 and 5.19 build every dataloader
        # here), its `drop_last` aside: no batch is dropped
        if not (getattr(self, "pad_rows", False) and isinstance(dataset, ds.Dataset)):
            return super()._get_dataloader(
                dataset,
                description,
                batch_size,
                sampler_fn,
                is_training,
                dataloader_key,
            )
        dataset = self._remove_unused_columns(dataset, description=description)
        seed = self.args.seed if self.args.data_seed is None else self.args.data_seed
        workers = self.args.dataloader_num_workers
        params = {
            "batch_sampler": TokenBudgetBatches(
                row_lengths(dataset),
                self.token_budget(is_training),
                seed=seed if is_training else None,
                n_passes=self.n_passes if is_training else 1,
            ),
            "collate_fn": self.data_collator,
            "num_workers": workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            # newer hf's only
            "multiprocessing_context": getattr(
                self.args, "dataloader_multiprocessing_context", None
            ),
            "in_order": getattr(self.args, "dataloader_in_order", True),
        }
        if is_training:
            params["worker_init_fn"] = functools.partial(
                seed_worker, num_workers=workers, rank=self.args.process_index
            )
        dataloader = self.accelerator.prepare(
            t.utils.data.DataLoader(dataset, **params)
        )
        # kept for later evaluations, as hf keeps its own, with persistent workers
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            self._eval_dataloaders = getattr(self, "_eval_dataloaders", dict()) | {
                dataloader_key: dataloader
            }
        return dataloader

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
        # `eval_loss` picks the best checkpoint and is what `cotorra tune`
        # minimizes; for the built-in custom loss, combine its dataset-level
        # term means rather than weighting every batch mean by LM-token counts
        means = self.term_means("eval")
        output.metrics.update(
            {f"{metric_key_prefix}_{name}": mean for name, mean in means.items()}
        )
        if isinstance(getattr(self.compute_loss_func, "__self__", None), Loss):
            output.metrics[f"{metric_key_prefix}_loss"] = self.custom_loss_mean(means)
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
        if packing == "pad":
            # its batches are filled to a budget of tokens, whatever hf's sampler
            # would have drawn: by a count of rows, drawn at random or in turn,
            # each of a short record's tokens would weigh far above a long one's
            # (`util.token_budget_batches`); hf's `group_by_length` is what the
            # budget does anyway
            strategy = self.cfg.training_args.get("train_sampling_strategy")
            if strategy not in (None, "group_by_length"):
                raise ValueError(
                    "`packing: pad` fills each batch to a budget of tokens, of rows "
                    "of about one length, which `training_args.train_sampling_"
                    f"strategy: {strategy}` can't change; leave it unset"
                )
            budget = self.cfg.get("tokens_per_batch")
            if budget is not None and budget < self.cfg.max_seq_len:
                raise ValueError(
                    f"`tokens_per_batch: {budget}` can't hold a row of "
                    f"`max_seq_len: {self.cfg.max_seq_len}` tokens"
                )
        elif self.cfg.get("tokens_per_batch") is not None:
            raise ValueError(
                "`tokens_per_batch` needs `packing: pad`, whose batches it fills; "
                "every other packing's rows are `max_seq_len` tokens long, batched "
                "by `training_args.per_device_train_batch_size`"
            )
        if self.cfg.get("doc_masking", False):
            # each record attends only to itself through position ids restarting
            # at every packed piece, which bfd's `doc_pos` provides; time-based
            # position ids would take their place
            if packing != "bfd":
                raise ValueError(
                    "`doc_masking` needs `packing: bfd`"
                    + (
                        "; `packing: pad` keeps each record in a row of its own "
                        "without it"
                        if packing == "pad"
                        else ""
                    )
                )
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
        # `packing: pad`'s rows, padded by the batch, are batched by a budget of
        # tokens, each of the loader's `n_epochs` passes apart
        self.trainer.pad_rows = packing == "pad"
        if self.trainer.pad_rows:
            if "time_based_rope" in self.cfg and is_flash_attention_requested(
                config=self.model.config
            ):
                raise ValueError(
                    "`packing: pad` with `time_based_rope` does not support flash "
                    "attention: singleton batches can split attention at timestamp "
                    "jumps; use sdpa or eager attention instead"
                )
            if self.trainer.args.world_size > 1:
                # accelerate shards a batch sampler's batches across processes,
                # and how it evens out batches of varying size varies by version
                raise ValueError(
                    "`packing: pad` doesn't yet train across processes (`torchrun`, "
                    "`accelerate launch`); run a single process"
                )
            self.trainer.n_passes = self.cfg.n_epochs
            self.trainer.tokens_per_batch = self.cfg.get("tokens_per_batch")
            self.trainer.max_seq_len = self.cfg.max_seq_len
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
        # `packing: pad`'s rows (util.record_chunks_iter) each hold a record at
        # its own length, padded here, on the right, to the batch's longest. A
        # causal model's tokens never attend to the padding after them, so only
        # the targets need masking, below; every other packing's rows are of one
        # length already
        pad_rows = self.cfg.get("packing", "concat") == "pad"

        def column(k: str) -> t.Tensor:
            rows = [x[k] for x in batch]
            return pad_sequence(rows, batch_first=True) if pad_rows else t.stack(rows)

        input_ids = column("input_ids")
        if pad_rows:
            n = t.tensor([len(x["input_ids"]) for x in batch])
            pad = t.arange(input_ids.shape[-1]) >= n[:, None]
        else:
            # padding left by `packing: bfd` (util.best_fit_packed_iter), if any
            pad = column("pad_mask").bool() if "pad_mask" in batch[0] else None
        f_set = {"input_ids": input_ids, "labels": input_ids}
        if "time_based_rope" in self.cfg:
            p_ids = column("s_elapsed") / self.cfg.time_based_rope.sec_per_pos_id
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
            f_set["position_ids"] = p_ids
            # every token attends to all those before it: given neither a mask nor
            # a KV cache (which gradient checkpointing turns off), transformers
            # starts a new packed sequence wherever these ids don't step by exactly
            # 1, as at every gap between timestamps. Flash attention drops a mask
            # of ones, though, and still reads a one-row batch's ids that way.
            # Ones over `packing: pad`'s padding too: the padding comes after
            # every token that counts, which causal attention keeps from seeing
            # it, and sdpa, given a mask of ones, keeps its causal fast path
            f_set["attention_mask"] = t.ones_like(input_ids)
        for name, options in self.head_options.items():
            target, weights = HEADS[name].target, HEADS[name].target_weight
            f_set[target] = column(target)
            if weights is not None and options.get("per_record", False):
                # the weights the loader gives each record, beside its target
                f_set[weights] = column(weights)
        if pad is not None:
            # padding, `packing: bfd`'s or `pad`'s, is never a target, neither
            # the lm's (-100, which every loss skips) nor a head's (nan, or -100
            # for a class). Nothing else needs masking: a piece ends short of its
            # row only where its record ends (a split record's other chunks each
            # fill a row), and a record's last token has no time to the next, so
            # a mixture time-to-next-token head never conditions a gap on another
            # record's token, or on the padding
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
