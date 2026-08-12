#!/usr/bin/env python3

"""
differentially privately train a model
"""

import contextlib
import pathlib
import warnings

import opacus
from omegaconf import OmegaConf
from transformers import TrainingArguments

from cotorra.trainer import Trainer, TrainerWithCustomLoss

warnings.filterwarnings("ignore", message=".*Full backward hook.*")


class TrainerWithCustomLossDP(TrainerWithCustomLoss):
    """Opacus-compatible trainer that can use a DP-wrapped train DataLoader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dp_train_dataloader = None

    def set_dp_train_dataloader(self, dataloader):
        self._dp_train_dataloader = dataloader

    def get_train_dataloader(self):
        if self._dp_train_dataloader is not None:
            return self._dp_train_dataloader
        return super().get_train_dataloader()

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "Opacus DP training currently requires gradient_accumulation_steps=1."
            )

        self.optimizer.zero_grad()
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        self.accelerator.backward(loss)
        return loss.detach()

    @contextlib.contextmanager
    def _unwrapped_model(self):
        """Temporarily expose the inner module, bypassing the DP wrapper."""
        wrapped, self.model = self.model, getattr(self.model, "_module", self.model)
        try:
            yield
        finally:
            self.model = wrapped

    def _save(self, output_dir, state_dict=None):
        with self._unwrapped_model():
            super()._save(output_dir, state_dict)

    def _load_best_model(self):
        with self._unwrapped_model():
            super()._load_best_model()


class TrainerDP(Trainer):
    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        noise_multiplier: float | None = None,
        max_grad_norm: float | None = None,
        **kwargs,
    ):
        super().__init__(
            training_cfg=training_cfg,
            processed_data_home=processed_data_home,
            output_home=output_home,
            privacy_parameters={
                k: v
                for k, v in {
                    "noise_multiplier": noise_multiplier,
                    "max_grad_norm": max_grad_norm,
                }.items()
                if v is not None
            },
            **kwargs,
        )

        self.trainer = TrainerWithCustomLossDP(
            model=self.model,
            data_collator=self.collate_fn,
            compute_loss_func=self.loss,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_home), **self.cfg.training_args
            ),
        )

        self.trainer.create_optimizer()

        # HuggingFace Trainer calls get_train_dataloader() internally, so we must
        # ensure get_train_dataloader() returns the Opacus-wrapped DataLoader.
        privacy_engine = opacus.PrivacyEngine()
        noise_multiplier = self.cfg.get("privacy_parameters", {}).get(
            "noise_multiplier", 1.0
        )
        max_grad_norm = self.cfg.get("privacy_parameters", {}).get("max_grad_norm", 1.0)
        self.logger.info(
            f"Making private with {noise_multiplier=} and {max_grad_norm=}"
        )
        dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
            module=self.trainer.model,
            optimizer=self.trainer.optimizer,
            data_loader=self.trainer.get_train_dataloader(),
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )
        self.trainer.model = dp_model
        self.trainer.optimizer = dp_optimizer
        self.trainer.set_dp_train_dataloader(dp_dataloader)
        self.model = self.trainer.model
        self.privacy_engine = privacy_engine

    def train(self, verbose=False):
        """Override train to properly handle saving the DP-wrapped model."""
        self.trainer.train()

        self.logger.info("For (epsilon, delta)-differential privacy:")
        for delta in [1e-5, 1e-4, 1e-3]:
            self.logger.info(
                "delta={delta} gives epsilon={epsilon:.3e}".format(
                    delta=delta,
                    epsilon=self.privacy_engine.accountant.get_epsilon(delta=delta),
                )
            )

        # Unwrap the model from GradSampleModule before saving
        unwrapped_model = self.trainer.model._module
        unwrapped_model.save_pretrained(self.output_home / f"mdl-{self.run_name}")

        with open(self.output_home / f"mdl-{self.run_name}-training.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if verbose:
            self.logger.summarize_trained_model(
                model=unwrapped_model,
                bos_token_id=self.bos_token_id,
                reverse=self.reverse_lookup,
            )


if __name__ == "__main__":
    self = TrainerDP(
        processed_data_home="./processed/ucmc",
        output_home="./output/ucmc",
        noise_multiplier=0.1,
    )
    self.train(verbose=True)
    # breakpoint()
