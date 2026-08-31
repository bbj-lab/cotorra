#!/usr/bin/env python3

"""tests for cotorra.trainer_dp.TrainerDP"""

import opacus
import pytest
import torch as t
from helpers import base_training_cfg, write_cfg
from transformers import AutoModelForCausalLM

from cotorra.trainer_dp import TrainerDP


@pytest.fixture(scope="module")
def dp_trainer(processed, tmp_path_factory) -> TrainerDP:
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("dp-cfg") / "training.yaml", base_training_cfg()
    )
    out = tmp_path_factory.mktemp("dp-output")
    return TrainerDP(
        training_cfg=cfg_path,
        processed_data_home=processed,
        output_home=out,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )


def test_privacy_parameter_overrides_are_applied(dp_trainer):
    assert dp_trainer.cfg.privacy_parameters.noise_multiplier == 0.5
    assert dp_trainer.cfg.privacy_parameters.max_grad_norm == 1.0


def test_model_and_engine_are_opacus_wrapped(dp_trainer):
    assert isinstance(dp_trainer.model, opacus.GradSampleModule)
    assert isinstance(dp_trainer.privacy_engine, opacus.PrivacyEngine)
    assert dp_trainer.trainer.model is dp_trainer.model


def test_get_train_dataloader_returns_the_dp_wrapped_loader(dp_trainer):
    assert (
        dp_trainer.trainer.get_train_dataloader()
        is dp_trainer.trainer._dp_train_dataloader
    )


def test_training_step_rejects_gradient_accumulation_other_than_one(dp_trainer):
    original = dp_trainer.trainer.args.gradient_accumulation_steps
    dp_trainer.trainer.args.gradient_accumulation_steps = 2
    try:
        with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
            dp_trainer.trainer.training_step(dp_trainer.model, {})
    finally:
        dp_trainer.trainer.args.gradient_accumulation_steps = original


def test_default_gradient_accumulation_passes_the_guard(dp_trainer):
    """the fast test config leaves it at 1, the only value Opacus supports"""
    assert dp_trainer.trainer.args.gradient_accumulation_steps == 1


def test_saving_temporarily_unwraps_the_grad_sample_module(dp_trainer, tmp_path):
    """`_save` must see the plain model, and must put the wrapper back"""
    wrapped = dp_trainer.trainer.model
    dp_trainer.trainer._save(str(tmp_path))
    assert dp_trainer.trainer.model is wrapped
    assert (tmp_path / "config.json").is_file()
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert reloaded.config.vocab_size == len(dp_trainer.tkzr_cfg.lookup)


@pytest.mark.slow
def test_train_saves_an_unwrapped_reloadable_model(processed, tmp_path_factory):
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("dp-train-cfg") / "training.yaml", base_training_cfg()
    )
    out = tmp_path_factory.mktemp("dp-train-output")
    trainer = TrainerDP(
        training_cfg=cfg_path,
        processed_data_home=processed,
        output_home=out,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    trainer.train()

    model_dir = out / f"mdl-{trainer.run_name}"
    assert model_dir.is_dir()
    assert (model_dir / "config.json").is_file()
    assert (out / f"mdl-{trainer.run_name}-training.yaml").is_file()

    # reloading only succeeds cleanly if save unwrapped the GradSampleModule
    # (its "_module." prefixed keys would otherwise mismatch the plain model)
    reloaded = AutoModelForCausalLM.from_pretrained(model_dir)
    sample = t.tensor(
        [[trainer.tkzr_cfg.lookup["BOS"], trainer.tkzr_cfg.lookup["EOS"]]]
    )
    logits = reloaded(sample).logits
    assert t.isfinite(logits).all()
