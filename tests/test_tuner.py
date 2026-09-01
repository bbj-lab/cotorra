#!/usr/bin/env python3

"""tests for cotorra.tuner.Tuner"""

import optuna
import pytest
from helpers import base_training_cfg, write_cfg

from cotorra.tuner import Tuner


def test_optuna_hp_space_suggests_expected_ranges():
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    space = Tuner.optuna_hp_space(trial)
    assert set(space.keys()) == {"learning_rate", "gradient_accumulation_steps"}
    assert 1e-4 <= space["learning_rate"] <= 5e-4
    assert 1 <= space["gradient_accumulation_steps"] <= 3


def test_optuna_hp_space_only_suggests_keys_training_arguments_accepts():
    from transformers import TrainingArguments

    study = optuna.create_study(direction="minimize")
    space = Tuner.optuna_hp_space(study.ask())
    for key in space:
        assert (
            hasattr(TrainingArguments, key)
            or key in TrainingArguments.__dataclass_fields__
        )


def test_tuner_inherits_the_trainer_wiring(processed, tmp_path_factory):
    """`Tuner` only adds the search; model/loss/collator must be the same"""
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("tuner-wiring-cfg") / "training.yaml",
        base_training_cfg(),
    )
    tuner = Tuner(
        training_cfg=cfg_path,
        processed_data_home=processed,
        output_home=tmp_path_factory.mktemp("tuner-wiring-output"),
    )
    assert tuner.trainer.compute_loss_func is tuner.loss
    assert tuner.trainer.data_collator == tuner.collate_fn
    assert tuner.cfg.tuning_args.n_trials == 1


@pytest.mark.slow
def test_train_runs_one_trial_and_saves_model_and_tuning_config(
    processed, tmp_path_factory
):
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("tuner-cfg") / "training.yaml", base_training_cfg()
    )
    out = tmp_path_factory.mktemp("tuner-output")
    tuner = Tuner(training_cfg=cfg_path, processed_data_home=processed, output_home=out)

    tuner.train()

    model_dir = out / f"mdl-{tuner.run_name}"
    assert model_dir.is_dir()
    assert (model_dir / "config.json").is_file()
    assert (out / f"mdl-{tuner.run_name}-tuning.yaml").is_file()
