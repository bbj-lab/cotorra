#!/usr/bin/env python3

"""tests for cotorra.tuner.Tuner"""

import optuna
import pytest
from helpers import base_training_cfg, default_cfg, write_cfg

from cotorra.tuner import DEFAULT_HP_SPACE, Tuner


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


def test_the_packaged_config_searches_the_default_space():
    """so a run without a config file searches what a config without
    `hp_space` does"""
    assert default_cfg("cotorra", "training")["tuning_args"]["hp_space"] == (
        DEFAULT_HP_SPACE
    )


def test_optuna_hp_space_samples_a_configured_space():
    hp_space = {
        "weight_decay": {"type": "float", "low": 0.0, "high": 0.1, "step": 0.05},
        "warmup_steps": {"type": "int", "low": 2, "high": 8, "log": True},
        "lr_scheduler_type": {"type": "categorical", "choices": ["linear", "cosine"]},
    }
    study = optuna.create_study(direction="minimize")
    for _ in range(5):
        space = Tuner.optuna_hp_space(study.ask(), hp_space)
        assert set(space.keys()) == set(hp_space)
        assert space["weight_decay"] in (0.0, 0.05, 0.1)
        assert isinstance(space["warmup_steps"], int)
        assert 2 <= space["warmup_steps"] <= 8
        assert space["lr_scheduler_type"] in ("linear", "cosine")


def _tuner(processed, tmp_path_factory, hp_space) -> Tuner:
    cfg = base_training_cfg()
    cfg["tuning_args"]["hp_space"] = hp_space
    return Tuner(
        training_cfg=write_cfg(
            tmp_path_factory.mktemp("tuner-hp-space-cfg") / "training.yaml", cfg
        ),
        processed_data_home=processed,
        output_home=tmp_path_factory.mktemp("tuner-hp-space-output"),
    )


def test_train_searches_the_configured_space(processed, tmp_path_factory, monkeypatch):
    """`hp_space` reaches the search as its sampler, not as an optuna kwarg"""
    hp_space = {"weight_decay": {"type": "float", "low": 0.0, "high": 0.1}}
    tuner = _tuner(processed, tmp_path_factory, hp_space)
    searched = {}

    def search(**kwargs):
        searched.update(kwargs)
        raise InterruptedError

    monkeypatch.setattr(tuner.trainer, "hyperparameter_search", search)
    with pytest.raises(InterruptedError):
        tuner.train()
    # still in the config, which the tuning yaml saved beside the model records
    assert "hp_space" in tuner.cfg.tuning_args
    assert searched["n_trials"] == 1
    trial = optuna.create_study(direction="minimize").ask()
    assert set(searched["hp_space"](trial)) == {"weight_decay"}


def test_train_rejects_a_field_training_arguments_lacks(
    processed, tmp_path_factory, monkeypatch
):
    hp_space = {"learning_rte": {"type": "float", "low": 1e-4, "high": 5e-4}}
    tuner = _tuner(processed, tmp_path_factory, hp_space)
    # it fails before searching, rather than once the trials have all run
    monkeypatch.setattr(tuner.trainer, "hyperparameter_search", pytest.fail)
    with pytest.raises(ValueError, match="learning_rte"):
        tuner.train()


def test_the_search_minimizes_the_eval_loss_alone():
    """not hf's default objective, which sums every metric evaluation reports
    once there's more than `eval_loss`, as there is with the custom loss's terms"""
    metrics = {
        "eval_loss": 3.0,
        "eval_x_ent_loss": 2.0,
        "eval_quantile_token_loss": 0.02,
        "epoch": 1.0,
    }
    assert Tuner.compute_objective(metrics) == 3.0


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
