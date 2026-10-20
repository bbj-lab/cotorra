#!/usr/bin/env python3

"""tests for cotorra.tuner.Tuner"""

import optuna
import pytest
from helpers import base_training_cfg, default_cfg, write_cfg
from omegaconf import OmegaConf
from transformers.trainer_utils import BestRun

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


def tuner_with(processed, tmp_path, tuning_args: dict, **kwargs) -> Tuner:
    """a `Tuner` whose config's `tuning_args` are `tuning_args`, writing under
    `tmp_path / "out"`"""
    cfg = base_training_cfg()
    cfg["tuning_args"] = tuning_args
    return Tuner(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=tmp_path / "out",
        **kwargs,
    )


def searched_with(tuner, monkeypatch) -> dict:
    """the keywords `Tuner.train` hands hf's search, which stops there"""
    searched = {}

    def search(**kwargs):
        searched.update(kwargs)
        raise InterruptedError

    monkeypatch.setattr(tuner.trainer, "hyperparameter_search", search)
    with pytest.raises(InterruptedError):
        tuner.train()
    return searched


def test_every_tuning_arg_but_the_space_reaches_the_search_as_it_is(
    processed, tmp_path, monkeypatch
):
    """`direction`, `backend` and `n_trials`, and any other key -- an optuna
    `timeout`, say -- as the keywords of hf's `hyperparameter_search`, beside
    the sampler built from `hp_space` and the objective"""
    tuning_args = {
        "direction": "maximize",
        "backend": "optuna",
        "n_trials": 3,
        "timeout": 600,
        "study_name": "a-study",
    }
    hp_space = {"weight_decay": {"type": "float", "low": 0.0, "high": 0.1}}
    tuner = tuner_with(processed, tmp_path, tuning_args | {"hp_space": hp_space})
    searched = searched_with(tuner, monkeypatch)
    assert set(searched) == set(tuning_args) | {"hp_space", "compute_objective"}
    assert {k: searched[k] for k in tuning_args} == tuning_args
    assert searched["compute_objective"] is Tuner.compute_objective


@pytest.mark.parametrize("how", ["absent", "empty"])
def test_without_a_space_the_search_samples_the_default_one(
    how, processed, tmp_path, monkeypatch
):
    """the learning rate and gradient accumulation, as before `hp_space` could
    be set -- whether the key is left out or left empty"""
    tuning_args = {"direction": "minimize", "backend": "optuna", "n_trials": 1}
    if how == "empty":
        tuning_args["hp_space"] = None
    searched = searched_with(tuner_with(processed, tmp_path, tuning_args), monkeypatch)
    study = optuna.create_study(direction="minimize")
    for _ in range(5):
        trial = study.ask()
        space = searched["hp_space"](trial)
        assert set(space) == set(DEFAULT_HP_SPACE)
        assert 1e-4 <= space["learning_rate"] <= 5e-4
        assert space["gradient_accumulation_steps"] in (1, 2, 3)
        assert trial.distributions["learning_rate"].log


def test_the_best_trials_hyperparameters_train_the_model_it_saves(
    processed, tmp_path, monkeypatch
):
    """set on hf's arguments before the final training, which then saves the
    model under mdl-<run_name>, beside the merged config as -tuning.yaml"""
    tuner = tuner_with(
        processed,
        tmp_path,
        {"direction": "minimize", "backend": "optuna", "n_trials": 1},
        overrides=["n_epochs=2"],
    )
    best = BestRun(
        run_id="1",
        objective=1.5,
        hyperparameters={"learning_rate": 3.3e-4, "gradient_accumulation_steps": 2},
    )
    monkeypatch.setattr(tuner.trainer, "hyperparameter_search", lambda **_: best)
    trained_with = []
    monkeypatch.setattr(
        tuner.trainer,
        "train",
        lambda **_: trained_with.append(
            (
                tuner.trainer.args.learning_rate,
                tuner.trainer.args.gradient_accumulation_steps,
            )
        ),
    )
    tuner.train()
    assert trained_with == [(3.3e-4, 2)]
    assert (tmp_path / "out" / "mdl-test-run" / "config.json").is_file()
    saved = OmegaConf.load(tmp_path / "out" / "mdl-test-run-tuning.yaml")
    assert saved == tuner.cfg and saved.n_epochs == 2


def test_tuning_verbosely_summarizes_the_tuned_model(processed, tmp_path, monkeypatch):
    tuner = tuner_with(
        processed,
        tmp_path,
        {"direction": "minimize", "backend": "optuna", "n_trials": 1},
    )
    best = BestRun(run_id="1", objective=1.0, hyperparameters={})
    monkeypatch.setattr(tuner.trainer, "hyperparameter_search", lambda **_: best)
    monkeypatch.setattr(tuner.trainer, "train", lambda **_: None)
    summaries = []
    monkeypatch.setattr(
        tuner.logger, "summarize_trained_model", lambda **kw: summaries.append(kw)
    )
    tuner.train()
    assert summaries == []
    tuner.train(verbose=True)
    [summary] = summaries
    lookup = tuner.tkzr_cfg.lookup
    assert summary["model"] is tuner.trainer.model
    assert summary["bos_token_id"] == lookup["BOS"]
    assert summary["reverse"] == {v: k for k, v in lookup.items()}


def test_optuna_hp_space_samples_an_int_on_its_step():
    study = optuna.create_study(direction="minimize")
    hp_space = {"warmup_steps": {"type": "int", "low": 1, "high": 9, "step": 4}}
    seen = {
        Tuner.optuna_hp_space(study.ask(), hp_space)["warmup_steps"] for _ in range(30)
    }
    assert seen <= {1, 5, 9} and len(seen) > 1


@pytest.mark.slow
def test_a_real_search_trains_the_final_model_with_hyperparameters_from_the_space(
    processed, tmp_path
):
    """two optuna trials over a categorical and an int field, then the final
    training, which runs with the best trial's choices"""
    hp_space = {
        "lr_scheduler_type": {"type": "categorical", "choices": ["constant", "cosine"]},
        "warmup_steps": {"type": "int", "low": 1, "high": 3},
    }
    tuning_args = {"direction": "minimize", "backend": "optuna", "n_trials": 2}
    tuner = tuner_with(processed, tmp_path, tuning_args | {"hp_space": hp_space})
    tuner.train()
    args = tuner.trainer.args
    assert args.lr_scheduler_type in ("constant", "cosine")
    assert args.warmup_steps in (1, 2, 3)
    assert (tmp_path / "out" / "mdl-test-run" / "config.json").is_file()
