#!/usr/bin/env python3

"""tests for cotorra.configurable.Configurable"""

import pytest
from omegaconf import OmegaConf

from cotorra.configurable import Configurable
from cotorra.logger import Logger


class _WithDefault(Configurable):
    default_file = "training.yaml"


def test_no_default_and_no_config_file_yields_empty_cfg():
    cfg_obj = Configurable()
    assert cfg_obj.config_file is None
    assert OmegaConf.to_container(cfg_obj.cfg) == {}
    assert isinstance(cfg_obj.logger, Logger)


def test_default_file_is_loaded_when_no_override_given():
    cfg_obj = _WithDefault()
    assert cfg_obj.cfg.max_seq_len == 4096
    assert cfg_obj.cfg.run_name == "cotorra-tuning"
    assert "time_based_rope" in cfg_obj.cfg


def test_config_file_replaces_rather_than_merges_with_default(tmp_path):
    """a user config that omits a key must not inherit it from the default"""
    custom = tmp_path / "training.yaml"
    custom.write_text(OmegaConf.to_yaml({"max_seq_len": 32}))

    cfg_obj = _WithDefault(custom)
    assert cfg_obj.cfg.max_seq_len == 32
    assert "model" not in cfg_obj.cfg
    assert "time_based_rope" not in cfg_obj.cfg


def test_kwargs_override_config_file(tmp_path):
    custom = tmp_path / "training.yaml"
    custom.write_text(OmegaConf.to_yaml({"max_seq_len": 32, "n_epochs": 1}))

    cfg_obj = _WithDefault(custom, max_seq_len=64)
    assert cfg_obj.cfg.max_seq_len == 64
    assert cfg_obj.cfg.n_epochs == 1


def test_none_valued_kwargs_are_ignored(tmp_path):
    custom = tmp_path / "training.yaml"
    custom.write_text(OmegaConf.to_yaml({"max_seq_len": 32}))

    cfg_obj = _WithDefault(custom, max_seq_len=None)
    assert cfg_obj.cfg.max_seq_len == 32


def test_kwargs_can_add_new_nested_config_blocks():
    cfg_obj = _WithDefault(tuning_args={"n_trials": 3})
    assert cfg_obj.cfg.tuning_args.n_trials == 3
    # untouched keys from the default file are still present
    assert cfg_obj.cfg.tuning_args.backend == "optuna"


def test_a_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _WithDefault(tmp_path / "not-here.yaml")


def test_kwargs_alone_populate_a_class_without_a_default_file():
    cfg_obj = Configurable(max_seq_len=8)
    assert cfg_obj.cfg.max_seq_len == 8


def test_paths_are_expanded_and_resolved(tmp_path, monkeypatch):
    """`~`-relative config paths must work, not just absolute ones"""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "training.yaml").write_text(OmegaConf.to_yaml({"max_seq_len": 7}))
    assert _WithDefault("~/training.yaml").cfg.max_seq_len == 7


def test_overrides_change_existing_keys_with_values_read_as_yaml():
    cfg = _WithDefault(
        overrides=[
            "training_args.learning_rate=1e-4",
            "n_epochs=2",
            "tokens_of_interest=[RESP//imv, LABEL//*]",
            "training_args={eval_steps: 0.5}",
        ]
    ).cfg
    assert cfg.training_args.learning_rate == pytest.approx(1e-4)
    assert isinstance(cfg.training_args.learning_rate, float)
    assert cfg.n_epochs == 2
    assert list(cfg.tokens_of_interest) == ["RESP//imv", "LABEL//*"]
    # a block merges into the one there rather than replacing it
    assert cfg.training_args.eval_steps == 0.5
    assert cfg.training_args.save_steps == 0.1


def test_an_override_adds_a_key_not_in_the_config():
    cfg = _WithDefault(
        overrides=[
            "tnt_objective.mixture_components=8",
            "tte_objective=",
            "training_args={bf16: true}",
            "wandb.entity=x",
        ]
    ).cfg
    assert cfg.tnt_objective.mixture_components == 8
    # an empty block, which turns its head on with every default
    assert "tte_objective" in cfg and cfg.tte_objective is None
    assert cfg.training_args.bf16 is True
    assert cfg.training_args.learning_rate == pytest.approx(2e-4)
    assert cfg.wandb.entity == "x"


def test_a_leading_plus_as_hydra_writes_an_addition_is_ignored():
    cfg = _WithDefault(overrides=["+n_epochs=2", "++max_seq_len=8", "+run=x"]).cfg
    assert cfg.n_epochs == 2
    assert cfg.max_seq_len == 8
    assert cfg.run == "x"


def test_a_tilde_override_deletes_a_key_where_null_leaves_it_present(tmp_path):
    """
    features toggle on the presence of their block, so setting one to null
    leaves it on, and deleting it is how to turn it off
    """
    assert "time_based_rope" in _WithDefault(overrides=["time_based_rope=null"]).cfg
    cfg = _WithDefault(overrides=["~time_based_rope", "~training_args.use_cache"]).cfg
    assert "time_based_rope" not in cfg
    assert "use_cache" not in cfg.training_args
    # an empty block is present, and deletable
    custom = tmp_path / "training.yaml"
    custom.write_text("max_seq_len: 32\ntte_objective:\n")
    assert "tte_objective" not in _WithDefault(custom, overrides=["~tte_objective"]).cfg
    with pytest.raises(KeyError):
        _WithDefault(overrides=["~tte_objective"])


@pytest.mark.parametrize("override", ["n_epochs", "+", "++n_epochs"])
def test_an_override_without_a_value_is_refused_rather_than_read_as_null(override):
    with pytest.raises(ValueError, match="can't read the override"):
        _WithDefault(overrides=[override])


def test_an_interpolated_preset_can_be_swapped_in_and_then_edited():
    cfg = _WithDefault(
        overrides=["model=${model_presets.qwen_3}", "model.model_args.hidden_size=256"]
    ).cfg
    assert cfg.model.model_name == "Qwen/Qwen3-1.7B-Base"
    assert cfg.model.model_args.hidden_size == 256
    assert cfg.model.model_args.num_key_value_heads == 4
    # the edit lands on a copy of the preset, not the preset itself
    assert cfg.model_presets.qwen_3.model_args.hidden_size == 512


def test_overrides_apply_in_order_and_kwargs_override_them():
    assert _WithDefault(overrides=["n_epochs=2", "n_epochs=3"]).cfg.n_epochs == 3
    assert _WithDefault(overrides=["n_epochs=2"], n_epochs=5).cfg.n_epochs == 5


def test_a_loaded_config_is_taken_as_is_and_left_untouched():
    """as a `Trainer` hands its merged config to its `Loader`"""
    loaded = _WithDefault(overrides=["max_seq_len=8"]).cfg
    again = Configurable(loaded, overrides=["max_seq_len=16"])
    assert again.cfg.max_seq_len == 16
    assert again.cfg.n_epochs == loaded.n_epochs
    assert loaded.max_seq_len == 8
