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
    cfg_obj = _WithDefault(privacy_parameters={"noise_multiplier": 0.5})
    assert cfg_obj.cfg.privacy_parameters.noise_multiplier == 0.5
    # untouched keys from the default file are still present
    assert cfg_obj.cfg.privacy_parameters.max_grad_norm == 1.0


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
