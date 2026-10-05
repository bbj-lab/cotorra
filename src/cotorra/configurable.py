#!/usr/bin/env python3

"""
configurable class with overridable defaults
"""

import collections.abc
import copy
import importlib.resources as resources
import pathlib

from omegaconf import DictConfig, OmegaConf

from cotorra.logger import Logger


def apply_overrides(cfg: DictConfig, overrides: collections.abc.Iterable[str]):
    """
    edits `cfg` in place with `overrides`, in order: `key=value` sets a key,
    adding it if `cfg` lacks it (a leading `+` or `++`, as Hydra writes an
    addition, is ignored), and `~key` deletes one, the only way to switch off a
    block toggled by its presence, since `key=null` leaves it present. Keys are
    dotted paths and values are read as yaml: `1e-4` is a float, `[a, b]` a
    list, and `{weight: 2.0}` a block, merged into any block already there. An
    interpolation such as `model=${model_presets.qwen_3}` resolves against the
    rest of `cfg`
    """
    for override in overrides:
        if override.startswith("~"):
            parent, _, leaf = override[1:].rpartition(".")
            del OmegaConf.select(cfg, parent)[leaf]  # `""` selects `cfg` itself
        elif "=" in override:
            cfg.merge_with(OmegaConf.from_dotlist([override.lstrip("+")]))
        else:  # which `from_dotlist` would read as `key=null`
            raise ValueError(f"can't read the override {override!r}: no `=`")


class Configurable:
    """
    takes a default configuration,
    allows the user to pass a configuration to override that,
    edits it with any command-line `overrides` (see `apply_overrides`), and
    finally considers keyword arguments that override all of these
    """

    default_file: str | None = None

    def __init__(
        self,
        config_file: pathlib.Path | str | DictConfig = None,
        overrides: collections.abc.Iterable[str] = None,
        **kwargs,
    ):
        self.config_file = config_file
        cfg = (
            # one already loaded, as a `Trainer` hands its own to its `Loader`
            copy.deepcopy(self.config_file)
            if isinstance(self.config_file, DictConfig)
            else OmegaConf.load(pathlib.Path(self.config_file).expanduser().resolve())
            if self.config_file is not None
            else OmegaConf.load(resources.files("cotorra.config") / self.default_file)
            if self.default_file is not None
            else OmegaConf.create()
        )
        apply_overrides(cfg, overrides or ())
        self.cfg = OmegaConf.merge(
            cfg, {k: v for k, v in kwargs.items() if v is not None}
        )

        self.logger = Logger()
