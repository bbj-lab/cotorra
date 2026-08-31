#!/usr/bin/env python3

"""config-building helpers shared by the fixtures in conftest and by tests"""

import atexit
import copy
import functools
import importlib.resources as resources
import json
import pathlib
import shutil
import tempfile

from omegaconf import OmegaConf

# On macOS the LightGBM wheel links `@rpath/libomp.dylib` against Homebrew's
# libomp, while torch and scikit-learn each bundle their own; with more than
# one OpenMP runtime resident, fitting a LightGBM model *segfaults the
# interpreter* rather than raising. `RepBasedScorer` uses LightGBM by default
# and `cotorra.cli` imports torch (via `Extractor`) first, so `cotorra
# rep-based-score` hits this too -- an environment problem rather than a
# cotorra bug, but one that would otherwise take the whole session down.
LIBOMP_HINT = (
    "torch and lightgbm resolve different libomp runtimes in this environment, "
    "which segfaults on fit; re-run with DYLD_LIBRARY_PATH="
    "<venv>/lib/python3.*/site-packages/torch/lib"
)

# a bare model config naming only `model_type`: everything else is filled in
# from `LlamaConfig`'s own defaults and then overridden by `model_args` below,
# exactly as a shipped preset would be
TINY_MODEL_CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}

TINY_MODEL_ARGS = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 512,
}


@functools.cache
def tiny_model_dir() -> pathlib.Path:
    """
    a throwaway directory holding just `TINY_MODEL_CONFIG` as `config.json`;
    `AutoConfig.from_pretrained` resolves a local path without consulting the
    hub, so `Trainer.model_init` builds a real `LlamaForCausalLM` (the
    architecture every shipped preset uses) with no network request and no
    access to a gated repository. Built once per session and cleaned up at
    exit, so the suite stays pure Python on disk.
    """
    home = pathlib.Path(tempfile.mkdtemp(prefix="cotorra-tiny-model-"))
    atexit.register(shutil.rmtree, home, True)
    (home / "config.json").write_text(json.dumps(TINY_MODEL_CONFIG))
    return home


def tiny_model() -> dict:
    """the `model:` block of a training config, pointing at `tiny_model_dir`"""
    return {"model_name": str(tiny_model_dir()), "model_args": dict(TINY_MODEL_ARGS)}


FAST_TRAINING_ARGS = {
    # a plain string, not a list: OmegaConf hands list-valued config entries to
    # `TrainingArguments` as `ListConfig`, which fails transformers' own
    # `isinstance(x, list)` checks; "none" is HF's own supported spelling for
    # "report to nothing" and stays a plain str through the OmegaConf merge
    "report_to": "none",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "save_strategy": "no",
    "eval_strategy": "no",
    "load_best_model_at_end": False,
    "logging_strategy": "no",
    "disable_tqdm": True,
    "max_steps": 4,
}


def default_cfg(package: str, name: str) -> dict:
    """a mutable copy of a shipped default config, e.g. ("cotorra", "training")"""
    return OmegaConf.to_container(
        OmegaConf.load(resources.files(f"{package}.config") / f"{name}.yaml"),
        resolve=True,
    )


def write_cfg(path: pathlib.Path, cfg: dict) -> pathlib.Path:
    """write a config mapping to `path` and return it"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    return path


def deep_update(base: dict, overrides: dict) -> dict:
    """recursively merge `overrides` into a copy of `base`"""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def base_training_cfg(**overrides) -> dict:
    """the shipped default training config, with a tiny model and fast training args"""
    cfg = default_cfg("cotorra", "training")
    cfg["model"] = tiny_model()
    cfg["max_seq_len"] = 16
    cfg["training_args"] = deep_update(cfg["training_args"], FAST_TRAINING_ARGS)
    cfg["wandb"] = {"project": "cotorra-tests", "run_name": "test-run"}
    cfg["run_name"] = "test-run"
    cfg["training_args"]["run_name"] = "test-run"
    cfg["tuning_args"] = {"direction": "minimize", "backend": "optuna", "n_trials": 1}
    return deep_update(cfg, overrides)


def base_extraction_cfg(**overrides) -> dict:
    """the shipped default extraction config, shrunk to the tiny model's context"""
    cfg = default_cfg("cotorra", "extraction")
    cfg["max_seq_len"] = 16
    cfg["extract"]["max_len"] = 16
    cfg["extract"]["batch_size"] = 8
    return deep_update(cfg, overrides)


def base_scoring_cfg(**overrides) -> dict:
    """the shipped default scoring config, shrunk to the tiny model's context"""
    cfg = default_cfg("cotorra", "scoring")
    cfg["max_seq_len"] = 16
    cfg["score"]["max_len"] = 16
    return deep_update(cfg, overrides)
