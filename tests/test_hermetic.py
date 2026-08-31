#!/usr/bin/env python3

"""guards the rootdir conftest: no HuggingFace cache or request escapes a run"""

import os
import pathlib

import datasets
import huggingface_hub
import pytest
import transformers
from helpers import tiny_model_dir

# the three libraries snapshot their cache locations into module-level
# constants at import time, so these are what actually got used -- not
# whatever the environment says now
CONSTANT_MODULES = {
    "huggingface_hub.constants": huggingface_hub.constants,
    "datasets.config": datasets.config,
    "transformers.utils.hub": transformers.utils.hub,
}

# absolute paths that are deliberately fixed and are not caches
NOT_A_CACHE = {"HF_JOBS_ARTIFACTS_MOUNT_PATH"}


def resolved_cache_paths() -> dict[str, pathlib.Path]:
    """
    every absolute filesystem path the three libraries resolved at import;
    discovered by scanning rather than hardcoded, so a constant added or
    renamed upstream is covered automatically instead of silently missed
    """
    found = {}
    for label, module in CONSTANT_MODULES.items():
        for name in dir(module):
            if name.startswith("_") or not name.isupper() or name in NOT_A_CACHE:
                continue
            value = getattr(module, name)
            if isinstance(value, (str, pathlib.Path)) and str(value).startswith("/"):
                found[f"{label}.{name}"] = pathlib.Path(value).resolve()
    return found


@pytest.fixture(scope="module")
def cache_home() -> pathlib.Path:
    home = os.environ.get("HF_HOME")
    if not home:
        pytest.fail(
            "HF_HOME is unset: the rootdir conftest.py did not run, so this "
            "session is using the developer's real HuggingFace caches"
        )
    return pathlib.Path(home).resolve()


def test_the_scan_finds_the_caches_it_is_meant_to_guard():
    """a scan that silently matched nothing would make the guard vacuous"""
    found = resolved_cache_paths()
    assert len(found) > 10
    for expected in (
        "huggingface_hub.constants.HF_HUB_CACHE",
        "huggingface_hub.constants.HF_TOKEN_PATH",
        "datasets.config.HF_DATASETS_CACHE",
        "transformers.utils.hub.HF_MODULES_CACHE",
    ):
        assert expected in found


def test_every_resolved_cache_path_is_inside_the_throwaway_home(cache_home):
    """fails loudly if an early import or a stale env var beat the conftest"""
    escaped = {
        name: path
        for name, path in resolved_cache_paths().items()
        if path != cache_home and cache_home not in path.parents
    }
    assert not escaped, f"cache paths outside {cache_home}: {escaped}"


def test_the_throwaway_home_is_where_the_run_actually_writes(cache_home, processed):
    """the relocation is real, not just a set of constants nobody consults"""
    from helpers import base_training_cfg, write_cfg

    from cotorra.loader import Loader

    cfg_path = write_cfg(cache_home / "probe-training.yaml", base_training_cfg())
    Loader(training_cfg=cfg_path, processed_data_home=processed).get_train_data()

    written = list((cache_home / "datasets").rglob("*.arrow"))
    assert written, f"nothing under {cache_home / 'datasets'}"


def test_hub_and_datasets_are_both_offline():
    """each library keeps its own copy of the flag; one alone leaves a gap"""
    assert huggingface_hub.constants.HF_HUB_OFFLINE
    assert huggingface_hub.constants.is_offline_mode()
    assert datasets.config.HF_HUB_OFFLINE


def test_dataset_download_counts_are_not_reported():
    """`load_dataset` otherwise HEADs s3.amazonaws.com once per call"""
    assert not datasets.config.HF_UPDATE_DOWNLOAD_COUNTS


def test_no_hub_token_is_visible():
    """nothing in the suite may authenticate as the developer"""
    assert not os.environ.get("HF_TOKEN")
    assert not os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert not pathlib.Path(huggingface_hub.constants.HF_TOKEN_PATH).exists()


def test_offline_mode_still_resolves_a_local_model_directory():
    """
    `cached_file` returns files from a local directory before it consults the
    cache or the hub, so `Trainer.model_init` works with the network cut off
    """
    cfg = transformers.AutoConfig.from_pretrained(str(tiny_model_dir()))
    assert cfg.model_type == "llama"
