#!/usr/bin/env python3

"""tests for cotorra.loader.Loader"""

import shutil
import time

import polars as pl
import pytest
from helpers import base_training_cfg, write_cfg

from cotorra.loader import Loader

SEQ_LEN = 16  # `base_training_cfg`'s max_seq_len


@pytest.fixture(scope="module")
def loader(processed, tmp_path_factory) -> Loader:
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-cfg") / "training.yaml", base_training_cfg()
    )
    return Loader(training_cfg=cfg_path, processed_data_home=processed)


@pytest.fixture(scope="module")
def loader_no_rope(processed, tmp_path_factory) -> Loader:
    cfg = base_training_cfg()
    del cfg["time_based_rope"]
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-no-rope-cfg") / "training.yaml", cfg
    )
    return Loader(training_cfg=cfg_path, processed_data_home=processed)


def test_splits_are_train_tuning_held_out(loader):
    assert loader.splits == ("train", "tuning", "held_out")


def test_derived_split_caches_are_written(loader, processed):
    for s in loader.splits:
        assert (processed / f"{s}_tokens_times.parquet").is_file()


def test_derived_split_caches_partition_the_subjects(loader, processed):
    """every subject lands in exactly the split `subject_splits` assigns it"""
    splits = pl.read_parquet(processed / "subject_splits.parquet")
    expected = dict(splits.group_by("split").len().iter_rows())
    for s in loader.splits:
        cached = pl.read_parquet(processed / f"{s}_tokens_times.parquet")
        assert cached.height == expected[s]
        assigned = set(splits.filter(pl.col("split") == s)["subject_id"].to_list())
        assert set(cached["subject_id"].to_list()) == assigned


def test_derived_split_caches_are_regenerated_when_tokens_are_newer(
    processed, tmp_path_factory
):
    """`Loader` compares mtimes, so a re-tokenized `tokens_times` must win"""
    home = tmp_path_factory.mktemp("loader-stale") / "processed"
    shutil.copytree(processed, home)
    cfg_path = write_cfg(home.parent / "training.yaml", base_training_cfg())
    Loader(training_cfg=cfg_path, processed_data_home=home)

    cache = home / "train_tokens_times.parquet"
    before = cache.stat().st_mtime
    time.sleep(0.01)  # coarser than the filesystem's mtime resolution
    (home / "tokens_times.parquet").touch()

    Loader(training_cfg=cfg_path, processed_data_home=home)
    assert cache.stat().st_mtime > before


def test_dataset_has_input_ids_and_s_elapsed_when_time_based_rope_configured(loader):
    for s in loader.splits:
        cols = loader.dataset[s].column_names
        assert "input_ids" in cols
        assert "s_elapsed" in cols
        assert "tokens" not in cols


def test_dataset_drops_s_elapsed_without_time_based_rope(loader_no_rope):
    for s in loader_no_rope.splits:
        assert loader_no_rope.dataset[s].column_names == ["input_ids"]


def test_for_inference_present_for_every_split_with_a_for_inference_file(
    loader, processed
):
    assert loader.inference_files
    for s in loader.splits:
        assert (processed / f"{s}_for_inference.parquet").is_file()
        assert s in loader.inference_files
    for s, ds_ in loader.for_inference.items():
        assert "input_ids" in ds_.column_names
        assert "s_elapsed_past" in ds_.column_names


def test_get_train_data_yields_fixed_length_torch_batches(loader):
    train = loader.get_train_data()
    assert len(train) > 0
    eg = train[0]
    assert set(eg.keys()) == {"input_ids", "s_elapsed"}
    assert eg["input_ids"].shape == (SEQ_LEN,)
    assert eg["s_elapsed"].shape == (SEQ_LEN,)


def test_get_tuning_data_yields_fixed_length_torch_batches(loader):
    tuning = loader.get_tuning_data()
    assert len(tuning) > 0
    assert tuning[0]["input_ids"].shape == (SEQ_LEN,)


def test_get_train_data_repeats_the_split_once_per_epoch(processed, tmp_path_factory):
    """
    `n_epochs` repeats the training split before chunking, so the number of
    fixed-length chunks scales with it (up to the single dropped remainder)
    """
    lengths = {}
    for n in (1, 2):
        cfg_path = write_cfg(
            tmp_path_factory.mktemp(f"epochs{n}") / "training.yaml",
            base_training_cfg(n_epochs=n),
        )
        loader_n = Loader(training_cfg=cfg_path, processed_data_home=processed)
        lengths[n] = len(loader_n.get_train_data())

    assert lengths[1] > 0
    assert lengths[2] == pytest.approx(2 * lengths[1], abs=1)


def test_get_tuning_data_ignores_n_epochs(processed, tmp_path_factory):
    """only the training split is repeated; evaluation must stay a single pass"""
    lengths = {}
    for n in (1, 2):
        cfg_path = write_cfg(
            tmp_path_factory.mktemp(f"tuning-epochs{n}") / "training.yaml",
            base_training_cfg(n_epochs=n),
        )
        loader_n = Loader(training_cfg=cfg_path, processed_data_home=processed)
        lengths[n] = len(loader_n.get_tuning_data())

    assert lengths[1] == lengths[2]
