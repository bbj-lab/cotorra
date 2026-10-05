#!/usr/bin/env python3

"""tests for cotorra.loader.Loader"""

import math
import shutil
import time

import polars as pl
import pytest
import torch as t
from helpers import base_training_cfg, write_cfg
from omegaconf import OmegaConf

from cotorra.loader import Loader, disposition_targets

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


def test_for_inference_is_none_when_no_inference_files_exist(
    processed, tmp_path_factory
):
    """
    `{split}_for_inference.parquet` is only needed by extract/score, so a
    processed directory holding nothing but training data must still load
    """
    home = tmp_path_factory.mktemp("loader-no-inference") / "processed"
    shutil.copytree(processed, home)
    for f in home.glob("*_for_inference.parquet"):
        f.unlink()

    cfg_path = write_cfg(home.parent / "training.yaml", base_training_cfg())
    loader = Loader(training_cfg=cfg_path, processed_data_home=home)

    assert loader.inference_files == {}
    assert loader.for_inference is None
    assert len(loader.get_train_data()) > 0


# --------------------------------------------- the time-to-next-token target


@pytest.fixture(scope="module")
def tte_home(processed, tmp_path_factory):
    """
    `processed` plus the `hours_to_end_time` column a `tte_objective`
    reads, which cocoa writes only under `include_hours_to_end_time` (off by
    default, and so in the shared fixture). The inference files are dropped
    rather than given a matching `hours_to_end_time_past`: only the training
    side is under test here
    """
    home = tmp_path_factory.mktemp("loader-tte") / "processed"
    shutil.copytree(processed, home)
    for f in home.glob("*_for_inference.parquet"):
        f.unlink()
    tt = home / "tokens_times.parquet"
    pl.read_parquet(tt).with_columns(
        hours_to_end_time=pl.col("times").list.eval(
            (pl.element().last() - pl.element()).dt.total_seconds().truediv(3600)
        )
    ).write_parquet(tt)
    return home


@pytest.fixture(scope="module")
def tnt_loader(processed, tmp_path_factory) -> Loader:
    """`tnt_objective` alone, on data cocoa tokenized without end times: the
    time-to-next-token target is derived, so it needs nothing they provide"""
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-tnt-cfg") / "training.yaml",
        base_training_cfg(tnt_objective={}),
    )
    return Loader(training_cfg=cfg_path, processed_data_home=processed)


@pytest.fixture(scope="module")
def every_head_loader(tte_home, tmp_path_factory) -> Loader:
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-every-head-cfg") / "training.yaml",
        base_training_cfg(tte_objective={}, tnt_objective={}, disposition_objective={}),
    )
    return Loader(training_cfg=cfg_path, processed_data_home=tte_home)


def test_split_caches_carry_the_hours_to_each_next_token(loader, processed):
    """
    what a time-to-next-token head trains on: the gap from each
    token to the one after it in the same record, derived for every run so the
    caches don't depend on the config; a record's last token has no successor
    and gets a nan
    """
    for s in loader.splits:
        cached = pl.read_parquet(processed / f"{s}_tokens_times.parquet")
        for times, hours in cached.select("times", "hours_to_next_token").iter_rows():
            expected = [
                (b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])
            ]
            assert hours[:-1] == pytest.approx(expected)
            assert math.isnan(hours[-1])


def test_split_caches_that_predate_the_derived_column_are_regenerated(
    processed, tmp_path_factory
):
    """
    a cache written before `hours_to_next_token` was derived is still newer
    than `tokens_times`, so the mtime check alone would keep it and starve an
    time-to-next-token head of its target
    """
    home = tmp_path_factory.mktemp("loader-predates") / "processed"
    shutil.copytree(processed, home)
    cfg_path = write_cfg(home.parent / "training.yaml", base_training_cfg())
    Loader(training_cfg=cfg_path, processed_data_home=home)

    cache = home / "train_tokens_times.parquet"
    pl.read_parquet(cache).drop("hours_to_next_token").write_parquet(cache)
    assert cache.stat().st_mtime > (home / "tokens_times.parquet").stat().st_mtime

    Loader(training_cfg=cfg_path, processed_data_home=home)
    assert "hours_to_next_token" in pl.read_parquet_schema(cache)


def test_each_target_is_loaded_only_for_its_objective(
    loader, tnt_loader, every_head_loader
):
    for s in loader.splits:
        assert loader.dataset[s].column_names == ["input_ids", "s_elapsed"]
        assert tnt_loader.dataset[s].column_names == [
            "input_ids",
            "s_elapsed",
            "hours_to_next_token",
        ]
        assert every_head_loader.dataset[s].column_names == [
            "input_ids",
            "s_elapsed",
            "hours_to_end_time",
            "hours_to_next_token",
            "disposition",
        ]


def test_packing_keeps_the_gap_across_a_record_boundary_masked(tnt_loader):
    """
    the nan on each record's last token -- its EOS -- survives packing and
    torch formatting, so the head is never scored on the jump from one
    record's end to the next one's start; every other token has a real gap
    """
    eos = tnt_loader.tokenizer_info.lookup.EOS
    train = tnt_loader.get_train_data()
    ids = t.cat([eg["input_ids"] for eg in train])
    hours = t.cat([eg["hours_to_next_token"] for eg in train])
    assert (ids == eos).any()
    assert hours[ids == eos].isnan().all()
    assert (hours[ids != eos] >= 0).all()


# ------------------------------------------------------ the disposition target


def test_disposition_targets_point_ahead_to_how_the_record_ends():
    """the class of the record's last disposition token for every token before
    it, and -100 from it on, its disposition no longer to come"""
    class_of = {7: 0, 8: 1}
    assert disposition_targets([1, 5, 6, 8, 2], class_of).tolist() == [
        1,
        1,
        1,
        -100,
        -100,
    ]
    assert disposition_targets([1, 7, 5, 8, 2], class_of).tolist()[:3] == [1, 1, 1]
    assert disposition_targets([1, 5, 6, 2], class_of).tolist() == [-100] * 4


def test_every_record_is_labeled_by_its_own_disposition(every_head_loader):
    """
    computed per record, before packing splits records across chunks: cocoa
    writes one `DSCG//*` token near each record's end, and every token before
    it carries that token's class -- in `classes` order, which follows the
    token ids
    """
    lookup = every_head_loader.tokenizer_info.lookup
    classes = every_head_loader.heads["disposition"]["classes"]
    assert classes == sorted(
        (k for k in lookup if k.startswith("DSCG//")), key=lookup.get
    )
    class_ids = [lookup[c] for c in classes]
    for row in every_head_loader.dataset["train"]:
        x, target = t.tensor(row["input_ids"]), t.tensor(row["disposition"])
        (at,) = t.isin(x, t.tensor(class_ids)).nonzero().flatten().tolist()
        assert (target[:at] == class_ids.index(x[at].item())).all()
        assert (target[at:] == -100).all()


def test_a_record_ending_in_none_of_the_classes_goes_unscored(
    processed, tmp_path_factory
):
    """with `DSCG//missing`, say, left out of `classes`; here every disposition
    but the commonest is"""
    lookup = OmegaConf.load(processed / "tokenizer.yaml").lookup
    tokens = pl.read_parquet(processed / "tokens_times.parquet")["tokens"].explode(
        empty_as_null=True
    )
    dscg = {lookup[k]: k for k in lookup if k.startswith("DSCG//")}
    commonest = tokens.filter(tokens.is_in(list(dscg))).mode()[0]
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-one-class-cfg") / "training.yaml",
        base_training_cfg(disposition_objective={"classes": [dscg[commonest]]}),
    )
    loader = Loader(training_cfg=cfg_path, processed_data_home=processed)
    scored = []
    for row in loader.dataset["train"]:
        scored.append(commonest in row["input_ids"])
        assert any(c != -100 for c in row["disposition"]) == scored[-1]
    assert any(scored) and not all(scored)
