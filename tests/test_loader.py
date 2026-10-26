#!/usr/bin/env python3

"""tests for cotorra.loader.Loader"""

import math
import shutil
import time

import datasets as ds
import numpy as np
import polars as pl
import pytest
import torch as t
from helpers import base_training_cfg, write_cfg
from omegaconf import OmegaConf

from cotorra.loader import SCORED, Loader, disposition_targets, record_weights
from cotorra.model import HEADS
from cotorra.util import best_fit_packed_iter

SEQ_LEN = 16  # `base_training_cfg`'s max_seq_len

# the heads whose objectives can weight each record alike
PER_RECORD = [name for name, head in HEADS.items() if head.target_weight]


def commonest_disposition(home) -> str:
    """the `DSCG//*` token that the most of `home`'s records end with"""
    lookup = OmegaConf.load(home / "tokenizer.yaml").lookup
    tokens = pl.read_parquet(home / "tokens_times.parquet")["tokens"].explode(
        empty_as_null=True
    )
    dscg = {lookup[k]: k for k in lookup if k.startswith("DSCG//")}
    return dscg[tokens.filter(tokens.is_in(list(dscg))).mode()[0]]


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
        assert ds_.column_names[:2] == ["subject_id", "input_ids"]
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


# ---------------------------------------------------- best-fit decreasing packing

PACKED_COLUMNS = ["input_ids", "s_elapsed", "pad_mask", "doc_pos"]


def loader_with(processed, tmp_path_factory, **overrides) -> Loader:
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-packing-cfg") / "training.yaml",
        base_training_cfg(**overrides),
    )
    return Loader(training_cfg=cfg_path, processed_data_home=processed)


@pytest.fixture(scope="module")
def bfd_loader(processed, tmp_path_factory) -> Loader:
    return loader_with(processed, tmp_path_factory, packing="bfd", n_epochs=2)


def unpadded(rows: ds.Dataset) -> list[tuple[int, float]]:
    """every token `rows` pack, beside its elapsed seconds, padding dropped"""
    return sorted(
        pair
        for eg in rows
        for pair in zip(
            eg["input_ids"][~eg["pad_mask"].bool()].tolist(),
            eg["s_elapsed"][~eg["pad_mask"].bool()].tolist(),
        )
    )


def in_records(dset: ds.Dataset) -> list[tuple[int, float]]:
    """every token of `dset`'s records, beside its elapsed seconds"""
    return sorted(
        (i, t.tensor(s, dtype=t.float32).item())
        for ids, secs in zip(dset["input_ids"], dset["s_elapsed"])
        for i, s in zip(ids, secs)
    )


def test_bfd_rows_are_fixed_length_and_flag_their_padding(bfd_loader):
    train = bfd_loader.get_train_data()
    assert train.column_names == PACKED_COLUMNS
    for eg in train:
        assert all(v.shape == (SEQ_LEN,) for v in eg.values())
    assert t.stack([eg["pad_mask"] for eg in train]).any()


def test_bfd_packing_trains_on_every_token_of_every_epoch(bfd_loader):
    """nothing is dropped, as concatenation drops its last partial row: each
    training token appears once an epoch, its elapsed time beside it"""
    assert unpadded(bfd_loader.get_train_data()) == sorted(
        2 * in_records(bfd_loader.dataset["train"])
    )


def test_bfd_hands_the_packer_the_split_and_its_epochs_to_pack_apart(
    bfd_loader, monkeypatch
):
    """the split as it is, not repeated: packed in one pass, the epochs' copies
    of a record would often share a row (see test_util), where the later copy
    could attend to the earlier"""
    seen = {}

    def from_generator(generator, gen_kwargs):
        seen.update(gen_kwargs, generator=generator)
        return ds.Dataset.from_dict({"input_ids": [[0]]})

    monkeypatch.setattr(ds.Dataset, "from_generator", staticmethod(from_generator))
    bfd_loader.get_train_data()
    assert seen["generator"] is best_fit_packed_iter
    assert seen["n_epochs"] == bfd_loader.cfg.n_epochs == 2
    assert len(seen["dset"]) == len(bfd_loader.dataset["train"])


def test_an_unknown_packing_is_refused(processed, tmp_path_factory):
    loader = loader_with(processed, tmp_path_factory, packing="best_fit")
    with pytest.raises(ValueError, match="`packing` is `concat` or `bfd`"):
        loader.get_train_data()


def test_bfd_alone_leaves_the_tuning_data_concatenated(bfd_loader):
    assert bfd_loader.get_tuning_data().column_names == ["input_ids", "s_elapsed"]


def test_doc_masking_evaluates_on_tuning_data_packed_as_training_is(
    processed, tmp_path_factory
):
    """so that `eval_loss` scores each record attending only to itself, as it
    trains, not given context it never trains on; by a seed of its own, so on
    the same rows whatever the training data drew first"""
    loader = loader_with(processed, tmp_path_factory, packing="bfd", doc_masking=True)
    first = loader.get_tuning_data()
    loader.get_train_data()
    again = loader.get_tuning_data()
    assert first.column_names == PACKED_COLUMNS
    assert all(t.equal(a["input_ids"], b["input_ids"]) for a, b in zip(first, again))
    assert unpadded(first) == in_records(loader.dataset["tuning"])


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
    default, and so in the shared fixture). The inference tables are left
    without a matching `hours_to_end_time_past`, which nothing reads
    """
    home = tmp_path_factory.mktemp("loader-tte") / "processed"
    shutil.copytree(processed, home)
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


def test_no_target_joins_the_inference_sets(
    every_head_loader, tte_home, processed_with_end_times, tmp_path_factory
):
    """
    the inference sets hold the model's context alone, whatever the objectives:
    with every head, inference tables without end times load as they are, and
    ones with them -- as cocoa winnows them under `include_hours_to_end_time`
    -- don't lend their `hours_to_end_time_past`, the outcome itself, to the
    batches `Extractor` writes back out
    """
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-end-times-cfg") / "training.yaml",
        base_training_cfg(tte_objective={}, tnt_objective={}, disposition_objective={}),
    )
    labeled = Loader(
        training_cfg=cfg_path, processed_data_home=processed_with_end_times
    )
    for loader, home, has_end_times in (
        (every_head_loader, tte_home, False),
        (labeled, processed_with_end_times, True),
    ):
        assert set(loader.heads) == {"tte", "tnt", "disposition"}
        assert set(loader.for_inference) == set(loader.splits)
        for s, ds_ in loader.for_inference.items():
            schema = pl.read_parquet_schema(home / f"{s}_for_inference.parquet")
            assert ("hours_to_end_time_past" in schema) == has_end_times
            assert ds_.column_names == ["subject_id", "input_ids", "s_elapsed_past"]


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


# --------------------------------------------------- the weights per record


def test_record_weights_share_the_mean_among_a_records_scored_tokens():
    """`mean / n` for a record with n tokens scored, `mean` being n's mean over
    the records with any, so that each one's scored tokens weigh `mean` between
    them; nothing for a record without any"""
    assert record_weights([4, 2, 0, 6]).tolist() == pytest.approx(
        [1.0, 2.0, 0.0, 2 / 3]
    )
    assert record_weights([0, 0]).tolist() == [0.0, 0.0]
    assert record_weights([]).tolist() == []


def test_each_head_counts_the_positions_its_loss_scores():
    """
    the time-to-event head's finite targets -- one past the end time included,
    which it scores as 0 hours left, but not a missing end time, arriving as a
    nan or a null -- and the disposition head's targets other than -100; every
    head that can weight each record alike has its count
    """
    assert list(SCORED) == PER_RECORD
    hours = pl.Series([[5.0, -1.0, math.nan, None], [None]])
    assert hours.list.eval(SCORED["tte"]).list.sum().to_list() == [2, 0]
    disposition = pl.Series([[2, 2, -100], [-100], [0]])
    assert disposition.list.eval(SCORED["disposition"]).list.sum().to_list() == [
        2,
        0,
        1,
    ]


# which of a record's tokens each head scores, as the tests see it
IS_SCORED = {
    "tte": lambda target: np.isfinite(np.asarray(target, dtype=np.float64)),
    "disposition": lambda target: np.asarray(target) != -100,
}


@pytest.fixture(scope="module")
def per_record_loader(tte_home, tmp_path_factory) -> Loader:
    """every head that can weight each record alike, weighting it, the
    disposition head with one class, so that the records ending otherwise go
    unscored"""
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-per-record-cfg") / "training.yaml",
        base_training_cfg(
            tte_objective={"per_record": True},
            disposition_objective={
                "per_record": True,
                "classes": [commonest_disposition(tte_home)],
            },
        ),
    )
    return Loader(training_cfg=cfg_path, processed_data_home=tte_home)


def test_each_weight_joins_its_target(per_record_loader):
    for s in per_record_loader.splits:
        assert per_record_loader.dataset[s].column_names == [
            "input_ids",
            "s_elapsed",
            "hours_to_end_time",
            "disposition",
            "hours_to_end_time_weight",
            "disposition_weight",
        ]


@pytest.mark.parametrize("name", PER_RECORD)
def test_every_record_weighs_the_same_however_long_it_runs(per_record_loader, name):
    """
    within each split, by its own records: a record's tokens share one weight,
    and its scored tokens' weights total the same over every record with any,
    and average 1 over the split's scored tokens; a record without any weighs
    nothing
    """
    head = HEADS[name]
    unscored = 0
    for s in per_record_loader.splits:
        counts, totals, weights = [], [], []
        for row in per_record_loader.dataset[s]:
            w = np.asarray(row[head.target_weight])
            scored = IS_SCORED[name](row[head.target])
            assert len(w) == len(row["input_ids"]) and len(set(w)) == 1
            if scored.any():
                counts.append(scored.sum())
                totals.append(w[scored].sum())
                weights.extend(w[scored])
            else:
                unscored += 1
                assert (w == 0).all()
        assert len(set(counts)) > 1
        assert totals == pytest.approx([totals[0]] * len(totals), rel=1e-5)
        assert np.mean(weights) == pytest.approx(1.0, rel=1e-5)
    # every record has a finite end time, but not every one ends in the class
    assert unscored == 0 if name == "tte" else unscored > 0


@pytest.mark.parametrize("name", PER_RECORD)
def test_packing_keeps_every_fragment_its_whole_records_weight(per_record_loader, name):
    """
    the tuning split packs in order, unshuffled, so its packed weights are its
    records' own run together -- records running past a chunk included, whose
    fragments weigh what they would whole
    """
    column = HEADS[name].target_weight
    tuning = per_record_loader.dataset["tuning"]
    assert max(len(row["input_ids"]) for row in tuning) > SEQ_LEN
    expected = [w for row in tuning for w in row[column]]
    packed = t.cat([eg[column] for eg in per_record_loader.get_tuning_data()])
    assert len(packed) == len(expected) // SEQ_LEN * SEQ_LEN
    assert t.allclose(packed, t.tensor(expected[: len(packed)], dtype=packed.dtype))


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
    commonest = commonest_disposition(processed)
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("loader-one-class-cfg") / "training.yaml",
        base_training_cfg(disposition_objective={"classes": [commonest]}),
    )
    loader = Loader(training_cfg=cfg_path, processed_data_home=processed)
    commonest_id = loader.tokenizer_info.lookup[commonest]
    scored = []
    for row in loader.dataset["train"]:
        scored.append(commonest_id in row["input_ids"])
        assert any(c != -100 for c in row["disposition"]) == scored[-1]
    assert any(scored) and not all(scored)


def test_a_loader_without_its_directory_is_refused_by_name():
    """`processed_data_home` defaults to None but has no fallback, and leaving it
    out used to fail at `None / "tokenizer.yaml"` without naming it"""
    with pytest.raises(TypeError, match=r"^Loader needs `processed_data_home`$"):
        Loader()
