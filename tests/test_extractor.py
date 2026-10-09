#!/usr/bin/env python3

"""tests for cotorra.extractor.Extractor"""

import fnmatch
import json
import math
import pathlib
import re
import shutil

import numpy as np
import polars as pl
import pytest
import torch as t
from helpers import base_extraction_cfg, base_scoring_cfg, write_cfg

from cotorra.extractor import HEAD_COLUMNS, Extractor
from cotorra.model import HEADS
from cotorra.scorer_rep_based import RepBasedScorer

SHARD_SIZE = 4


@pytest.fixture
def extractor(
    extraction_cfg_path, processed, fake_model_home, tmp_path_factory
) -> Extractor:
    out = tmp_path_factory.mktemp("extract-output")
    return Extractor(
        extraction_cfg=extraction_cfg_path,
        processed_data_home=processed,
        model_home=fake_model_home,
        output_home=out,
    )


@pytest.fixture
def sample_batch(extractor):
    eos = extractor.model.config.eos_token_id
    return {
        "input_ids": [t.tensor([1, 2, 3, eos]), t.tensor([1, 2, eos])],
        "s_elapsed_past": [
            t.tensor([0.0, 60.0, 120.0, 180.0]),
            t.tensor([0.0, 60.0, 120.0]),
        ],
    }


def test_collate_fn_pads_and_adds_position_ids(extractor, sample_batch):
    out = extractor.collate_fn(sample_batch)
    assert out["input_ids"].shape == (2, 4)
    assert out["input_ids"][1, -1].item() == extractor.model.config.eos_token_id
    assert out["position_ids"].shape == (2, 4)


def test_collate_fn_matches_the_trainers_position_id_convention(
    extractor, sample_batch
):
    """
    the same elapsed-seconds-over-`sec_per_pos_id` plus arange convention as
    `Trainer.collate_fn`; if the two ever drift, a model trained with
    time-based RoPE gets position ids it never saw at extraction time
    """
    out = extractor.collate_fn(sample_batch)
    sec_per_pos_id = extractor.cfg.time_based_rope.sec_per_pos_id
    expected = t.tensor([0.0, 60.0, 120.0, 180.0]) / sec_per_pos_id + t.arange(4)
    # `collate_fn` places its output on the model's device (mps/cuda if present)
    assert t.allclose(out["position_ids"][0].cpu(), expected)


def test_collate_fn_omits_position_ids_without_time_based_rope(
    processed, fake_model_home, tmp_path_factory, sample_batch
):
    cfg = base_extraction_cfg()
    del cfg["time_based_rope"]
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("extract-no-rope-cfg") / "extraction.yaml", cfg
    )
    extractor = Extractor(
        extraction_cfg=cfg_path,
        processed_data_home=processed,
        model_home=fake_model_home,
        output_home=tmp_path_factory.mktemp("extract-no-rope-output"),
    )
    assert extractor.collate_fn(sample_batch)["position_ids"] is None


def test_overrides_reach_the_loader_as_well_as_the_extractor(
    extraction_cfg_path, processed, fake_model_home, tmp_path_factory
):
    """the loader gets the extractor's merged config, not the file it came from"""
    extractor = Extractor(
        extraction_cfg=extraction_cfg_path,
        processed_data_home=processed,
        model_home=fake_model_home,
        output_home=tmp_path_factory.mktemp("extract-override-output"),
        overrides=["~time_based_rope"],
    )
    assert extractor.loader.cfg == extractor.cfg
    for ds_ in extractor.loader.for_inference.values():
        assert "s_elapsed_past" not in ds_.column_names


def test_extract_final_pools_the_hidden_state_at_the_last_real_token(
    extractor, sample_batch
):
    batch = extractor.extract_final(dict(sample_batch))
    features = batch["features"]
    hidden_size = extractor.model.config.hidden_size
    assert features.shape == (2, hidden_size)
    assert np.isfinite(features).all()


def test_extract_final_all_times_pads_beyond_each_sequence_with_nan(
    extractor, sample_batch
):
    batch = extractor.extract_final(dict(sample_batch), all_times=True)
    features = batch["features"]
    hidden_size = extractor.model.config.hidden_size
    assert features.shape == (2, extractor.cfg.max_seq_len, hidden_size)

    # mirror Extractor's own "last real (pre-EOS) token" computation to find
    # each row's fill boundary, rather than hardcoding indices
    collated = extractor.collate_fn(dict(sample_batch))
    eos = extractor.model.config.eos_token_id
    hits = collated["input_ids"] == eos
    last_real = t.where(
        hits.any(dim=-1),
        hits.long().argmax(dim=-1) - 1,
        collated["input_ids"].shape[-1] - 1,
    )

    for i, pos in enumerate(last_real.tolist()):
        assert not np.isnan(features[i, pos]).any()
        assert np.isnan(features[i, pos + 1]).all()


def test_extract_final_all_times_agrees_with_the_final_pooling_at_that_token(
    extractor, sample_batch
):
    """the two modes must read the same hidden state, only shaped differently"""
    final = extractor.extract_final(dict(sample_batch))["features"]
    all_times = extractor.extract_final(dict(sample_batch), all_times=True)["features"]

    collated = extractor.collate_fn(dict(sample_batch))
    hits = collated["input_ids"] == extractor.model.config.eos_token_id
    last_real = t.where(
        hits.any(dim=-1),
        hits.long().argmax(dim=-1) - 1,
        collated["input_ids"].shape[-1] - 1,
    )
    for i, pos in enumerate(last_real.tolist()):
        np.testing.assert_allclose(all_times[i, pos], final[i])


def test_extract_writes_one_features_file_per_split(extractor):
    extractor.extract()
    for split in extractor.loader.splits:
        f = (
            extractor.output_home
            / f"features-{split}-{extractor.model_home.name}.parquet"
        )
        assert f.is_file()
        df = pl.read_parquet(f)
        assert "features" in df.columns
        assert df.height > 0


def test_extract_writes_one_row_per_inference_subject(extractor, processed):
    """in the inference table's order, each row named by its `subject_id`"""
    extractor.extract()
    for split in extractor.loader.splits:
        written = pl.read_parquet(
            extractor.output_home
            / f"features-{split}-{extractor.model_home.name}.parquet"
        )
        expected = pl.read_parquet(processed / f"{split}_for_inference.parquet")
        assert written.columns[0] == "subject_id"
        assert written["subject_id"].equals(expected["subject_id"])


def test_inference_tables_without_subject_ids_extract_without_them(
    processed, fake_model_home, tmp_path
):
    """`subject_id` is optional in the inference tables: without one, the
    features are written as before, a row per inference row, just unnamed"""
    home = tmp_path / "processed"
    shutil.copytree(processed, home)
    for split in ("train", "tuning", "held_out"):
        f = home / f"{split}_for_inference.parquet"
        pl.read_parquet(f).drop("subject_id").write_parquet(f)
    out = tmp_path / "out"
    out.mkdir()
    extractor = Extractor(
        extraction_cfg=write_cfg(tmp_path / "extraction.yaml", base_extraction_cfg()),
        processed_data_home=home,
        model_home=fake_model_home,
        output_home=out,
    )
    extractor.extract()
    for split in extractor.loader.splits:
        written = pl.read_parquet(
            out / f"features-{split}-{fake_model_home.name}.parquet"
        )
        assert written.columns == ["input_ids", "s_elapsed_past", "features"]
        expected = pl.read_parquet(home / f"{split}_for_inference.parquet")
        assert written.height == expected.height


def test_extract_all_times_writes_differently_named_files(extractor):
    extractor.extract(all_times=True)
    for split in extractor.loader.splits:
        f = (
            extractor.output_home
            / f"features-all-{split}-{extractor.model_home.name}.parquet"
        )
        assert f.is_file()


def test_extract_shards_are_named_and_globbed_as_the_scorer_expects(
    processed, fake_model_home, tmp_path_factory
):
    """
    a small `shard_size` splits each split across `-<i>-of-<n>` files;
    `RepBasedScorer` reads them back through a `features-<split>*-<model>`
    glob, so the pieces must together cover every subject exactly once, in the
    inference table's order
    """
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("shard-cfg") / "extraction.yaml", base_extraction_cfg()
    )
    out = tmp_path_factory.mktemp("shard-output")
    extractor = Extractor(
        extraction_cfg=cfg_path,
        processed_data_home=processed,
        model_home=fake_model_home,
        output_home=out,
        extract={"max_len": 16, "batch_size": 8, "shard_size": SHARD_SIZE},
    )
    extractor.extract()

    sharded = 0
    for split in extractor.loader.splits:
        shards = sorted(out.glob(f"features-{split}*-{fake_model_home.name}.parquet"))
        expected = pl.read_parquet(processed / f"{split}_for_inference.parquet")
        scanned = pl.concat([pl.read_parquet(s) for s in shards])
        assert scanned["subject_id"].equals(expected["subject_id"])
        if expected.height > SHARD_SIZE:
            sharded += 1
            assert len(shards) == math.ceil(expected.height / SHARD_SIZE)
            assert all("-of-" in s.name for s in shards)
        else:  # a single shard keeps the unsuffixed name
            assert [s.name for s in shards] == [
                f"features-{split}-{fake_model_home.name}.parquet"
            ]
    assert sharded, "no split was large enough to actually shard"


def test_pad_token_id_falls_back_to_eos_when_the_model_has_none(extractor):
    """
    a checkpoint saved from `Trainer` carries no `pad_token_id`, and
    `pad_sequence` needs one; the constructor backfills it from EOS
    """
    assert extractor.model.config.pad_token_id == extractor.model.config.eos_token_id


def test_a_row_is_read_at_its_last_token_whatever_pad_id_the_model_has(extractor):
    """
    padded with EOS, not the model's own pad id, which a gemma-based checkpoint
    sets to 0: the winnowed prompts hold no EOS, so padded with anything else a
    row shorter than its batch would be read at the batch's last column, after a
    run of pads, and its features would depend on its batchmates
    """
    extractor.model.config.pad_token_id = 0  # as `Gemma3TextConfig` defaults it
    long, short = t.arange(3, 9), t.arange(3, 6)
    assert extractor.model.config.eos_token_id not in long

    def batch(*rows):
        return {
            "input_ids": list(rows),
            "s_elapsed_past": [row.float() * 60.0 for row in rows],
        }

    together = extractor.extract_final(batch(long, short))["features"]
    alone = extractor.extract_final(batch(short))["features"]
    np.testing.assert_allclose(together[1], alone[0], rtol=1e-2, atol=1e-2)
    every = extractor.extract_final(batch(long, short), all_times=True)["features"]
    assert np.isfinite(every[1, : len(short)]).all()
    assert np.isnan(every[1, len(short) :]).all()


def test_collate_fn_truncation_keeps_the_oldest_tokens(extractor):
    """
    `collate_fn` slices `x[:max_len]`, so an over-long prompt loses its most
    *recent* tokens -- the opposite of `GenerativeScorer`, whose
    `prompt_overflow: truncate_left` keeps the last `max_len`. Because
    `extract_final` pools at the last surviving token, a subject whose history
    exceeds `extract.max_len` gets features describing the start of their
    timeline rather than the prediction time. Pinned rather than fixed:
    changing it changes every feature table cotorra has ever written.
    """
    max_len = extractor.cfg.extract.max_len
    prompt = t.arange(3 * max_len)
    out = extractor.collate_fn(
        {"input_ids": [prompt], "s_elapsed_past": [prompt.float() * 60.0]}
    )
    assert out["input_ids"].shape == (1, max_len)
    assert t.equal(out["input_ids"][0].cpu(), prompt[:max_len])
    assert not t.equal(out["input_ids"][0].cpu(), prompt[-max_len:])


def test_extract_final_pools_at_the_last_position_when_no_eos_survives(extractor):
    """
    the common path in practice: every prompt in a real timeline is longer
    than `extract.max_len`, so truncation drops the EOS and `extract_final`
    falls back to the final position rather than the token before an EOS
    """
    ids = t.arange(4)
    assert extractor.model.config.eos_token_id not in ids
    batch = {"input_ids": [ids], "s_elapsed_past": [t.arange(4).float() * 60.0]}

    final = extractor.extract_final(dict(batch))["features"]
    all_times = extractor.extract_final(dict(batch), all_times=True)["features"]

    np.testing.assert_allclose(all_times[0, len(ids) - 1], final[0])
    assert np.isnan(all_times[0, len(ids)]).all()


def test_dropping_time_based_rope_changes_the_extracted_features(
    extractor, sample_batch, processed, fake_model_home, tmp_path_factory
):
    """
    the `time_based_rope` block is not cosmetic: without it the model sees
    plain 0..n-1 position ids, so a model trained *with* time-based RoPE has
    to be extracted with it too (and at the same `sec_per_pos_id`) -- this
    pins that the mismatch is observable rather than silently harmless
    """
    cfg = base_extraction_cfg()
    del cfg["time_based_rope"]
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("extract-rope-off-cfg") / "extraction.yaml", cfg
    )
    plain = Extractor(
        extraction_cfg=cfg_path,
        processed_data_home=processed,
        model_home=fake_model_home,
        output_home=tmp_path_factory.mktemp("extract-rope-off-output"),
    )

    with_rope = extractor.extract_final(dict(sample_batch))["features"]
    without_rope = plain.extract_final(dict(sample_batch))["features"]
    assert not np.allclose(with_rope, without_rope)


# ----------------------------------------------- the secondary heads' columns

ALL_HEADS = ("tte", "tnt", "disposition")
# what `extract` writes with no head asked for
PLAIN_COLUMNS = ["subject_id", "input_ids", "s_elapsed_past", "features"]
PLAIN_FEATURES = {False: pl.List(pl.Float16), True: pl.List(pl.List(pl.Float64))}
# what the error a model without a head raises calls it
DESCRIPTIONS = {
    "tte": "time-to-event",
    "tnt": "time-to-next-token",
    "disposition": "discharge-disposition",
}


def head_columns(extractor: Extractor, name: str) -> list[str]:
    """the columns head `name` adds, as the model at hand names them"""
    if name == "disposition":
        return [f"{c}_prob" for c in extractor.model.heads["disposition"].classes]
    return {"tte": ["time_to_event"], "tnt": ["time_to_next_token"]}[name]


def gathered(extractor: Extractor, batch: dict) -> list[int]:
    """each row's gathered position -- the last before its EOS, or else the
    last -- found as `extract_final` finds it"""
    input_ids = extractor.collate_fn(dict(batch))["input_ids"]
    hits = input_ids == extractor.model.config.eos_token_id
    return t.where(
        hits.any(dim=-1), hits.long().argmax(dim=-1) - 1, input_ids.shape[-1] - 1
    ).tolist()


def forward(extractor: Extractor, batch: dict):
    """the model's own forward over the batch `extract_final` reads"""
    with t.no_grad():
        return extractor.model(
            **extractor.collate_fn(dict(batch)), output_hidden_states=True
        )


def written(extractor: Extractor, split: str, all_times: bool) -> pathlib.Path:
    a = "-all" if all_times else ""
    name = extractor.model_home.name
    return extractor.output_home / f"features{a}-{split}-{name}.parquet"


@pytest.fixture
def extract_from(extraction_cfg_path, processed, tmp_path_factory):
    """an `Extractor` reading `processed`, or another home, through the model at
    `model_home`, and writing to a directory of its own"""

    def make(model_home, processed_data_home=processed, **kwargs) -> Extractor:
        return Extractor(
            extraction_cfg=extraction_cfg_path,
            processed_data_home=processed_data_home,
            model_home=model_home,
            output_home=tmp_path_factory.mktemp("extract-heads-output"),
            **kwargs,
        )

    return make


@pytest.fixture
def ending_batch(tokenizer_cfg) -> dict:
    """two rows that end in an EOS, the second short enough to be padded, so
    that each is gathered short of the last position: at 4 and at 1"""
    bos, eos = tokenizer_cfg.lookup.BOS, tokenizer_cfg.lookup.EOS
    return {
        "subject_id": ["a", "b"],
        "input_ids": [t.tensor([bos, 5, 6, 7, 8, eos]), t.tensor([bos, 9, eos])],
        "s_elapsed_past": [t.arange(6) * 600.0, t.arange(3) * 600.0],
    }


def test_every_head_has_its_columns():
    """a head added to `HEADS` needs its columns in `HEAD_COLUMNS` too"""
    assert list(HEAD_COLUMNS) == list(HEADS)


@pytest.mark.parametrize("carried", [(), ALL_HEADS], ids=["stock", "every-head"])
def test_with_no_head_asked_for_the_features_table_holds_the_plain_columns(
    carried, extract_from, fake_model_home, model_with_heads, ending_batch
):
    """the columns and dtypes `extract` writes, from a stock model and from one
    carrying every head alike -- `--all-times` features in float64, as they have
    always been"""
    extractor = extract_from(
        model_with_heads(*carried, mixture=True) if carried else fake_model_home
    )
    for all_times in (False, True):
        batch = extractor.extract_final(dict(ending_batch), all_times=all_times)
        assert list(batch) == PLAIN_COLUMNS
        extractor.extract(all_times=all_times)
        for split in extractor.loader.splits:
            schema = pl.read_parquet_schema(written(extractor, split, all_times))
            assert list(schema) == PLAIN_COLUMNS
            assert schema["features"] == PLAIN_FEATURES[all_times]


def test_the_heads_leave_the_features_untouched(
    extract_from, model_with_heads, ending_batch
):
    """read from the model's own hidden states, not from the float16 copy the
    features become, and written after them"""
    extractor = extract_from(model_with_heads(*ALL_HEADS, mixture=True))
    for all_times in (False, True):
        plain = extractor.extract_final(dict(ending_batch), all_times=all_times)
        flagged = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=ALL_HEADS
        )
        assert flagged["features"].dtype == plain["features"].dtype
        np.testing.assert_array_equal(flagged["features"], plain["features"])

        extractor.extract(all_times=all_times)
        tables = {
            split: pl.read_parquet(written(extractor, split, all_times))
            for split in extractor.loader.splits
        }
        extractor.extract(all_times=all_times, heads=ALL_HEADS)
        for split, table in tables.items():
            flagged = pl.read_parquet(written(extractor, split, all_times))
            assert flagged.select(PLAIN_COLUMNS).equals(table)


@pytest.mark.parametrize(
    "name, mixture",
    [("tte", False), ("tnt", False), ("tnt", True), ("disposition", False)],
    ids=["tte", "tnt", "tnt-mixture", "disposition"],
)
def test_each_head_adds_its_own_columns_and_no_others(
    name, mixture, extract_from, model_with_heads, ending_batch
):
    """from a model carrying every head, only the one asked for: one float32
    value per row, or with `all_times` one per position, laid out as those
    features are -- in the batch and in the written table alike"""
    extractor = extract_from(model_with_heads(*ALL_HEADS, mixture=mixture))
    expected = head_columns(extractor, name)
    for all_times in (False, True):
        batch = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=[name]
        )
        assert list(batch) == PLAIN_COLUMNS + expected
        for column in expected:
            assert batch[column].dtype == np.float32
            assert batch[column].shape == (
                (2, extractor.cfg.max_seq_len) if all_times else (2,)
            )

        extractor.extract(all_times=all_times, heads=[name])
        dtype = pl.List(pl.Float32) if all_times else pl.Float32
        for split in extractor.loader.splits:
            schema = pl.read_parquet_schema(written(extractor, split, all_times))
            assert list(schema) == PLAIN_COLUMNS + expected
            assert all(schema[column] == dtype for column in expected)


@pytest.mark.parametrize("name", ["tte", "tnt", "disposition"])
def test_a_heads_values_are_the_models_own_at_the_gathered_position(
    name, extract_from, model_with_heads, ending_batch
):
    """
    what the model's own forward predicts where the features come from -- each
    row's last token before its EOS -- in hours, or as probabilities over the
    classes. The positions either side predict values well apart, so reading
    either of them instead would fail
    """
    extractor = extract_from(model_with_heads(*ALL_HEADS))
    batch = extractor.extract_final(dict(ending_batch), heads=[name])
    out = forward(extractor, ending_batch)
    every_position = {
        "tte": {"time_to_event": out.tte_pred.float().expm1()},
        "tnt": {"time_to_next_token": out.tnt_pred.float().expm1()},
        "disposition": dict(
            zip(
                head_columns(extractor, "disposition"),
                out.disposition_pred.float().softmax(dim=-1).unbind(dim=-1),
            )
        ),
    }[name]
    rows, positions = np.arange(2), np.array(gathered(extractor, ending_batch))
    assert positions.tolist() == [4, 1]
    for column, values in every_position.items():
        values = values.cpu().numpy()
        np.testing.assert_allclose(batch[column], values[rows, positions], rtol=1e-5)
        for off_by_one in (positions - 1, positions + 1):
            assert not np.allclose(
                batch[column], values[rows, off_by_one], rtol=1e-2
            ), column


@pytest.mark.parametrize("mixture", [False, True], ids=["point", "mixture"])
def test_every_positions_values_lead_up_to_the_final_one_then_turn_nan(
    mixture, extract_from, model_with_heads, ending_batch
):
    """with `all_times`, each position up to the gathered one holds its own
    value -- the last of them the one extracted without it -- and every later
    one a nan, as the features do"""
    extractor = extract_from(model_with_heads(*ALL_HEADS, mixture=mixture))
    final = extractor.extract_final(dict(ending_batch), heads=ALL_HEADS)
    every = extractor.extract_final(dict(ending_batch), all_times=True, heads=ALL_HEADS)
    to_event = forward(extractor, ending_batch).tte_pred.float().expm1().cpu().numpy()
    columns = [c for name in ALL_HEADS for c in head_columns(extractor, name)]
    for i, pos in enumerate(gathered(extractor, ending_batch)):
        for column in columns:
            np.testing.assert_allclose(
                every[column][i, pos], final[column][i], rtol=1e-5
            )
            assert np.isfinite(every[column][i, : pos + 1]).all()
            assert np.isnan(every[column][i, pos + 1 :]).all()
        np.testing.assert_allclose(
            every["time_to_event"][i, : pos + 1], to_event[i, : pos + 1], rtol=1e-5
        )


@pytest.mark.parametrize("bias", [-50.0, 30.0])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_hours_are_never_negative_and_stay_finite_whatever_the_bias(
    dtype, bias, extract_from, model_with_heads, ending_batch
):
    """a time head pushed far below zero gives hours of nearly 0, never fewer,
    and one pushed to 30 log1p-hours ~1e13 of them, well past the 65504 at which
    float16 would overflow, from a bfloat16 checkpoint as from a float32 one"""
    extractor = extract_from(model_with_heads("tte", "tnt", dtype=dtype))
    with t.no_grad():
        for name in ("tte", "tnt"):
            extractor.model.heads[name].linear.bias.fill_(bias)
    for all_times in (False, True):
        batch = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=["tte", "tnt"]
        )
        for column in ("time_to_event", "time_to_next_token"):
            values = batch[column][~np.isnan(batch[column])]
            assert values.size and values.dtype == np.float32
            assert np.isfinite(values).all() and (values >= 0).all()
            if bias > 0:
                assert values.min() > 65504


@pytest.mark.parametrize("bias", [-50.0, 50.0])
def test_a_mixture_heads_hours_stay_finite_and_capped_whatever_the_bias(
    bias, extract_from, model_with_heads, ending_batch
):
    """never the same timestamp, and components centred and spread far beyond
    any a trained head would give: still hours between 0 and the million a draw
    is capped at"""
    extractor = extract_from(model_with_heads("tnt", mixture=True))
    with t.no_grad():
        raw = extractor.model.heads["tnt"].proj[-1].bias
        raw[0] = -50.0  # the zero part's logit
        raw[1 + 3 :] = bias  # the 3 components' means and widths
    for all_times in (False, True):
        hours = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=["tnt"]
        )["time_to_next_token"]
        values = hours[~np.isnan(hours)]
        assert values.size and np.isfinite(values).all()
        assert (values >= 0).all() and (values <= 1e6 * (1 + 1e-5)).all()


@pytest.mark.parametrize(
    "classes",
    [None, ("DSCG//home", "DSCG//expired")],
    ids=["every-disposition", "two-classes"],
)
def test_disposition_probabilities_are_a_distribution_over_the_models_classes(
    classes, extract_from, model_with_heads, ending_batch
):
    """
    a column per class the checkpoint names, in its order -- token-id order,
    whatever order the patterns came in -- each a probability, summing to 1
    across them; every `DSCG//*` token by default, and just two classes for a
    head trained on two
    """
    extractor = extract_from(model_with_heads("disposition", classes=classes))
    lookup = extractor.tkzr_cfg.lookup
    patterns = classes or ("DSCG//*",)
    model_classes = extractor.model.heads["disposition"].classes
    assert model_classes == sorted(
        (tok for tok in lookup if any(fnmatch.fnmatch(tok, p) for p in patterns)),
        key=lookup.get,
    )
    assert len(model_classes) == (2 if classes else 4)
    columns = [f"{c}_prob" for c in model_classes]
    for all_times in (False, True):
        batch = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=["disposition"]
        )
        assert list(batch) == PLAIN_COLUMNS + columns
        probs = np.stack([batch[c] for c in columns], axis=-1)
        probs = probs[~np.isnan(probs).any(axis=-1)]
        assert len(probs) == (2 + 5 if all_times else 2)  # positions 0-4 and 0-1
        assert ((probs >= 0) & (probs <= 1)).all()
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    extractor.extract(heads=["disposition"])
    for split in extractor.loader.splits:
        schema = pl.read_parquet_schema(written(extractor, split, False))
        assert list(schema) == PLAIN_COLUMNS + columns


def test_disposition_columns_follow_the_order_the_checkpoint_lists_its_classes(
    extract_from, model_with_heads, ending_batch, tmp_path
):
    """
    each class's column holds the probability of the logit the head gives it,
    named in the checkpoint's own order -- here neither token-id order nor
    alphabetical, which coincide in the synthetic vocabulary, so that reading
    the classes in either of those orders instead would mislabel the columns
    """
    home = tmp_path / "mdl-unordered"
    shutil.copytree(model_with_heads("disposition"), home)
    config = json.loads((home / "config.json").read_text())
    classes = config["heads"]["disposition"]["classes"]
    classes = [classes[2], classes[0], classes[3], classes[1]]
    assert classes not in (sorted(classes), sorted(classes, reverse=True))
    config["heads"]["disposition"]["classes"] = classes
    (home / "config.json").write_text(json.dumps(config))
    extractor = extract_from(home)
    batch = extractor.extract_final(dict(ending_batch), heads=["disposition"])
    assert list(batch) == PLAIN_COLUMNS + [f"{c}_prob" for c in classes]
    probs = forward(extractor, ending_batch).disposition_pred.float().softmax(-1)
    rows, positions = np.arange(2), np.array(gathered(extractor, ending_batch))
    for k, c in enumerate(classes):
        np.testing.assert_allclose(
            batch[f"{c}_prob"], probs.cpu().numpy()[rows, positions, k], rtol=1e-5
        )


@pytest.mark.parametrize(
    "carried, asked",
    [
        ((), "tte"),
        ((), "tnt"),
        ((), "disposition"),
        (("tnt", "disposition"), "tte"),
        (("tte", "disposition"), "tnt"),
        (("tte", "tnt"), "disposition"),
    ],
    ids=["stock-tte", "stock-tnt", "stock-disposition", "no-tte", "no-tnt", "no-disp"],
)
def test_a_head_the_model_lacks_is_refused_before_anything_is_written(
    carried,
    asked,
    extract_from,
    fake_model_home,
    model_with_heads,
    ending_batch,
    monkeypatch,
):
    """a stock model carries no head at all, a `cotorra` one only those it was
    trained with; asking for another fails on the model, before a batch is
    extracted, let alone a table written"""
    extractor = extract_from(model_with_heads(*carried) if carried else fake_model_home)
    message = re.escape(
        f"this model carries no {DESCRIPTIONS[asked]} head; train it with a "
        f"`{asked}_objective` to have one"
    )
    began = lambda *args, **kwargs: pytest.fail("extraction began")
    with monkeypatch.context() as patched:
        patched.setattr(extractor, "extract_final", began)
        for all_times in (False, True):
            with pytest.raises(ValueError, match=message):
                extractor.extract(all_times=all_times, heads=[*carried, asked])
    monkeypatch.setattr(extractor, "collate_fn", began)
    for all_times in (False, True):
        with pytest.raises(ValueError, match=message):
            extractor.extract_final(
                dict(ending_batch), all_times=all_times, heads=[asked]
            )
    assert list(extractor.output_home.iterdir()) == []


def test_a_head_of_no_known_name_is_refused_naming_those_there_are(
    extract_from, model_with_heads
):
    extractor = extract_from(model_with_heads(*ALL_HEADS))
    message = re.escape(
        "no secondary head is named ['mpp']; the heads are "
        "['tte', 'tnt', 'disposition']"
    )
    with pytest.raises(ValueError, match=message):
        extractor.extract(heads=["tte", "mpp"])
    assert list(extractor.output_home.iterdir()) == []


def test_a_head_may_be_named_alone_or_more_than_once(
    extract_from, model_with_heads, ending_batch
):
    extractor = extract_from(model_with_heads("tte"))
    alone = extractor.extract_final(dict(ending_batch), heads="tte")
    twice = extractor.extract_final(dict(ending_batch), heads=["tte", "tte"])
    assert list(alone) == list(twice) == PLAIN_COLUMNS + ["time_to_event"]
    np.testing.assert_array_equal(alone["time_to_event"], twice["time_to_event"])


def test_heads_may_come_as_a_one_shot_iterable(extract_from, model_with_heads):
    """checked once, then read for every batch of every split, so taken in as a
    list first rather than used up by the check"""
    extractor = extract_from(model_with_heads("tte"))
    extractor.extract(heads=(name for name in ["tte"]))
    for split in extractor.loader.splits:
        schema = pl.read_parquet_schema(written(extractor, split, False))
        assert list(schema) == PLAIN_COLUMNS + ["time_to_event"]


def test_head_columns_come_in_the_order_the_heads_are_asked_for(
    extract_from, model_with_heads, ending_batch
):
    """from the Python API, in the order `heads` lists them, in the batch and in
    the table alike; the command line always asks in the same order"""
    extractor = extract_from(model_with_heads(*ALL_HEADS))
    asked = ["disposition", "tnt", "tte"]
    batch = extractor.extract_final(dict(ending_batch), heads=asked)
    expected = [c for name in asked for c in head_columns(extractor, name)]
    assert list(batch) == PLAIN_COLUMNS + expected
    extractor.extract(heads=asked)
    for split in extractor.loader.splits:
        schema = pl.read_parquet_schema(written(extractor, split, False))
        assert list(schema) == PLAIN_COLUMNS + expected


def test_a_mixture_head_gives_its_mean_given_the_history_alone(
    extract_from, model_with_heads, ending_batch
):
    """
    the model's own `predict_hours_to_next_token` at the gathered position: a
    mean over every token that might come next, and so not what the head gives
    conditioned on the token that does follow the gathered one, the EOS
    """
    extractor = extract_from(model_with_heads("tnt", mixture=True))
    rows, positions = np.arange(2), np.array(gathered(extractor, ending_batch))
    model = extractor.model
    with t.no_grad():
        hidden = forward(extractor, ending_batch).hidden_states[-1][rows, positions]
        expected = model.predict_hours_to_next_token(hidden).cpu().numpy()
        eos = t.full((2,), model.config.eos_token_id, device=hidden.device)
        given_eos = model.tnt_distribution(hidden, eos).mean_log1p_hours().expm1()
    for all_times in (False, True):
        hours = extractor.extract_final(
            dict(ending_batch), all_times=all_times, heads=["tnt"]
        )["time_to_next_token"]
        if all_times:
            hours = hours[rows, positions]
        assert not np.isnan(hours).any()
        np.testing.assert_allclose(hours, expected, rtol=1e-5)
        assert not np.allclose(hours, given_eos.cpu().numpy(), rtol=1e-2)


def test_time_to_event_needs_no_end_times(extract_from, model_with_heads, processed):
    """the head's prediction comes from the checkpoint alone: `processed` holds
    no end times, and so no target a `tte_objective` could train on"""
    assert "hours_to_end_time" not in pl.read_parquet_schema(
        processed / "tokens_times.parquet"
    )
    extractor = extract_from(model_with_heads("tte"))
    extractor.extract(heads=["tte"])
    for split in extractor.loader.splits:
        assert "hours_to_end_time_past" not in pl.read_parquet_schema(
            processed / f"{split}_for_inference.parquet"
        )
        to_event = pl.read_parquet(written(extractor, split, False))["time_to_event"]
        assert to_event.dtype == pl.Float32 and to_event.is_finite().all()


def test_every_shard_carries_the_same_head_columns(
    extract_from, model_with_heads, processed
):
    """
    `datasets` types each column off the first batch it maps, so every batch of
    every shard has to give the same columns in the same dtype -- a last batch
    of one row too -- or a glob over the shards, as downstream readers scan
    them, would fail
    """
    extractor = extract_from(
        model_with_heads(*ALL_HEADS, mixture=True),
        extract={"max_len": 16, "batch_size": 3, "shard_size": SHARD_SIZE},
    )
    name = extractor.model_home.name
    columns = [c for head in ALL_HEADS for c in head_columns(extractor, head)]
    for all_times in (False, True):
        extractor.extract(all_times=all_times, heads=ALL_HEADS)
        a, sharded = "-all" if all_times else "", 0
        for split in extractor.loader.splits:
            pattern = f"features{a}-{split}*-{name}.parquet"
            shards = sorted(extractor.output_home.glob(pattern))
            schemas = [pl.read_parquet_schema(s) for s in shards]
            assert all(schema == schemas[0] for schema in schemas)
            assert list(schemas[0]) == PLAIN_COLUMNS + columns
            for column in columns:
                assert schemas[0][column] == (
                    pl.List(pl.Float32) if all_times else pl.Float32
                )
            scanned = pl.scan_parquet(extractor.output_home / pattern).collect()
            expected = pl.read_parquet(processed / f"{split}_for_inference.parquet")
            assert scanned.height == expected.height
            sharded += len(shards) > 1
        assert sharded, "no split was large enough to actually shard"


def test_features_with_head_columns_still_score(
    processed, model_with_heads, target_token, tmp_path
):
    """`RepBasedScorer` reads only `features` from the tables it globs, so the
    head columns alongside change nothing for it"""
    home = tmp_path / "processed"
    shutil.copytree(processed, home)
    model_home = model_with_heads(*ALL_HEADS, mixture=True)
    extractor = Extractor(
        extraction_cfg=write_cfg(tmp_path / "extraction.yaml", base_extraction_cfg()),
        processed_data_home=home,
        model_home=model_home,
    )
    extractor.extract(heads=ALL_HEADS)
    scorer = RepBasedScorer(
        scoring_cfg=write_cfg(
            tmp_path / "scoring.yaml",
            base_scoring_cfg(score={"target_tokens": [target_token]}),
        ),
        processed_data_home=home,
        model_home=model_home,
        estimator_type="logistic",
    )
    for split, features in scorer.features.items():
        rows = pl.read_parquet(home / f"{split}_for_inference.parquet").height
        assert features.shape == (rows, extractor.model.config.hidden_size)
    scores = scorer.score_label(target_token=target_token)
    assert np.isfinite(scores[~np.isnan(scores)]).all()


def test_a_bfloat16_checkpoint_gives_float32_columns(
    extract_from, model_with_heads, ending_batch
):
    """
    as the shipped presets' checkpoints load: `.numpy()` refuses bfloat16, and
    hours or probabilities worked out in it keep 3 significant digits, so each
    head's output is taken up to float32 first -- as it is here, from the same
    hidden states
    """
    extractor = extract_from(
        model_with_heads(*ALL_HEADS, mixture=True, dtype="bfloat16")
    )
    model = extractor.model
    assert next(model.parameters()).dtype == t.bfloat16
    rows, positions = np.arange(2), np.array(gathered(extractor, ending_batch))
    with t.no_grad():
        hidden = forward(extractor, ending_batch).hidden_states[-1][rows, positions]
        probs = model.heads["disposition"].predict(hidden).float().softmax(dim=-1)
        expected = {
            "time_to_event": model.heads["tte"].predict(hidden).float().expm1(),
            "time_to_next_token": model.predict_hours_to_next_token(hidden),
            **dict(zip(head_columns(extractor, "disposition"), probs.unbind(dim=-1))),
        }
    batch = extractor.extract_final(dict(ending_batch), heads=ALL_HEADS)
    every = extractor.extract_final(dict(ending_batch), all_times=True, heads=ALL_HEADS)
    for column, values in expected.items():
        assert batch[column].dtype == every[column].dtype == np.float32
        np.testing.assert_allclose(batch[column], values.cpu().numpy(), rtol=1e-6)
        assert np.isfinite(every[column][~np.isnan(every[column])]).all()
    extractor.extract(heads=ALL_HEADS)
    for split in extractor.loader.splits:
        schema = pl.read_parquet_schema(written(extractor, split, False))
        assert all(schema[column] == pl.Float32 for column in expected)
