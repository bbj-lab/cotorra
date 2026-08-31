#!/usr/bin/env python3

"""tests for cotorra.extractor.Extractor"""

import math

import numpy as np
import polars as pl
import pytest
import torch as t
from helpers import base_extraction_cfg, write_cfg

from cotorra.extractor import Extractor

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
    assert out["input_ids"][1, -1].item() == extractor.model.config.pad_token_id
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
    extractor.extract()
    for split in extractor.loader.splits:
        written = pl.read_parquet(
            extractor.output_home
            / f"features-{split}-{extractor.model_home.name}.parquet"
        )
        expected = pl.read_parquet(processed / f"{split}_for_inference.parquet")
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
    glob, so the pieces must together cover every subject exactly once
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
        assert sum(pl.read_parquet(s).height for s in shards) == expected.height
        if expected.height > SHARD_SIZE:
            sharded += 1
            assert len(shards) == math.ceil(expected.height / SHARD_SIZE)
            assert all("-of-" in s.name for s in shards)
        else:  # a single shard keeps the unsuffixed name
            assert [s.name for s in shards] == [
                f"features-{split}-{fake_model_home.name}.parquet"
            ]
    assert sharded, "no split was large enough to actually shard"
