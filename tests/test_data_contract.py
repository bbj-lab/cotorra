#!/usr/bin/env python3

"""
tests for the input contract cotorra expects from cocoa: if these fail, the
synthetic fixture (or cocoa itself) has drifted, and every downstream stage
test is testing the wrong shape of data
"""

import polars as pl
import pytest

SPLITS = ("train", "tuning", "held_out")


def test_tokenizer_lookup_has_the_required_special_tokens(tokenizer_cfg):
    assert "lookup" in tokenizer_cfg
    for special in ("BOS", "EOS"):
        assert special in tokenizer_cfg.lookup


def test_tokenizer_ids_are_a_contiguous_range_from_zero(tokenizer_cfg):
    """`Trainer.model_init` sets vocab_size to len(lookup), so ids must fit"""
    ids = sorted(int(v) for v in tokenizer_cfg.lookup.values())
    assert ids == list(range(len(ids)))


def test_tokens_times_has_the_expected_columns(processed):
    schema = pl.read_parquet_schema(processed / "tokens_times.parquet")
    assert {"subject_id", "tokens", "times"} <= set(schema)


def test_tokens_and_times_are_the_same_length_per_subject(processed):
    df = pl.read_parquet(processed / "tokens_times.parquet")
    lengths = df.select(
        (pl.col("tokens").list.len() == pl.col("times").list.len()).all()
    ).item()
    assert lengths


def test_subject_splits_only_uses_the_three_known_splits(processed):
    splits = pl.read_parquet(processed / "subject_splits.parquet")
    assert set(splits["split"].unique()) == set(SPLITS)


@pytest.mark.parametrize("split", SPLITS)
def test_for_inference_carries_tokens_past_and_paired_label_columns(processed, split):
    df = pl.read_parquet(processed / f"{split}_for_inference.parquet")
    assert "tokens_past" in df.columns
    assert "s_elapsed_past" in df.columns
    labelled = [c.removesuffix("_past") for c in df.columns if c.endswith("_past")]
    for token in labelled:
        if token == "tokens" or token == "s_elapsed":
            continue
        assert f"{token}_future" in df.columns


def test_every_token_id_is_within_the_tokenizer_vocabulary(processed, tokenizer_cfg):
    ids = (
        pl.read_parquet(processed / "tokens_times.parquet")
        .select(pl.col("tokens").explode())
        .to_series()
    )
    assert ids.min() >= 0
    assert ids.max() < len(tokenizer_cfg.lookup)
