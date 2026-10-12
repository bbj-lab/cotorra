#!/usr/bin/env python3

"""
tests for cotorra.scorer_generative.GenerativeScorer; the module imports
`quick_sco_re` (the `[gen]` extra, installed from git) at load, so everything
below the laziness check is skipped unless that extra is present
"""

import subprocess
import sys
import textwrap

import polars as pl
import pytest
from helpers import base_scoring_cfg, write_cfg
from omegaconf import OmegaConf

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

VOCAB = {
    "UNK": 0,
    "BOS": 1,
    "EOS": 2,
    "TRUNC": 3,
    "PAD": 4,
    "DSCG//expired": 5,
    "DSCG//home": 6,
    "TIME//1h-2h": 7,
    "RESP//imv": 8,
}


def test_cli_does_not_import_the_generative_scorer_or_quick_sco_re():
    """
    `GenerativeScorer` pulls in the optional (heavy, git-installed) `[gen]`
    deps at module load, so `cli.py` imports it inside the command body; a
    base install must be able to run every other command without them
    """
    probe = textwrap.dedent("""
        import sys
        from typer.testing import CliRunner
        from cotorra.cli import app

        assert CliRunner().invoke(app, ["--help"]).exit_code == 0
        leaked = [m for m in ("cotorra.scorer_generative", "quick_sco_re", "sglang")
                  if m in sys.modules]
        assert not leaked, leaked
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def generation_config_builder():
    quick_sco_re = pytest.importorskip(
        "quick_sco_re", reason="requires the [gen] extra"
    )
    from cotorra.scorer_generative import build_generation_config

    return build_generation_config, quick_sco_re


def build(generation_config_builder, **overrides):
    build_generation_config, _ = generation_config_builder
    cfg = OmegaConf.create(base_scoring_cfg(**overrides))
    tracked = ["DSCG//expired", "RESP//imv"]
    return build_generation_config(
        cfg, VOCAB, tracked, [VOCAB[name] for name in tracked]
    )


def test_end_tokens_resolve_to_ids(generation_config_builder):
    gen_cfg = build(generation_config_builder)
    assert gen_cfg.end_token_ids == {VOCAB["EOS"]}


def test_end_token_prefixes_expand_over_the_vocabulary(generation_config_builder):
    gen_cfg = build(
        generation_config_builder, generation={"end_tokens": {"prefixes": ["DSCG"]}}
    )
    assert gen_cfg.end_token_ids == {
        VOCAB["EOS"],
        VOCAB["DSCG//expired"],
        VOCAB["DSCG//home"],
    }


def test_unknown_token_names_are_ignored_rather_than_crashing(
    generation_config_builder,
):
    gen_cfg = build(
        generation_config_builder,
        score={"end_tokens": ["EOS", "NOT//A//TOKEN"], "suppressed_tokens": ["PAD"]},
    )
    assert gen_cfg.end_token_ids == {VOCAB["EOS"]}
    assert gen_cfg.suppressed_ids == [VOCAB["PAD"]]


def test_time_stopping_is_off_by_default(generation_config_builder):
    gen_cfg = build(generation_config_builder)
    assert gen_cfg.trunc_id is None
    assert gen_cfg.max_time is None
    assert gen_cfg.token_id_to_minutes == {}


def test_time_stopping_maps_token_bounds_to_their_geometric_mean(
    generation_config_builder,
):
    gen_cfg = build(
        generation_config_builder,
        generation={
            "time_stopping": {
                "enabled": True,
                "trunc_token": "TRUNC",
                "max_time_minutes": 1440,
                "time_token_bounds": {"TIME//1h-2h": [60, 120]},
            }
        },
    )
    assert gen_cfg.trunc_id == VOCAB["TRUNC"]
    assert gen_cfg.max_time == 1440
    assert gen_cfg.token_id_to_minutes[VOCAB["TIME//1h-2h"]] == pytest.approx(
        (60 * 120) ** 0.5
    )


def test_time_stopping_rejects_a_trunc_token_absent_from_the_vocabulary(
    generation_config_builder,
):
    with pytest.raises(ValueError, match="not found in vocabulary"):
        build(
            generation_config_builder,
            generation={
                "time_stopping": {"enabled": True, "trunc_token": "NOT//A//TOKEN"}
            },
        )


def test_time_stopping_unsuppresses_the_trunc_token(generation_config_builder):
    """a suppressed TRUNC could never be forced, defeating the time horizon"""
    gen_cfg = build(
        generation_config_builder,
        score={"suppressed_tokens": ["PAD", "TRUNC"]},
        generation={
            "time_stopping": {
                "enabled": True,
                "trunc_token": "TRUNC",
                "max_time_minutes": 1440,
            }
        },
    )
    assert VOCAB["TRUNC"] not in gen_cfg.suppressed_ids
    assert VOCAB["PAD"] in gen_cfg.suppressed_ids


def test_verbose_save_all_summarizes_the_scores_it_writes(
    processed, target_token, tmp_path, monkeypatch
):
    """
    `save_all` assembles its table as a `pl.DataFrame`, but
    `Logger.summarize_preds` collects what it selects, which only a
    `pl.LazyFrame` can, so `-v` used to crash at the very end of a generation
    run, its scores already written. Built on the synthetic data, with the
    SGLang engine and the generation itself stubbed out
    """
    pytest.importorskip("quick_sco_re", reason="requires the [gen] extra")
    from cotorra import scorer_generative

    monkeypatch.setattr(scorer_generative, "create_engine", lambda **_: None)
    cfg = base_scoring_cfg(score={"target_tokens": [target_token]})
    scorer = scorer_generative.GenerativeScorer(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", cfg),
        processed_data_home=processed,
        model_home=tmp_path / "mdl-stub",
        output_home=tmp_path,
    )
    columns = [f"{target_token}_{m}_score" for m in ("mc", "scope", "reach")]
    future = scorer.ds[f"{target_token}_future"].to_numpy().astype(float)

    async def score():
        return {column: future for column in columns}

    scorer.score = score
    scorer.save_all(verbose=True)

    assert set(columns) <= set(pl.read_parquet(scorer.output_home).columns)


@pytest.mark.parametrize(
    "path",
    [
        ("generation",),
        ("generation", "end_tokens"),
        ("generation", "end_tokens", "prefixes"),
        ("generation", "time_stopping"),
        ("generation", "time_stopping", "time_token_bounds"),
        ("score", "end_tokens"),
        ("score", "suppressed_tokens"),
    ],
    ids="/".join,
)
def test_a_block_or_list_left_empty_reads_as_if_absent(generation_config_builder, path):
    """an entry left empty in the yaml parses to None, which `.get(key, default)`
    returns in place of its default, so each of these used to fail on `None.get`
    or on iterating None; time stopping is on, so that its bounds are read"""
    build_generation_config, _ = generation_config_builder
    tracked = ["DSCG//expired", "RESP//imv"]
    built = {}
    for how in ("empty", "absent"):
        cfg = base_scoring_cfg(
            generation={
                "time_stopping": {
                    "enabled": True,
                    "trunc_token": "TRUNC",
                    "time_token_bounds": {"TIME//1h-2h": [60, 120]},
                }
            }
        )
        *parents, leaf = path
        parent = cfg
        for key in parents:
            parent = parent[key]
        if how == "empty":
            parent[leaf] = None
        else:
            del parent[leaf]
        built[how] = build_generation_config(
            OmegaConf.create(cfg), VOCAB, tracked, [VOCAB[name] for name in tracked]
        )
    assert built["empty"] == built["absent"]


def scorer_with(scoring_cfg: dict, processed, tmp_path, monkeypatch):
    """a `GenerativeScorer` built on the synthetic data under `scoring_cfg`,
    with the SGLang engine stubbed out"""
    pytest.importorskip("quick_sco_re", reason="requires the [gen] extra")
    from cotorra import scorer_generative

    monkeypatch.setattr(scorer_generative, "create_engine", lambda **_: None)
    return scorer_generative.GenerativeScorer(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", scoring_cfg),
        processed_data_home=processed,
        model_home=tmp_path / "mdl-stub",
        output_home=tmp_path,
    )


@pytest.mark.parametrize(
    "blocks, methods",
    [
        ({"generation": None, "engine": None}, ["M1", "M2"]),
        ({"generation": {"methods": None}}, ["M1", "M2"]),
        ({"generation": {"methods": []}}, []),  # explicit, so kept
    ],
    ids=["empty-blocks", "empty-methods", "no-methods"],
)
def test_empty_generation_and_engine_blocks_take_their_defaults(
    blocks, methods, processed, target_token, tmp_path, monkeypatch
):
    """as in `build_generation_config`, the scorer read `generation` and `engine`
    with `.get(key, {})`, which returns a block left empty as None, and an empty
    `methods:` likewise, which `"M2" in self.methods` then failed on"""
    cfg = base_scoring_cfg(score={"target_tokens": [target_token]}, **blocks)
    scorer = scorer_with(cfg, processed, tmp_path, monkeypatch)
    assert scorer.methods == methods
    assert scorer.chunk_size == 64


def test_a_generative_scorer_without_its_directories_is_refused_by_name(tmp_path):
    """both default to None, a holdover from when the config named them, and
    leaving one out used to fail inside `pathlib` without saying which"""
    pytest.importorskip("quick_sco_re", reason="requires the [gen] extra")
    from cotorra.scorer_generative import GenerativeScorer

    with pytest.raises(TypeError, match=r"^GenerativeScorer needs `model_home`$"):
        GenerativeScorer(processed_data_home=tmp_path)
    with pytest.raises(
        TypeError,
        match=r"^GenerativeScorer needs `processed_data_home` and `model_home`$",
    ):
        GenerativeScorer()
