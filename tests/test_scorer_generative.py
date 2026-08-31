#!/usr/bin/env python3

"""
tests for cotorra.scorer_generative.GenerativeScorer; the module imports
`quick_sco_re` (the `[gen]` extra, installed from git) at load, so everything
below the laziness check is skipped unless that extra is present
"""

import subprocess
import sys
import textwrap

import pytest
from helpers import base_scoring_cfg
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
