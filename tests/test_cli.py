#!/usr/bin/env python3

"""tests for cotorra.cli, the package's only public entry point"""

import shutil

import polars as pl
import pytest
from helpers import base_scoring_cfg, base_training_cfg, write_cfg
from typer.testing import CliRunner

from cotorra.cli import app

runner = CliRunner()

COMMANDS = ("train", "tune", "extract", "generative-score", "rep-based-score")


def test_top_level_help_lists_every_pipeline_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_help(command):
    """`-h`/`--help` must work without importing the optional heavy deps"""
    result = runner.invoke(app, [command, "-h"])
    assert result.exit_code == 0


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_requires_processed_data_home(command):
    result = runner.invoke(app, [command])
    assert result.exit_code != 0


def test_rep_based_score_rejects_an_unknown_estimator(processed, fake_model_home):
    result = runner.invoke(
        app,
        # fmt: off
        [
            "rep-based-score",
            "-p",
            str(processed),
            "-m",
            str(fake_model_home),
            "-e",
            "random-forest",
        ],
        # fmt: on
    )
    assert result.exit_code == 2
    assert "random-forest" in result.output


@pytest.mark.slow
def test_train_writes_a_model_and_reports_its_path(processed, tmp_path):
    cfg_path = write_cfg(tmp_path / "training.yaml", base_training_cfg())
    out = tmp_path / "output"
    out.mkdir()

    result = runner.invoke(
        app,
        # fmt: off
        ["train", "-t", str(cfg_path), "-p", str(processed), "-o", str(out)],
        # fmt: on
    )

    assert result.exit_code == 0, result.output
    assert (out / "mdl-test-run").is_dir()
    assert (out / "mdl-test-run-training.yaml").is_file()


@pytest.mark.slow
def test_extract_then_rep_based_score_end_to_end(
    processed, fake_model_home, target_token, tmp_path
):
    """
    the documented two-step workflow: `extract` must leave features where
    `rep-based-score` looks for them, under the names it globs for
    """
    home = tmp_path / "processed"
    shutil.copytree(processed, home)
    scoring_cfg = write_cfg(
        tmp_path / "scoring.yaml",
        base_scoring_cfg(score={"target_tokens": [target_token]}),
    )

    extracted = runner.invoke(
        app,
        # fmt: off
        ["extract", "-p", str(home), "-m", str(fake_model_home)],
        # fmt: on
    )
    assert extracted.exit_code == 0, extracted.output
    assert (home / f"features-held_out-{fake_model_home.name}.parquet").is_file()

    scored = runner.invoke(
        app,
        # fmt: off
        [
            "rep-based-score",
            "-s",
            str(scoring_cfg),
            "-p",
            str(home),
            "-m",
            str(fake_model_home),
            "-e",
            "logistic",
        ],
        # fmt: on
    )
    assert scored.exit_code == 0, scored.output

    scores = home / f"scores-rep-based-{fake_model_home.name}.parquet"
    assert scores.is_file()
    assert f"{target_token}_rep_score" in pl.read_parquet(scores).columns


def _report(output: str, after: str) -> str:
    """
    the command's own closing summary, with all whitespace removed.

    Slicing at the `✓ ... completed` marker keeps upstream chatter (tqdm's
    `Map:` bars, fsspec's `open file:` records) out of the assertion -- an
    earlier version of this matched a debug log that happened to name the same
    path. Dropping whitespace then makes the check immune to rich wrapping a
    long tmp_path across lines, since paths contain none.
    """
    _, marker, tail = output.rpartition(after)
    assert marker, f"{after!r} not found in command output:\n{output}"
    return "".join(tail.split())


def test_extract_reports_the_directory_it_wrote_to(
    processed, fake_model_home, tmp_path
):
    """`extract` used to report `processed_data_home`, misnaming `-o` runs"""
    out = tmp_path / "features"
    out.mkdir()
    result = runner.invoke(
        app,
        # fmt: off
        ["extract", "-p", str(processed), "-m", str(fake_model_home), "-o", str(out)],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    assert (out / f"features-held_out-{fake_model_home.name}.parquet").is_file()
    report = _report(result.output, "Extraction completed")
    assert "".join(str(out).split()) in report
