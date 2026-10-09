#!/usr/bin/env python3

"""tests for cotorra.cli, the package's only public entry point"""

import json
import re
import shutil
import types

import click
import polars as pl
import pytest
from helpers import base_scoring_cfg, base_training_cfg, write_cfg
from omegaconf import OmegaConf
from typer.testing import CliRunner

import cotorra.cli
from cotorra.cli import app
from cotorra.model import HEADS

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


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_takes_config_overrides(command):
    """typer >= 0.27 no longer upper-cases an argument's name in its usage line,
    and typer < 0.26 under click 8.5 drops the argument's help from `-h`"""
    result = runner.invoke(app, [command, "-h"])
    assert result.exit_code == 0
    assert "[overrides]..." in result.output.lower()
    assert "Config overrides" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_hands_its_overrides_to_its_config(command, tmp_path):
    """an unreadable override fails as the config loads, before any data is read"""
    if command == "generative-score":
        pytest.importorskip("quick_sco_re", reason="requires the [gen] extra")
    second = "-o" if command in ("train", "tune") else "-m"
    result = runner.invoke(
        app,
        # fmt: off
        [command, "-p", str(tmp_path), second, str(tmp_path), "no_such_key"],
        # fmt: on
    )
    assert isinstance(result.exception, ValueError), result.output
    assert "no_such_key" in str(result.exception)


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
def test_train_saves_its_overrides_with_the_model(processed, tmp_path):
    cfg_path = write_cfg(tmp_path / "training.yaml", base_training_cfg())
    out = tmp_path / "output"
    out.mkdir()

    result = runner.invoke(
        app,
        # fmt: off
        [
            "train",
            "-t",
            str(cfg_path),
            "-p",
            str(processed),
            "-o",
            str(out),
            "run_name=overridden",
            "training_args.learning_rate=1e-3",
        ],
        # fmt: on
    )

    assert result.exit_code == 0, result.output
    assert (out / "mdl-overridden").is_dir()
    saved = OmegaConf.load(out / "mdl-overridden-training.yaml")
    assert saved.training_args.learning_rate == pytest.approx(1e-3)


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


HEAD_FLAGS = {
    "tte": ("--time-to-event", "-t"),
    "tnt": ("--time-to-next-token", "-n"),
    "disposition": ("--discharge-disposition", "-d"),
}


def test_extract_help_lists_a_flag_for_each_head():
    """every head in `HEADS`, its long and short flag side by side -- read at a
    width the long names fit, and unstyled, whatever the terminal"""
    assert list(HEAD_FLAGS) == list(HEADS)
    result = runner.invoke(app, ["extract", "-h"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for long, short in HEAD_FLAGS.values():
        assert re.search(rf"{long}\s+{short}\s", click.unstyle(result.output)), long


@pytest.fixture
def extractions(monkeypatch) -> list[dict]:
    """
    swaps `Extractor` in the command for a stand-in that only records what it
    is built with and asked to extract, so that what the flags hand on can be
    checked without a model or any data
    """
    calls = []

    class Recording:
        def __init__(self, **kwargs):
            self.built_with = kwargs
            self.output_home = kwargs["output_home"]
            self.loader = types.SimpleNamespace(splits=("train", "tuning", "held_out"))

        def extract(self, **kwargs):
            calls.append({"built_with": self.built_with, "extract": kwargs})

    monkeypatch.setattr(cotorra.cli, "Extractor", Recording)
    return calls


@pytest.mark.parametrize(
    "flags, heads",
    [
        ([], []),
        (["-t"], ["tte"]),
        (["--time-to-event"], ["tte"]),
        (["-n"], ["tnt"]),
        (["--time-to-next-token"], ["tnt"]),
        (["-d"], ["disposition"]),
        (["--discharge-disposition"], ["disposition"]),
        (["-d", "-n", "-t"], ["tte", "tnt", "disposition"]),
        (["-a", "-d", "-t"], ["tte", "disposition"]),
    ],
    ids=str,
)
def test_each_head_flag_reaches_the_extractor_as_its_head(
    flags, heads, extractions, tmp_path
):
    """as the heads' names in `HEADS`, always in the same order -- time to
    event, time to next token, disposition -- whatever order the flags came in,
    and alongside `--all-times`"""
    result = runner.invoke(
        app, ["extract", "-p", str(tmp_path), "-m", str(tmp_path), *flags]
    )
    assert result.exit_code == 0, result.output
    [call] = extractions
    assert call["extract"] == {"all_times": "-a" in flags, "heads": heads}


def test_a_head_flag_among_the_overrides_still_counts(extractions, tmp_path):
    """the overrides trail the command, but a flag placed among them is still
    read as a flag rather than as one of them"""
    result = runner.invoke(
        app,
        # fmt: off
        [
            "extract",
            "-p",
            str(tmp_path),
            "-m",
            str(tmp_path),
            "extract.batch_size=2",
            "-n",
            "max_seq_len=8",
            "-t",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    [call] = extractions
    assert call["extract"]["heads"] == ["tte", "tnt"]
    assert call["built_with"]["overrides"] == ["extract.batch_size=2", "max_seq_len=8"]


def test_extract_with_every_head_flag_writes_their_columns(
    processed, model_with_heads, tmp_path
):
    """every head's columns land in each split's table, after the features, in
    float32 and finite -- the disposition head's named for the classes its
    checkpoint lists"""
    model_home = model_with_heads("tte", "tnt", "disposition", mixture=True)
    heads = json.loads((model_home / "config.json").read_text())["heads"]
    classes = heads["disposition"]["classes"]
    out = tmp_path / "features"
    out.mkdir()
    result = runner.invoke(
        app,
        # fmt: off
        [
            "extract",
            "-p",
            str(processed),
            "-m",
            str(model_home),
            "-o",
            str(out),
            "-t",
            "-n",
            "-d",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    for split in ("train", "tuning", "held_out"):
        table = pl.read_parquet(out / f"features-{split}-{model_home.name}.parquet")
        assert table.columns == [
            "subject_id",
            "input_ids",
            "s_elapsed_past",
            "features",
            "time_to_event",
            "time_to_next_token",
            *(f"{c}_prob" for c in classes),
        ]
        assert table.height > 0
        assert all(
            table[c].dtype == pl.Float32 and table[c].is_finite().all()
            for c in table.columns[4:]
        )


def test_extract_refuses_a_head_flag_for_a_head_the_model_lacks(
    processed, fake_model_home, tmp_path
):
    """a stock model has no head to extract, and says so before writing a
    feature table"""
    out = tmp_path / "features"
    out.mkdir()
    result = runner.invoke(
        app,
        # fmt: off
        [
            "extract",
            "-p",
            str(processed),
            "-m",
            str(fake_model_home),
            "-o",
            str(out),
            "-n",
        ],
        # fmt: on
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "this model carries no time-to-next-token head" in str(result.exception)
    assert list(out.iterdir()) == []
