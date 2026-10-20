#!/usr/bin/env python3

"""tests for cotorra.cli, the package's only public entry point"""

import importlib.metadata
import json
import pathlib
import re
import shutil
import sys
import types

import click
import numpy as np
import polars as pl
import pytest
from helpers import base_scoring_cfg, base_training_cfg, write_cfg
from omegaconf import OmegaConf
from typer.testing import CliRunner

import cotorra.cli
from cotorra.cli import app
from cotorra.model import HEADS
from cotorra.scorer_rep_based import EstimatorType

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
def test_each_command_hands_its_overrides_to_its_config(command, tmp_path, request):
    """an unreadable override fails as the config loads, before any data is read"""
    if command == "generative-score":  # importable without the [gen] extra
        request.getfixturevalue("scorer_generative")
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
    assert call["extract"] == {
        "all_times": "-a" in flags,
        "heads": heads,
        "token_nll": False,
    }


@pytest.mark.parametrize(
    "flags, heads",
    [(["-l"], []), (["--token-nll"], []), (["-a", "-l", "-t"], ["tte"])],
    ids=str,
)
def test_the_token_nll_flag_reaches_the_extractor(flags, heads, extractions, tmp_path):
    """on its own, or alongside `--all-times` and a head's flag"""
    result = runner.invoke(
        app, ["extract", "-p", str(tmp_path), "-m", str(tmp_path), *flags]
    )
    assert result.exit_code == 0, result.output
    [call] = extractions
    assert call["extract"] == {
        "all_times": "-a" in flags,
        "heads": heads,
        "token_nll": True,
    }


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


def test_extract_with_the_token_nll_flag_writes_a_list_per_row(
    processed, fake_model_home, tmp_path
):
    """from a stock model, which carries no head but has its own likelihood: a
    float32 list last in each split's table, as long as the row's tokens"""
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
            "-l",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    for split in ("train", "tuning", "held_out"):
        table = pl.read_parquet(
            out / f"features-{split}-{fake_model_home.name}.parquet"
        )
        assert table.columns[-1] == "token_nll_bits"
        assert table["token_nll_bits"].dtype == pl.List(pl.Float32)
        assert table.height > 0
        assert table["token_nll_bits"].list.len().equals(table["input_ids"].list.len())


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


def test_extract_reports_its_directory_once(extractions, tmp_path):
    """it used to print the same line once per split, its loop variable unused"""
    out = tmp_path / "features"
    result = runner.invoke(
        app, ["extract", "-p", str(tmp_path), "-m", str(tmp_path), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    report = _report(result.output, "Extraction completed")
    assert report.count("".join(str(out).split())) == 1


@pytest.mark.parametrize("command, stage", [("train", "Trainer"), ("tune", "Tuner")])
def test_the_model_is_reported_under_the_name_it_was_saved_by(
    command, stage, monkeypatch, tmp_path
):
    """the closing summary printed `mdl-{cfg.run_name}`, but the stage saves
    under its `run_name`, which falls back on `wandb.run_name`: a config without
    a top-level `run_name` failed on that last line, the model already saved"""

    class Trained:
        def __init__(self, **kwargs):
            self.cfg = OmegaConf.create({"wandb": {"run_name": "from-wandb"}})
            self.run_name = "from-wandb"
            self.output_home = tmp_path

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(cotorra.cli, stage, Trained)
    result = runner.invoke(app, [command, "-p", str(tmp_path), "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = _report(result.output, "completed in")
    assert "".join(str(tmp_path / "mdl-from-wandb").split()) in report


# ------------------------------------------- every option reaches its stage


class Recording:
    """
    a stand-in for a stage a command builds -- `Trainer`, `Tuner`, either
    scorer -- that records what it was built with and what it was then asked to
    do, so what each flag hands on can be checked without a model or any data
    """

    built: list["Recording"] = []

    def __init__(self, **kwargs):
        self.built_with, self.asked = kwargs, []
        self.run_name = "recorded"
        self.output_home = pathlib.Path(kwargs.get("output_home") or "default-home")
        self.built.append(self)

    def train(self, **kwargs):
        self.asked.append(("train", kwargs))

    def save_all(self, **kwargs):
        self.asked.append(("save_all", kwargs))


@pytest.fixture
def recorded(monkeypatch, request) -> list[Recording]:
    """every stage a command builds swapped for a `Recording`, the generative
    scorer's on the module `scorer_generative` gives, from which the command
    imports it once it runs"""
    built = []
    monkeypatch.setattr(Recording, "built", built)
    for stage in ("Trainer", "Tuner", "RepBasedScorer"):
        monkeypatch.setattr(cotorra.cli, stage, Recording)
    scorer_generative = request.getfixturevalue("scorer_generative")
    monkeypatch.setattr(scorer_generative, "GenerativeScorer", Recording)
    return built


TRAINING_FLAGS = {
    "short": {"config": "-t", "resume": "-r", "verbose": "-v"},
    "long": {
        "config": "--training-config",
        "resume": "--resume-from-checkpoint",
        "verbose": "--verbose",
    },
}


@pytest.mark.parametrize("spelled", list(TRAINING_FLAGS))
@pytest.mark.parametrize("command", ["train", "tune"])
def test_every_training_option_reaches_its_stage(command, spelled, recorded, tmp_path):
    """the config as a path, both directories, the overrides, and `--verbose`
    -- and for `train`, `--resume-from-checkpoint` -- in short or long form"""
    flags, cfg = TRAINING_FLAGS[spelled], tmp_path / "training.yaml"
    args = [command, flags["config"], str(cfg), "-p", "data", "-o", "out"]
    args += [flags["verbose"], "n_epochs=2", "~wandb"]
    args += [flags["resume"]] if command == "train" else []
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    [stage] = recorded
    assert stage.built_with == {
        "training_cfg": cfg,
        "processed_data_home": "data",
        "output_home": "out",
        "overrides": ["n_epochs=2", "~wandb"],
    }
    assert isinstance(stage.built_with["training_cfg"], pathlib.Path)
    trained = {"verbose": True} | (
        {"resume_from_checkpoint": True} if command == "train" else {}
    )
    assert stage.asked == [("train", trained)]
    report = _report(result.output, "completed in")
    assert "".join(str(pathlib.Path("out") / "mdl-recorded").split()) in report


@pytest.mark.parametrize("command", ["train", "tune"])
def test_a_training_option_left_out_takes_its_default(command, recorded):
    """the packaged config, a fresh start rather than a resumed one, and quiet"""
    result = runner.invoke(app, [command, "-p", "data", "-o", "out"])
    assert result.exit_code == 0, result.output
    [stage] = recorded
    assert stage.built_with["training_cfg"] is None
    assert not stage.built_with["overrides"]
    trained = {"verbose": False} | (
        {"resume_from_checkpoint": False} if command == "train" else {}
    )
    assert stage.asked == [("train", trained)]


@pytest.mark.parametrize("command", ["generative-score", "rep-based-score"])
def test_every_scoring_option_reaches_its_scorer(command, recorded, tmp_path):
    """the config as a path, the three directories -- and for `rep-based-score`
    the one to train on, and the estimator -- the overrides, and `--verbose`"""
    cfg = tmp_path / "scoring.yaml"
    args = [command, "-s", str(cfg), "-p", "data", "-m", "mdl", "-o", "out", "-v"]
    extra = (
        ["-t", "elsewhere", "-e", "logistic-z"] if command == "rep-based-score" else []
    )
    result = runner.invoke(app, args + extra + ["score.max_len=8"])
    assert result.exit_code == 0, result.output
    [scorer] = recorded
    expected = {
        "scoring_cfg": cfg,
        "processed_data_home": "data",
        "model_home": "mdl",
        "output_home": "out",
        "overrides": ["score.max_len=8"],
    }
    if command == "rep-based-score":
        expected |= {"training_home": "elsewhere", "estimator_type": "logistic-z"}
    assert scorer.built_with == expected
    assert isinstance(scorer.built_with["scoring_cfg"], pathlib.Path)
    assert scorer.asked == [("save_all", {"verbose": True})]
    report = _report(result.output, "scoring completed in")
    assert "".join(str(pathlib.Path("out")).split()) in report


@pytest.mark.parametrize("command", ["generative-score", "rep-based-score"])
def test_a_scoring_option_left_out_takes_its_default(command, recorded):
    """the packaged config, scores beside the inference tables (the scorer's
    own default, which None leaves it to), trained on the same directory, by
    lightGBM, and quiet"""
    result = runner.invoke(app, [command, "-p", "data", "-m", "mdl"])
    assert result.exit_code == 0, result.output
    [scorer] = recorded
    assert scorer.built_with["scoring_cfg"] is None
    assert scorer.built_with["output_home"] is None
    assert not scorer.built_with["overrides"]
    if command == "rep-based-score":
        assert scorer.built_with["training_home"] is None
        assert scorer.built_with["estimator_type"] == "lightGBM"
    assert scorer.asked == [("save_all", {"verbose": False})]


@pytest.mark.parametrize("estimator", list(EstimatorType), ids=lambda e: e.value)
def test_each_estimator_reaches_the_scorer_as_its_value(estimator, recorded):
    """a plain string: `score_label` dispatches on `str(estimator_type).lower()`,
    and an `EstimatorType` member would fall through to lightGBM there"""
    result = runner.invoke(
        app, ["rep-based-score", "-p", "data", "-m", "mdl", "-e", estimator.value]
    )
    assert result.exit_code == 0, result.output
    [scorer] = recorded
    assert type(scorer.built_with["estimator_type"]) is str
    assert scorer.built_with["estimator_type"] == estimator.value


def test_extract_takes_its_config_and_directories_in_long_form(extractions, tmp_path):
    """the config as a path, and `--output-home` None when left out, so that the
    extractor writes beside the inference tables"""
    cfg = tmp_path / "extraction.yaml"
    result = runner.invoke(
        app,
        # fmt: off
        [
            "extract",
            "--extraction-config",
            str(cfg),
            "--processed-data-home",
            "data",
            "--model-home",
            "mdl",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    [call] = extractions
    assert call["built_with"] == {
        "extraction_cfg": cfg,
        "processed_data_home": "data",
        "model_home": "mdl",
        "output_home": None,
        "overrides": call["built_with"]["overrides"],
    }
    assert isinstance(call["built_with"]["extraction_cfg"], pathlib.Path)
    assert not call["built_with"]["overrides"]


REQUIRED = {
    "train": ("--processed-data-home", "--output-home"),
    "tune": ("--processed-data-home", "--output-home"),
    "extract": ("--processed-data-home", "--model-home"),
    "generative-score": ("--processed-data-home", "--model-home"),
    "rep-based-score": ("--processed-data-home", "--model-home"),
}


@pytest.mark.parametrize("command", list(REQUIRED))
def test_each_required_option_left_out_is_refused_by_name(command, tmp_path):
    """before anything is built, with click's usage error naming the flag"""
    for missing in REQUIRED[command]:
        given = [
            arg
            for flag in REQUIRED[command]
            if flag != missing
            for arg in (flag, str(tmp_path))
        ]
        result = runner.invoke(app, [command, *given], env={"COLUMNS": "200"})
        assert result.exit_code == 2, result.output
        output = click.unstyle(result.output)
        assert "Missing option" in output and missing in output, output


def test_the_top_level_help_names_the_installed_version():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert f"(v{importlib.metadata.version('cotorra')})" in result.output


def test_main_runs_the_app(monkeypatch, capsys):
    """the `cotorra` entry point pyproject.toml names"""
    monkeypatch.setattr(sys, "argv", ["cotorra", "--help"])
    with pytest.raises(SystemExit) as exited:
        cotorra.cli.main()
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert "Usage" in out and all(command in out for command in COMMANDS)


def test_generative_score_writes_its_scores_and_reports_where(
    scorer_generative, processed, target_token, tmp_path, monkeypatch
):
    """
    end to end on the synthetic data, sglang faked: every M1 trajectory gives
    each outcome an inline SCOPE estimate of 0.25 and REACH of 0.5, and no
    outcome is ever generated, so each patient's mc, scope and reach are 0,
    0.25 and 0.5, written under `--output-home` and reported there
    """
    qsr = sys.modules[scorer_generative.GeneratedTrajectory.__module__]

    async def generate(engine, config, prompts, methods, stop_at_tracked_events=True):
        n = len(config.tracked_ids)
        return [
            qsr.GeneratedTrajectory(
                patient_idx=i,
                sample_idx=s,
                traj_type=qsr.TrajectoryType.M1,
                prompt_len=len(prompt),
                output_ids=[],
                timeline_terminating_id=None,
                scope_estimates=np.full(n, 0.25),
                reach_estimates=np.full(n, 0.5),
                occurred_flag=np.zeros(n, dtype=bool),
                inline_tracked_ids=list(config.tracked_ids),
            )
            for i, prompt in enumerate(prompts)
            for s in range(config.n_samp)
        ]

    monkeypatch.setattr(scorer_generative, "create_engine", lambda **kwargs: kwargs)
    monkeypatch.setattr(scorer_generative, "generate_trajectories", generate)
    cfg = base_scoring_cfg(score={"target_tokens": [target_token], "n_samp": 2})
    out = tmp_path / "scores"
    out.mkdir()
    result = runner.invoke(
        app,
        # fmt: off
        [
            "generative-score",
            "-s",
            str(write_cfg(tmp_path / "scoring.yaml", cfg)),
            "-p",
            str(processed),
            "-m",
            str(tmp_path / "mdl-stub"),
            "-o",
            str(out),
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    scores = out / "scores-generative-mdl-stub.parquet"
    table = pl.read_parquet(scores)
    held_out = pl.read_parquet(processed / "held_out_for_inference.parquet")
    assert table.height == held_out.height
    unhad = table.filter(~held_out[f"{target_token}_past"])
    assert unhad.height > 0
    for kind, value in {"mc": 0.0, "scope": 0.25, "reach": 0.5}.items():
        assert unhad[f"{target_token}_{kind}_score"].to_list() == [value] * unhad.height
    report = _report(result.output, "Generative scoring completed")
    assert "".join(str(scores).split()) in report
