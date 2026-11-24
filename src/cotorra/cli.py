#!/usr/bin/env python3

"""
CLI for cotorra - configurable training for generative event models
"""

import pathlib
import platform
import re
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, requires, version
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console

from cotorra.baselines import (
    BagOfTokensScorer,
    Baseline,
    BaselineEstimator,
    TsfreshScorer,
)
from cotorra.extractor import Extractor
from cotorra.scorer_generative import GenerativeScorer
from cotorra.scorer_rep_based import EstimatorType, RepBasedScorer
from cotorra.trainer import Trainer
from cotorra.tuner import Tuner

__version__ = version("cotorra")

app = typer.Typer(
    name="cotorra",
    help=f"Configurable training for generative event models (v{__version__})",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

# trailing every command: edits to its config, applied by `apply_overrides`
Overrides = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="Config overrides: key=value sets a key, adding it if need be, and "
        "~key deletes one",
        show_default=False,
    ),
]


def show_versions(show: bool) -> None:
    """
    print the versions of cotorra, python, and each dependency cotorra requires,
    with the cuda that torch was built for and the accelerator it finds, and exit;
    for `--version`, so a bug report can paste them
    """
    if not show:
        return
    import torch as t  # loaded already, by the stages imported above

    if t.cuda.is_available():
        gpus = Counter(map(t.cuda.get_device_name, range(t.cuda.device_count())))
        accelerator = ", ".join(f"{n} x {gpu}" for gpu, n in gpus.items())
    elif t.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "none"
    found = {
        "cotorra": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": t.version.cuda or "none",
        "accelerator": accelerator,
    }
    for requirement in requires("cotorra") or ():
        spec, _, marker = requirement.partition(";")
        if "extra" in marker:  # an optional dependency, of one of the extras
            continue
        name = re.match(r"[\w.-]+", spec.strip())[0]
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    width = max(map(len, found))
    for name, v in found.items():
        typer.echo(f"{name:<{width}}  {v}")
    raise typer.Exit()


@app.callback()
def options(
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=show_versions,
            is_eager=True,
            help="Show the versions of cotorra, its dependencies, Python, and CUDA, "
            "and exit.",
        ),
    ] = False,
) -> None:
    """the options `cotorra` takes before a command"""


@app.command()
def train(
    training_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--training-config",
            "-t",
            help="Training configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
    ] = ...,
    resume_from_checkpoint: Annotated[
        bool,
        typer.Option(
            "--resume-from-checkpoint",
            "-r",
            help="Try to resume training from the latest checkpoint in --output-home.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging")
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Train a model on tokenized data. For tokenization, consult the cocoa package.
    """
    with console.status("[bold green]Training model..."):
        t0 = time.perf_counter()
        trainer = Trainer(
            training_cfg=training_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
            overrides=overrides,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint, verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Training completed in {t1 - t0:.2f}s.")
        out_path = trainer.output_home / f"mdl-{trainer.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def tune(
    training_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--training-config",
            "-t",
            help="Training configuration file (overrides default)",
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
            show_default=False,
        ),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Output directory for trained models",
            show_default=False,
        ),
    ] = ...,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging")
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Run hyperparameter tuning while training a model.
    """
    with console.status("[bold green]Tuning model..."):
        t0 = time.perf_counter()
        tuner = Tuner(
            training_cfg=training_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
            overrides=overrides,
        )
        tuner.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tuning completed in {t1 - t0:.2f}s.")
        out_path = tuner.output_home / f"mdl-{tuner.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def extract(
    extraction_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--extraction-config",
            "-e",
            help="Extraction configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    model_home: Annotated[
        str,
        typer.Option(
            "--model-home", "-m", help="Directory of the trained model to extract from"
        ),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Output directory for extracted features, "
            "defaults to processed-data-home",
            show_default=False,
        ),
    ] = None,
    all_times: Annotated[
        bool,
        typer.Option(
            "--all-times",
            "-a",
            help="Extract features for all time steps (instead of just the final one)?",
        ),
    ] = False,
    time_to_event: Annotated[
        bool,
        typer.Option(
            "--time-to-event",
            "-t",
            help="Add the time-to-event head's predicted hours to the features?",
        ),
    ] = False,
    time_to_next_token: Annotated[
        bool,
        typer.Option(
            "--time-to-next-token",
            "-n",
            help="Add the time-to-next-token head's predicted hours, given the "
            "history alone, to the features?",
        ),
    ] = False,
    discharge_disposition: Annotated[
        bool,
        typer.Option(
            "--discharge-disposition",
            "-d",
            help="Add the disposition head's probability of each discharge "
            "disposition to the features?",
        ),
    ] = False,
    token_nll: Annotated[
        bool,
        typer.Option(
            "--token-nll",
            "-l",
            help="Add each token's negative log2-likelihood under the model, given "
            "the tokens before it, as a list as long as the tokens?",
        ),
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Extract representations from a trained model.
    """
    with console.status("[bold green]Extracting representations..."):
        t0 = time.perf_counter()
        extractor = Extractor(
            extraction_cfg=extraction_config,
            processed_data_home=processed_data_home,
            model_home=model_home,
            output_home=output_home,
            overrides=overrides,
        )
        extractor.extract(
            all_times=all_times,
            heads=[
                head
                for head, wanted in (
                    ("tte", time_to_event),
                    ("tnt", time_to_next_token),
                    ("disposition", discharge_disposition),
                )
                if wanted
            ],
            token_nll=token_nll,
        )
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Extraction completed in {t1 - t0:.2f}s.")
        print(f"  Features: [cyan]{extractor.output_home}[/cyan]")


@app.command()
def generative_score(
    scoring_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--scoring-config",
            "-s",
            help="Scoring configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    model_home: Annotated[
        str,
        typer.Option(
            "--model-home", "-m", help="Directory of the trained model to score with"
        ),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Output directory for scores, defaults to processed-data-home",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging")
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Generate SCORE/REACH metrics from a trained model and save them to parquet.
    """
    with console.status("[bold green]Generative scoring on held-out data..."):
        t0 = time.perf_counter()
        scorer = GenerativeScorer(
            scoring_cfg=scoring_config,
            processed_data_home=processed_data_home,
            model_home=model_home,
            output_home=output_home,
            overrides=overrides,
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Generative scoring completed in {t1 - t0:.2f}s.")
        print(f"  Scores: [cyan]{scorer.output_home}[/cyan]")


@app.command()
def rep_based_score(
    scoring_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--scoring-config",
            "-s",
            help="Scoring configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    model_home: Annotated[
        str,
        typer.Option(
            "--model-home", "-m", help="Directory of the trained model to score with"
        ),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Output directory for scores, defaults to processed-data-home",
            show_default=False,
        ),
    ] = None,
    training_home: Annotated[
        Optional[str],
        typer.Option(
            "--training-home",
            "-t",
            help="Use features and labels extracted here to train the model (transfer)",
            show_default=False,
        ),
    ] = None,
    estimator_type: Annotated[
        EstimatorType,
        typer.Option(
            "--estimator", "-e", help="Estimator to use for rep-based scoring"
        ),
    ] = EstimatorType.lightgbm,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging")
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Generate rep-based scores for the token-based outcomes of interest.
    Note: this requires that features have already been extracted and saved
    """

    with console.status("[bold green]Rep-based scoring on held-out data..."):
        t0 = time.perf_counter()
        scorer = RepBasedScorer(
            scoring_cfg=scoring_config,
            processed_data_home=processed_data_home,
            model_home=model_home,
            output_home=output_home,
            training_home=training_home,
            estimator_type=estimator_type.value,
            overrides=overrides,
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Rep-based scoring completed in {t1 - t0:.2f}s.")
        print(f"  Scores: [cyan]{scorer.output_home}[/cyan]")


@app.command()
def baseline_score(
    scoring_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--scoring-config",
            "-s",
            help="Scoring configuration file (overrides default)",
            show_default=False,
        ),
    ] = None,
    processed_data_home: Annotated[
        str,
        typer.Option("--processed-data-home", "-p", help="Processed data directory"),
    ] = ...,
    output_home: Annotated[
        Optional[str],
        typer.Option(
            "--output-home",
            "-o",
            help="Output directory for scores, defaults to processed-data-home",
            show_default=False,
        ),
    ] = None,
    training_home: Annotated[
        Optional[str],
        typer.Option(
            "--training-home",
            "-t",
            help="Fit the estimator on the train and tuning splits here (transfer)",
            show_default=False,
        ),
    ] = None,
    baseline: Annotated[
        Baseline,
        typer.Option(
            "--baseline",
            "-b",
            help="Features to fit on: counts or one-hot indicators of each "
            "timeline's tokens before the threshold, or tsfresh's minimal features "
            "of its meds.parquet rows before it",
        ),
    ] = Baseline.counts,
    estimator_type: Annotated[
        BaselineEstimator,
        typer.Option("--estimator", "-e", help="Estimator to fit on the features"),
    ] = BaselineEstimator.lightgbm,
    n_jobs: Annotated[
        Optional[int],
        typer.Option(
            "--n-jobs",
            "-j",
            help="Processes tsfresh extracts features in (0 for none but this "
            "one), half the CPUs by default",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging")
    ] = False,
    overrides: Overrides = None,
) -> None:
    """
    Generate baseline scores for the token-based outcomes of interest, from
    features of each timeline's past that need no trained model.
    """
    if n_jobs is not None and baseline is not Baseline.tsfresh:
        raise typer.BadParameter(
            "only the tsfresh baseline extracts in processes", param_hint="'--n-jobs'"
        )

    with console.status("[bold green]Baseline scoring on held-out data..."):
        t0 = time.perf_counter()
        dirs = dict(
            scoring_cfg=scoring_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
            training_home=training_home,
        )
        scorer = (
            TsfreshScorer(
                **dirs,
                estimator_type=estimator_type.value,
                n_jobs=n_jobs,
                overrides=overrides,
            )
            if baseline is Baseline.tsfresh
            else BagOfTokensScorer(
                **dirs,
                featurization=baseline.value,
                estimator_type=estimator_type.value,
                overrides=overrides,
            )
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Baseline scoring completed in {t1 - t0:.2f}s.")
        print(f"  Scores: [cyan]{scorer.output_home}[/cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
