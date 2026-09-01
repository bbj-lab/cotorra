#!/usr/bin/env python3

"""
CLI for cotorra - configurable training for generative event models
"""

import pathlib
import time
from importlib.metadata import version
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console

from cotorra.extractor import Extractor
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
            is_flag=True,
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging", is_flag=True)
    ] = False,
):
    """
    Train a model on tokenized data. For tokenization, consult the cocoa package.
    """
    with console.status("[bold green]Training model..."):
        t0 = time.perf_counter()
        trainer = Trainer(
            training_cfg=training_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint, verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Training completed in {t1 - t0:.2f}s.")
        out_path = trainer.output_home / f"mdl-{trainer.cfg.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def train_private(
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
    noise_multiplier: Annotated[
        Optional[float],
        typer.Option(
            "--noise-multiplier",
            "-n",
            help="Noise multiplier (overrides configuration)",
            show_default=False,
        ),
    ] = None,
    max_grad_norm: Annotated[
        Optional[float],
        typer.Option(
            "--max-grad-norm",
            "-m",
            help="Max grad norm (overrides configuration)",
            show_default=False,
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging", is_flag=True)
    ] = False,
):
    """
    Train a model with differential privacy on tokenized data.
    """
    from cotorra.trainer_dp import TrainerDP

    with console.status("[bold green]Training model with differential privacy..."):
        t0 = time.perf_counter()
        trainer = TrainerDP(
            training_cfg=training_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        trainer.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] DP training completed in {t1 - t0:.2f}s.")
        out_path = trainer.output_home / f"mdl-{trainer.cfg.run_name}"
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
        bool, typer.Option("--verbose", "-v", help="Verbose logging", is_flag=True)
    ] = False,
):
    """
    Run hyperparameter tuning while training a model.
    """
    with console.status("[bold green]Tuning model..."):
        t0 = time.perf_counter()
        tuner = Tuner(
            training_cfg=training_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        tuner.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Tuning completed in {t1 - t0:.2f}s.")
        out_path = tuner.output_home / f"mdl-{tuner.cfg.run_name}"
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
            is_flag=True,
        ),
    ] = False,
):
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
        )
        extractor.extract(all_times=all_times)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Extraction completed in {t1 - t0:.2f}s.")
        for split in extractor.loader.splits:
            print(f" Output in: {extractor.output_home}")


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
        bool, typer.Option("--verbose", "-v", help="Verbose logging", is_flag=True)
    ] = False,
):
    """
    Generate SCORE/REACH metrics from a trained model and save them to parquet.
    """
    from cotorra.scorer_generative import GenerativeScorer  # only loads if called

    with console.status("[bold green]Generative scoring on held-out data..."):
        t0 = time.perf_counter()
        scorer = GenerativeScorer(
            scoring_cfg=scoring_config,
            processed_data_home=processed_data_home,
            model_home=model_home,
            output_home=output_home,
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
    estimator_type: Annotated[
        EstimatorType,
        typer.Option(
            "--estimator", "-e", help="Estimator to use for rep-based scoring"
        ),
    ] = EstimatorType.lightgbm,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose logging", is_flag=True)
    ] = False,
):
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
            estimator_type=estimator_type.value,
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Rep-based scoring completed in {t1 - t0:.2f}s.")
        print(f"  Scores: [cyan]{scorer.output_home}[/cyan]")


def main():
    app()


if __name__ == "__main__":
    main()
