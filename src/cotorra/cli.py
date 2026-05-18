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
from cotorra.scorer_rep_based import RepBasedScorer
from cotorra.trainer import Trainer
from cotorra.tuner import Tuner

__version__ = version("cotorra")

app = typer.Typer(
    name="cotorra",
    help=f"Configurable training for generative event models (v{__version__})",
)
console = Console()


@app.command()
def train(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    model_config: Annotated[
        Optional[pathlib.Path],
        typer.Option("--model-config", help="Model configuration file"),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
):
    """
    Train a model on tokenized data. For tokenization, consult the cocoa package.
    """
    with console.status("[bold green]Training model..."):
        t0 = time.perf_counter()
        trainer = Trainer(
            main_cfg=main_config,
            mdl_cfg=model_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        trainer.train(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Training completed in {t1 - t0:.2f}s.")
        out_path = trainer.output_home / f"mdl-{trainer.cfg.run_name}"
        print(f"  Model: [cyan]{out_path}[/cyan]")


@app.command()
def tune(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    model_config: Annotated[
        Optional[pathlib.Path],
        typer.Option("--model-config", help="Model configuration file"),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
):
    """
    Run hyperparameter tuning while training a model.
    """
    with console.status("[bold green]Tuning model..."):
        t0 = time.perf_counter()
        tuner = Tuner(
            main_cfg=main_config,
            mdl_cfg=model_config,
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
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for trained models"),
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
            main_cfg=main_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        extractor.extract(all_times=all_times)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Extraction completed in {t1 - t0:.2f}s.")
        for split in extractor.loader.splits:
            a = "-all" if all_times else ""
            output = extractor.processed_data_home / f"features{a}-{split}.parquet"
            print(f" Output: {output}")


@app.command()
def generative_score(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    output_home: Annotated[
        Optional[str],
        typer.Option("--output-home", "-o", help="Output directory for score files"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
):
    """
    Generate SCORE/REACH metrics from a trained model and save them to parquet.
    """
    from cotorra.scorer_generative import GenerativeScorer  # only loads if called

    with console.status("[bold green]Generative scoring on held-out data..."):
        t0 = time.perf_counter()
        scorer = GenerativeScorer(
            main_cfg=main_config,
            processed_data_home=processed_data_home,
            output_home=output_home,
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Generative scoring completed in {t1 - t0:.2f}s.")
        out_path = (
            scorer.output_home
            / f"scores-generative-{scorer.cfg.wandb.run_name}.parquet"
        )
        print(f"  Scores: [cyan]{out_path}[/cyan]")


@app.command()
def rep_based_score(
    main_config: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "--main-config", "-m", help="Main configuration file (overrides default)"
        ),
    ] = None,
    processed_data_home: Annotated[
        Optional[str],
        typer.Option(
            "--processed-data-home",
            "-p",
            help="Processed data directory (overrides config)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Verbose logging for collate", is_flag=True
        ),
    ] = False,
    ridge: Annotated[
        bool,
        typer.Option("--ridge", help="Use ridge regression (alpha=0.1) instead of LightGBM", is_flag=True),
    ] = False,
):
    """
    Generate rep-based scores for the token-based outcomes of interest.
    """

    with console.status("[bold green]Rep-based scoring on held-out data..."):
        t0 = time.perf_counter()
        scorer = RepBasedScorer(
            main_cfg=main_config, processed_data_home=processed_data_home, ridge=ridge
        )
        scorer.save_all(verbose=verbose)
        t1 = time.perf_counter()
        print(f"\n[green]✓[/green] Rep-based scoring completed in {t1 - t0:.2f}s.")
        scorer_tag = "ridge" if ridge else "rep-based"
        out_path = (
            scorer.processed_data_home
            / f"scores-{scorer_tag}-{scorer.cfg.run_name}.parquet"
        )
        print(f"  Scores: [cyan]{out_path}[/cyan]")


def main():
    app()


if __name__ == "__main__":
    main()
