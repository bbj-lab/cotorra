# CLI

Cotorra ships a command-line interface, `cotorra`, that drives every stage of the
modeling pipeline: training a generative event model, extracting its
representations, and turning it into predictions. Each stage has its own command.
Tokenized inputs are produced upstream by the
[cocoa](https://github.com/bbj-lab/cocoa) package.

## Commands

| Command                    | What it does                                                          |
| -------------------------- | --------------------------------------------------------------------- |
| `cotorra train`            | Train a causal language model on tokenized timelines.                 |
| `cotorra tune`             | Train while searching over hyperparameters.                           |
| `cotorra extract`          | Extract hidden-state representations from a trained model.            |
| `cotorra generative-score` | Score held-out timelines by autoregressive generation.                |
| `cotorra rep-based-score`  | Score held-out timelines with a classifier fit on extracted features. |

Every command accepts `--processed-data-home` / `-p` (the directory of tokenized
inputs and intermediate artifacts) and an optional config file that overrides the
packaged default for that stage (`-t` for training, `-e` for extraction, `-s` for
scoring). Most also take `--output-home` / `-o` and `--verbose` / `-v`; the
extraction and scoring commands additionally require `--model-home` / `-m`, the
trained model to run.

Run any command with `-h` / `--help` to see its full set of options:

```sh
cotorra --help
cotorra train --help
```

## Typical usage

Train a model, extract its representations, and score held-out data:

```sh
cotorra train \
    --processed-data-home ./processed/mimic \
    --output-home ./models \
    --verbose

cotorra extract \
    --processed-data-home ./processed/mimic \
    --model-home ./models/mdl-<run_name>

cotorra rep-based-score \
    --processed-data-home ./processed/mimic \
    --model-home ./models/mdl-<run_name>
```

Or score directly from the trained model by autoregressive generation, instead of
fitting a classifier on extracted representations:

```sh
cotorra generative-score \
    --processed-data-home ./processed/mimic \
    --model-home ./models/mdl-<run_name>
```

---

::: cotorra.cli
