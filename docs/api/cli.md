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
| `cotorra baseline-score`   | Score held-out timelines with a baseline needing no trained model.    |

Every command accepts `--processed-data-home` / `-p` (the directory of tokenized
inputs and intermediate artifacts) and an optional config file that overrides the
packaged default for that stage (`-t` for training, `-e` for extraction, `-s` for
scoring). Most also take `--output-home` / `-o` and `--verbose` / `-v`; the
extraction and scoring commands additionally require `--model-home` / `-m`, the
trained model to run (all but `baseline-score`, which needs none).

To change individual config keys for one run, list them after the command's
options, as in
`cotorra train -p ./processed/mimic -o ./models n_epochs=2 '~time_based_rope'`;
see [Overriding config keys](../guide/configuration.md#overriding-config-keys)
for the syntax.

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

To set those scores against baselines that need no trained model, score the same
outcomes from a bag of each timeline's tokens before the threshold (`counts` or
`one-hot`), or from tsfresh's minimal features of its `meds.parquet` rows before
it (`tsfresh`):

```sh
cotorra baseline-score \
    --processed-data-home ./processed/mimic \
    --baseline counts \
    --estimator logistic
```

## Command reference

Each command's `--help`:

### `cotorra train`

```
 Usage: cotorra train [OPTIONS] [overrides]...

 Train a model on tokenized data. For tokenization, consult the cocoa
 package.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --training-config         -t      <path>  Training configuration     │
│                                              file (overrides default)   │
│ *  --processed-data-home     -p      <str>   Processed data directory   │
│                                              (overrides config)         │
│                                              [required]                 │
│ *  --output-home             -o      <str>   Output directory for       │
│                                              trained models             │
│                                              [required]                 │
│    --resume-from-checkpoint  -r              Try to resume training     │
│                                              from the latest checkpoint │
│                                              in --output-home.          │
│    --verbose                 -v              Verbose logging            │
│    --help                    -h              Show this message and      │
│                                              exit.                      │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cotorra tune`

```
 Usage: cotorra tune [OPTIONS] [overrides]...

 Run hyperparameter tuning while training a model.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --training-config      -t      <path>  Training configuration file   │
│                                           (overrides default)           │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           (overrides config)            │
│                                           [required]                    │
│ *  --output-home          -o      <str>   Output directory for trained  │
│                                           models                        │
│                                           [required]                    │
│    --verbose              -v              Verbose logging               │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cotorra extract`

```
 Usage: cotorra extract [OPTIONS] [overrides]...

 Extract representations from a trained model.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --extraction-config      -e      <path>  Extraction configuration    │
│                                             file (overrides default)    │
│ *  --processed-data-home    -p      <str>   Processed data directory    │
│                                             [required]                  │
│ *  --model-home             -m      <str>   Directory of the trained    │
│                                             model to extract from       │
│                                             [required]                  │
│    --output-home            -o      <str>   Output directory for        │
│                                             extracted features,         │
│                                             defaults to                 │
│                                             processed-data-home         │
│    --all-times              -a              Extract features for all    │
│                                             time steps (instead of just │
│                                             the final one)?             │
│    --time-to-event          -t              Add the time-to-event       │
│                                             head's predicted hours to   │
│                                             the features?               │
│    --time-to-next-token     -n              Add the time-to-next-token  │
│                                             head's predicted hours,     │
│                                             given the history alone, to │
│                                             the features?               │
│    --discharge-disposition  -d              Add the disposition head's  │
│                                             probability of each         │
│                                             discharge disposition to    │
│                                             the features?               │
│    --token-nll              -l              Add each token's negative   │
│                                             log2-likelihood under the   │
│                                             model, given the tokens     │
│                                             before it, as a list as     │
│                                             long as the tokens?         │
│    --help                   -h              Show this message and exit. │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cotorra generative-score`

```
 Usage: cotorra generative-score [OPTIONS] [overrides]...

 Generate SCORE/REACH metrics from a trained model and save them to
 parquet.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --scoring-config       -s      <path>  Scoring configuration file    │
│                                           (overrides default)           │
│ *  --processed-data-home  -p      <str>   Processed data directory      │
│                                           [required]                    │
│ *  --model-home           -m      <str>   Directory of the trained      │
│                                           model to score with           │
│                                           [required]                    │
│    --output-home          -o      <str>   Output directory for scores,  │
│                                           defaults to                   │
│                                           processed-data-home           │
│    --verbose              -v              Verbose logging               │
│    --help                 -h              Show this message and exit.   │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cotorra rep-based-score`

```
 Usage: cotorra rep-based-score [OPTIONS] [overrides]...

 Generate rep-based scores for the token-based outcomes of interest. Note:
 this requires that features have already been extracted and saved

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --scoring-config    -s      <path>               Scoring             │
│                                                     configuration file  │
│                                                     (overrides default) │
│ *  --processed-data-…  -p      <str>                Processed data      │
│                                                     directory           │
│                                                     [required]          │
│ *  --model-home        -m      <str>                Directory of the    │
│                                                     trained model to    │
│                                                     score with          │
│                                                     [required]          │
│    --output-home       -o      <str>                Output directory    │
│                                                     for scores,         │
│                                                     defaults to         │
│                                                     processed-data-home │
│    --training-home     -t      <str>                Use features and    │
│                                                     labels extracted    │
│                                                     here to train the   │
│                                                     model (transfer)    │
│    --estimator         -e      <k-NN|lightGBM|logi  Estimator to use    │
│                                stic|logistic-z|log  for rep-based       │
│                                istic-CV|logistic-C  scoring             │
│                                V-z|XGBoost>         [default: lightGBM] │
│    --verbose           -v                           Verbose logging     │
│    --help              -h                           Show this message   │
│                                                     and exit.           │
╰─────────────────────────────────────────────────────────────────────────╯
```

### `cotorra baseline-score`

```
 Usage: cotorra baseline-score [OPTIONS] [overrides]...

 Generate baseline scores for the token-based outcomes of interest, from
 features of each timeline's past that need no trained model.

╭─ Arguments ─────────────────────────────────────────────────────────────╮
│   overrides      <str>  Config overrides: key=value sets a key, adding  │
│                         it if need be, and ~key deletes one             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────╮
│    --scoring-config    -s      <path>               Scoring             │
│                                                     configuration file  │
│                                                     (overrides default) │
│ *  --processed-data-…  -p      <str>                Processed data      │
│                                                     directory           │
│                                                     [required]          │
│    --output-home       -o      <str>                Output directory    │
│                                                     for scores,         │
│                                                     defaults to         │
│                                                     processed-data-home │
│    --training-home     -t      <str>                Fit the estimator   │
│                                                     on the train and    │
│                                                     tuning splits here  │
│                                                     (transfer)          │
│    --baseline          -b      <counts|one-hot|tsf  Features to fit on: │
│                                resh>                counts or one-hot   │
│                                                     indicators of each  │
│                                                     timeline's tokens   │
│                                                     before the          │
│                                                     threshold, or       │
│                                                     tsfresh's minimal   │
│                                                     features of its     │
│                                                     meds.parquet rows   │
│                                                     before it           │
│                                                     [default: counts]   │
│    --estimator         -e      <lightGBM|logistic|  Estimator to fit on │
│                                XGBoost>             the features        │
│                                                     [default: lightGBM] │
│    --n-jobs            -j      <int>                Processes tsfresh   │
│                                                     extracts features   │
│                                                     in (0 for none but  │
│                                                     this one), half the │
│                                                     CPUs by default     │
│    --verbose           -v                           Verbose logging     │
│    --help              -h                           Show this message   │
│                                                     and exit.           │
╰─────────────────────────────────────────────────────────────────────────╯
```

---

::: cotorra.cli
