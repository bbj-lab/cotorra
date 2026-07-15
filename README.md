<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cotorra/master/img/cotorra.png" width="400" style="display: block;
margin: 0 auto; -webkit-mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );
  mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );"/>
</p>

# Cotorra: a configurable trainer

[![PyPI Version](https://img.shields.io/pypi/v/cotorra)](https://pypi.org/project/cotorra/)
[![DOI](https://raw.githubusercontent.com/burkh4rt/cotorra/master/img/1193885071.svg)](https://doi.org/10.5281/zenodo.20414127)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/cotorra/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/cotorra)

> 🦜 the wild parakeet of Chicago's south side

## About

This repo provides a configurable trainer for generative event models on
tokenized timelines. _Cotorra_ is a Spanish term for a small-to-medium sized
parrot, particularly the Monk parakeet. Monk parakeets were introduced to the
south side of Chicago, where they have flourished. [^1] It benefits from previous
experience training foundation models on tokenized electronic health records.
[^2] [^3] [^4] [^5]

Given a dataset of tokenized timelines, this package trains a model to predict
the next token in a subject's timeline given their history up to that point, and
then uses the trained model to extract representations and score outcomes of
interest. It does all of this in a configurable way.

<!-- cards-anchor -->

## Installation

Install the latest release from PyPI:

```sh
pip install "cotorra" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

This installs the `cotorra` command. To work from source instead (e.g. to use
generative scoring or for development):

```sh
git clone git@github.com:bbj-lab/cotorra.git
cd cotorra
python -m venv .venv
. .venv/bin/activate
pip install -e ".[gen]" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## Inputs

Suppose you have a dataset of tokenized timelines `tokens_times.parquet` as a
parquet table with columns:

- `subject_id`
- `tokens` — the integer token sequence for the subject's timeline.
- `times` — a parallel list of timestamps, one per token, indicating when each
  event occurred.

The table will look something like this:

```
┌────────────────────┬─────────────────┬─────────────────────────────────┐
│ subject_id         ┆ tokens          ┆ times                           │
│ ---                ┆ ---             ┆ ---                             │
│ str                ┆ list[u32]       ┆ list[datetime[μs]]              │
╞════════════════════╪═════════════════╪═════════════════════════════════╡
│ 20002103           ┆ [20, 350, … 21] ┆ [2116-05-08 02:45:00, 2116-05-… │
│ 20008372           ┆ [20, 350, … 21] ┆ [2110-10-30 13:03:00, 2110-10-… │
│ …                  ┆ …               ┆ …                               │
│ 29994865           ┆ [20, 364, … 21] ┆ [2111-01-28 21:49:00, 2111-01-… │
└────────────────────┴─────────────────┴─────────────────────────────────┘
```

You also have a `tokenizer.yaml`, a plain yaml file that contains information
about the configuration, learned vocabulary, and bins. This file is sufficient to
reconstitute the tokenizer object. We only need this file to contain a lookup
table:

```yaml
lookup:
  UNK: 0
  ADMN//direct: 1
  ADMN//ed: 2
  ADMN//elective: 3
  AGE//age_Q0: 4
  ...
```

Finally, we need `subject_splits.parquet` which is a table listing out all
subject_id's and their corresponding split assignment (with splits: `train`,
`tuning`, and `held_out`). This table may include additional demographic
information provided as pass-through-columns to
[cocoa](https://github.com/bbj-lab/cocoa).

```
┌────────────┬──────────┐
│ subject_id ┆ split    │
│ ---        ┆ ---      │
│ str        ┆ str      │
╞════════════╪══════════╡
│ 21081215   ┆ train    │
│ 20302177   ┆ train    │
│ …          ┆ …        │
│ 28150003   ┆ held_out │
│ 22151813   ┆ held_out │
└────────────┴──────────┘
```

For extraction and scoring workflows, we also need split-specific inference
tables in the same `processed_data_home` directory:

- `train_for_inference.parquet`
- `tuning_for_inference.parquet`
- `held_out_for_inference.parquet`

These tables are expected to include at least:

- `tokens_past` (the model context used for extraction/scoring)
- `s_elapsed_past` (if using `time_based_rope`)
- token-specific label columns such as `<TOKEN>_past` and `<TOKEN>_future` used
  by generative and representation-based scoring.

The `cocoa winnow` command provides these.

<!-- prettier-ignore-start -->
> [!TIP]
> For getting your data to this point, check out our configurable
> collator / tokenizer: [☕️ cocoa](https://github.com/bbj-lab/cocoa)
<!-- prettier-ignore-end -->

Each command below is driven by a YAML config. The package ships a default for
each command under `src/cotorra/config/`, which you can override by passing a
config file via the appropriate CLI flag.

## (1) Training

The trainer consumes the tokenized timelines and fits a causal language model to
predict the next token in each subject's timeline. It:

1. Builds a next-token-prediction dataset from `tokens_times.parquet` and the
   subject splits.
2. Initializes a HuggingFace causal LM from a preset (or a custom architecture
   config).
3. Optionally applies custom losses that upweight quantile-boundary tokens or
   tokens of clinical interest.
4. Optionally uses time-aware rotary position embeddings so that position ids
   reflect elapsed time rather than token index.
5. Trains the model — optionally with differential privacy
   (`cotorra train-private`) or hyperparameter tuning (`cotorra tune`) — and
   saves it.

Training is driven by a YAML config (the package ships a default; see
[`./src/cotorra/config/training.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/training.yaml))
that specifies:

- **model**:
    - **model_name**: Name or path of the HuggingFace model (e.g.,
      `meta-llama/Llama-3.2-1B`).
    - **model_args**: Model architecture parameters passed directly to
      HuggingFace's
      [`AutoConfig`](https://huggingface.co/docs/transformers/en/model_doc/auto).

    _Note: The bundled config defines reusable model presets under
    `model_presets`._

- **max_seq_len**: Maximum sequence length for model input.
- **n_epochs**: Number of epochs (handled in the dataloader, not the trainer).
- **run_name**: Name for the current run (referenced by `wandb` and
  `training_args`).
- **tokens_of_interest**: List of special tokens to upweight during training
  (referenced by loss config). Supports patterns specified with fnmatch.
- **wandb**:
    - **project**: Weights & Biases project name for experiment tracking.
    - **run_name**: Name for the current run.
- **custom_loss**: Boolean flag to enable custom loss functions (default:
  `false`).
- **quantile_token_loss** _(optional)_: Upweights loss on quantile boundary
  tokens.
    - **qt_weight**: Weight multiplier for quantile tokens.
- **label_weighted_loss** _(optional)_: Upweights loss on specific tokens of
  clinical interest.
    - **tokens_of_interest**: List of token labels to upweight. Supports patterns
      specified with fnmatch.
    - **toi_weight**: Weight multiplier applied to those tokens.
- **time_based_rope** _(optional)_: Enables time-aware rotary position
  embeddings.
    - **sec_per_pos_id**: Number of seconds represented by one position id
      increment.
- **training_args**: Arguments passed to HuggingFace's
  [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).
- **tuning_args**: Arguments passed to HuggingFace's
  [`hyperparameter_search`](https://huggingface.co/docs/transformers/hpo_train?backends=Optuna)
  when `cotorra tune` is called.

### Model presets

We offer the following presets:

| designator     | base model                | # params w/ ~1340-token vocab |
| -------------- | ------------------------- | ----------------------------- |
| `llama_32`     | `meta-llama/Llama-3.2-1B` | ~76.9M                        |
| `llama_32_mid` | `meta-llama/Llama-3.2-1B` | ~8.2M                         |
| `qwen_3`       | `Qwen/Qwen3-1.7B-Base`    | ~74.1M                        |
| `qwen_3_mid`   | `Qwen/Qwen3-1.7B-Base`    | ~8.4M                         |
| `gemma_3`      | `google/gemma-3-1b-pt`    | ~75.7M                        |
| `gemma_3_mid`  | `google/gemma-3-1b-pt`    | ~7.8M                         |

Use the `model` key to select one of these presets and then override any
individual `model_args` entries as needed.

<!-- prettier-ignore-start -->
> [!TIP]
> Training supports the `--resume-from-checkpoint` (`-r`) flag. When set,
> `cotorra train` will attempt to resume from the latest HuggingFace checkpoint
> saved under `--output-home`. If no checkpoint is found (or resumption fails),
> it automatically falls back to training from scratch — so the flag is safe to
> pass unconditionally in scripts. Use `save_steps` in `training_args` in the
> [training.yaml](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/training.yaml) file to control the frequency
> of checkpointing.
<!-- prettier-ignore-end -->

### Differential privacy

We wrap [opacus](https://opacus.ai) to support training with differential privacy
(see `cotorra train-private`). The following relevant parameters can be modified
in the configuration:

```yaml
privacy_parameters:
    noise_multiplier: !!float 1.0
    max_grad_norm: !!float 1.0
```

### Outputs

- `mdl-<run_name>/` — the trained model, saved under `--output-home` in
  HuggingFace format (via `save_pretrained`), ready to be passed as
  `--model-home` to `extract` and the scoring commands.
- `mdl-<run_name>-training.yaml` — the resolved training configuration used for
  the run.

## (2) Extraction

The extractor loads a trained model and computes hidden-state representations of
each subject's context, suitable for representation-based scoring or downstream
tasks. It:

1. Loads the trained model from `--model-home` and the split-specific inference
   tables.
2. Runs the model over each subject's `tokens_past` context.
3. Extracts the hidden-state representation at the final position by default, or
   at all time steps when `--all-times` is set.
4. Writes one feature table per split (optionally sharded).

Extraction is driven by a YAML config (the package ships a default; see
[`./src/cotorra/config/extraction.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/extraction.yaml))
that specifies:

- **max_seq_len**: Maximum sequence length.
- **time_based_rope** _(optional)_: Enables time-aware position ids during
  extraction (must match the setting used at training time).
    - **sec_per_pos_id**: Number of seconds represented by one position id
      increment.
- **extract**:
    - **max_len**: Maximum input length (tokens) during extraction.
    - **batch_size**: Batch size for inference.
    - **shard_size** _(optional)_: Number of samples per output parquet shard.
      Omit to write a single file per split.

### Outputs

- `features-<split>-<model_name>.parquet` — extracted representations for each
  split (`train`, `tuning`, `held_out`). With `--all-times`, files are named
  `features-all-<split>-<model_name>.parquet`; when `shard_size` is set, each
  split is written across `-<index>-of-<count>` shards. These files are the input
  to `cotorra rep-based-score`.

## (3) Scoring

Scoring uses a trained model to produce outcome scores for the tokens of
interest. Two complementary approaches are provided:

**Generative scoring** (`cotorra generative-score`) Monte Carlo samples future
trajectories directly from the model. It:

1. Loads the trained model and held-out inference data.
2. Samples future trajectories for each target token.
3. Computes MC, SCOPE, and REACH scores per target token.

Note this depends on the
[quick-sco-re](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git)
package.

**Representation-based scoring** (`cotorra rep-based-score`) fits a lightweight
estimator on extracted features (run `cotorra extract` first). It:

1. Loads the extracted features and label columns.
2. Fits the chosen estimator on the training split.
3. Predicts outcome probabilities for the held-out split.

Both are driven by a YAML config (the package ships a default; see
[`./src/cotorra/config/scoring.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/scoring.yaml))
that specifies:

- **run_name**: Name for the current run, used to label output files.
- **tokens_of_interest**: List of token-based outcomes of interest. Supports
  patterns specified with fnmatch. (Referenced by target tokens.)
- **score**:
    - **max_len**: Maximum input length (tokens) during scoring.
    - **n_samp**: Number of Monte Carlo samples per input per trajectory type.
    - **target_tokens**: Token-based outcomes of interest to score. Supports
      patterns specified with fnmatch.
    - **end_tokens**: Tokens that naturally terminate a generated sequence (e.g.
      `EOS`).
    - **suppressed_tokens**: Tokens to suppress via logit bias during generation
      (e.g. `PAD`).
    - **trunc_id**: Token id forced after the time horizon is exceeded.
    - **max_time**: Maximum time horizon in minutes.
    - **batch_size**: Batch size for inference.

### Outputs

- `scores-generative-<model_name>.parquet` — held-out scores from
  `generative-score`, with a `<TOKEN>_mc_score`, `<TOKEN>_scope_score`, and
  `<TOKEN>_reach_score` column for each target token.
- `scores-rep-based-<model_name>.parquet` — held-out scores from
  `rep-based-score`, with a `<TOKEN>_rep_score` column for each target token.

## Usage

We provide a CLI:

```
 Usage: cotorra [OPTIONS] COMMAND [ARGS]...

 Configurable training for generative event models (v26.6.1)

╭─ Options ───────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current      │
│                                 shell.                                  │
│ --show-completion               Show completion for the current shell,  │
│                                 to copy it or customize the             │
│                                 installation.                           │
│ --help                -h        Show this message and exit.             │
╰─────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────╮
│ train             Train a model on tokenized data. For tokenization,    │
│                   consult the cocoa package.                            │
│ train-private     Train a model with differential privacy on tokenized  │
│                   data.                                                 │
│ tune              Run hyperparameter tuning while training a model.     │
│ extract           Extract representations from a trained model.         │
│ generative-score  Generate SCORE/REACH metrics from a trained model and │
│                   save them to parquet.                                 │
│ rep-based-score   Generate rep-based scores for the token-based         │
│                   outcomes of interest.                                 │
│                   Note: this requires that features have already been   │
│                   extracted and saved                                   │
╰─────────────────────────────────────────────────────────────────────────╯
```

with commands:

- `cotorra train`

    ```
    Usage: cotorra train [OPTIONS]

    Train a model on tokenized data. For tokenization, consult the cocoa
    package.

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --training-config         -t      PATH  Training configuration file  │
    │                                            (overrides default)          │
    │ *  --processed-data-home     -p      TEXT  Processed data directory     │
    │                                            (overrides config)           │
    │                                            [required]                   │
    │ *  --output-home             -o      TEXT  Output directory for trained │
    │                                            models                       │
    │                                            [required]                   │
    │    --resume-from-checkpoint  -r            Try to resume training from  │
    │                                            the latest checkpoint in     │
    │                                            --output-home.               │
    │    --verbose                 -v            Verbose logging              │
    │    --help                    -h            Show this message and exit.  │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra tune`

    ```
    Usage: cotorra tune [OPTIONS]

    Run hyperparameter tuning while training a model.

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --training-config      -t      PATH  Training configuration file     │
    │                                         (overrides default)             │
    │ *  --processed-data-home  -p      TEXT  Processed data directory        │
    │                                         (overrides config)              │
    │                                         [required]                      │
    │ *  --output-home          -o      TEXT  Output directory for trained    │
    │                                         models                          │
    │                                         [required]                      │
    │    --verbose              -v            Verbose logging                 │
    │    --help                 -h            Show this message and exit.     │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra train-private`

    ```
    Usage: cotorra train-private [OPTIONS]

    Train a model with differential privacy on tokenized data.

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --training-config      -t      PATH   Training configuration file    │
    │                                          (overrides default)            │
    │ *  --processed-data-home  -p      TEXT   Processed data directory       │
    │                                          (overrides config)             │
    │                                          [required]                     │
    │ *  --output-home          -o      TEXT   Output directory for trained   │
    │                                          models                         │
    │                                          [required]                     │
    │    --noise-multiplier     -n      FLOAT  Noise multiplier (overrides    │
    │                                          configuration)                 │
    │    --max-grad-norm        -m      FLOAT  Max grad norm (overrides       │
    │                                          configuration)                 │
    │    --verbose              -v             Verbose logging                │
    │    --help                 -h             Show this message and exit.    │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra extract`

    ```
    Usage: cotorra extract [OPTIONS]

    Extract representations from a trained model.

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --extraction-config    -e      PATH  Extraction configuration file   │
    │                                         (overrides default)             │
    │ *  --processed-data-home  -p      TEXT  Processed data directory        │
    │                                         [required]                      │
    │ *  --model-home           -m      TEXT  Directory of the trained model  │
    │                                         to extract from                 │
    │                                         [required]                      │
    │    --output-home          -o      TEXT  Output directory for extracted  │
    │                                         features, defaults to           │
    │                                         processed-data-home             │
    │    --all-times            -a            Extract features for all time   │
    │                                         steps (instead of just the      │
    │                                         final one)?                     │
    │    --help                 -h            Show this message and exit.     │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra generative-score`

    ```
    Usage: cotorra generative-score [OPTIONS]

    Generate SCORE/REACH metrics from a trained model and save them to
    parquet.

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --scoring-config       -s      PATH  Scoring configuration file      │
    │                                         (overrides default)             │
    │ *  --processed-data-home  -p      TEXT  Processed data directory        │
    │                                         [required]                      │
    │ *  --model-home           -m      TEXT  Directory of the trained model  │
    │                                         to score with                   │
    │                                         [required]                      │
    │    --output-home          -o      TEXT  Output directory for scores,    │
    │                                         defaults to processed-data-home │
    │    --verbose              -v            Verbose logging                 │
    │    --help                 -h            Show this message and exit.     │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra rep-based-score` (note: you need to run `extract` first)

    ```
    Usage: cotorra rep-based-score [OPTIONS]

    Generate rep-based scores for the token-based outcomes of interest. Note:
    this requires that features have already been extracted and saved

    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --scoring-config    -s      PATH                 Scoring             │
    │                                                     configuration file  │
    │                                                     (overrides default) │
    │ *  --processed-data-…  -p      TEXT                 Processed data      │
    │                                                     directory           │
    │                                                     [required]          │
    │ *  --model-home        -m      TEXT                 Directory of the    │
    │                                                     trained model to    │
    │                                                     score with          │
    │                                                     [required]          │
    │    --output-home       -o      TEXT                 Output directory    │
    │                                                     for scores,         │
    │                                                     defaults to         │
    │                                                     processed-data-home │
    │    --estimator         -e      [k-NN|lightGBM|logi  Estimator to use    │
    │                                stic|logistic-z|log  for rep-based       │
    │                                istic-CV|logistic-C  scoring             │
    │                                V-z|XGBoost]         [default: lightGBM] │
    │    --verbose           -v                           Verbose logging     │
    │    --help              -h                           Show this message   │
    │                                                     and exit.           │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

[^1]:
    L. Gersony, "The Quiet Victory of Chicago’s Monk Parakeets," _The Chicago
    Maroon_, 23 January 2022,
    [https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/](https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/)

[^2]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^3]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188

[^4]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^5]:
    I. Lee, L. Solo, M. Burkhart, B. Ramadan, W. Parker, & B. Beaulieu-Jones,
    Representation before training: a fixed-budget benchmark for generative
    medical event models,
    [arXiv:2604.16775](https://doi.org/10.48550/arXiv.2604.16775)

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "output" \
 --exclude "wandb" \
 --exclude ".venv" \
 --exclude ".idea" \
 ~/Documents/chicago/cotorra \
 bbj-lab1:~
```

```
for d in data-raw processed; do ln -s /mnt/bbj-lab/users/burkh4rt/$d $d; done
ds='mimic-icu'
cotorra train \
		--training-config src/cotorra/config/training.yaml \
		--processed-data-home processed/mimic-icu \
		--output-home output/test \
    --resume-from-checkpoint \
    --verbose
```

Send to randi:
```
for d in data-raw processed; do
	ln -s /gpfs/data/bbj-lab/users/burkh4rt/$d $d
done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cotorra \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

```
srun -p gpuq \
 --gres=gpu:1 \
 --time=8:00:00 \
 --job-name=adhoc \
 --pty bash -i
. .venv/bin/activate
```

Send to pypi:
```
rm -rf dist
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/*
```

Make docs:
```
mkdocs build
mkdocs serve --dev-addr 127.0.0.1:8002
```

Make tag:
```
git tag -s v26.6.1 -m "docs may be read"
```

-->
