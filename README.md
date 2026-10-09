<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cotorra/master/img/cotorra.png" alt="friendly parrot" width="400" />
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
[^2] [^3] [^4] [^5] [^6]

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
[cocoa-tokenizer](https://pypi.org/project/cocoa-tokenizer/).

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

- `subject_id` (optional; carried into the extracted feature tables)
- `tokens_past` (the model context used for extraction/scoring)
- `s_elapsed_past` (if using `time_based_rope`)
- token-specific label columns such as `<TOKEN>_past` and `<TOKEN>_future` used
  by generative and representation-based scoring.

The `cocoa winnow` command provides these.

<!-- prettier-ignore-start -->
> [!TIP]
> For getting your data to this point, check out our configurable
> collator / tokenizer: [☕️ cocoa-tokenizer](https://pypi.org/project/cocoa-tokenizer/)
<!-- prettier-ignore-end -->

Each command below is driven by a YAML config. The package ships a default for
each command under `src/cotorra/config/`, which you can override by passing a
config file via the appropriate CLI flag, and whose individual keys you can
change on the command line (see
[Overriding config keys](#overriding-config-keys)).

## (1) Training

The trainer consumes the tokenized timelines and fits a causal language model to
predict the next token in each subject's timeline. It:

1. Builds a next-token-prediction dataset from `tokens_times.parquet` and the
   subject splits.
2. Initializes a HuggingFace causal LM from a preset (or a custom architecture
   config).
3. Optionally applies custom losses that teach the ordering of quantile bins or
   emphasize tokens of clinical interest.
4. Optionally uses time-aware rotary position embeddings so that position ids
   reflect elapsed time rather than token index.
5. Optionally trains secondary heads alongside next-token prediction: time to
   event, time to next token, and discharge disposition, in any combination.
6. Trains the model — optionally with hyperparameter tuning (`cotorra tune`) —
   and saves it.

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
- **quantile_token_loss** _(optional)_: Teaches the model that a code's quantile
  bins are ordered. Cross-entropy alone counts predicting `Q9` for a true `Q3` as
  no worse than predicting `Q4`. For each quantile token, this adds the squared
  error between the bin the model expects (the probability-weighted midpoint of
  the code's bins on a 0–1 quantile scale) and the bin that came, averaged over
  quantile tokens. It only moves probability among a code's bins, never making
  the code itself likelier, and like cross-entropy it's minimized by the true
  probabilities. Works with fused (`LAB//sodium_Q3`) and unfused (`LAB//sodium`
  then `Q3`) tokenizations.
    - **qt_weight**: Weight on the term (the bundled config uses `50.0`).
- **balanced_toi_loss** _(optional)_: Upweights getting specific tokens of
  clinical interest right, by adding a binary cross-entropy term on whether the
  next token is one of them (scored with the total probability the model puts on
  them). Like cross-entropy, it's minimized by the true probabilities, so it
  doesn't make those tokens likelier.
    - **tokens_of_interest**: List of token labels in the set. Supports patterns
      specified with fnmatch.
    - **bce_weight**: Weight multiplier for the term (default: `1.0`; the bundled
      config uses `20.0`).

    _Note:_ it replaces `label_weighted_loss`, which is deprecated: configs using
    it still train as before, but with a warning, since weighting the
    cross-entropy by its target makes the model predict those tokens more often.

- **tte_objective** _(optional)_: Trains a time-to-event (TTE) head that
  predicts, at each token, the log1p-hours remaining until the end of the record
  (a token recorded after the end time counts as 0 hours remaining). Requires an
  `hours_to_end_time` column in `tokens_times.parquet`. See
  [Training secondary heads](#training-secondary-heads).
    - **weight**: Weight on the term (default: `1.0`).
- **tnt_objective** _(optional)_: Trains a time-to-next-token (TNT) head that
  predicts, at each token, the log1p-hours until the next token, making the model
  a marked point process. Its target is derived from `times`, and a record's last
  token is left out.
    - **weight**: Weight on the term (default: `1.0`).
    - **mixture_components** _(optional)_: Makes the head predict a distribution
      rather than a single value: the chance that the next token shares this
      one's timestamp, plus a mixture of this many log-normals over the positive
      gap, conditioned on which token comes next and trained by likelihood.
      Needed to sample realistic times. Left unset, the head predicts log1p-hours
      by squared error.
- **disposition_objective** _(optional)_: Trains a supervised head that predicts,
  at each token, the record's discharge disposition, read off its `DSCG//*`
  token, by cross-entropy.
    - **weight**: Weight on the term (default: `1.0`).
    - **classes**: The dispositions to predict among, as token labels or fnmatch
      patterns (default: `["DSCG//*"]`). A record whose disposition matches none
      of them isn't scored.
- **time_based_rope** _(optional)_: Enables time-aware rotary position
  embeddings.
    - **sec_per_pos_id**: Number of seconds represented by one position id
      increment.

    _Note:_ SGLang (behind `generative-score`) and vLLM number positions by token
    index and can't take time-based ones, so they run a model trained with this
    on positions it never saw.

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

### Training secondary heads

Every model learns to predict the next token. Alongside that, it can train any
combination of three secondary heads, each with its own config block:

| block                   | head                     | predicts, at each token           | target                          |
| ----------------------- | ------------------------ | --------------------------------- | ------------------------------- |
| `tte_objective`         | time to event (TTE)      | hours until the record's end time | `hours_to_end_time`, from cocoa |
| `tnt_objective`         | time to next token (TNT) | hours until the next token        | derived from `times`            |
| `disposition_objective` | discharge disposition    | how the record ends               | the record's `DSCG//*` token    |

With any of them set, training writes a `cotorra` model: the selected preset plus
those heads, each trained jointly with next-token prediction.

The TNT head turns the model into a marked point process: the language-modelling
head predicts _what_ the next token is, and the TNT head _when_ it arrives. It
comes in two kinds. By default it predicts a single value per token (log1p-hours,
by squared error). With `mixture_components` set, it predicts a distribution over
the gap instead, given which token comes next, which is what sampling realistic
times needs.

The disposition head is supervised. At each token it predicts how the record will
end, for example the chance that the patient dies before discharge
(`DSCG//expired`). Its target comes from the record's own `DSCG//*` token: every
token before it is scored against that disposition, and the tokens from it on
aren't, since by then the disposition is known. A record whose disposition
matches none of `classes` isn't scored at all. `DSCG//*` includes
`DSCG//missing`; to leave it out, list the classes you want instead.

To train one or more:

1. **Configure training.** A config passed with `--training-config` replaces the
   shipped default rather than merging into it, so start from a full copy of
   [training.yaml](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/training.yaml)
   and add (or uncomment) the blocks for the heads you want:

    ```yaml
    tte_objective:
        weight: !!float 1.0
    tnt_objective:
        weight: !!float 1.0
        mixture_components: !!int 8 # optional: a distribution over the gap
    disposition_objective:
        weight: !!float 1.0
        classes: ["DSCG//*"] # optional: fnmatch patterns
    ```

    Each weight defaults to `1.0`, so empty blocks work too. With
    `custom_loss: true` (as shipped), the weighted terms are added to the custom
    next-token loss; otherwise the model adds them to HuggingFace's standard loss
    itself.

2. **Tokenize with end times (TTE head only).** The TTE head trains on an
   `hours_to_end_time` column, which cocoa writes only when asked. Copy cocoa's
   [default tokenization config](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/tokenization.yaml),
   set `include_hours_to_end_time: !!bool true`, and rerun tokenization and
   winnowing. Winnowing carries the column into the `*_for_inference.parquet`
   tables as `hours_to_end_time_past`, which cotorra also expects.

    ```sh
    cocoa tokenize -c tokenization.yaml -p processed/
    cocoa winnow -p processed/
    ```

    Skip this step for the other heads: cotorra derives their targets from
    `tokens_times.parquet`.

3. **Train** as usual:

    ```sh
    cotorra train -t training.yaml -p processed/ -o output/
    ```

    The startup log names the heads, e.g.
    `<model_name> (with tte, tnt, disposition heads)`, and the run writes
    `mdl-<run_name>/` as a `cotorra` model.

4. **Use the trained model.** Importing `cotorra.model` registers the `cotorra`
   model type, so `AutoModelForCausalLM.from_pretrained` loads it. The time heads
   predict non-negative values on the log1p-hours scale, so `expm1` converts them
   back to hours, never fewer than 0; the disposition head gives logits over its
   classes:

    ```python
    import torch as t
    from transformers import AutoModelForCausalLM

    import cotorra.model  # noqa: F401 -- registers the `cotorra` model type

    model = AutoModelForCausalLM.from_pretrained("output/mdl-<run_name>").eval()
    input_ids = t.tensor([[1, 5, 7, 9]])
    s_elapsed = t.tensor([[0.0, 0.0, 600.0, 3600.0]])  # seconds since start
    sec_per_pos_id = 300  # as set under `time_based_rope` for training
    position_ids = s_elapsed / sec_per_pos_id + t.arange(input_ids.shape[-1])

    with t.inference_mode():
        out = model(input_ids=input_ids, position_ids=position_ids)
    next_token_logits = out.logits[:, -1]
    if "tte" in model.heads:
        hours_to_end_time = out.tte_pred.expm1()  # (batch, seq_len)
    if "tnt" in model.heads:  # a point head; a mixture head leaves it None
        hours_to_next_token = out.tnt_pred.expm1()  # (batch, seq_len)
    if "disposition" in model.heads:
        classes = model.heads["disposition"].classes  # ["DSCG//expired", ...]
        p_disposition = out.disposition_pred.softmax(-1)  # (batch, seq_len, n)
        p_expired = p_disposition[..., classes.index("DSCG//expired")]
    ```

    Pass `position_ids` only if the model was trained with `time_based_rope`,
    built the way training builds them (above). Each head's prediction at
    position `i` is made having read tokens up to `i`: the TTE head's is the
    hours from token `i` to the end of the record, the TNT head's the wait from
    token `i` to token `i + 1`, and the disposition head's the chance of each way
    the record could end. Read at a prompt's last token (an inference table's
    `tokens_past`), they condition on everything so far.

    A mixture TNT head leaves `out.tnt_pred` empty, since its prediction depends
    on which token comes next; `model.tnt_distribution(hidden_states, next_ids)`
    gives the distribution, with `sample()`, `log_prob(hours)`,
    `point_estimate()` and `mean_log1p_hours()`. For a prediction from the
    history alone, which is what `cotorra extract --time-to-next-token` writes,
    `model.predict_hours_to_next_token(hidden_states)` averages the mean
    log1p-hours given each possible next token over the model's own next-token
    probabilities, then converts it to hours; for a point head it gives
    `out.tnt_pred.float().expm1()`. Both methods take the last hidden states,
    `out.hidden_states[-1]` from a forward pass with `output_hidden_states=True`.

Of the other stages, `cotorra extract` can add the heads' predictions to the
feature tables it writes (see [Extraction](#2-extraction)); `rep-based-score`
fits on the features alone, and `generative-score` hasn't been tested with a
`cotorra` model.

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
4. Optionally adds the predictions of the model's secondary heads at the same
   positions: the hours until the record's end time (`--time-to-event`), the
   hours until the next token (`--time-to-next-token`), and the probability of
   each discharge disposition (`--discharge-disposition`). Each needs a `cotorra`
   model carrying that head (see
   [Training secondary heads](#training-secondary-heads)).
5. Writes one feature table per split (optionally sharded).

Extraction is driven by a YAML config (the package ships a default; see
[`./src/cotorra/config/extraction.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/extraction.yaml))
that specifies:

- **max_seq_len**: Maximum sequence length.
- **time_based_rope** _(optional)_: Enables time-aware position ids during
  extraction (must match the setting used at training time).
    - **sec_per_pos_id**: Number of seconds represented by one position id
      increment.
- **extract**:
    - **max_len**: Maximum input length (tokens) during extraction. A longer
      context keeps its first `max_len` tokens, so its features and head columns
      are read there, short of the prediction point.
    - **batch_size**: Batch size for inference.
    - **shard_size** _(optional)_: Number of samples per output parquet shard.
      Omit to write a single file per split.

### Outputs

- `features-<split>-<model_name>.parquet` — extracted representations for each
  split (`train`, `tuning`, `held_out`), in a `features` column, one row per row
  of `<split>_for_inference.parquet` and in its order, led by its `subject_id`
  where it has one. With `--all-times`, files are named
  `features-all-<split>-<model_name>.parquet`; when `shard_size` is set, each
  split is written across `-<index>-of-<count>` shards. These files are the input
  to `cotorra rep-based-score`, which reads only `features`. The head flags add
  float32 columns after `features`, always in this order:
    - `time_to_event` (`--time-to-event`): the hours until the record's end time,
      `expm1` of the TTE head's log1p-hours.
    - `time_to_next_token` (`--time-to-next-token`): the hours until the next
      token, predicted from the history alone. For a point head that's `expm1` of
      its log1p-hours; a mixture head predicts the gap given the next token, so
      its mean log1p-hours given each possible next token is first averaged over
      the model's own next-token probabilities, rather than conditioned on the
      token that actually follows. That average costs a vocabulary's worth of
      head evaluations per position: little at the final position, but with
      `--all-times` it can take several times as long as the forward pass.
    - `<class>_prob` (`--discharge-disposition`): the probability of each
      discharge disposition, one column per class of the disposition head (e.g.
      `DSCG//expired_prob`), in the head's order,
      `model.heads["disposition"].classes`: token-id order, whatever order the
      training config listed `classes` in. They sum to 1.

    Each holds a value per row, read where `features` is; with `--all-times`, a
    list per row of length `max_seq_len`, holding a value at each position up to
    the final one and NaN after it, as `features` does. The heads and their
    classes come from the saved model, so the extraction config needs no
    `*_objective` block. A flag for a head the model doesn't carry stops
    extraction with an error once the model is loaded, before any feature table
    is written; a model trained without any objective is a stock HuggingFace
    model and carries none.

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

We support two types of model transfer for rep-based scoring:

1. You can apply a model trained on one dataset to a second dataset to obtain
   train, tuning, and held-out features, and then pair these features with labels
   from the second dataset to train supervised classifiers that can be applied to
   the held-out set of the second set.

2. You can also apply a model trained on one dataset to obtain train and tuning
   features for that first dataset and learn supervised classifiers using the
   labels from this first dataset. You can then apply the model to a second
   dataset to extract features and then apply the supervised classifier to the
   held-out features of this second set. For this second type of transfer, use
   the `--training-home` option offered by this command.

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
      patterns specified with fnmatch. Every matching vocabulary token becomes an
      outcome that `generative-score` samples trajectories for.
    - **end_tokens**: Tokens that naturally terminate a generated sequence (e.g.
      `EOS`).
    - **suppressed_tokens**: Tokens to suppress via logit bias during generation
      (e.g. `PAD`).
    - **batch_size**: Batch size for inference.
- **generation** _(generative-score only; modeled on the `pipeline_config.yaml`
  used by
  [quick-sco-re](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git))_:
    - **temperature**: Sampling temperature (1.0 for proper Monte Carlo
      estimation).
    - **score_inline**: If true, tracks every `score.target_tokens` outcome's
      SCOPE/REACH estimate in a single generation pass instead of a separate pass
      per outcome.
    - **methods**: Which trajectory types to run — some subset of `[M1, M2]`.
    - **end_tokens.prefixes**: Additional end-token prefixes (e.g. `DSCG`), on
      top of `score.end_tokens`.
    - **time_stopping** _(optional time-based truncation)_:
        - **enabled**: Turns on simulated-time-horizon stopping.
        - **trunc_token**: Token forced once the time horizon is exceeded.
        - **max_time_minutes**: Maximum simulated time horizon, in minutes.
        - **time_check_interval**: Tokens generated between time-horizon checks.
        - **time_token_bounds**: Map of time-bin token name to `[lo, hi]` minute
          bounds (e.g. `"TIME//1h-2h": [60, 120]`), used to estimate elapsed time
          from generated tokens.
- **engine** _(generative-score only)_:
    - **mem_fraction**: Fraction of GPU memory reserved for the inference engine.
    - **patient_chunk_size**: Number of patients generated per batch.
- **prompt_overflow** _(generative-score only)_: How to handle `tokens_past`
  longer than `score.max_len` — `"drop"` skips the patient, `"truncate_left"`
  keeps the last `max_len` tokens.

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
    Usage: cotorra train [OPTIONS] [OVERRIDES]...

    Train a model on tokenized data. For tokenization, consult the cocoa
    package.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   [overrides]...      TEXT  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
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
    Usage: cotorra tune [OPTIONS] [OVERRIDES]...

    Run hyperparameter tuning while training a model.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   [overrides]...      TEXT  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
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

- `cotorra extract`

    ```
    Usage: cotorra extract [OPTIONS] [OVERRIDES]...

    Extract representations from a trained model.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   [overrides]...      TEXT  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ───────────────────────────────────────────────────────────────╮
    │    --extraction-config      -e      PATH  Extraction configuration file │
    │                                           (overrides default)           │
    │ *  --processed-data-home    -p      TEXT  Processed data directory      │
    │                                           [required]                    │
    │ *  --model-home             -m      TEXT  Directory of the trained      │
    │                                           model to extract from         │
    │                                           [required]                    │
    │    --output-home            -o      TEXT  Output directory for          │
    │                                           extracted features, defaults  │
    │                                           to processed-data-home        │
    │    --all-times              -a            Extract features for all time │
    │                                           steps (instead of just the    │
    │                                           final one)?                   │
    │    --time-to-event          -t            Add the time-to-event head's  │
    │                                           predicted hours to the        │
    │                                           features?                     │
    │    --time-to-next-token     -n            Add the time-to-next-token    │
    │                                           head's predicted hours, given │
    │                                           the history alone, to the     │
    │                                           features?                     │
    │    --discharge-disposition  -d            Add the disposition head's    │
    │                                           probability of each discharge │
    │                                           disposition to the features?  │
    │    --help                   -h            Show this message and exit.   │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

- `cotorra generative-score`

    ```
    Usage: cotorra generative-score [OPTIONS] [OVERRIDES]...

    Generate SCORE/REACH metrics from a trained model and save them to
    parquet.

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   [overrides]...      TEXT  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
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
    Usage: cotorra rep-based-score [OPTIONS] [OVERRIDES]...

    Generate rep-based scores for the token-based outcomes of interest. Note:
    this requires that features have already been extracted and saved

    ╭─ Arguments ─────────────────────────────────────────────────────────────╮
    │   [overrides]...      TEXT  Config overrides: key=value sets a key,     │
    │                             adding it if need be, and ~key deletes one  │
    ╰─────────────────────────────────────────────────────────────────────────╯
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
    │    --training-home     -t      TEXT                 Use features and    │
    │                                                     labels extracted    │
    │                                                     here to train the   │
    │                                                     model (transfer)    │
    │    --estimator         -e      [k-NN|lightGBM|logi  Estimator to use    │
    │                                stic|logistic-z|log  for rep-based       │
    │                                istic-CV|logistic-C  scoring             │
    │                                V-z|XGBoost]         [default: lightGBM] │
    │    --verbose           -v                           Verbose logging     │
    │    --help              -h                           Show this message   │
    │                                                     and exit.           │
    ╰─────────────────────────────────────────────────────────────────────────╯
    ```

### Overriding config keys

To change a few keys without writing a new config file, list them after a
command's options, as dotted paths into its config:

```sh
cotorra train -t my-training.yaml -p processed/mimic -o output \
    training_args.learning_rate=1e-4 n_epochs=2 \
    tnt_objective.mixture_components=8 '~time_based_rope' \
    'model=${model_presets.qwen_3}' model.model_args.hidden_size=256
```

The syntax borrows from
[Hydra's](https://hydra.cc/docs/advanced/override_grammar/basic/):

- `key=value` sets a key, adding it if it isn't there, e.g. to train a secondary
  head whose block is commented out. Keys aren't checked against the config, so a
  misspelled one is added quietly and does nothing. A leading `+` or `++`, as
  Hydra marks an addition, is accepted and ignored.
- `~key` deletes a key.

Overrides apply in order to the config file passed (or to the packaged default
when none is), and training saves the result with the model, as
`mdl-<run_name>-training.yaml`. Values are read as YAML: `1e-4` is a float,
`[a, b]` a list (lists are replaced whole), and `{weight: 2.0}` a block, merged
into any block already there.

A few things to know:

- Setting a key to `null` doesn't turn its feature off. Blocks such as
  `time_based_rope`, `balanced_toi_loss` and the `*_objective` heads switch on
  when present, even when empty: `tte_objective=` trains the TTE head with its
  defaults. Delete a block with `~` to switch it off.
- YAML anchors make copies, not links. The packaged configs copy `run_name`,
  `max_seq_len` and `tokens_of_interest` into other blocks, so `run_name=foo`
  renames the model to `mdl-foo` but leaves `wandb.run_name` and
  `training_args.run_name` as they were. Override each copy you mean to change.
- `${...}` interpolations resolve after the overrides, so with the packaged
  `model_presets`, `'model=${model_presets.qwen_3}'` selects another preset, and
  a later `model.model_args` override edits that selection.
- Quote anything containing `$`, which the shell would otherwise expand, and, in
  zsh, anything starting with `~`, which zsh reads as a directory name and fails
  on.

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

[^6]:
    M. Burkhart, L. Solo, I. Lee, S. Charles, Z. Liao, K. Chhikara, D. Therese,
    W.-T. Liao, C. Gao, W. Parker, & B. Beaulieu-Jones, Federated generative
    event models for tokenized electronic health records,
    [arXiv:2608.02939](https://doi.org/10.48550/arXiv.2608.02939)

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
git tag -s v26.6.3 -m "bootstrapping utilities"
```

-->
