# Training

The trainer consumes the tokenized timelines and fits a causal language model to
predict the next token in each subject's timeline. It:

1. Builds a next-token-prediction dataset from `tokens_times.parquet` and the
   subject splits, packing the records into fixed-length rows, optionally whole
   and each attending only to itself, or giving each record a row of its own.
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
- **packing** _(optional)_: How training records are packed into rows of
  `max_seq_len` tokens: `concat` (default) joins them end to end, `bfd` packs
  them whole, padding the room left over, and `pad` gives each a row of its own,
  padding each batch to its longest row. See [Packing records](packing.md).
- **doc_masking** _(optional)_: With `packing: bfd`, keeps each packed record
  from attending to the others in its row (default: `false`). See
  [Packing records](packing.md).
- **attn_implementation** _(optional)_: The attention implementation the model is
  built with, e.g. `sdpa`, `eager` or, with
  [flash-attn](https://github.com/Dao-AILab/flash-attention) installed,
  `flash_attention_2` (default: transformers' own choice, usually `sdpa`).
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
  [Training secondary heads](secondary-heads.md).
    - **weight**: Weight on the term (default: `1.0`).
    - **per_record**: Weights each record alike, rather than each token (default:
      `false`). Unweighted, every token's error counts the same, so a record
      counts in proportion to its length. With `true`, each of a record's `n`
      tokens is weighted by `mean / n`, where `mean` is the average `n` across
      that split's records. Every record then counts the same however long it
      runs, and the loss stays on its usual scale, since the weights average 1
      over each split's tokens. A short record's tokens each weigh more. The
      tuning split is weighted the same way, so in `eval_loss`, which
      `cotorra tune` minimizes, the term averages over records as well.
- **tnt_objective** _(optional)_: Trains a time-to-next-token (TNT) head that
  predicts, at each token, the log1p-hours until the next token, making the model
  a marked point process. Its target is derived from `times`, and a record's last
  token is left out. See [Training secondary heads](secondary-heads.md).
    - **weight**: Weight on the term (default: `1.0`).
    - **mixture_components** _(optional)_: Makes the head predict a distribution
      rather than a single value: the chance that the next token shares this
      one's timestamp, plus a mixture of this many log-normals (at least 1) over
      the positive gap, conditioned on which token comes next and trained by
      likelihood. Needed to sample realistic times. Left unset, the head predicts
      log1p-hours by squared error.
- **disposition_objective** _(optional)_: Trains a supervised head that predicts,
  at each token, the record's discharge disposition, read off its `DSCG//*`
  token, by cross-entropy. See [Training secondary heads](secondary-heads.md).
    - **weight**: Weight on the term (default: `1.0`).
    - **classes**: The dispositions to predict among, as token labels or fnmatch
      patterns (default: `["DSCG//*"]`). A record whose disposition matches none
      of them isn't scored.
    - **per_record**: Weights each record alike, rather than each token (default:
      `false`), as for `tte_objective`. The head is scored on the tokens before a
      record's disposition token, so unweighted, a record counts in proportion to
      how many there are. With `true`, each of them is weighted by `mean / n`,
      `n` counting them and `mean` averaging `n` across that split's scored
      records.
- **time_based_rope** _(optional)_: Enables time-aware rotary position
  embeddings.
    - **sec_per_pos_id**: Number of seconds represented by one position id
      increment.

    _Note:_ SGLang (behind `generative-score`) and vLLM number positions by token
    index and can't take time-based ones, so they run a model trained with this
    on positions it never saw.

- **training_args**: Arguments passed to HuggingFace's
  [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

    _Note:_ Don't set `label_smoothing_factor`. HuggingFace smooths only its own
    loss, so with `custom_loss: true` the setting does nothing, and cotorra
    warns. With `custom_loss: false` and a secondary head, HuggingFace would drop
    the heads' terms and score each token against itself, so cotorra refuses it.
    Only a model without heads, on `custom_loss: false`, is smoothed as intended.

- **tuning_args**: Arguments passed to HuggingFace's
  [`hyperparameter_search`](https://huggingface.co/docs/transformers/hpo_train?backends=Optuna)
  when `cotorra tune` is called.
    - **hp_space**: The search space, a block per `training_args` field to tune.
      Its `type` (`float`, `int` or `categorical`) picks Optuna's
      `trial.suggest_<type>`, and its other keys are that method's arguments
      (`low`, `high`, `step`, `log`; `choices` for `categorical`). Without it,
      the search covers `learning_rate` (1e-4 to 5e-4, log-scaled) and
      `gradient_accumulation_steps` (1 to 3).
- **prune_trial_checkpoints** _(optional)_: Deletes each trial's checkpoints
  (`run-*/checkpoint-*` under `--output-home`) as the trial ends, when
  `cotorra tune` is called (default: `false`). The model it saves is trained
  afresh after the search, so nothing it needs is lost, only disk space saved.

## Model presets

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

## Outputs

- `mdl-<run_name>/` — the trained model, saved under `--output-home` in
  HuggingFace format (via `save_pretrained`), ready to be passed as
  `--model-home` to `extract` and the scoring commands.
- `mdl-<run_name>-training.yaml` — the resolved training configuration used for
  the run.
