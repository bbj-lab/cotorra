# Extraction

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
   model carrying that head (see [Training secondary heads](secondary-heads.md)).
5. Optionally scores every token of the context by its negative log2-likelihood
   under the model, given the tokens before it (`--token-nll`).
6. Writes one feature table per split (optionally sharded).

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

## Outputs

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

    `--token-nll` adds one more column, last: `token_nll_bits`, a float32 list
    per row as long as its `input_ids` (the row's `tokens_past`). Entry i is that
    token's negative log2-likelihood in bits given the tokens before it, −log₂
    p(tokenᵢ | token₀, …, tokenᵢ₋₁), from the model's own next-token
    probabilities. Entry 0 is zero: the first token is always `BOS`, so it
    carries no information. Every entry past `extract.max_len` is NaN, since the
    model never sees those tokens. The column is the same with or without
    `--all-times`, and every model has it, stock HuggingFace models included.
    Skipping any NaNs (e.g. with `np.nansum`), a row's sum is the context's total
    surprisal in bits, and the mean of its entries after the first is log₂ of the
    model's perplexity on it. The probabilities are the model's as trained: a
    model trained with the deprecated `label_weighted_loss` overpredicts its
    tokens of interest, so it scores them as less surprising than they are.
