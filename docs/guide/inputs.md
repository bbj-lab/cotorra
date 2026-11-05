# Inputs

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

The tsfresh baseline (`cotorra baseline-score --baseline tsfresh`) reads one more
table, the `meds.parquet` cocoa's collator writes, with a row for each event:
`subject_id`, `time`, `code`, `numeric_value` and `text_value`. It takes each
timeline's rows up to its threshold as the inference tables place it, so these
need `subject_id` too, and the `times` of each timeline's tokens, both of which
`cocoa winnow` writes.

<!-- prettier-ignore-start -->
> [!TIP]
> For getting your data to this point, check out our configurable
> collator / tokenizer: [☕️ cocoa-tokenizer](https://pypi.org/project/cocoa-tokenizer/)
<!-- prettier-ignore-end -->
