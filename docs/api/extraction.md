# Extraction

The extraction stage runs a trained model over timelines and captures the
representations it computes, so that a lightweight classifier can later be fit on
them (see [`RepBasedScorer`](scoring.md#repbasedscorer)). This is the
representation-based counterpart to autoregressive
[generative scoring](scoring.md#generativescorer).

## `Extractor`

`Extractor` loads a saved model, moves it to the best available device (CUDA,
MPS, or CPU), and runs it in inference mode over the timelines in each split's
inference table (`train`, `tuning`, `held_out`). For each timeline it reads out
the model's last hidden-layer activations — by default the single vector at the
final token before the first `EOS` (the model's summary of the timeline up to the
prediction point), or, when `all_times` is set, the full sequence of per-token
representations padded with NaN to `max_seq_len`. Results are written to
`output_home` as a parquet table per split, sharded once a split exceeds
`extract.shard_size`, with a row per timeline and its representation in a
`features` column, ready to be consumed downstream.

Given `heads` (any of `"tte"`, `"tnt"`, `"disposition"`), it adds those secondary
heads' predictions at the same positions, laid out as `features` is, as float32
columns after it in the order `heads` lists them: `time_to_event` and
`time_to_next_token` in hours, the latter predicted from the history alone
(`CotorraForCausalLM.predict_hours_to_next_token`), and a `<class>_prob`
probability for each of the disposition head's classes. Asking for a head the
model doesn't carry raises a `ValueError` before anything is written.

::: cotorra.extractor.Extractor
