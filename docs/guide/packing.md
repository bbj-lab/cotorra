# Packing records

Training packs the subjects' timelines into rows of up to `max_seq_len` tokens,
in one of three ways:

- `packing: concat`, the default, joins the shuffled records end to end and cuts
  the result every `max_seq_len` tokens. Records straddle rows, and the last,
  partial row is dropped.
- `packing: bfd` packs the records whole, by best-fit decreasing
  ([Ding et al., 2024](https://arxiv.org/abs/2404.10830)). A record is split only
  when it's longer than `max_seq_len`, into chunks of at most that length. Each
  chunk, longest first, goes into the row with the least room left that still
  fits it. Nothing is dropped: the room left over is padded instead, and the
  padding is never a target, for next-token prediction or for a secondary head.
  With `n_epochs` above 1, each epoch is packed apart, so that no row holds a
  record twice, where the later copy could attend to the earlier one.
- `packing: pad` gives each record a row of its own. A record is split only when
  it's longer than `max_seq_len`, into chunks of at most that length, each a row
  of its own. The rows keep their own lengths, and each batch is padded to its
  longest, the padding never a target. Only a last chunk of a single token is
  dropped, since it has no next token to predict. See
  [A record to a row](#a-record-to-a-row).

With `concat` or `bfd`, a token attends to the records packed before it in its
row. With `packing: bfd`, `doc_masking: true` stops that. Each packed record
attends only to itself, its position ids count from 0, and its first token isn't
a target, since it would be predicted from an unrelated record. Evaluation then
packs the tuning split the same way, so that `eval_loss`, which picks the best
checkpoint and which `cotorra tune` minimizes, scores the model as it trains.
Without `doc_masking`, the tuning split is concatenated, whichever the `packing`.

```yaml
packing: bfd
doc_masking: !!bool true
# attn_implementation: flash_attention_2 # faster on a GPU, with flash-attn installed
```

A few things to know:

- `packing: bfd` needs `training_args.remove_unused_columns: false`, as the
  packaged config sets it. Otherwise HuggingFace drops the columns marking the
  padding, so cotorra refuses to train without it.
- `doc_masking` needs no particular attention implementation. For `sdpa` (the
  default) and `eager`, transformers builds a block-diagonal mask from the
  position ids that restart at each record. For `flash_attention_2`, whether set
  by `attn_implementation` or in `model.model_args`, cotorra flattens each batch
  into a single row, which flash attention's variable-length kernels split where
  the position ids restart. `nn.DataParallel`, which splits a batch by its rows,
  can't take a single row, so cotorra refuses flash attention under `doc_masking`
  when one process sees several GPUs: launch one process per GPU instead, with
  `torchrun` or `accelerate launch`. Either way, the model's KV cache has to be
  off, so the saved model carries `use_cache: false`. To generate from it with a
  cache, set `model.generation_config.use_cache = True` after loading it.
- Under `doc_masking`, each chunk of a record longer than `max_seq_len` attends
  only to itself, so its tokens can't see where their record began.
- `doc_masking` doesn't yet work with the secondary heads or `time_based_rope`.

## A record to a row

With `packing: pad`, each record attends only to itself, with no need for
`doc_masking`. It starts its row, so its position ids count from 0, or with
`time_based_rope` from its first token's time, as they do when `extract` reads
it. Unlike `doc_masking`, it works with `time_based_rope` and with the secondary
heads.

Training draws the rows with HuggingFace's length-grouped sampler, so that each
batch holds rows of about the same length and pads them little. The sampler
shuffles the rows, sorts each run of 50 batches longest first, and puts the
longest row in the first batch, so that a batch too large for memory fails at the
first step. It's seeded by `training_args.data_seed`, or `seed` without one, so
that every process and every trial of `cotorra tune` draws the same order. To
draw the rows another way, set `training_args.train_sampling_strategy` to another
of HuggingFace's samplers, e.g. `random`. Evaluation takes the tuning split's
rows longest first, always in that order, so that `eval_loss`, which picks the
best checkpoint and which `cotorra tune` minimizes, is computed on the same
batches every time.

```yaml
packing: pad
max_seq_len: !!int 16384
training_args:
    per_device_train_batch_size: !!int 2 # 2 × 16384 tokens at most, as 8 × 4096
    per_device_eval_batch_size: !!int 2
```

A few things to know:

- A batch holds up to `per_device_train_batch_size` × `max_seq_len` tokens, as
  with the other packings, so a larger `max_seq_len` calls for a smaller batch.
  Most batches hold far fewer tokens, since only the longest records fill a row.
- A step's loss averages over its batch's tokens, and a batch of short records
  holds far fewer of them than a batch of long ones. Each token of a short record
  therefore counts for more than each token of a long one, so that each row
  counts roughly alike, where `concat` counts each token alike. An epoch also
  takes more, smaller steps than `concat`'s, so the learning rate may need
  retuning.
- Each chunk of a record longer than `max_seq_len` starts a row of its own, so
  its tokens can't see where their record began.
- With `time_based_rope` and `flash_attention_2`, keep every batch above one row:
  flash attention drops a mask of ones, and it reads a single row's time-based
  position ids as several sequences, splitting attention wherever they jump.
