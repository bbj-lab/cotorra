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

Each batch holds as many rows as fit `tokens_per_batch` tokens once they're
padded to the batch's longest: a few long records, or hundreds of short ones.
That way every step trains on about as many tokens, as with the other packings. A
step's loss is a mean over its batch's tokens, and every step counts alike, so a
batch of a fixed number of rows would make each token of a short record count for
far more than each token of a long one. A record's length is its outcome (a short
stay makes a short record), so that would teach the model that records end sooner
than they do.

Each of the `n_epochs` passes over the records is batched on its own, so a
record's copies never share a batch. Within a pass, the rows are sorted by
length, ties in random order, and cut into batches shortest first, so that each
batch holds rows of about one length and pads them little. The batches are then
shuffled, and the one holding the longest row goes first, so that a budget too
large for memory fails at the first step. The order is seeded by
`training_args.data_seed`, or `seed` without one, so that every trial of
`cotorra tune` draws the same batches. Evaluation takes the tuning split's rows
longest first, batched the same way. Each logged `eval_*` term is a mean over the
positions it scores, rather than HuggingFace's mean over the batches weighted by
their rows: over the tokens for cross-entropy, the quantile tokens for
`quantile_token_loss`, and each secondary head's usable targets for its head. Its
`eval_loss`, which picks the best checkpoint and which `cotorra tune` minimizes,
weighs those means together as training's loss weighs a batch's. The other
packings keep HuggingFace's `eval_loss`.

```yaml
packing: pad
max_seq_len: !!int 16384
tokens_per_batch: !!int 65536 # 4 rows of 16384 at most, or more shorter ones
```

A few things to know:

- `tokens_per_batch` defaults to `training_args.per_device_train_batch_size` ×
  `max_seq_len`, the most a batch of the other packings holds, and to
  `per_device_eval_batch_size` × `max_seq_len` for evaluation. When set, it's
  used for both. Like the batch sizes, it's per GPU: under `nn.DataParallel`,
  which splits each batch's rows across the GPUs, a batch holds as many rows as
  keep every GPU's share within it, however unevenly they divide.
- Keep `tokens_per_batch` at 4 × `max_seq_len` or more. A row just over half the
  budget takes a batch alone, half empty, so its tokens count for up to twice as
  much. At 4 rows or more, a batch of rows of about one length fills to some 80%
  of its budget at least. Very short records, of a few dozen tokens, are rare
  enough that a batch of them spans a wide range of lengths and fills less, so
  their tokens still count for somewhat more.
- An epoch takes about as many steps as `concat`'s with batches of
  `tokens_per_batch` tokens, but the batches hold whole records, and their
  padding isn't scored.
- `training_args.train_sampling_strategy` can't apply: cotorra refuses `random`
  or `sequential` under `packing: pad`, and takes `group_by_length` as what it
  does anyway.
- `training_args.auto_find_batch_size` can't shrink the batches, as it would try
  to after running out of memory, so cotorra refuses it under `packing: pad`.
  Lower `tokens_per_batch` instead.
- `packing: pad` doesn't yet train across processes (`torchrun`,
  `accelerate launch`). How accelerate shares out batches of varying sizes across
  processes differs by version, so cotorra refuses it. One process on one or more
  GPUs works.
- Each chunk of a record longer than `max_seq_len` starts a row of its own, so
  its tokens can't see where their record began.
- With `time_based_rope`, cotorra refuses `flash_attention_2` under
  `packing: pad`. Flash attention drops a mask of ones and reads a single row's
  time-based position ids as several sequences, splitting attention wherever they
  jump, and a row over half of `tokens_per_batch` takes a batch alone, as can the
  longest rows of an epoch's last batch. Use `sdpa` (the default) or `eager`.
