# Configuration

Each command is driven by a YAML config. The package ships a default for each
command under `src/cotorra/config/`, which you can override by passing a config
file via the appropriate CLI flag, and whose individual keys you can change on
the command line (see [Overriding config keys](#overriding-config-keys)).

## Overriding config keys

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
