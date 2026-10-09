# Training

The training stage fits a fresh causal language model on tokenized timelines.
Both trainers read a `training.yaml` configuration, pull the training and
validation splits out of the processed data, and wrap HuggingFace's `Trainer` so
that sequences of event tokens are modeled autoregressively (each token predicts
the next). Time-aware position ids for [time-based RoPE](../index.md#1-training)
and a custom loss are wired in when the configuration requests them. On
completion the model weights and the exact configuration used are written to
`output_home`.

## `Trainer`

The workhorse. `Trainer` builds a model _from scratch_ from a HuggingFace
architecture preset (its vocabulary, `BOS`, and `EOS` come from
`tokenizer.yaml`), packs the training and tuning splits into fixed-length
sequences, and runs standard next-token training. Its collate function ties
`labels` to `input_ids` for causal language modeling and, when time-based RoPE is
configured, derives per-token `position_ids` from elapsed time. Runs are tracked
in Weights & Biases, and `train()` can resume from the latest checkpoint.

::: cotorra.trainer.Trainer

## `Tuner`

`Tuner` extends [`Trainer`](#trainer) to run an [Optuna](https://optuna.org)
hyperparameter search (over learning rate and gradient-accumulation steps) before
the final training run. The best trial's hyperparameters are copied back onto the
trainer, the model is retrained with them, and the weights and configuration are
saved as usual.

::: cotorra.tuner.Tuner
