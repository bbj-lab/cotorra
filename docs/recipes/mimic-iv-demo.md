# The MIMIC-IV demo, end to end: from tokenized timelines to predictions

[cocoa's MIMIC-IV demo](https://cocoa-tokenizer.readthedocs.io/latest/recipes/mimic-iv-demo/)
turns the
[MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/),
[^1] which anyone can download, into tokenized timelines, one for each of its 310
hospitalizations. This recipe picks up where that one leaves off. We'll train a
small model on the timelines, take from it a representation of each
hospitalization's first 6 hours, fit classifiers that predict from those
representations what comes later, such as a positive screen for delirium, and set
their predictions against baselines that need no model. On Linux with an NVIDIA
GPU, we'll also predict the same outcomes by sampling from the model. With 208
hospitalizations to learn from and 63 held out, the demo illustrates the process
more than it makes a useful model.

Every command runs from the `mimic-iv-demo` directory that cocoa's recipe leaves,
with the timelines in `processed/demo`. You'll need Python 3.11 to 3.14 and, on a
Mac, Apple silicon (PyTorch no longer builds for Intel Macs). The numbers below
come from cotorra 26.11.1 and cocoa 26.11.0 on an Apple M5 Pro, where steps 1 to
5 take about 5 minutes on its GPU, or 9 on its CPU alone; yours will differ a
little. They write about 160 MB, most of it to `output/demo`, and some 30 MB to
Hugging Face's datasets cache (`~/.cache/huggingface/datasets`, unless `HF_HOME`
puts it elsewhere).

## 0. Install cotorra

Install cotorra in cocoa's environment, or in a new one, since it installs cocoa
too:

```sh
pip install -U 'cotorra>=26.11.1' \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
```

pip takes the newest PyTorch on either index, PyPI's. On a Mac, it runs on the
GPU of Apple silicon; on Linux, it's built for CUDA 13.0, which needs an NVIDIA
GPU from the Turing generation or later and a driver from the 580 series or later
(`nvidia-smi` shows yours). cotorra trains on an NVIDIA GPU if PyTorch finds one,
else on a Mac's GPU, and else on the CPU, so with an NVIDIA GPU, check that
`python -c 'import torch; print(torch.ones(1, device="cuda"))'` prints
`tensor([1.], device='cuda:0')`. If it fails instead, install again, best in a
new environment, since pip would leave behind the CUDA 13 libraries it installed
first. The environment takes about 1.5 GB of disk on a Mac, and several times
that on Linux. To run cocoa's recipe in a new environment too, install
`'cotorra[demo]>=26.11.1'` this way before you start it: the `demo` extra brings
the clifpy and DuckDB that `cocoa get-demo` needs.

## 1. Cut the timelines at 6 hours

```sh
cocoa winnow -p processed/demo threshold.duration_s=21600
```

cocoa's recipe cut each timeline 24 hours after its first event, into the past
that the steps below predict from and the future in which each outcome comes or
doesn't. By then, though, many outcomes have already come, leaving few to
predict: of the 57 held-out hospitalizations it kept, 24 had gone to the ICU, and
only 2 of the others went later. This command cuts the timelines at 6 hours
(21,600 seconds) instead, rewriting `processed/demo/train_for_inference.parquet`
and its like for `tuning` and `held_out`. Every timeline runs past 6 hours, so
they keep all 208, 39, and 63 hospitalizations, and more of each outcome comes
after the cut: a positive screen for delirium, for one, in 9 of the 62 held out
without one in their first 6 hours, against 5 of 52 at the 24-hour cut.

## 2. Train a small model

```sh
cotorra train -p processed/demo -o output/demo \
    'model=${model_presets.llama_32_mid}' \
    model.model_name=HuggingFaceTB/SmolLM2-135M \
    model.model_args.num_key_value_heads=8 \
    model.model_args.dtype=float32 \
    max_seq_len=1024 n_epochs=20 training_args.learning_rate=2.5e-3 \
    training_args.report_to=none '~time_based_rope' run_name=demo
```

`-p` names the directory of timelines, and `-o` the directory to write to. The
arguments after them fit cotorra's shipped training config, written for the full
MIMIC-IV on a GPU, to the demo (`key=value` sets a key, and `~key` deletes one;
the quotes keep the shell from reading `$` and `~`):

- `'model=${model_presets.llama_32_mid}'` picks the smaller of the config's Llama
  presets, 6 layers 256 wide. A preset takes the rest of its architecture, though
  not its weights, from a model on Hugging Face:
  `model.model_name=HuggingFaceTB/SmolLM2-135M` takes it from the open SmolLM2
  rather than the gated Llama 3.2, and `model.model_args.num_key_value_heads=8`
  gives each of the preset's 8 attention heads keys and values of its own:
  SmolLM2's 3 don't divide 8, and training would stop at its first step. The
  model has 8,153,088 parameters.
- `model.model_args.dtype=float32` trains in float32 rather than SmolLM2's
  bfloat16, which rounds away many of training's small updates and is far slower
  on a CPU.
- `max_seq_len=1024` trains on the timelines laid end to end, 1,024 tokens at a
  time rather than 4,096, which runs out of memory on a laptop with 24 GB, and
  `n_epochs=20` goes through them 20 times, in 560 steps of 8 sequences.
- `training_args.learning_rate=2.5e-3` raises the shipped 2e-4, at which a model
  this small learns slowly, to about the learning rate that a search with
  `cotorra tune` found best (see [Next steps](#next-steps)).
- `training_args.report_to=none` keeps training from reporting to Weights &
  Biases, which would ask you to log in, and `'~time_based_rope'` numbers
  positions by token alone, rather than by time too, as SGLang does in step 6.

Every tenth of the way, cotorra scores the model on the 39 tuning
hospitalizations and prints a line such as this one, the last:

```
{'eval_loss': '4.518', 'eval_x_ent_loss': '2.248', 'eval_balanced_toi_loss': '0.01488', 'eval_quantile_token_loss': '0.03946', ...}
```

`eval_x_ent_loss` is the cross-entropy of each next token, in nats: 7.0 for a
model that guesses evenly among the vocabulary's 1,115 tokens, 5.9 for one that
knows only how common each token is, and 2.9 for one that also knows how often
each token follows each other one. Here it falls from 5.3 at the first evaluation
to 2.25 at the last. `eval_loss`, which training minimizes, adds 50 times
`eval_quantile_token_loss` and 20 times `eval_balanced_toi_loss`, two terms that
the shipped config asks for.

Training takes about 3 minutes on the M5 Pro's GPU, using about 6 GB of memory,
and about 7 minutes on its CPU; on a slower machine, `n_epochs=10` halves the
time, for a model that ends near 2.9. cotorra writes the model to
`output/demo/mdl-demo/` and the config it trained with to
`output/demo/mdl-demo-training.yaml`, and Hugging Face's trainer leaves its best
checkpoint, with the optimizer's state, in `output/demo/checkpoint-560/`. The
warnings are harmless: on a Mac, PyTorch notes as every `cotorra` command starts
that `Redirects are currently not supported in MacOs`, and warns at each
evaluation that the
`'pin_memory' argument is set as true but not supported on MPS now`; without a
token, the Hugging Face Hub may warn that
`You are sending unauthenticated requests to the HF Hub`; and as training reloads
its best checkpoint, transformers warns of
`missing keys in the checkpoint model loaded: ['lm_head.weight']`, since the
model's output layer shares its weights with its token embeddings.

## 3. Extract representations

```sh
cotorra extract -p processed/demo -m output/demo/mdl-demo '~time_based_rope'
```

For each hospitalization, `cotorra extract` runs the model over its first 6 hours
(its `tokens_past`, at most 325 tokens) and keeps the final layer's hidden state
at the last token: a representation of the hospitalization at the 6-hour mark,
256 numbers long. It writes them, as `features`, to
`processed/demo/features-train-mdl-demo.parquet` and its like for `tuning` and
`held_out`, a row for each row of the split's `<split>_for_inference.parquet`.
`'~time_based_rope'` numbers positions as in training. It takes a few seconds,
and datasets' warning that a transform `couldn't be hashed properly` is harmless.

## 4. Predict outcomes from the representations

```sh
cotorra rep-based-score -p processed/demo -m output/demo/mdl-demo --verbose
```

The shipped scoring config names 15 outcomes: `XFR-IN//icu`, `RESP//imv`,
`DSCG//expired`, `ASMT//cam_total_yes`, and 11 `LABEL//` tokens. For each,
`cotorra rep-based-score` fits a LightGBM classifier to the representations of
the training hospitalizations in which the outcome didn't come in the first 6
hours, to predict whether it comes later, and scores the held-out ones, writing
`processed/demo/scores-rep-based-mdl-demo.parquet`, with a `<token>_rep_score`
column for each outcome. With `--verbose`, it logs a 95% bootstrap interval for
each outcome's ROC AUC, PR AUC, and Brier score on the held-out hospitalizations
scored for it, as for `ASMT//cam_total_yes`, a positive screen for delirium by
the Confusion Assessment Method, which 9 of the 62 scored for it go on to have:

```
INFO     🦜 [18:08:55EST] tt='ASMT//cam_total_yes'
INFO     🦜 [18:08:55EST] method='rep'
INFO     🦜 [18:08:57EST] roc_auc: [0.695 0.943]
INFO     🦜 [18:08:57EST] pr_auc : [0.151 0.765]
INFO     🦜 [18:08:57EST] brier  : [0.056 0.202]
```

With so few hospitalizations, most outcomes come later in a handful of them at
most, and their intervals run wide. `LABEL//hyperna_init` and `LABEL//hypok_init`
come later in none of the held-out hospitalizations, so their ROC AUCs are
`[nan nan]`, after warnings that cotorra is
`using stratified sampling due to rarity` and numpy's
`All-NaN slice encountered`. On data this small, LightGBM also warns thousands of
times that it found `No further splits with positive gain`. Scoring takes about
half a minute. `--estimator logistic-CV-z` fits logistic regression instead,
writing over the same scores file unless `-o` names another directory, which has
to exist.

## 5. Set the scores against baselines

```sh
cotorra baseline-score -p processed/demo --verbose
cotorra baseline-score -p processed/demo --baseline tsfresh --verbose
```

`cotorra baseline-score` fits the same classifier to the same hospitalizations,
on features that need no model. The first command counts how many times each of
the vocabulary's tokens comes in each hospitalization's first 6 hours, in any
order, and writes `processed/demo/scores-bag-of-tokens-counts-lightGBM.parquet`.
The second summarizes the events that cocoa tokenized, in
`processed/demo/meds.parquet`, with [tsfresh](https://tsfresh.readthedocs.io)'s
minimal features: ten statistics of each code's values, such as
`LAB-RES//sodium`'s mean and maximum, which the tokens keep only as deciles, and
a count of each code without values. It writes
`processed/demo/scores-tsfresh-lightGBM.parquet`. With `--verbose`, each logs the
same intervals as step 4, which for `ASMT//cam_total_yes` come to:

| features                 | ROC AUC         | PR AUC          | Brier           |
| ------------------------ | --------------- | --------------- | --------------- |
| step 4's representations | `[0.695 0.943]` | `[0.151 0.765]` | `[0.056 0.202]` |
| token counts             | `[0.568 0.937]` | `[0.132 0.662]` | `[0.066 0.217]` |
| tsfresh                  | `[0.761 0.95 ]` | `[0.163 0.755]` | `[0.069 0.224]` |

Every interval for ROC AUC lies above 0.5, but with 9 positive screens among the
62, they overlap too much for any of the three to do clearly better than the
others. The baselines don't depend on the model, so they come out the same from
run to run. Each command takes about half a minute, and `--estimator logistic` or
`--estimator XGBoost` fits another classifier, to a scores file named for it.

## 6. Predict outcomes generatively (Linux with an NVIDIA GPU)

`cotorra generative-score` predicts the same outcomes from the model itself.
SGLang samples 100 continuations of each held-out hospitalization's first 6
hours, and from them come three estimates of each outcome's probability: the
share of samples in which it comes (`mc`); SCOPE (`scope`), which adds up the
probability the model gives the outcome at each step of a sample, until it comes;
and REACH (`reach`), 1 − ∏(1 − p), the chance that it comes at least once along a
sample.[^2] The step needs Linux with glibc 2.34 or later (Ubuntu 22.04 or RHEL
9, say), Python 3.11 to 3.13, and an NVIDIA GPU from the Turing generation or
later, with a driver from the 580 series or later: SGLang ships only for those,
and brings its own PyTorch, for CUDA 13.0. cotorra's `gen` extra installs SGLang
and quick-sco-re, which comes from GitHub, and since a package on PyPI can't
depend on one from GitHub, cotorra's package there leaves the extra out. Install
cotorra from GitHub too, and with `processed/demo` and `output/demo/mdl-demo`
copied over if you trained elsewhere, run:

```sh
pip install 'cotorra[gen] @ git+https://github.com/bbj-lab/cotorra.git@v26.11.1'
cotorra generative-score -p processed/demo -m output/demo/mdl-demo \
    score.max_len=1024 --verbose
```

`score.max_len=1024` stops each sample, the first 6 hours included, at the 1,024
tokens that the model saw at once in training, which cuts short any sample that
would run longer, as the timelines of 10 of the 63 held-out hospitalizations do.
`cotorra generative-score` writes
`processed/demo/scores-generative-mdl-demo.parquet`, with a `<token>_mc_score`,
`<token>_scope_score`, and `<token>_reach_score` column for each outcome, and
with `--verbose` logs the same intervals as step 4, for each of the three.

## Next steps

- `cotorra tune` searches for better hyperparameters with Optuna, training a
  model for each trial and then one more with the best trial's, so that this
  search, of 4 trials over learning rates from 5e-4 to 5e-3 and 1 to 3 steps of
  gradient accumulation, takes about five times as long as step 2:

    ```sh
    cotorra tune -t output/demo/mdl-demo-training.yaml -p processed/demo \
        -o output/demo run_name=demo-tuned tuning_args.n_trials=4 \
        tuning_args.hp_space.learning_rate.low=5e-4 \
        tuning_args.hp_space.learning_rate.high=5e-3
    ```

    `-t` starts from step 2's config, overrides and all. To score the tuned
    model, use `mdl-demo-tuned` wherever steps 3, 4, and 6 say `mdl-demo`.

- Every model learns to predict the next token, and it can learn secondary heads
  alongside: added to step 2's overrides (with a `run_name` of its own), an empty
  `disposition_objective=` block trains one that predicts how each
  hospitalization ends, and `cotorra extract -d` then adds its probability of
  each of the demo's 9 `DSCG//` tokens, as a column beside `features`. A model
  with a head is saved as a `cotorra` model, which SGLang doesn't know, so keep
  step 6 to one without. [Training secondary heads](../guide/secondary-heads.md)
  covers it, and the heads for the time to the next token and to the end of the
  hospitalization; the last needs cocoa to write `hours_to_end_time`, which its
  shipped config doesn't.
- `cotorra extract -l` adds each token's surprisal under the model, in bits,
  given the tokens before it, as a `token_nll_bits` column.[^3]
- The shipped configs number positions by time as well as by token. To train that
  way, leave `'~time_based_rope'` out of steps 2 and 3 both; step 6 then runs the
  model on positions it never saw.
- [Transfer learning](transfer-learning.md) applies a model trained on one
  dataset to others, and cocoa's
  [tokenizer-transfer recipe](https://cocoa-tokenizer.readthedocs.io/latest/recipes/tokenizer-transfer/)
  splits the demo into such a pair, tokenized with one vocabulary.
- With credentialed access to MIMIC-IV, the
  [development workflow](development-workflow.md) runs these steps on its CLIF
  version, MIMIC-IV-Ext-CLIF, with the shipped configs, on a GPU; it needs the
  Hugging Face and Weights & Biases logins that step 2 avoided.

[^1]:
    A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. A. Celi, & R. Mark,
    [MIMIC-IV Clinical Database Demo (version 2.2)](https://doi.org/10.13026/dp1f-ex47),
    PhysioNet (2023), RRID:SCR_007345

[^2]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^3]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188
