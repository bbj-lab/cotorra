## The MIMIC-IV demo, end to end: from tokenized timelines to predictions

[cocoa's MIMIC-IV demo](https://github.com/bbj-lab/cocoa/blob/master/recipes/mimic-iv-demo.md)
turns the
[MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/),
which anyone can download, into tokenized timelines, one for each of its 275
hospitalizations. This recipe picks up where that one leaves off. We'll train a
small model on the timelines and tune its hyperparameters, take from the model a
representation of each hospitalization's first 24 hours, and fit classifiers that
predict, from those representations, what comes later. On a Linux machine with an
NVIDIA GPU, we'll also predict the same outcomes generatively, by sampling the
rest of each timeline from the model. Steps 1 to 4 take about 17 minutes on a
laptop's GPU, 14 of them tuning.

With 187 hospitalizations to learn from, a model can only learn so much, and with
50 held out, among which most outcomes come a handful of times or not at all, its
scores can only tell us so much. The point is to run each stage of cotorra on
data that anyone can get, and to see what each one makes. The demo's license,
which cocoa's recipe describes, covers what you make from it here too; if you
publish anything made from it, cite it, along with MIMIC-IV and PhysioNet, as its
page asks.[^1]

You'll need the `mimic-iv-demo` working directory that cocoa's recipe leaves,
with the timelines in `processed/demo`; every command below runs from there.
You'll also need Python 3.11 or later, Apple silicon if you're on a Mac (PyTorch
no longer builds for Intel Macs), and cotorra 26.10.1 or later. The numbers below
come from cotorra 26.10.1 on an Apple M5 Pro. Yours will differ with your
hardware, a little even from run to run on the same GPU, and from step 2 on with
the hyperparameters that tuning happens to try. On a Mac, the environment takes
about 1.6 GB of disk, and on Linux several times that, for PyTorch's CUDA
libraries. What the steps below write takes about 560 MB, mostly checkpoints in
`output/demo`, and some 30 MB of it in Hugging Face's datasets cache
(`~/.cache/huggingface/datasets`, unless `HF_HOME` puts it elsewhere), outside
the working directory; the cache grows when you train with other settings, since
each new `max_seq_len`, `n_epochs`, or secondary head caches its own packing of
the timelines, as does numbering positions by time.

### 0. Install cotorra

Install cotorra in the environment you installed cocoa in, or in a new one, since
it installs cocoa too:

```sh
pip install -U 'cotorra>=26.10.0' \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
```

pip takes the newest PyTorch it finds on either index. The first holds builds for
CUDA 12.8, but none newer than PyTorch 2.11, so on Linux pip takes PyPI's newer
build, for CUDA 13.0, which needs an NVIDIA driver from the 580 series or later
(`nvidia-smi` shows yours) and a GPU from NVIDIA's Turing generation or later; on
a Mac, it takes PyPI's build, which runs on the GPU of Apple silicon. Training
and extraction run on an NVIDIA GPU if PyTorch finds one, else on a Mac's GPU,
and else on the CPU, so on a machine with an NVIDIA GPU, check that
`python -c 'import torch; print(torch.ones(1, device="cuda"))'` prints
`tensor([1.], device='cuda:0')`. If it fails instead, install again, best in a
new environment, since pip would leave behind the CUDA 13 libraries it installed
first. On a GPU older than Turing (a V100, say), for which PyTorch warns that it
`does not include kernels for this GPU`, make
`https://download.pytorch.org/whl/cu126` the first index: PyTorch's builds for
CUDA 12.6 are the only ones that still run on such a GPU. On a newer GPU with a
driver older than the 580 series, add `'torch<2.12'` after `'cotorra>=26.10.0'`:
pip then takes PyTorch 2.11's build for CUDA 12.8 from the first index.

On a Mac, run one more command. cotorra imports LightGBM and XGBoost, whose
builds for macOS look for Homebrew's OpenMP runtime (its `libomp` package), while
PyTorch brings its own. Without Homebrew's, every `cotorra` command fails as it
starts, with `Library not loaded: @rpath/libomp.dylib`; with it, more than one
OpenMP runtime ends up in one process, and step 4 crashes Python with a
segmentation fault as it fits its first classifier. Pointing macOS's dynamic
linker at PyTorch's copy avoids both:

```sh
export DYLD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib"
```

The setting lasts as long as your shell. macOS drops it on the way into the
programs it protects, such as `/bin/sh` and `/usr/bin/time`, so a script that
runs cotorra has to set it itself.

### 1. Train a small model

```sh
cotorra train -p processed/demo -o output/demo \
    'model=${model_presets.llama_32_mid}' \
    model.model_name=HuggingFaceTB/SmolLM2-135M \
    model.model_args.num_key_value_heads=8 \
    model.model_args.dtype=float32 \
    max_seq_len=1024 n_epochs=20 \
    training_args.report_to=none '~time_based_rope' run_name=demo
```

`-p` names the directory of timelines, and `-o` the directory to write to. The
arguments after them change keys of cotorra's shipped training config, which is
written for the full MIMIC-IV on a GPU: `key=value` sets a key, and `~key`
deletes one. Here they fit it to the demo:

- `'model=${model_presets.llama_32_mid}'` picks the smaller of the config's two
  Llama presets, with 6 layers 256 wide, in place of the default, `llama_32`,
  with 9 layers 1,024 wide. A preset takes the rest of its architecture from a
  model on Hugging Face, only its configuration and not its weights, since
  cotorra trains from scratch. Both Llama presets take it from Meta's Llama 3.2,
  whose repository is gated: you'd have to accept its license and log in.
  `model.model_name=HuggingFaceTB/SmolLM2-135M` takes it from SmolLM2 instead, an
  open model with Llama's architecture, and
  `model.model_args.num_key_value_heads=8` gives each of the preset's 8 attention
  heads keys and values of its own, as Llama 3.2's configuration would. That one
  is needed: SmolLM2's configuration has 3, which don't divide 8, and without the
  override training stops at its first step with
  `Number of heads in key and value must divide the number of heads`. The model
  has 8,149,248 parameters.
- `model.model_args.dtype=float32` keeps the model's weights, and the optimizer's
  state, in float32. SmolLM2's configuration would have them in bfloat16, in
  which many of the small updates that training makes to them get rounded away:
  trained that way, the model ends step 1 at a cross-entropy of 5.0 nats rather
  than 3.8. bfloat16 is also slow on a CPU: on the M5 Pro's, training takes about
  6 minutes in float32, and more than 6 hours in bfloat16. Step 2 inherits the
  setting from the config that step 1 saves.
- `max_seq_len=1024` trains on the timelines, laid end to end, 1,024 tokens at a
  time rather than 4,096, which runs out of memory on a laptop with 24 GB. The
  first 24 hours of a hospitalization, which steps 3 to 5 read, run to 860 tokens
  at most.
- `n_epochs=20` goes through the 187 training timelines, 210,552 tokens, 20
  times: 4,112 sequences of 1,024 tokens, in 514 steps of 8. The shipped config's
  1 would make 205 sequences, in 26 steps.
- `training_args.report_to=none` keeps training from reporting to Weights &
  Biases, which would ask you to log in.
- `'~time_based_rope'` numbers the model's positions by token alone, rather than
  by token and time (a position for every 5 minutes), since SGLang, which samples
  from the model in step 5, numbers them by token.
- `run_name=demo` names the model.

Quote the first override, so that the shell leaves its `$` alone, and the
`~time_based_rope`, which zsh would read as a directory.

cotorra first splits the timelines into
`processed/demo/train_tokens_times.parquet` and its like for `tuning` and
`held_out`, and then trains. Every tenth of the way, it scores the model on the
33 tuning hospitalizations and prints a line such as this one, the last:

```
{'eval_loss': '6.54', 'eval_x_ent_loss': '3.842', 'eval_balanced_toi_loss': '0.01512', 'eval_quantile_token_loss': '0.0479', ...}
```

`eval_x_ent_loss` is the cross-entropy of each next token, in nats: 7.0 for a
model that guesses evenly among the vocabulary's 1,100 tokens, 5.8 for one that
knows only how common each token is, and 2.9 for one that also knows how often
each token follows each other one. Here it falls from 6.1 at the first evaluation
to 3.8 at the last: at the shipped learning rate, 2e-4, a model this small learns
slowly, and step 2 looks for a better learning rate. `eval_loss`, the loss that
training minimizes, here on the tuning hospitalizations, adds two terms that the
shipped config asks for: 50 times `eval_quantile_token_loss`, which teaches the
model that a code's deciles are ordered, and 20 times `eval_balanced_toi_loss`, a
binary cross-entropy on whether the next token is one of the outcomes that step 4
predicts, scored by the total probability the model gives them.

Training takes under three minutes, using about 5 GB of memory. On a Mac, PyTorch
warns at each evaluation that the
`'pin_memory' argument is set as true but not supported on MPS now`. Without a
token, the Hugging Face Hub may warn that
`You are sending unauthenticated requests to the HF Hub`, which is fine, since
SmolLM2's configuration is open. As training ends, it reloads its best
checkpoint, and transformers warns of
`missing keys in the checkpoint model loaded: ['lm_head.weight']`, which is fine
too: the model's output layer shares its weights with its token embeddings, so
the checkpoint stores them once. cotorra writes the model to
`output/demo/mdl-demo/`, in Hugging Face's format, and the config it trained
with, overrides and all, to `output/demo/mdl-demo-training.yaml`. Hugging Face's
trainer also leaves the checkpoint it reloaded (here its last), with the
optimizer's state, in `output/demo/checkpoint-514/`, about 93 MB.

### 2. Tune its hyperparameters

```sh
cotorra tune -t output/demo/mdl-demo-training.yaml -p processed/demo \
    -o output/demo run_name=demo-tuned tuning_args.n_trials=4 \
    tuning_args.hp_space.learning_rate.low=5e-4 \
    tuning_args.hp_space.learning_rate.high=5e-3
```

`-t` starts from the config that step 1 saved, with its overrides, rather than
from the shipped one. `cotorra tune` hands the search to Optuna, by way of
Hugging Face's `hyperparameter_search`: each trial trains a new model with
hyperparameters that Optuna draws from `tuning_args.hp_space`, and is scored by
its last `eval_loss`. The shipped space covers learning rates from 1e-4 to 5e-4,
on a log scale, and 1 to 3 steps of gradient accumulation (with 2, each update
averages the gradients of 2 batches of 8 sequences, so there are half as many
updates, and the progress bar counts 257 steps rather than 514). The overrides
name the tuned model `demo-tuned`, so that it doesn't overwrite step 1's, move
the learning rates up, to 5e-4 to 5e-3, and run 4 trials rather than 10. Optuna
logs each trial as it finishes, as in:

```
[I …] Trial 0 finished with value: 4.667490005493164 and parameters: {'learning_rate': 0.0009584108029995487, 'gradient_accumulation_steps': 1}. Best is trial 0 with value: 4.667490005493164.
```

Here its four trials drew, and scored:

| trial | learning rate | gradient accumulation | `eval_loss` |
| ----- | ------------- | --------------------- | ----------- |
| 0     | 9.58e-4       | 1                     | 4.667       |
| 1     | 1.50e-3       | 3                     | 5.113       |
| 2     | 1.19e-3       | 2                     | 4.848       |
| 3     | 1.36e-3       | 3                     | 5.102       |

Optuna draws its first 10 trials at random, and Hugging Face doesn't seed it, so
yours will differ, but each of these did far better than step 1's 6.54. cotorra
then trains once more, with the best trial's hyperparameters, and writes the
model to `output/demo/mdl-demo-tuned/` and the config it searched with to
`output/demo/mdl-demo-tuned-tuning.yaml`. That config still holds step 1's
learning rate and gradient accumulation; the ones the search chose appear in
Optuna's log, not there. The last run's last evaluation puts its cross-entropy,
2.33, below the 2.9 of knowing how often each token follows each other one:

```
{'eval_loss': '4.664', 'eval_x_ent_loss': '2.326', 'eval_balanced_toi_loss': '0.01524', 'eval_quantile_token_loss': '0.04067', ...}
```

Here its best evaluation came just before, at step 468, with an `eval_loss` of
4.658, so that is the checkpoint `mdl-demo-tuned/` holds, and the one the trainer
leaves in `output/demo/checkpoint-468/`, in place of step 1's. Each trial leaves
its own best checkpoint in `output/demo/run-0/` to `run-3/`; the steps below read
only `mdl-demo-tuned/`, so you can delete the checkpoints once tuning is done.
Tuning takes about 14 minutes, under three for each trial and for the last run.
To skip it, use `mdl-demo` wherever the steps below say `mdl-demo-tuned`, in the
commands and in the names of the files they write.

### 3. Extract representations

```sh
cotorra extract -p processed/demo -m output/demo/mdl-demo-tuned '~time_based_rope'
```

cocoa's winnowing kept 169 hospitalizations in the training split, 30 in tuning,
and 50 held out. For each, `cotorra extract` runs the model over the first 24
hours, its `tokens_past`, and takes the final layer's hidden state at the last of
those tokens: a representation of the hospitalization at the 24-hour mark, 256
numbers long. The shipped extraction config numbers positions by time too, so
`'~time_based_rope'` again has them numbered by token, as in training.
`cotorra extract` writes `processed/demo/features-train-mdl-demo-tuned.parquet`,
and its like for `tuning` and `held_out`, with a row for each row of the split's
`<split>_for_inference.parquet`, in the same order: its `subject_id`, its
`tokens_past` (as `input_ids`), and the representation (as `features`). It takes
a few seconds, and datasets warns that a transform
`couldn't be hashed properly, a random hash was used instead`, which only keeps
it from reusing a cached result, as cotorra doesn't anyway.

### 4. Predict outcomes from the representations

```sh
cotorra rep-based-score -p processed/demo -m output/demo/mdl-demo-tuned --verbose
```

The shipped scoring config's `target_tokens` match 15 of the vocabulary's tokens:
`XFR-IN//icu`, `RESP//imv`, `DSCG//expired`, `ASMT//cam_total_yes`, and 11
`LABEL//` tokens. For each, `cotorra rep-based-score` fits a LightGBM classifier
to the representations of the training hospitalizations in which the outcome
didn't come in the first 24 hours, to predict whether it comes later, and applies
the classifier to the held-out hospitalizations. `cotorra rep-based-score` writes
`processed/demo/scores-rep-based-mdl-demo-tuned.parquet`, the held-out inference
table with a `<token>_rep_score` column for each outcome, NaN where the outcome
came in the first 24 hours. With `--verbose`, it logs a 95% bootstrap interval
for each outcome's ROC AUC, PR AUC, and Brier score on the held-out
hospitalizations, as for `LABEL//sepsis_onset`, which comes after the first 24
hours in 34 of the 50:

```
INFO     🦜 [19:27:46EDT] tt='LABEL//sepsis_onset'
INFO     🦜 [19:27:46EDT] method='rep'
INFO     🦜 [19:27:47EDT] roc_auc: [0.528 0.864]
INFO     🦜 [19:27:47EDT] pr_auc : [0.644 0.949]
INFO     🦜 [19:27:47EDT] brier  : [0.151 0.36 ]
```

Every other outcome is rarer. `DSCG//expired` comes in 3 of the 50 held-out
hospitalizations, and its interval for ROC AUC runs from 0.596 to 0.939. For an
outcome that comes later in only one or two of the hospitalizations scored for
it, the interval can fall anywhere, as `RESP//imv`'s does, from 0.053 to 0.289:
11 of the 50 are on invasive ventilation in the first 24 hours, and only 1 of the
other 39 goes on it later. `LABEL//hyperna_init` and `LABEL//hypok_init` come in
none of the held-out hospitalizations, so their ROC AUCs are `[nan nan]`, and
their PR AUCs, `[0.5 0.5]`, mean nothing either, after a warning that cotorra is
`using stratified sampling due to rarity` and, the first time, numpy's
`RuntimeWarning: All-NaN slice encountered`. On data this small, LightGBM also
warns thousands of times that it found `No further splits with positive gain`,
and about a thousand times that it
`Stopped training because there are no more leaves that meet the split requirements`.
Scoring takes about half a minute.

To fit another classifier, pass `--estimator`, such as
`--estimator logistic-CV-z` for logistic regression on standardized
representations, with the strength of its L2 penalty chosen by cross-validation.
It overwrites the scores file; to keep both, make a directory
(`mkdir -p processed/demo/logistic`) and pass it with
`-o processed/demo/logistic`. In place of LightGBM's warnings, it warns that some
cross-validation folds hold no case of an outcome.

### 5. Predict outcomes generatively (Linux with an NVIDIA GPU)

`cotorra generative-score` predicts the same outcomes from the model itself.
SGLang samples 100 continuations of each held-out hospitalization's first 24
hours, and from them come three estimates of each outcome's probability: the
share of samples in which it comes (`mc`); SCOPE (`scope`), which adds up the
probability the model gives the outcome at each step of a sample, until it comes;
and REACH (`reach`), the chance, 1 − ∏(1 − p), that it comes at least once along
a sample without it.[^2] For REACH, SGLang samples again the rest of each sample
in which the outcome came, from just before it, with the outcome forbidden, so it
draws more than 100 continuations in all.

The step needs Linux with glibc 2.34 or later (Ubuntu 22.04 or RHEL 9, say),
Python 3.11 to 3.13, and an NVIDIA GPU from the Turing generation or later, with
a driver from the 580 series or later, whatever you did in step 0: SGLang ships
only for those Pythons and that glibc, and brings its own PyTorch, for CUDA 13.0,
in place of yours. It also needs two more packages, SGLang and quick-sco-re,
which cotorra's `gen` extra installs. quick-sco-re installs from GitHub, and
since a package on PyPI can't depend on one from GitHub, cotorra's package there
leaves the extra out; install cotorra from GitHub too:

```sh
pip install 'cotorra[gen] @ git+https://github.com/bbj-lab/cotorra.git@v26.10.1'
```

Then, with `processed/demo` and `output/demo/mdl-demo-tuned` copied over if you
trained elsewhere, run:

```sh
cotorra generative-score -p processed/demo -m output/demo/mdl-demo-tuned \
    score.max_len=1024 --verbose
```

`score.max_len=1024` stops each sample, counting the first 24 hours, just short
of 1,024 tokens (quick-sco-re keeps 9 in reserve), the most that the model saw at
once in training. That cuts short any sample that would run longer, as the
timelines of 10 of the 50 held-out hospitalizations do, and with it any outcome
that would have come after the cut. `cotorra generative-score` writes
`processed/demo/scores-generative-mdl-demo-tuned.parquet`, with a
`<token>_mc_score`, `<token>_scope_score`, and `<token>_reach_score` column for
each outcome, and with `--verbose` logs the same summaries as step 4, for each of
the three.

### Next steps

- Every model learns to predict the next token, and it can learn secondary heads
  alongside. Added to step 1's overrides, an empty `disposition_objective=` block
  trains one that predicts how each hospitalization ends, among the demo's 9
  `DSCG//` tokens, and `cotorra extract -d` then adds its probability of each, as
  a column beside `features` (`DSCG//expired_prob`, say), which step 4 doesn't
  read. Give that run its own `run_name`, and keep step 5 to a model without
  heads: one with a head is saved as a `cotorra` model, which SGLang doesn't
  know, and `generative-score` hasn't been tested with one.
  [Training secondary heads](https://github.com/bbj-lab/cotorra#training-secondary-heads)
  covers it, and the heads for the time to the next token and to the end of the
  hospitalization; the last needs cocoa to write `hours_to_end_time`, which it
  does only with `include_hours_to_end_time`, off in the shipped tokenization
  config that the demo uses.
- `cotorra extract -l` adds each token's surprisal under the model, in bits,
  given the tokens before it, as a `token_nll_bits` column.[^3]
- The shipped configs number positions by time as well as by token. To train that
  way, leave `'~time_based_rope'` out of steps 1 and 3 both, so that the
  positions line up; step 5 then runs the model on positions it never saw.
- [Transfer learning](transfer-learning.md) trains a model on one dataset and
  applies it to others, and cocoa's
  [tokenizer-transfer recipe](https://github.com/bbj-lab/cocoa/blob/master/recipes/tokenizer-transfer.md)
  splits the demo into such a pair, its patients treated in 2011–2013 and in
  2014–2016, tokenized with one vocabulary.
- With credentialed access to MIMIC-IV, the
  [development workflow](development-workflow.md) runs these steps on its CLIF
  version, MIMIC-IV-Ext-CLIF, with the shipped configs, on a GPU; it needs the
  Hugging Face login (for its default preset) and the Weights & Biases one that
  step 1 avoided.

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
