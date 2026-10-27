# CLAUDE.md

Guidance for working in this repository. The [README.md](README.md) is the
user-facing reference (data formats, every CLI flag, config keys, model presets);
this file covers what you need to _work on the code_ — architecture, conventions,
and gotchas that aren't obvious from the source.

## What this is

`cotorra` is a configurable trainer for generative event models over tokenized
timelines (built for foundation models on tokenized electronic health records).
Given tokenized subject timelines, it trains a causal LM to predict the next
token, then uses that model to extract representations and score outcomes. It
ships as a PyPI package exposing a single `cotorra` CLI. It is the downstream
sibling of [cocoa](https://github.com/bbj-lab/cocoa), which produces the
tokenized inputs cotorra consumes.

## Pipeline

Everything is driven through the `cotorra` CLI
([src/cotorra/cli.py](src/cotorra/cli.py), a Typer app):

```
train / tune   →  extract  →  rep-based-score
               ↘  generative-score
```

- **train / tune** — fit a causal LM; write `mdl-<run_name>/` (HuggingFace
  `save_pretrained` format) + `mdl-<run_name>-training.yaml` under
  `--output-home`.
- **extract** — run a trained model over inference contexts, write hidden-state
  feature tables to `--output-home` (default: `--processed-data-home`) as
  `features[-all]-<split>[-<i>-of-<n>]-<model_name>.parquet` (`-all` with
  `--all-times`; the shard suffix once a split exceeds `extract.shard_size`), a
  row per row of `<split>_for_inference.parquet`, in its order and led by its
  `subject_id` (kept wherever every split's table has one). `-t`/`-n`/`-d`
  (`--time-to-event`, `--time-to-next-token`, `--discharge-disposition`) add
  float32 head columns after `features`, read at its positions (a value per row,
  or with `--all-times` a `max_seq_len` list, NaN after the row's final position,
  as `features` is): `time_to_event` and `time_to_next_token` in hours, and a
  `<class>_prob` per disposition class; `HEAD_COLUMNS` in
  [extractor.py](src/cotorra/extractor.py) maps each head name to its columns.
  `-l` (`--token-nll`) adds `token_nll_bits` last, from the LM's logits rather
  than a head, so any model has it: each token's −log₂ p given those before it, a
  float32 list as long as the row's `input_ids` whatever `--all-times` (0 at the
  first token, a `BOS` carrying no information; NaN past `extract.max_len`).
- **rep-based-score** — fit a lightweight sklearn/boosting estimator on extracted
  features (requires `extract` first). It globs `features-<split>*-…` from
  `--processed-data-home`, so features extracted to another `--output-home`
  aren't found; `--training-home` (transfer) fits on another directory's
  train/tuning features and labels and scores this one's `held_out`.
- **generative-score** — Monte-Carlo sample trajectories to compute
  MC/SCOPE/REACH scores (does not need `extract`).

## Architecture

Each pipeline stage is a class that subclasses `Configurable`
([src/cotorra/configurable.py](src/cotorra/configurable.py)). The CLI is a thin
wrapper that instantiates the class and calls one method; the real logic lives in
the classes.

| Module                                                   | Class                                   | Role                                                         |
| -------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| [configurable.py](src/cotorra/configurable.py)           | `Configurable`                          | base: loads/merges config, holds a `Logger`                  |
| [loader.py](src/cotorra/loader.py)                       | `Loader`                                | splits `tokens_times.parquet` by subject, builds HF datasets |
| [trainer.py](src/cotorra/trainer.py)                     | `Trainer`, `TrainerWithCustomLoss`      | model init + HF Trainer                                      |
| [tuner.py](src/cotorra/tuner.py)                         | `Tuner`                                 | `Trainer` + Optuna hyperparameter search                     |
| [loss.py](src/cotorra/loss.py)                           | `Loss`                                  | custom losses (quantile-token, balanced-TOI) + head terms    |
| [model.py](src/cotorra/model.py)                         | `CotorraForCausalLM`, `Head` subclasses | causal LM carrying any combination of secondary heads        |
| [extractor.py](src/cotorra/extractor.py)                 | `Extractor`                             | hidden-state and head-prediction extraction                  |
| [scorer_rep_based.py](src/cotorra/scorer_rep_based.py)   | `RepBasedScorer`, `EstimatorType`       | estimator-on-features scoring                                |
| [scorer_generative.py](src/cotorra/scorer_generative.py) | `GenerativeScorer`                      | SCOPE/REACH generative scoring                               |
| [logger.py](src/cotorra/logger.py)                       | `Logger`                                | rich logging + bootstrap-CI eval summaries                   |
| [util.py](src/cotorra/util.py)                           | —                                       | batching helpers, `bootstrap_ci`                             |

Inheritance matters: `Tuner` extends `Trainer`, which extends `Configurable`;
`Loss` and `Logger` stand alone. Changing `Trainer.__init__` or `collate_fn`
affects tuning too.

In [model.py](src/cotorra/model.py), a model is a backbone plus any combination
of secondary heads, each a `Head` subclass registered in `HEADS` by its `name`
(`tte`, `tnt`, `disposition`). That name keys the head's `<name>_objective`
training block, its entry in `CotorraConfig.heads`, its `<name>_pred` /
`<name>_loss` output fields, and its `extract` columns in `HEAD_COLUMNS`; its
`target` names the batch column it's scored against, and its `target_weight`
(`hours_to_end_time_weight`, `disposition_weight`; the TNT heads have none) the
per-position weights it's scored with under `per_record`, which `forward` pops
alongside the targets and `Head.forward` refuses to go without, or to take
unasked. Inheritance is for what heads share: `TteHead` and `TntPointHead` are
`Log1pHoursHead`s, and `TntPointHead` and `TntMixtureHead` are `TntHead`s, the
interface sampling and extraction use. The combinations live in one class,
`CotorraForCausalLM`, which builds a `ModuleDict` of the heads its config lists,
so adding an objective takes a `Head` subclass in `HEADS` plus its two output
fields, a `HEAD_COLUMNS` entry and an `extract` flag (tests hold both to
`HEADS`'s keys), not a new model class; weighting one per record takes a
`target_weight`, a `per_record` option, and a `SCORED` entry in
[loader.py](src/cotorra/loader.py) (which positions its loss scores), which tests
hold to the heads with a `target_weight`. `head_options` turns the training
config's blocks into the heads' options (resolving the disposition `classes`
patterns against the vocab) and is the one place `Trainer`, `Loader` and `Loss`
read them from. `Trainer.model_init` builds a stock HF model when no objective is
set. Importing `cotorra.model` registers `cotorra` with the HF auto classes, so
anything that loads a saved `mdl-<run_name>/` (as `Extractor` does) has to import
it first.

Point heads (`Log1pHoursHead`: `TteHead`, and the default TNT head) are an
`nn.Linear` read through a softplus, so their log1p-hours, and the hours `expm1`
turns them into, are never negative; a clamp would leave a head that drifts below
zero without a gradient. The TNT head is such a point estimate (squared error on
log1p-hours) unless `mixture_components` is set. Then it is a `TntMixtureHead`:
given the hidden state _and the next token's embedding_ (layer-normed first: a
raw embedding sits ~50x below the normed hidden state, and unnormalized the head
learned the zero-gap rate but not which tokens arrive at once), it emits a
`ZeroInflatedLogNormalMixture` (an exact-zero gap for same-timestamp tokens, else
a log-normal mixture), trained by NLL and leaving `tnt_pred` empty.
`CotorraForCausalLM.predict_hours_to_next_token` (what `extract -n` writes) is
either kind's prediction from the history alone, through
`TntHead.mean_log1p_hours`: a point head's output, or a mixture head's mean
log1p-hours given each next token, averaged over the LM head's softmax and never
conditioned on the observed or likeliest one (the per-token mean is
`ZeroInflatedLogNormalMixture.mean_log1p_hours`, by Gauss-Hermite quadrature).
That costs a vocabulary's worth of head evaluations per position, so
`TntMixtureHead.mean_log1p_hours` splits `proj`'s first layer over `[h; norm(e)]`
to compute the vocabulary's half once per call and chunks positions by
`chunk_elements`; a change to `proj`'s layout has to change it too. Every head is
scored unshifted (position i against its own target) over every position,
whatever `logits_to_keep` trims, and each loss is summed then normalized so a
batch with nothing to score still gives its head a gradient.

### Configuration model (important)

`Configurable.__init__` builds `self.cfg` (an OmegaConf object) in three layers,
each overriding the one before:

1. a user config file passed via the relevant CLI flag (`--training-config`,
   etc.) or, only when none is passed, the class's packaged default YAML
   (`default_file`, in [src/cotorra/config/](src/cotorra/config/)); a
   `DictConfig` passed instead is copied as is,
2. `overrides`: the `key=value` arguments trailing a CLI command, applied by
   `apply_overrides` (`key=value` sets a key, adding it if absent, so a typo adds
   a stray key rather than failing; `~key` deletes one; a Hydra-style leading
   `+`/`++` is ignored),
3. keyword args passed to the constructor (only non-`None` values).

A passed config file _replaces_ the packaged default rather than merging into it,
so a key the file omits is absent from `self.cfg`, not filled in from the
default, though an override can still add it. The CLI passes `overrides=` to each
class, which forwards it to `Configurable` through `**kwargs`. Read optional keys
defensively with `self.cfg.get(...)` or `"key" in self.cfg` — several features
(`time_based_rope`, `quantile_token_loss`, `balanced_toi_loss`, and the
`*_objective` blocks) are toggled purely by _presence_ of their config block, not
a boolean, so an override setting one to `null` leaves it on and `~key` is what
turns it off.

`Trainer` and `Extractor` hand their `Loader` their merged `self.cfg`, not the
config file's path: the `Loader` does the chunking (`max_seq_len`), the epoch
repeats (`n_epochs`) and the column selection (`time_based_rope`, the heads), so
reloading the file would silently drop overrides and kwargs. Any new class built
from another's config should be handed `self.cfg` the same way.

**Configs must stay backwards-compatible.** An old config, written before a key
existed and so missing it, must run exactly as it did before. The rule covers
only what has been released to PyPI (check against the latest release tag, e.g.
`git merge-base --is-ancestor <commit> v26.9.0`); anything unreleased, even on
`main`, can change without an option to keep its old behavior. Since nothing
fills in a missing key, the fallback in the code decides what happens, not the
packaged YAML:

- a new feature stays off when its block is absent (the presence toggle does this
  for free);
- a new key inside an existing block falls back to the old behavior, e.g.
  `(self.cfg.block or {}).get("key", <old value>)` (`or {}` because a block left
  empty in the YAML parses to `None`);
- an existing key keeps its meaning and default, and an optional key never
  becomes required;
- a new block added to a packaged YAML ships commented out, as the `*_objective`
  blocks do, so runs without a config file are unchanged too.

The same goes for the model configs saved with checkpoints (`CotorraConfig`, and
each head's options in it): a new field needs a default under which older
checkpoints load and behave as before.

Fixing a bug so a key does what it documents isn't a compatibility break: with
`custom_loss: true`, `gradient_accumulation_steps` used to sum its micro-batches'
gradients rather than average them (HF leaves that to a custom loss), and now
averages them. It weights each micro-batch alike, where HF's stock path weights
each scored token alike; the two agree while every micro-batch scores as many
tokens, as concatenated rows do and `packing: bfd`'s needn't. With
`custom_loss: false`, a model with heads added each head's mean, as it was, to
HF's LM term: a sum over the step's `num_items_in_batch` tokens, for which HF
skips its division by the micro-batches and multiplies by the processes (to undo
DDP's averaging), so the heads' weights were multiplied by both in training,
though not in evaluation. Now `CotorraForCausalLM.forward` weights each head's
term by its micro-batch's share of those tokens, so that it scales as the LM term
does under any of HF's scalings. Transformers 5.5.4 counts every unmasked label,
not the shifted ones the loss scores, so its count runs one token a row high,
shrinking both terms alike by about one part in `max_seq_len`.

Deprecating a key follows the same rule: `label_weighted_loss` (it made the model
predict its tokens of interest more often) is still honored, and
`Trainer.__init__` issues a `FutureWarning` (not a `DeprecationWarning`, which
Python hides outside `__main__`). The one deliberate exception is the packaged
`training.yaml`, which swapped its active `label_weighted_loss` block for
`balanced_toi_loss`, so runs without a config file changed.

The three packaged config files
([training.yaml](src/cotorra/config/training.yaml),
[extraction.yaml](src/cotorra/config/extraction.yaml),
[scoring.yaml](src/cotorra/config/scoring.yaml)) are the source of truth for
defaults and are documented key-by-key in the README. `training.yaml` defines
reusable `model_presets` (YAML anchors) and selects one via the `model:` key.

## Conventions

- **Formatting/linting: ruff only.** Line length 88, double quotes, `E`/`F`/`I`
  (isort) rules, `skip-magic-trailing-comma`. Config in
  [pyproject.toml](pyproject.toml). Run before committing:
    ```sh
    ruff format .
    ruff check . --fix
    ```
    First-party imports for isort: `cocoa`, `cotorra`, `coreopsis`. Non-Python
    files (md/yaml/toml/json) are formatted by prettier (`proseWrap: always`,
    printWidth 81; 100 for yaml) and taplo — see
    [.prettierrc.toml](.prettierrc.toml) / [.taplo.toml](.taplo.toml). Markdown
    uses 4-space tabs and an 81-col ruler.
- **`import torch as t`** everywhere — match it, don't `import torch`.
- **One-line module docstring** at the top of each file describing its single
  purpose; keep it.
- **`if __name__ == "__main__":` blocks are dev scratch harnesses**, not tests or
  entry points. They often hardcode local paths (`./processed/mimic`, for which
  cocoa's MIMIC-IV demo builds a public stand-in, `processed/demo`, the one
  [recipes/mimic-iv-demo.md](recipes/mimic-iv-demo.md) trains on) and a trailing
  `# breakpoint()`. Leave them; they're for interactive debugging.
- **Tests: `pytest`** (config in [pyproject.toml](pyproject.toml); install with
  `pip install -e ".[test]"`). CI
  ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs it on every
  push and pull request, and weekly, on Python 3.11 through 3.14, and once more
  on 3.11 with each dependency at the lowest version
  [pyproject.toml](pyproject.toml) allows. It doesn't run ruff at all, neither
  `ruff check` nor `ruff format`, so a style slip never fails it; that's left to
  the ruff run before committing. It installs torch's CPU wheels from PyTorch's
  own index, so it never runs CUDA or MPS code, nor the `[gen]` extra. The
  3.11–3.14 jobs also install `pytest-cov` (CI-only, not in the `test` extra) and
  upload coverage to [Codecov](https://app.codecov.io/gh/bbj-lab/cotorra) by
  OIDC, as cocoa's do, so no token secret is needed (fork PRs upload tokenless;
  Codecov refuses a tokenless upload from any other branch). Coverage is
  informational only: the upload step can't fail the job (`continue-on-error`),
  and [codecov.yml](codecov.yml) makes its statuses always pass and waits for all
  four uploads before reporting (`after_n_builds`; keep it equal to the matrix
  size). Don't add a coverage threshold (`--cov-fail-under`, a status `target`)
  or make the upload required. Keep the matrix in step with `requires-python`,
  and raise a floor in `pyproject.toml` when code starts needing a newer
  dependency, or the lowest job fails. On Linux that job gets torch 2.6; the
  darwin-only 2.8 floor is for MPS, where 12 extraction and sampling tests fail
  under 2.6. `requires-python` stops short of 3.15 because PyPI and the CUDA
  index have no torch wheels for it, and the CPU index's segfault loading a saved
  model (seen on macOS with 3.15.0rc2). Still run the suite locally before
  committing; `-m "not slow"` drops the handful of real (tiny) training loops.
  `pythonpath` puts `src` ahead of site-packages so a run always exercises the
  working tree even when a non-editable `cotorra` sits in the venv — don't drop
  it. The suite needs neither network nor real data:
  [tests/synth.py](tests/synth.py) writes CLIF-like raw tables the fixtures push
  through the real cocoa collate → tokenize → winnow pipeline, and the model
  under test is a tiny randomly-initialized `LlamaForCausalLM`. Without the
  `[gen]` extra, the `scorer_generative` fixture imports
  `cotorra.scorer_generative` against
  [tests/fake_quick_sco_re.py](tests/fake_quick_sco_re.py), a copy of
  `quick_sco_re`'s dataclasses whose engine and generation are placeholders the
  tests swap for fakes (sglang's part, faked either way), so the generative
  scorer is tested in CI too; a guard test holds the copy to the real package
  wherever the extra is installed, and is the one test that skips without it.
- **Logging** goes through `self.logger` (the parrot-emoji `Logger`), not
  `print`. The CLI itself uses `rich.print`/`Console` for status spinners and the
  `✓ …completed` summaries.

## Data contract (inputs)

Stages read from `--processed-data-home`. Expected files (produced by cocoa's
`pipeline`/`winnow`):

- `tokens_times.parquet` — `subject_id`, `tokens` (list[u32]), `times`
  (list[datetime]), plus `hours_to_end_time` (list[float], written under cocoa's
  `include_hours_to_end_time`) when training with `tte_objective`. Only training
  reads its rows, but `Loader` asserts the file exists and `Extractor` builds a
  `Loader`, so `extract` needs it too.
- `subject_splits.parquet` — `subject_id` → `split` ∈ {`train`, `tuning`,
  `held_out`}.
- `tokenizer.yaml` — must contain a `lookup:` map (token label → int id), incl.
  `BOS`, `EOS`; `quantile_token_loss` also reads `cfg.n_bins`.
- `{train,tuning,held_out}_for_inference.parquet` — for extract/score; include
  `tokens_past`, optional `subject_id` and `s_elapsed_past`, and `<TOKEN>_past` /
  `<TOKEN>_future` label columns. `Loader` (`Trainer`'s too, when they're
  present) keeps only their context — `subject_id`, `tokens_past` as `input_ids`,
  and `s_elapsed_past` under `time_based_rope` — never a head's target or
  anything else that gives the outcome away (`hours_to_end_time_past`,
  `end_time`, `s_total_duration`, the future): `Extractor.extract_final` writes
  the batch it reads back out beside the features.

`Loader` derives per-split `{split}_tokens_times.parquet` caches, adding
`s_elapsed` and `hours_to_next_token` (the TNT target, nan on each record's last
token), and regenerates them when `tokens_times.parquet` is newer or a cache
predates a derived column. The disposition target depends on the config's
`classes`, so it isn't cached: `Loader` labels each record (its last matching
`DSCG//*` token's class before that token, -100 from it on) with `datasets.map`
before packing splits records across chunks. Under an objective's `per_record`
(the TTE's or the disposition's) it weights each record the same way: `mean / n`
on each of its tokens, `n` counting those its head scores (a finite
`hours_to_end_time`; a disposition other than -100) and `mean` being `n`'s
average over the split's records with any, so that a fragment keeps its whole
record's weight, and the weights average 1 over each split's scored tokens,
keeping the loss on its unweighted scale. It counts them off the dataset itself,
the split caches lacking the disposition target. Token-set selectors
(`tokens_of_interest`, `target_tokens`) support **fnmatch patterns** (e.g.
`LABEL//*`), resolved against the tokenizer vocab.

`Loader.get_train_data` packs the train split into `max_seq_len` rows by
`packing`: `concat` (the default, `util.batched_iter`) runs the shuffled,
epoch-repeated records together and drops the last partial row; `bfd`
(`util.best_fit_packed_iter`) cuts only records longer than a row and places each
chunk whole by best-fit decreasing, padding each row with zeros and adding
`pad_mask` and `doc_pos` (each token's position in its piece, restarting at 0 for
every piece and for the padding). It packs each of `n_epochs` passes apart, in an
order of its own: packed in one pass, a record's copies, all of one length, sit
side by side among the chunks sorted by length and land in the same rows, where
the later attends to the earlier. `Trainer.collate_fn` keeps the padding from
being any loss's target (-100 for the labels and the disposition, nan for the
time heads); nothing else needs masking, since a piece ends short of its row only
where its record ends (a split record's other chunks each fill a row), whose last
token has no TNT target, so a mixture TNT head never conditions on another
record's token. `doc_masking` (`bfd` only, and not yet with heads or
`time_based_rope`) passes `doc_pos` as `position_ids` and labels each piece's
first token -100; for flash attention, read off the built model (so
`model.model_args` counts too), it flattens the batch into one row, which
`Trainer.__init__` refuses under `nn.DataParallel`. The tuning split is then
`bfd`-packed too, by its own fixed seed, so `eval_loss` scores the masked format
the model trains on (without `doc_masking` it stays concatenated).

## Gotchas

- **Install requires the PyTorch CUDA index.** Use the two-index `pip install`
  from the README; a plain `pip install cotorra` will fail to resolve torch
  correctly. Development install is `pip install -e ".[gen]"`.
- **`generative-score` needs the `[gen]` extra** (`quick-sco-re`, `sglang`). It's
  declared in [pyproject.toml](pyproject.toml)'s `optional-dependencies`, but
  `quick-sco-re` comes from git, so only a source install gets it (the release
  workflow drops the extra from what it publishes; see
  [Build, docs, release](#build-docs-release)). `GenerativeScorer` imports
  `quick_sco_re` at module load, so [cli.py](src/cotorra/cli.py) imports it
  _lazily_ inside the command. Preserve that laziness so the base install works
  without these heavy/optional deps.
- **`sec_per_pos_id` isn't saved with the model.** `extract` has to be told the
  `time_based_rope` spacing the model was trained with; it's in
  `mdl-<run_name>-training.yaml`.
- **`time_based_rope` must match between training and extraction.** If a model
  was trained with time-based RoPE, the extraction config must enable it too
  (same `sec_per_pos_id`), or position ids won't line up. `generative-score`
  can't honor it at all: `GenerativeScorer` hands SGLang only `tokens_past`, and
  SGLang assigns sequential positions.
- **`extract`'s head flags read the saved model, not the config.** `-t`/`-n`/`-d`
  need a `cotorra` checkpoint whose `config.heads` carries the head; a model
  trained without objectives is a stock HF one with none, and `Extractor.extract`
  refuses it (`_check_heads`) before extracting a batch or writing a table,
  though `Extractor.__init__` has by then loaded the model and built the `Loader`
  (which may refresh the split caches). `*_objective` blocks in the extraction
  config don't enable them: they'd make its `Loader` load the heads' training
  targets (unused by `extract`), so a `tte_objective` there fails without
  `hours_to_end_time` in `tokens_times.parquet`.
- **`remove_unused_columns: false`** in `training_args` is required — otherwise
  HF drops the `s_elapsed` column that time-based RoPE needs, and
  `packing: bfd`'s `pad_mask` and `doc_pos` (`Trainer.__init__` refuses `bfd`
  without it; HF's own default is `true`).
- **transformers reads position ids that don't step by exactly 1 as packed
  sequences.** Given no `attention_mask` and no KV cache, `create_causal_mask`
  (`find_packed_sequence_indices`, in every transformers cotorra allows) starts a
  new sequence wherever consecutive `position_ids` don't differ by 1, and masks
  attention across them, per row (verified for sdpa and eager; flex needs
  `torch.compile`, which macOS CPU lacks). `doc_masking` relies on it, so
  `model_init` turns the cache off under it, and `collate_fn` flattens the batch
  for flash attention, whose varlen path takes a single row only.
  `time_based_rope`'s ids jump at every gap between timestamps, so both collators
  pass an all-ones `attention_mask` beside them: without it, a model run without
  a cache (gradient checkpointing turns it off; a checkpoint may be saved with
  `use_cache: false`) splits attention at every jump. Flash attention drops a
  mask of ones, though, and derives a one-row batch's sequences from its position
  ids anyway (its helpers give wrong segments on 5.19 and fail on 5.5.4; the
  kernel itself is untested, flash-attn needing CUDA), so time-based RoPE with
  flash attention is only sound for batches of more than one row. HF sets
  `training_args.use_cache` on the model it builds in `__init__` (the one
  `built_trainer.model` saves in the tests), not on the one `model_init` rebuilds
  in `train()`, which keeps its `AutoConfig`'s.
- **HF's `label_smoothing_factor` only smooths HF's own loss.** On
  `custom_loss: true` it does nothing (`Trainer.__init__` warns). On
  `custom_loss: false` with heads, HF pops `labels` to smooth the logits itself,
  so the model adds no head term, and since `CotorraForCausalLM` isn't in HF's
  `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` it smooths without shifting the labels;
  `Trainer.__init__` refuses that. A stock model on `custom_loss: false` is
  smoothed correctly.
- **A label of -100 isn't a target** (`bfd`'s padding, `doc_masking`'s piece
  starts). Cross-entropy skips it by `ignore_index`, but a custom-loss term that
  looks labels up in a per-token table (`label_to_cat`, `balanced_toi_flag`) has
  to clamp and mask them: a -100 index wraps around to a token from the end of
  the vocabulary.
- **`--resume-from-checkpoint` is safe to pass unconditionally**: `Trainer.train`
  falls back to training from scratch if no checkpoint is found.
- **`processed` is a symlink** (`-> ../cocoa/processed`) and `output/`, `wandb/`,
  `.venv/`, `dist/`, `build/`, `site/` are local/generated — not tracked, don't
  edit as source.

## Build, docs, release

```sh
# docs (mkdocs-material; config in mkdocs.yml, sources in docs/)
mkdocs build
mkdocs serve --dev-addr 127.0.0.1:8001

# release: bump pyproject.toml `version`, commit, then push a signed tag for it
git tag -s vXX.X.X -m "message"
git push origin vXX.X.X
```

Pushing a `v*` tag to `bbj-lab/cotorra` runs
[publish.yml](.github/workflows/publish.yml), which builds the sdist and wheel
and uploads them to PyPI and TestPyPI by trusted publishing (OIDC, as for
Codecov, so no token secret), in parallel jobs through the `pypi` and `testpypi`
environments, so neither index's failure blocks the other; TestPyPI skips a
version it already has. It fails unless the tag is `v` + `version`, points to the
commit built, and is signed (`git tag -s`, not a lightweight tag) with burkh4rt's
release key: it imports the public keys GitHub serves for that account and
accepts only the one whose fingerprint it pins (`RELEASE_KEY`), so letting anyone
else release means changing that. Each file gets two Sigstore-signed
attestations, both keyless (signed for the workflow's OIDC identity): SLSA build
provenance from the build job, kept by GitHub
(`gh attestation verify <file> --repo bbj-lab/cotorra`), and the publish action's
PEP 740 attestations, uploaded beside it to each index
(`pypi-attestations verify pypi --repository https://github.com/bbj-lab/cotorra pypi:<file>`).
PyPI no longer takes PGP signatures, so Sigstore is the only signing it shows.
PyPI refuses any requirement that's a direct URL reference, even in an extra, so
before building it drops every extra holding one: `gen`, whose `quick-sco-re`
comes from git. Don't build and upload by hand: a plain `python -m build` keeps
`gen`, and PyPI rejects the upload. Once `quick-sco-re` is on PyPI, pin it by
version and `gen` ships with no change to the workflow.

`docs/recipes/` and top-level `recipes/` are kept in sync (the docs include the
recipe markdown). `docs/index.md` likewise includes `README.md`, so README edits
change the docs home page, and `docs/api/` renders docstrings via mkdocstrings.
The end-to-end dev workflow (cocoa → tune → score) is in
[recipes/development-workflow.md](recipes/development-workflow.md).
[recipes/mimic-iv-demo.md](recipes/mimic-iv-demo.md) runs train → tune → extract
→ rep-based-score on the public MIMIC-IV demo, with the shipped configs and a few
overrides, and quotes what they make of it (parameter count, losses, Optuna's
trials, AUC intervals, timings, warnings), so rerun it after changing a shipped
config, a loss, the tuner, or the scorers, without a Hugging Face token, as the
public would.
