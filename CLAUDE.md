# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**Cotorra** (`cotorra` on PyPI) is a configurable trainer for generative event
models over tokenized timelines, built for foundation models on tokenized
electronic health records. Given tokenized subject timelines, it trains a causal
LM to predict the next token, then uses that model to extract representations and
score outcomes. It ships a CLI (`cotorra`) and is driven by YAML config. It is
the downstream sibling of [cocoa](https://github.com/bbj-lab/cocoa), which
produces the tokenized inputs cotorra consumes. The [README.md](README.md) is the
landing page on GitHub and PyPI, and the user guide ([docs/guide/](docs/guide/))
and the CLI reference ([docs/api/cli.md](docs/api/cli.md)) are the user-facing
reference (data formats, every CLI flag, config keys, model presets); this file
covers what you need to _work on the code_ — architecture, conventions, and
gotchas that aren't obvious from the source.

Each stage is a `cotorra` command ([cli.py](src/cotorra/cli.py), a Typer app) run
by a `Configurable` subclass, with a shipped default config in
[src/cotorra/config/](src/cotorra/config/):
[training.yaml](src/cotorra/config/training.yaml) for training and tuning,
[extraction.yaml](src/cotorra/config/extraction.yaml) for extraction, and
[scoring.yaml](src/cotorra/config/scoring.yaml) for the three scorers.

```
train / tune   →  extract  →  rep-based-score
               ↘  generative-score

baseline-score   (no model)
```

1. **Train / tune** ([trainer.py](src/cotorra/trainer.py),
   [tuner.py](src/cotorra/tuner.py)) — fit a causal LM: `Trainer` builds the
   model and trains it through `TrainerWithCustomLoss`, a subclass of HF's
   Trainer, and `Tuner` adds an Optuna hyperparameter search. → `mdl-<run_name>/`
   (HuggingFace `save_pretrained` format) and `mdl-<run_name>-training.yaml`,
   under `--output-home`
2. **Extract** (`Extractor`, [extractor.py](src/cotorra/extractor.py)) — run a
   trained model over inference contexts, writing hidden-state feature tables to
   `--output-home` (default: `--processed-data-home`), a row per row of
   `<split>_for_inference.parquet`, in its order and led by its `subject_id`
   (kept wherever every split's table has one). `-t`/`-n`/`-d`
   (`--time-to-event`, `--time-to-next-token`, `--discharge-disposition`) add
   float32 head columns after `features`, read at its positions (a value per row,
   or with `--all-times` a `max_seq_len` list, NaN after the row's final
   position, as `features` is): `time_to_event` and `time_to_next_token` in
   hours, and a `<class>_prob` per disposition class; `HEAD_COLUMNS` in
   [extractor.py](src/cotorra/extractor.py) maps each head name to its columns.
   `-l` (`--token-nll`) adds `token_nll_bits` last, from the LM's logits rather
   than a head, so any model has it: each token's −log₂ p given those before it,
   a float32 list as long as the row's `input_ids` whatever `--all-times` (0 at
   the first token, a `BOS` carrying no information; NaN past `extract.max_len`).
   → `features[-all]-<split>[-<i>-of-<n>]-<model_name>.parquet` (`-all` with
   `--all-times`; the shard suffix once a split exceeds `extract.shard_size`)
3. **Score** held-out timelines, three ways:
    - `rep-based-score` (`RepBasedScorer`, its estimators an `EstimatorType`,
      [scorer_rep_based.py](src/cotorra/scorer_rep_based.py)) fits a lightweight
      sklearn/boosting estimator on extracted features (requires `extract`
      first). It globs `features-<split>*-…` from `--processed-data-home`, so
      features extracted to another `--output-home` aren't found;
      `--training-home` (transfer) fits on another directory's train/tuning
      features and labels and scores this one's `held_out`.
    - `generative-score` (`GenerativeScorer`,
      [scorer_generative.py](src/cotorra/scorer_generative.py)) Monte-Carlo
      samples trajectories to compute MC/SCOPE/REACH scores (it doesn't need
      `extract`).
    - `baseline-score` (`BagOfTokensScorer` and `TsfreshScorer`, both
      `BaselineScorer`s, [baselines.py](src/cotorra/baselines.py)) fits the same
      estimators on features of the timelines alone, needing no model:
      `--baseline counts`/`one-hot`, a bag of `tokens_past`, writing
      `scores-bag-of-tokens-<featurization>-<estimator>`, or `tsfresh`, tsfresh's
      minimal features of `meds.parquet` up to the threshold, writing
      `scores-tsfresh-<estimator>`; `--training-home` as for `rep-based-score`.

Every stage reads cocoa's outputs from `--processed-data-home` (see **Inputs**
under Architecture notes). Around the stages:

| Module                                         | Class                                   | Role                                                         |
| ---------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| [configurable.py](src/cotorra/configurable.py) | `Configurable`                          | base: loads/merges config, holds a `Logger`                  |
| [loader.py](src/cotorra/loader.py)             | `Loader`                                | splits `tokens_times.parquet` by subject, builds HF datasets |
| [loss.py](src/cotorra/loss.py)                 | `Loss`                                  | custom losses (quantile-token, balanced-TOI) + head terms    |
| [model.py](src/cotorra/model.py)               | `CotorraForCausalLM`, `Head` subclasses | causal LM carrying any combination of secondary heads        |
| [logger.py](src/cotorra/logger.py)             | `Logger`                                | rich logging + bootstrap-CI eval summaries                   |
| [util.py](src/cotorra/util.py)                 | —                                       | batching helpers, `bootstrap_ci`                             |
| [openmp.py](src/cotorra/openmp.py)             | —                                       | on macOS, re-runs the CLI (and pytest) with torch's libomp   |

## Commands

```sh
# Dev install (Python 3.11 to 3.14), with torch from PyTorch's CUDA 12.8 index
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all]' \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
# ...without an NVIDIA GPU, use https://download.pytorch.org/whl/cpu as the index;
# '.[all,gen]' adds the sglang generative-score samples with (Linux, an NVIDIA
# GPU, Python 3.11 or 3.12; on 3.13 pip builds outlines-core from source, which
# takes rust, gcc, g++, libssl-dev and pkg-config)

# Train, then extract and score (every command reads -p <processed-data-home>)
cotorra train -p <processed-data-home> -o <output-home> [--verbose]
cotorra tune -p <processed-data-home> -o <output-home>  # with a hyperparameter search
cotorra extract -p <processed-data-home> -m <model-home> [-t -n -d -l] [--all-times]
cotorra rep-based-score -p <processed-data-home> -m <model-home> [-e <estimator>]
cotorra generative-score -p <processed-data-home> -m <model-home>  # needs [gen]
cotorra baseline-score -p <processed-data-home> -b counts|one-hot|tsfresh
cotorra train -t <config.yaml> ...  # replaces the shipped default; -e for extract's,
                                    # -s for the scorers'
cotorra <command> ... n_epochs=2 '~time_based_rope'  # trailing overrides edit keys
cotorra <command> -h                # help
cotorra --version                   # versions of cotorra, its dependencies, python,
                                    # and cuda, and the gpu

# Format + lint (run before committing; CI doesn't)
ruff format .
ruff check . --fix

# Tests (pytest, synthetic data only)
pytest
pytest -m "not slow"                # skips the few real (tiny) training loops

# Docs (zensical, configured in zensical.toml; published to readthedocs)
zensical build [--strict]           # --strict fails on a broken link or anchor
zensical serve --dev-addr 127.0.0.1:8001

# Release: bump pyproject.toml `version`, commit, then push a signed tag for it
git tag -s vYY.M.patch -m "message"
git push origin vYY.M.patch
```

There's a pytest suite in [tests/](tests/) — run it with `pytest` (config in
[pyproject.toml](pyproject.toml); install with `pip install -e ".[test]"`). CI
([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs it on every
push and pull request, and weekly, on Python 3.11 through 3.14, and once more on
3.11 with each dependency at the lowest version [pyproject.toml](pyproject.toml)
allows. A push or pull request touching only `joss-submission/` (the JOSS paper)
skips it, and [draft-pdf.yml](.github/workflows/draft-pdf.yml) compiles the paper
to a PDF artifact instead. It doesn't run ruff at all, neither `ruff check` nor
`ruff format`, so a style slip never fails it; that's left to the ruff run before
committing. It installs torch's CPU wheels from PyTorch's own index, so it never
runs CUDA or MPS code, nor the `[gen]` extra. The 3.11–3.14 jobs also install
`pytest-cov` (CI-only, not in the `test` extra) and upload coverage to
[Codecov](https://app.codecov.io/gh/bbj-lab/cotorra) by OIDC, as cocoa's do, so
no token secret is needed (fork PRs upload tokenless; Codecov refuses a tokenless
upload from any other branch). Coverage is informational only: the upload step
can't fail the job (`continue-on-error`), and [codecov.yml](codecov.yml) makes
its statuses always pass and waits for all four uploads before reporting
(`after_n_builds`; keep it equal to the matrix size). Don't add a coverage
threshold (`--cov-fail-under`, a status `target`) or make the upload required.
Keep the matrix in step with `requires-python`, and raise a floor in
`pyproject.toml` when code starts needing a newer dependency, or the lowest job
fails. On Linux that job gets torch 2.6; the darwin-only 2.8 floor is for MPS,
where 12 extraction and sampling tests fail under 2.6. `requires-python` stops
short of 3.15 because PyPI and the CUDA index have no torch wheels for it, and
the CPU index's segfault loading a saved model (seen on macOS with 3.15.0rc2).
Still run the suite locally before committing; `-m "not slow"` drops the handful
of real (tiny) training loops. `pythonpath` puts `src` ahead of site-packages so
a run always exercises the working tree even when a non-editable `cotorra` sits
in the venv — don't drop it.

The suite needs neither network nor real data: [tests/synth.py](tests/synth.py)
writes CLIF-like raw tables the fixtures in
[tests/conftest.py](tests/conftest.py) push through the real cocoa collate →
tokenize → winnow pipeline, and the model under test is a tiny
randomly-initialized `LlamaForCausalLM`; the config builders that fixtures and
tests share live in [tests/helpers.py](tests/helpers.py). The generative scorer's
tests run against quick-sco-re itself, which every install has, with its engine
and generation (sglang's part) swapped for fakes, so they run in CI too, without
the `[gen]` extra. When you change behavior, add or update a pytest case first.

`if __name__ == "__main__":` blocks are dev scratch harnesses, not tests or entry
points, and coverage skips them (see `[tool.coverage]` in
[pyproject.toml](pyproject.toml)). They often hardcode local paths
(`./processed/mimic`, for which cocoa's MIMIC-IV demo builds a public stand-in,
`processed/demo`, the one
[docs/recipes/mimic-iv-demo.md](docs/recipes/mimic-iv-demo.md) trains on) and a
trailing `# breakpoint()`. Leave them; they're for interactive debugging.

## Architecture notes

- **Every stage is a `Configurable`, and the CLI is thin**: it instantiates the
  stage's class and calls one method, so the real logic lives in the classes.
  Inheritance matters: `Tuner` extends `Trainer`, which extends `Configurable`;
  `Loss` and `Logger` stand alone. Changing `Trainer.__init__` or `collate_fn`
  affects tuning too.
- **Config resolution** ([configurable.py](src/cotorra/configurable.py)):
  `Configurable.__init__` builds `self.cfg` (an OmegaConf object) in three
  layers, each overriding the one before:
    1. a user config file passed via the relevant CLI flag (`--training-config`,
       etc.) or, only when none is passed, the class's packaged default YAML
       (`default_file`, in [src/cotorra/config/](src/cotorra/config/)); a
       `DictConfig` passed instead is copied as is,
    2. `overrides`: the `key=value` arguments trailing a CLI command, applied by
       `apply_overrides` (see below),
    3. keyword args passed to the constructor (only non-`None` values).

    A passed config file _replaces_ the packaged default rather than merging into
    it, so a key the file omits is absent from `self.cfg`, not filled in from the
    default, though an override can still add it (this is why the
    backward-compatibility rule under Conventions matters). The CLI passes
    `overrides=` to each class, which forwards it to `Configurable` through
    `**kwargs`.

- **Overrides** (`apply_overrides`, the same function as cocoa's): `key=value`
  sets a key, adding it if absent, so a typo adds a stray key rather than
  failing; `~key` deletes one; a Hydra-style leading `+`/`++` is ignored. Read
  optional keys defensively with `self.cfg.get(...)` or `"key" in self.cfg` —
  several features (`time_based_rope`, `quantile_token_loss`,
  `balanced_toi_loss`, and the `*_objective` blocks) are toggled purely by
  _presence_ of their config block, not a boolean, so an override setting one to
  `null` leaves it on and `~key` is what turns it off.
- **`Trainer` and `Extractor` hand their `Loader` their merged `self.cfg`**, not
  the config file's path: the `Loader` does the chunking (`max_seq_len`), the
  epoch repeats (`n_epochs`) and the column selection (`time_based_rope`, the
  heads), so reloading the file would silently drop overrides and kwargs. Any new
  class built from another's config should be handed `self.cfg` the same way.
- **The packaged configs are the source of truth for defaults.**
  [training.yaml](src/cotorra/config/training.yaml),
  [extraction.yaml](src/cotorra/config/extraction.yaml), and
  [scoring.yaml](src/cotorra/config/scoring.yaml) are documented key-by-key in
  the user guide. `training.yaml` defines reusable `model_presets` (YAML anchors)
  and selects one via the `model:` key.
- **A model is a backbone plus any combination of secondary heads**
  ([model.py](src/cotorra/model.py)), each a `Head` subclass registered in
  `HEADS` by its `name` (`tte`, `tnt`, `disposition`). That name keys the head's
  `<name>_objective` training block, its entry in `CotorraConfig.heads`, its
  `<name>_pred` / `<name>_loss` output fields, and its `extract` columns in
  `HEAD_COLUMNS`; its `target` names the batch column it's scored against, and
  its `target_weight` (`hours_to_end_time_weight`, `disposition_weight`; the TNT
  heads have none) the per-position weights it's scored with under `per_record`,
  which `forward` pops alongside the targets and `Head.forward` refuses to go
  without, or to take unasked. Inheritance is for what heads share: `TteHead` and
  `TntPointHead` are `Log1pHoursHead`s, and `TntPointHead` and `TntMixtureHead`
  are `TntHead`s, the interface sampling and extraction use. The combinations
  live in one class, `CotorraForCausalLM`, which builds a `ModuleDict` of the
  heads its config lists, so adding an objective takes a `Head` subclass in
  `HEADS` plus its two output fields, a `HEAD_COLUMNS` entry and an `extract`
  flag (tests hold both to `HEADS`'s keys), not a new model class; weighting one
  per record takes a `target_weight`, a `per_record` option, and a `SCORED` entry
  in [loader.py](src/cotorra/loader.py) (which positions its loss scores), which
  tests hold to the heads with a `target_weight`. `head_options` turns the
  training config's blocks into the heads' options (resolving the disposition
  `classes` patterns against the vocab) and is the one place `Trainer`, `Loader`
  and `Loss` read them from. `Trainer.model_init` builds a stock HF model when no
  objective is set. Importing `cotorra.model` registers `cotorra` with the HF
  auto classes, so anything that loads a saved `mdl-<run_name>/` (as `Extractor`
  does) has to import it first.
- **Point heads never go negative; a mixture TNT head conditions on the next
  token.** Point heads (`Log1pHoursHead`: `TteHead`, and the default TNT head)
  are an `nn.Linear` read through a softplus, so their log1p-hours, and the hours
  `expm1` turns them into, are never negative; a clamp would leave a head that
  drifts below zero without a gradient. The TNT head is such a point estimate
  (squared error on log1p-hours) unless `mixture_components` is set. Then it is a
  `TntMixtureHead`: given the hidden state _and the next token's embedding_
  (layer-normed first: a raw embedding sits ~50x below the normed hidden state,
  and unnormalized the head learned the zero-gap rate but not which tokens arrive
  at once), it emits a `ZeroInflatedLogNormalMixture` (an exact-zero gap for
  same-timestamp tokens, else a log-normal mixture), trained by NLL and leaving
  `tnt_pred` empty. `CotorraForCausalLM.predict_hours_to_next_token` (what
  `extract -n` writes) is either kind's prediction from the history alone,
  through `TntHead.mean_log1p_hours`: a point head's output, or a mixture head's
  mean log1p-hours given each next token, averaged over the LM head's softmax and
  never conditioned on the observed or likeliest one (the per-token mean is
  `ZeroInflatedLogNormalMixture.mean_log1p_hours`, by Gauss-Hermite quadrature).
  That costs a vocabulary's worth of head evaluations per position, so
  `TntMixtureHead.mean_log1p_hours` splits `proj`'s first layer over
  `[h; norm(e)]` to compute the vocabulary's half once per call and chunks
  positions by `chunk_elements`; a change to `proj`'s layout has to change it
  too. Every head is scored unshifted (position i against its own target) over
  every position, whatever `logits_to_keep` trims, and each loss is summed then
  normalized so a batch with nothing to score still gives its head a gradient.
  It's normalized by the count of the positions `Head.scored` marks, which the
  head's `loss` reads too, so that `TrainerWithCustomLoss` can count them to
  average the head's term over a whole evaluation (see `packing: pad`, below); a
  new head implements `scored` beside `loss`.
- **Inputs.** Stages read from `--processed-data-home` the files cocoa's
  `pipeline`/`winnow` produce:
    - `tokens_times.parquet` — `subject_id`, `tokens` (list[u32]), `times`
      (list[datetime]), plus `hours_to_end_time` (list[float], written under
      cocoa's `include_hours_to_end_time`) when training with `tte_objective`.
      Only training reads its rows, but `Loader` asserts the file exists and
      `Extractor` builds a `Loader`, so `extract` needs it too.
    - `subject_splits.parquet` — `subject_id` → `split` ∈ {`train`, `tuning`,
      `held_out`}.
    - `tokenizer.yaml` — must contain a `lookup:` map (token label → int id),
      incl. `BOS`, `EOS`; `quantile_token_loss` also reads `cfg.n_bins`. Read it
      with `load_yaml` from [configurable.py](src/cotorra/configurable.py), never
      a bare `OmegaConf.load`: from omegaconf 2.4, that refuses a document of
      more than 10,000 nodes unless told otherwise, and each word takes four (in
      `lookup` and `counts`), so any real vocabulary breaks it; `load_yaml` lifts
      the limit where the installed omegaconf has one.
    - `{train,tuning,held_out}_for_inference.parquet` — for extract/score;
      include `tokens_past`, optional `subject_id` and `s_elapsed_past`, and
      `<TOKEN>_past` / `<TOKEN>_future` label columns. `Loader` (`Trainer`'s too,
      when they're present) keeps only their context — `subject_id`,
      `tokens_past` as `input_ids`, and `s_elapsed_past` under `time_based_rope`
      — never a head's target or anything else that gives the outcome away
      (`hours_to_end_time_past`, `end_time`, `s_total_duration`, the future):
      `Extractor.extract_final` writes the batch it reads back out beside the
      features.
    - `meds.parquet` — cocoa's collated rows (`subject_id`, `time`, `code`,
      `numeric_value`, `text_value`), which only `TsfreshScorer` reads: each
      subject's rows up to the last token of its inference row's `tokens_past`
      (`baselines.past_meds`), the rows that past was tokenized from when the
      threshold is a time. Like `BagOfTokensScorer`, which bags `tokens_past`
      alone, it never reads an inference table's `tokens` or `times` past that
      cut: they hold the whole timeline, future and all.
- **`Loader` caches each split and weights its records.** It derives per-split
  `{split}_tokens_times.parquet` caches, adding `s_elapsed` and
  `hours_to_next_token` (the TNT target, nan on each record's last token), and
  regenerates them when `tokens_times.parquet` is newer or a cache predates a
  derived column. The disposition target depends on the config's `classes`, so it
  isn't cached: `Loader` labels each record (its last matching `DSCG//*` token's
  class before that token, -100 from it on) with `datasets.map` before packing
  splits records across chunks. Under an objective's `per_record` (the TTE's or
  the disposition's) it weights each record the same way: `mean / n` on each of
  its tokens, `n` counting those its head scores (a finite `hours_to_end_time`; a
  disposition other than -100) and `mean` being `n`'s average over the split's
  records with any, so that a fragment keeps its whole record's weight, and the
  weights average 1 over each split's scored tokens, keeping the loss on its
  unweighted scale. It counts them off the dataset itself, the split caches
  lacking the disposition target. Token-set selectors (`tokens_of_interest`,
  `target_tokens`) support **fnmatch patterns** (e.g. `LABEL//*`), resolved
  against the tokenizer vocab.
- **`packing` decides how `Loader.get_train_data` fills the train split's
  `max_seq_len` rows**: `concat` (the default, `util.batched_iter`) runs the
  shuffled, epoch-repeated records together and drops the last partial row; `bfd`
  (`util.best_fit_packed_iter`) cuts only records longer than a row and places
  each chunk whole by best-fit decreasing, padding each row with zeros and adding
  `pad_mask` and `doc_pos` (each token's position in its piece, restarting at 0
  for every piece and for the padding). It packs each of `n_epochs` passes apart,
  in an order of its own: packed in one pass, a record's copies, all of one
  length, sit side by side among the chunks sorted by length and land in the same
  rows, where the later attends to the earlier. `Trainer.collate_fn` keeps the
  padding from being any loss's target (-100 for the labels and the disposition,
  nan for the time heads); nothing else needs masking, since a piece ends short
  of its row only where its record ends (a split record's other chunks each fill
  a row), whose last token has no TNT target, so a mixture TNT head never
  conditions on another record's token. `doc_masking` (`bfd` only, and not yet
  with heads or `time_based_rope`) passes `doc_pos` as `position_ids` and labels
  each piece's first token -100; for flash attention, read off the built model
  (so `model.model_args` counts too), it flattens the batch into one row, which
  `Trainer.__init__` refuses under `nn.DataParallel`. The tuning split is then
  `bfd`-packed too, by its own fixed seed, so `eval_loss` scores the masked
  format the model trains on (without `doc_masking` it stays concatenated).
- **`packing: pad` batches by a budget of tokens.** `pad`
  (`util.record_chunks_iter`) cuts records as `bfd` does but makes each chunk a
  row of its own, unpadded, each of `n_epochs` passes in an order of its own, and
  drops a last chunk of one token: with no next token to predict, a batch of such
  rows alone would give HF's own loss no target, a nan in `eval_loss` and a 0/0
  in a step; `Trainer.collate_fn` pads each batch on the right to its longest
  row, with zeros masked as `bfd`'s padding is, and each row's first label masked
  too (see 5.5.4's count, under Conventions). It keeps time-based RoPE's all-ones
  `attention_mask` over the padding: causal attention keeps every token from the
  padding after it, and sdpa, given ones, keeps its causal fast path, where a
  mask with zeros would have it build a (rows, 1, len, len) one. The mixture TNT
  head needs no mask either: a row ends short of `max_seq_len` only where its
  record ends, and no batch pads past `max_seq_len`.
  `TrainerWithCustomLoss._get_dataloader` (where HF 5.5.4 and 5.19 alike build
  every dataloader) hands `pad`'s rows to a `DataLoader` of its own, whose
  `batch_sampler` is `TokenBudgetBatches`: `util.token_budget_batches` fills each
  batch to `tokens_per_batch` padded tokens (`Trainer.token_budget`; by default
  the batch size × `max_seq_len`; per GPU, as `nn.DataParallel` splits a batch's
  k rows ceil(k / `n_gpu`) to a GPU, each GPU's share held to it), each of the
  loader's `n_epochs` passes apart, sorted by length over the whole pass and cut
  shortest first, the batches then shuffled, the one holding the longest row
  first; seeded by `data_seed` or `seed`, and redrawn by `set_epoch`, which
  accelerate (≥1.3) forwards to a batch sampler. Why a budget rather than a count
  of rows: a step's loss is a mean over its batch's tokens and every step counts
  alike (marabou's runs clip every step, their gradient norms ~100×
  `max_grad_norm: 1`, so rescaling a step's loss changes nothing), so a few rows
  of one length to a batch weigh each record about alike; a record's length being
  its outcome, that tilts a generative model toward records that end soon (on a
  NICU split, the discharge rate it was fit to rose 3.4×). Megabatches, as HF's
  length-grouped sampler sorts, would leave the rare short rows a batch spanning
  a wide range of lengths, badly filled, so the sort is over the whole pass,
  fairseq's way. Under `pad`, `tally` weights the logged terms by the tokens a
  batch's LM scores rather than its rows, and in evaluation by the positions each
  term scores: `Loss.custom_loss` records each LM term's count beside it
  (`counts`), from the mask it averages over, and a head's is its
  `Head.scored`'s. `evaluation_loop`'s `eval_loss` is then `Loss.combine` of
  those means (the one place the objective's weights are applied, to a batch's
  terms in training too) or, for a loss function other than `Loss`'s, the loss's
  own mean over the tokens scored; HF's weighs batch means by rows, which a batch
  of hundreds of short records would dominate. Every other packing keeps HF's
  `eval_loss`, and terms weighted by rows, as released. `Trainer.__init__`
  refuses `pad` with `train_sampling_strategy` other than `group_by_length`, with
  more than one process (how accelerate shards batches of varying size differs by
  version), with `auto_find_batch_size` (which halves the batch size HF hands a
  dataloader, which the budget doesn't read, so every retry would rebuild the
  same batches), or with time-based RoPE under flash attention (see the gotcha
  below), and refuses `tokens_per_batch` without `pad` or below `max_seq_len`.

## Conventions

- **Style**: ruff, line length 88, double quotes, LF,
  `skip-magic-trailing-comma`. Lint set is `E,F,I` (isort included; first-party =
  `cocoa`, `cotorra`), configured in [pyproject.toml](pyproject.toml). Non-Python
  files (md/yaml/toml/json) are formatted by prettier (`proseWrap: always`,
  printWidth 81; 100 for yaml) and taplo — see
  [.prettierrc.toml](.prettierrc.toml) / [.taplo.toml](.taplo.toml). Markdown
  uses 4-space tabs and an 81-col ruler.
- **`import torch as t`** everywhere — match it, don't `import torch`.
- Files open with `#!/usr/bin/env python3` and a short lowercase module docstring
  describing the file's purpose; keep it. Method docstrings are terse and
  lowercase. Match the surrounding terseness.
- **Type hints**: when writing a new function, try to hint its parameters and
  return (`-> None` included), but only where the hints come out clean: short,
  readable, and accurate. If a precise hint would sprawl over several lines, or
  need a `cast`, a `# type: ignore`, or reshaped code to satisfy a type checker
  (none runs here), use a simpler one like `dict[str, Any]` or leave it off.
  Spell out `X | None` whenever the default is `None`, and hint a list read from
  a config `collections.abc.Sequence`, since OmegaConf hands it over as a
  `ListConfig`. Annotations are evaluated at import on Python 3.11, so quote
  forward references and skip 3.12+ names like `typing.override`; in `cli.py`,
  typer reads a command's parameter annotations, so changing one changes the CLI.
- **Logging** goes through `self.logger` (the parrot-emoji `Logger`), not
  `print`. The CLI itself uses `rich.print`/`Console` for status spinners and the
  `✓ …completed` summaries; a new CLI command goes in
  [cli.py](src/cotorra/cli.py) as a typer command that reports the same way.
- **Configs stay backward compatible with what's on PyPI.** An old config,
  written before a key existed and so missing it, must run exactly as it did
  before. The rule covers only what has been released to PyPI (check against the
  latest release tag, e.g. `git merge-base --is-ancestor <commit> v26.9.0`);
  anything unreleased, even on `main`, can change without an option to keep its
  old behavior. Since nothing fills in a missing key, the fallback in the code
  decides what happens, not the packaged YAML:
    - a new feature stays off when its block is absent (the presence toggle does
      this for free);
    - a new key inside an existing block falls back to the old behavior, e.g.
      `(self.cfg.block or {}).get("key", <old value>)` (`or {}` because a block
      left empty in the YAML parses to `None`);
    - an existing key keeps its meaning and default, and an optional key never
      becomes required;
    - a new block added to a packaged YAML ships commented out, as the
      `*_objective` blocks do, so runs without a config file are unchanged too;
    - the same goes for the model configs saved with checkpoints
      (`CotorraConfig`, and each head's options in it): a new field needs a
      default under which older checkpoints load and behave as before;
    - add a pytest case showing that a config without the new key produces the
      old output.

    Fixing a bug so a key does what it documents isn't a compatibility break:
    with `custom_loss: true`, `gradient_accumulation_steps` used to sum its
    micro-batches' gradients rather than average them (HF leaves that to a custom
    loss), and now averages them. It weights each micro-batch alike, where HF's
    stock path weights each scored token alike; the two agree while every
    micro-batch scores as many tokens, as concatenated rows do, `packing: pad`'s,
    filled to a budget of tokens, about do, and `bfd`'s needn't. With
    `custom_loss: false`, a model with heads added each head's mean, as it was,
    to HF's LM term: a sum over the step's `num_items_in_batch` tokens, for which
    HF skips its division by the micro-batches and multiplies by the processes
    (to undo DDP's averaging), so the heads' weights were multiplied by both in
    training, though not in evaluation. Now `CotorraForCausalLM.forward` weights
    each head's term by its micro-batch's share of those tokens, so that it
    scales as the LM term does under any of HF's scalings. Transformers 5.5.4
    counts every unmasked label, not the shifted ones the loss scores, so its
    count runs one token a row high, shrinking both terms alike by about one part
    in `max_seq_len`. `packing: pad`, whose rows are records of any length (where
    a target too many to each would shrink a short record's loss far more than a
    long one's), masks each row's first label in `collate_fn`, which no loss
    reads, so that its count is right on every version.

    Deprecating a key follows the same rule: `label_weighted_loss` (it made the
    model predict its tokens of interest more often) is still honored, and
    `Trainer.__init__` issues a `FutureWarning` (not a `DeprecationWarning`,
    which Python hides outside `__main__`). The one deliberate exception is the
    packaged `training.yaml`, which swapped its active `label_weighted_loss`
    block for `balanced_toi_loss`, so runs without a config file changed.

- **Versioning is CalVer** `YY.M.patch` (e.g. `26.11.0`); releases are signed git
  tags `vYY.M.patch`, published to PyPI. Not every tag was published: 26.6.0 was
  the first PyPI release, and `v26.4.0` never shipped. `__version__` comes from
  installed package metadata, not a literal. Pushing a `v*` tag to
  `bbj-lab/cotorra` runs [publish.yml](.github/workflows/publish.yml), which
  builds the sdist and wheel and uploads them to PyPI and TestPyPI by trusted
  publishing (OIDC, as for Codecov, so no token secret), in parallel jobs through
  the `pypi` and `testpypi` environments, so neither index's failure blocks the
  other; TestPyPI skips a version it already has. It fails unless the tag is
  `v` + `version`, points to the commit built, and is signed (`git tag -s`, not a
  lightweight tag) with burkh4rt's release key: it imports the public keys GitHub
  serves for that account and accepts only the one whose fingerprint it pins
  (`RELEASE_KEY`), so letting anyone else release means changing that. Each file
  gets two Sigstore-signed attestations, both keyless (signed for the workflow's
  OIDC identity): SLSA build provenance from the build job, kept by GitHub
  (`gh attestation verify <file> --repo bbj-lab/cotorra`), and the publish
  action's PEP 740 attestations, uploaded beside it to each index
  (`pypi-attestations verify pypi --repository https://github.com/bbj-lab/cotorra pypi:<file>`).
  PyPI no longer takes PGP signatures, so Sigstore is the only signing it shows.
  PyPI refuses any requirement that's a direct URL reference (`name @ git+…`),
  even in an extra, so the build job refuses one before building anything; it
  builds `pyproject.toml` as it stands, `gen` included, so keep every
  requirement, the extras' too, to a package on PyPI.
- Keep the user guide in [docs/guide/](docs/guide/) (a page per stage, plus
  packing, secondary heads, and configuration; scoring is a section of its own,
  [docs/guide/scoring/](docs/guide/scoring/): an overview holding the shared
  `scoring.yaml` keys and a page each for representation-based, generative,
  supervised-finetuning, and baseline scoring; it documents every config key),
  the recipes in [docs/recipes/](docs/recipes/), the CLI reference in
  [docs/api/cli.md](docs/api/cli.md) (whose help blocks are pasted
  `COLUMNS=75 cotorra <command> --help` output, to repaste when a flag changes;
  the rest of `docs/api/` renders docstrings via mkdocstrings),
  [README.md](README.md), and the shipped configs in sync when behavior changes.
  The scoring overview's figure,
  [docs/img/scoring-approaches.svg](docs/img/scoring-approaches.svg), is the JOSS
  paper's Figure 1, converted from its PDF by `pdftocairo -svg`, which turns its
  Gotham text into outlines so that it renders without the font; convert a
  revised PDF the same way. The recipes, a page each with its `nav` entry, link
  the guide relatively (`../guide/<page>.md`), which works on GitHub and the site
  alike. Top-level `recipes/` holds only forwarding notes for links to the old
  paths (v26.11.0's README links `recipes/mimic-iv-demo.md` on GitHub).
  `docs/index.md` includes three sections of `README.md` (`intro`, `install`,
  `references`, each between `<!-- --8<-- [start:<name>] -->` and `[end:<name>]`
  comments), so editing them changes the docs home page; snippets' `check_paths`
  fails the build if a marker goes missing. The rest of the README, a landing
  page whose links into the docs are absolute
  (`https://cotorra.readthedocs.io/latest/…`) since GitHub and PyPI render it,
  isn't in the docs. The end-to-end dev workflow (cocoa → tune → score) is in
  [docs/recipes/development-workflow.md](docs/recipes/development-workflow.md).
  [docs/recipes/mimic-iv-demo.md](docs/recipes/mimic-iv-demo.md) re-cuts cocoa's
  public MIMIC-IV demo at 6 hours and runs train → extract → rep-based-score →
  baseline-score on it (tune is left to its next steps, for slower laptops), with
  the shipped configs and a few overrides, and quotes what they make of it
  (parameter count, losses, AUC intervals, timings, warnings), so rerun it after
  changing a shipped config, a loss, the scorers or the baselines, or cocoa's
  demo (whose timelines it counts), without a Hugging Face token, as the public
  would.

## Gotchas

- Don't commit anything under `data-raw/`, `processed/`, `output/`, or `wandb/`
  (real patient data, and the models and runs trained on it; gitignored).
  `processed` is a symlink (`-> ../cocoa/processed`), and `.venv/`, `dist/`,
  `build/`, `site/` are local/generated too — not tracked, don't edit as source.
- Editing a shipped `src/cotorra/config/*.yaml` changes the default behavior for
  every user — usually you want a separate config passed with `-t`, `-e`, or `-s`
  instead.
- **Install requires the PyTorch CUDA index.** Use the two-index `pip install`
  from the README (or the dev install under Commands); a plain
  `pip install cotorra` will fail to resolve torch correctly.
- **`generative-score` needs the `[gen]` extra's sglang** to start its engine.
  [quick-sco-re](https://pypi.org/project/quick-sco-re/), which does the
  sampling, is a base dependency: from 0.3.0 it imports sglang only in
  `create_engine`, so [cli.py](src/cotorra/cli.py) imports `GenerativeScorer`
  with the other stages, and `gen` holds just sglang (which pins an exact torch
  and ships for Linux alone), as `quick-sco-re[sglang]` plus cotorra's own floor.
  `test_neither_the_cli_nor_the_scorer_imports_sglang` holds the CLI and the
  scorer to importing without it; were either to import it at load, a base
  install would run no command at all. `create_engine` refuses a model with
  heads, which SGLang can't load; `quick_sco_re.export_backbone` writes its
  backbone, a stock checkpoint, which `test_model.py` holds to the same logits.
  sglang ships wheels only for Linux with glibc 2.34 or later and Python 3.11 to
  3.13, and pins outlines 0.1.11, whose outlines-core 0.1.26 has wheels only to
  3.12, so on 3.13 pip compiles it, which takes Rust, gcc, g++, and OpenSSL's
  headers with pkg-config (the MIMIC-IV demo's generative step says how).
  Wherever sglang has no wheel (macOS, Python 3.14, older glibc), pip resolves
  `cotorra[gen]` to 26.11.0, the last release without a `gen` extra, with only a
  warning, so the docs' install commands carry `>=26.11.1`, which makes that an
  error instead; keep the floor.
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
  flash attention is only sound for batches of more than one row:
  `Trainer.__init__` refuses it under `packing: pad`, which puts a row over half
  of `tokens_per_batch` in a batch alone and can leave an epoch's last batch a
  single long row, and any packing gets one-row batches from
  `per_device_train_batch_size: 1`. HF sets `training_args.use_cache` on the
  model it builds in `__init__` (the one `built_trainer.model` saves in the
  tests), not on the one `model_init` rebuilds in `train()`, which keeps its
  `AutoConfig`'s.
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
- **tsfresh spawns its workers.** `TsfreshScorer`'s `n_jobs` (tsfresh's; half the
  CPUs by default) starts processes that, where Python spawns them (macOS, and
  Linux from 3.14), import the calling script again, so a script without an
  `if __name__ == "__main__":` guard recurses; the tests pass `n_jobs=0`, which
  extracts in-process. tsfresh is imported inside `baselines.tsfresh_features`,
  as importing it takes a second (half a minute the first time).
- **On macOS, `cotorra` and pytest run themselves again.** LightGBM's and
  XGBoost's macOS wheels link `@rpath/libomp.dylib`, which their rpaths resolve
  to Homebrew's libomp, while torch loads the one in its `lib`. Without
  Homebrew's, importing either booster fails; with it, a process holds two OpenMP
  runtimes and segfaults in them: as a booster fits after torch was imported, or
  as torch next trains after a booster was imported, so that one test module
  importing a booster at collection took the whole suite down. Which crash a
  command meets depends on its import order (`cli.py` importing `baselines`
  before `extractor` spares `rep-based-score` its), so don't rely on it. The
  dynamic linker reads `DYLD_LIBRARY_PATH` only as a process starts, so
  [openmp.py](src/cotorra/openmp.py) re-execs the process with torch's `lib`
  first on it: `cotorra.openmp:main` is the console script, re-running before it
  imports `cotorra.cli` (whose `main` stays, unguarded, for `python -m`), and
  `tests/conftest.py`'s `pytest_configure` re-runs pytest, having stopped
  pytest's capture, whose files the new process would otherwise write to. Neither
  re-runs under `COTORRA_OPENMP_RERUN` (which the re-run sets, so that an
  interpreter macOS protects, dropping `DYLD_*`, can't loop it; a user may set it
  too), under a debugger (`pydevd`) or in a Jupyter kernel; there
  `conftest.boosters_usable` still skips the boosters' fits, though not torch's
  crash after their import. Keep `openmp.py` to the standard library, as it runs
  before anything that loads a booster. CI never runs macOS, so there only
  `test_openmp.py`'s fakes exercise it; its one real re-run skips off macOS.
- **`--resume-from-checkpoint` is safe to pass unconditionally**: `Trainer.train`
  falls back to training from scratch if no checkpoint is found.
- The docs build with [Zensical](https://zensical.org) from
  [zensical.toml](zensical.toml), formatted like `pyproject.toml` (taplo, with
  4-space indents). Taplo sorts the keys between blank lines and comments, so a
  comment on a key goes inline or above its table. The theme is the `modern`
  variant, which draws the header and tabs in the page's colors, so
  [extra.css](docs/stylesheets/extra.css) paints them maroon itself, by
  redefining the page's colors inside them. It also swaps the theme's greys for
  UChicago's neutrals (black, the three greystones, white), taking light
  greystone at half strength, `#ececec`, for the larger surfaces. Admonitions
  take the UChicago colors of cocoa's visualizer, a link turns dark brick under
  the pointer, and a card on the home page has a maroon icon, and a maroon border
  under the pointer. It hides every page's table of contents, both the sidebar on
  the right and the button that stands in for it on a narrow screen (the modern
  variant never repeats it in the navigation), so a page's text takes the width;
  the home page's `hide: toc` front matter predates that rule.
  [docs/index.md](docs/index.md) wraps the README's `intro` in a `.hero` block,
  and the stylesheet centers its title, the paragraph right after it (the
  badges), and its blockquote (the tagline), so keep them in that order. The
  `glightbox` plugin zooms an image on click; the README's hero opts out with
  `class="off-glb"`. cocoa's docs share the stylesheet, overrides, and theme
  settings, so a change to how one site looks goes to the other too. Zensical
  runs its own versions of the plugins listed there and silently skips any it
  lacks: so far just `git-revision-date-localized`, so the footer's "Last
  updated" line ([copyright.html](overrides/partials/copyright.html)) stays empty
  until Zensical ships it. It caches builds in `.cache/` and doesn't notice when
  the docs dependencies change, so run `zensical build --clean` after upgrading
  them; a stale cache once passed a build that a clean one failed.
