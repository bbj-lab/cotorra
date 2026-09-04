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
  feature tables (`features-<split>-<model_name>.parquet`).
- **rep-based-score** — fit a lightweight sklearn/boosting estimator on extracted
  features (requires `extract` first).
- **generative-score** — Monte-Carlo sample trajectories to compute
  MC/SCOPE/REACH scores (does not need `extract`).

## Architecture

Each pipeline stage is a class that subclasses `Configurable`
([src/cotorra/configurable.py](src/cotorra/configurable.py)). The CLI is a thin
wrapper that instantiates the class and calls one method; the real logic lives in
the classes.

| Module                                                   | Class                              | Role                                                         |
| -------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| [configurable.py](src/cotorra/configurable.py)           | `Configurable`                     | base: loads/merges config, holds a `Logger`                  |
| [loader.py](src/cotorra/loader.py)                       | `Loader`                           | splits `tokens_times.parquet` by subject, builds HF datasets |
| [trainer.py](src/cotorra/trainer.py)                     | `Trainer`, `TrainerWithCustomLoss` | model init + HF Trainer                                      |
| [tuner.py](src/cotorra/tuner.py)                         | `Tuner`                            | `Trainer` + Optuna hyperparameter search                     |
| [loss.py](src/cotorra/loss.py)                           | `Loss`                             | custom losses (quantile-token, label-weighted)               |
| [extractor.py](src/cotorra/extractor.py)                 | `Extractor`                        | hidden-state extraction                                      |
| [scorer_rep_based.py](src/cotorra/scorer_rep_based.py)   | `RepBasedScorer`, `EstimatorType`  | estimator-on-features scoring                                |
| [scorer_generative.py](src/cotorra/scorer_generative.py) | `GenerativeScorer`                 | SCOPE/REACH generative scoring                               |
| [logger.py](src/cotorra/logger.py)                       | `Logger`                           | rich logging + bootstrap-CI eval summaries                   |
| [util.py](src/cotorra/util.py)                           | —                                  | batching helpers, `bootstrap_ci`                             |

Inheritance matters: `Tuner` extends `Trainer`, which extends `Configurable`;
`Loss` and `Logger` stand alone. Changing `Trainer.__init__` or `collate_fn`
affects tuning too.

### Configuration model (important)

`Configurable.__init__` merges three layers via OmegaConf, later overriding
earlier:

1. the class's packaged default YAML (`default_file`, in
   [src/cotorra/config/](src/cotorra/config/)),
2. a user config file passed via the relevant CLI flag (`--training-config`,
   etc.),
3. keyword args passed to the constructor (only non-`None` values).

So CLI flags reach config by being threaded as kwargs to the constructor. The
merged result is `self.cfg` (an OmegaConf object). Read optional keys defensively
with `self.cfg.get(...)` or `"key" in self.cfg` — several features
(`time_based_rope`, `quantile_token_loss`, `label_weighted_loss`) are toggled
purely by _presence_ of their config block, not a boolean.

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
  entry points. They often hardcode local paths (`./processed/mimic`) and a
  trailing `# breakpoint()`. Leave them; they're for interactive debugging.
- **Tests: `pytest`** (config in [pyproject.toml](pyproject.toml); install with
  `pip install -e ".[test]"`). No CI, so run it locally before committing;
  `-m "not slow"` drops the handful of real (tiny) training loops. `pythonpath`
  puts `src` ahead of site-packages so a run always exercises the working tree
  even when a non-editable `cotorra` sits in the venv — don't drop it. The suite
  needs neither network nor real data: [tests/synth.py](tests/synth.py) writes
  CLIF-like raw tables the fixtures push through the real cocoa collate →
  tokenize → winnow pipeline, and the model under test is a tiny
  randomly-initialized `LlamaForCausalLM`.
  [tests/test_scorer_generative.py](tests/test_scorer_generative.py) skips
  without the `[gen]` extra.
- **Logging** goes through `self.logger` (the parrot-emoji `Logger`), not
  `print`. The CLI itself uses `rich.print`/`Console` for status spinners and the
  `✓ …completed` summaries.

## Data contract (inputs)

Stages read from `--processed-data-home`. Expected files (produced by cocoa's
`pipeline`/`winnow`):

- `tokens_times.parquet` — `subject_id`, `tokens` (list[u32]), `times`
  (list[datetime]); training only.
- `subject_splits.parquet` — `subject_id` → `split` ∈ {`train`, `tuning`,
  `held_out`}.
- `tokenizer.yaml` — must contain a `lookup:` map (token label → int id), incl.
  `BOS`, `EOS`.
- `{train,tuning,held_out}_for_inference.parquet` — for extract/score; include
  `tokens_past`, optional `s_elapsed_past`, and `<TOKEN>_past` / `<TOKEN>_future`
  label columns.

`Loader` derives per-split `{split}_tokens_times.parquet` caches and regenerates
them when `tokens_times.parquet` is newer. Token-set selectors
(`tokens_of_interest`, `target_tokens`) support **fnmatch patterns** (e.g.
`LABEL//*`), resolved against the tokenizer vocab.

## Gotchas

- **Install requires the PyTorch CUDA index.** Use the two-index `pip install`
  from the README; a plain `pip install cotorra` will fail to resolve torch
  correctly. Development install is `pip install -e ".[gen]"`.
- **`generative-score` needs the `[gen]` extra** (`quick-sco-re`, `sglang`),
  which is installed from git and only available from source. Note it's commented
  out in [pyproject.toml](pyproject.toml)'s `optional-dependencies`;
  `GenerativeScorer` imports `quick_sco_re` at module load, so
  [cli.py](src/cotorra/cli.py) imports it _lazily_ inside the command. Preserve
  that laziness so the base install works without these heavy/optional deps.
- **`time_based_rope` must match between training and extraction.** If a model
  was trained with time-based RoPE, the extraction config must enable it too
  (same `sec_per_pos_id`), or position ids won't line up.
- **`remove_unused_columns: false`** in `training_args` is required — otherwise
  HF drops the `s_elapsed` column that time-based RoPE needs.
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

# build + publish to PyPI
rm -rf dist && python3 -m build && python3 -m twine upload --repository pypi dist/*

# release tag (signed); version lives in pyproject.toml `version`
git tag -s vXX.X.X -m "message"
```

`docs/recipes/` and top-level `recipes/` are kept in sync (the docs include the
recipe markdown). The end-to-end dev workflow (cocoa → tune → score) is in
[recipes/development-workflow.md](recipes/development-workflow.md).
