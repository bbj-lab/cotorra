# Contributing to Cotorra

Thank you for helping to improve Cotorra! Bug reports, questions, typo fixes, and
new features are all welcome. ✨ Please keep discussions kind, patient, and
constructive: our contributors come from medicine, statistics, and software, and
we want everyone to feel welcome.

## Before you start

Read the [README](README.md) and the
[documentation](https://cotorra.readthedocs.io). The
[MIMIC-IV demo](recipes/mimic-iv-demo.md) is the quickest way to see what each
stage does, and [CLAUDE.md](CLAUDE.md) maps the codebase's architecture and
conventions for human contributors and coding agents alike. Cotorra trains on the
tokenized timelines that [Cocoa](https://github.com/bbj-lab/cocoa) produces, so a
problem with how your data was collated or tokenized may belong there instead.

New to open source? GitHub has guides to
[finding ways to contribute](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github),
[setting up Git](https://docs.github.com/en/get-started/git-basics/set-up-git),
the
[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow),
and
[pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests).

## What we're looking for

We welcome bug reports (ideally with a small example that reproduces the problem)
and fixes; corrections to the README, recipes, and API docs; new recipes; new
configuration options, model presets, training objectives, and scoring
estimators; and tests for behavior that isn't covered yet.

Please open an issue to discuss it first if you want to:

- change a shipped default config in [src/cotorra/config/](src/cotorra/config/),
  which changes the output for every user
- add a pipeline stage or CLI command
- change what an existing config produces, or how a saved model loads
- add a dependency, since the base install is already large

If you aren't sure whether your idea fits, open an issue and ask.

## Patient data

**Never put real patient data in an issue, a pull request, a commit, or a
screenshot,** nor anything derived from it: a model trained on it, or the
features, scores, and timelines that model produces. Git ignores `data-raw/`,
`processed/`, `output/`, and `wandb/` to guard against accidents. To reproduce a
problem, use the synthetic data that [tests/synth.py](tests/synth.py) generates,
or the public MIMIC-IV demo.

## Issues

[Search the existing issues](https://github.com/bbj-lab/cotorra/issues) first. If
yours is new, open one, and for a bug include:

- your Cotorra (`pip show cotorra`), Python, and PyTorch versions, and your GPU
  and the CUDA version PyTorch was built for, if you have one
- the command you ran, with any configs and overrides you passed
- the full traceback, and what you expected to happen

To work on an existing issue, leave a comment so that others know, and ask any
questions there.

## Making changes

1. Fork the repository, clone your fork, and create a branch from `main`.
2. Install Cotorra in development mode (Python 3.11 to 3.14):

    ```sh
    python -m venv .venv && . .venv/bin/activate
    pip install -e '.[all]' \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple
    ```

    Without an NVIDIA GPU, use `https://download.pytorch.org/whl/cpu` as the
    index. To work on `generative-score`, install `'.[all,gen]'`, which needs
    Linux, Python 3.11 to 3.13, and an NVIDIA GPU (see the
    [MIMIC-IV demo](recipes/mimic-iv-demo.md)); without it, its tests run against
    a stand-in, `tests/fake_quick_sco_re.py`, and only the one holding that
    stand-in to the real package skips.

As you work:

- **Test what you change.** The tests push synthetic data through Cocoa and train
  a tiny, randomly initialized model, so `pytest` needs no real data, GPU, or
  network. `pytest -m "not slow"` skips the few real training loops while you
  iterate.
- **Stay backward compatible** with the releases on PyPI. A config that worked
  with a released version must keep working, with the same output, without any
  key you add, and older saved models must load and behave as before. Give each
  new feature or key a fallback to the old behavior, and test it. The
  "Configuration model" section of [CLAUDE.md](CLAUDE.md) has the details.
- **Update the docs** to match: the [README](README.md) (which documents every
  config key and CLI flag), the [recipes](recipes/), the [API docs](docs/api/),
  and the shipped configs. Edit recipes in [recipes/](recipes/); a new one also
  needs a stub in [docs/recipes/](docs/recipes/) and a `nav` entry in
  [mkdocs.yml](mkdocs.yml).
- **Match the surrounding style.** Ruff formats and lints the Python (line length
  88, double quotes); the code imports `torch as t` and logs through
  `self.logger` rather than `print`. Prettier wraps the Markdown, following
  [.prettierrc.toml](.prettierrc.toml).

## Pull requests

Before you push, format, lint, and run the full test suite, slow tests included:

```sh
ruff format .
ruff check . --fix
pytest
```

Review your own diff for leftover debugging lines or stray files, then open a
pull request against `main` on
[bbj-lab/cotorra](https://github.com/bbj-lab/cotorra):

- Describe what you changed and why, and
  [link any issue it resolves](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue),
  e.g. with `Closes #123`.
- [Allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
  so that a maintainer can update your branch before merging.
- CI runs `pytest` on Python 3.11 through 3.14 and again with each dependency at
  the oldest version [pyproject.toml](pyproject.toml) allows. It doesn't run
  ruff, so formatting and linting are up to you before you push. It runs on CPUs
  only, so if your change touches GPU code, say how you tested it there. Codecov
  then reports the change in coverage, but only for information: its checks
  always pass.
- A maintainer will review it and may suggest changes or leave comments. Mark
  each conversation as resolved once you've addressed it.

Cotorra is open source under the [MIT License](LICENSE.md), and by opening a pull
request you agree that your contribution is licensed under it too. Once merged,
it ships with the next release of [`cotorra`](https://pypi.org/project/cotorra/)
and its [documentation](https://cotorra.readthedocs.io). Thank you! 🎉
