<!-- --8<-- [start:intro] -->

<p align="center">
<img src="https://raw.githubusercontent.com/burkh4rt/cotorra/master/img/cotorra.png" alt="friendly parrot" width="400" class="off-glb" />
</p>

# Cotorra: a configurable trainer

[![PyPI Version](https://img.shields.io/pypi/v/cotorra)](https://pypi.org/project/cotorra/)
[![DOI](https://raw.githubusercontent.com/burkh4rt/cotorra/master/img/1193885071.svg)](https://doi.org/10.5281/zenodo.20414127)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/bbj-lab/cotorra/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/bbj-lab/cotorra)
[![tests](https://github.com/bbj-lab/cotorra/actions/workflows/tests.yml/badge.svg)](https://github.com/bbj-lab/cotorra/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/bbj-lab/cotorra/graph/badge.svg?token=93WBN0HCUV)](https://codecov.io/gh/bbj-lab/cotorra)

> 🦜 the wild parakeet of Chicago's south side

## About

This repo provides a configurable trainer for generative event models on
tokenized timelines. It also supports three types of downstream inference on
held-out data: (i) representation-based, (ii) generative, and (iii) inference via
supervised finetuning. _Cotorra_ is a Spanish term for a small-to-medium sized
parrot, particularly the Monk parakeet. Monk parakeets were introduced to the
south side of Chicago, where they have flourished. [^1] The library benefits from
previous experience training foundation models on tokenized electronic health
records. [^2] [^3] [^4] [^5] [^6]

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:install] -->

## Installation

Install the latest release from PyPI:

```sh
pip install "cotorra" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

This installs the `cotorra` command. To see it run end to end on public data,
follow the
[MIMIC-IV demo](https://cotorra.readthedocs.io/latest/recipes/mimic-iv-demo/).

`cotorra generative-score` also needs SGLang, which the `gen` extra installs
(`pip install "cotorra[gen]>=26.11.1"`) on Linux with an NVIDIA GPU and Python
3.11 or 3.12, or 3.13 once a Rust compiler and a few build tools are in place;
see the MIMIC-IV demo's
[generative step](https://cotorra.readthedocs.io/latest/recipes/mimic-iv-demo/#6-predict-outcomes-generatively-linux-with-an-nvidia-gpu).

On a Mac, LightGBM and XGBoost look for Homebrew's OpenMP runtime, while PyTorch
brings its own: without Homebrew's, importing cotorra fails, and with it, a
process holding both runtimes can crash. The `cotorra` command sees to this
itself, starting again with PyTorch's runtime first on `DYLD_LIBRARY_PATH`. To
import cotorra in Python of your own, such as a notebook, set that before Python
starts:

```sh
export DYLD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib"
```

<!-- --8<-- [end:install] -->

## Usage

Each stage of the pipeline is a `cotorra` command, driven by a YAML config:

```
train / tune  →  extract  →  rep-based-score
               ↘  generative-score

baseline-score (transformer-free)
```

The [documentation](https://cotorra.readthedocs.io/latest/) covers:

- [Inputs](https://cotorra.readthedocs.io/latest/guide/inputs/): the tokenized
  timelines and tables the commands read, as
  [cocoa](https://pypi.org/project/cocoa-tokenizer/) writes them.
- [Training](https://cotorra.readthedocs.io/latest/guide/training/): fitting a
  model with `cotorra train` or `cotorra tune`, including how records are
  [packed into rows](https://cotorra.readthedocs.io/latest/guide/packing/) and
  the
  [secondary heads](https://cotorra.readthedocs.io/latest/guide/secondary-heads/)
  it can train alongside next-token prediction.
- [Extraction](https://cotorra.readthedocs.io/latest/guide/extraction/):
  hidden-state representations and head predictions from a trained model, with
  `cotorra extract`.
- [Scoring](https://cotorra.readthedocs.io/latest/guide/scoring/): outcome scores
  for held-out data, from an estimator fit on extracted features
  (`cotorra rep-based-score`), from sampled trajectories
  (`cotorra generative-score`), or from a supervised head; and baselines that
  need no model (`cotorra baseline-score`).
- [Configuration](https://cotorra.readthedocs.io/latest/guide/configuration/):
  the YAML config each command reads, and how to change its keys on the command
  line.
- The [CLI reference](https://cotorra.readthedocs.io/latest/api/cli/), with every
  command's options, and the
  [recipes](https://cotorra.readthedocs.io/latest/recipes/), worked examples from
  a run on public data to transfer learning.

## Contributing

Bug reports, fixes, recipes, and new features are all welcome. The
[contributing guide](https://github.com/bbj-lab/cotorra/blob/main/CONTRIBUTING.md)
explains how to report an issue, set up a development install, and open a pull
request.

<!-- --8<-- [start:references] -->

[^1]:
    L. Gersony, "The Quiet Victory of Chicago’s Monk Parakeets," _The Chicago
    Maroon_, 23 January 2022,
    [https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/](https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/)

[^2]:
    M. Burkhart, B. Ramadan, Z. Liao, K. Chhikara, J. Rojas, W. Parker, & B.
    Beaulieu-Jones, Foundation models for electronic health records:
    representation dynamics and transferability,
    [arXiv:2504.10422](https://doi.org/10.48550/arXiv.2504.10422)

[^3]:
    M. Burkhart, B. Ramadan, L. Solo, W. Parker, & B. Beaulieu-Jones,
    [Quantifying surprise in clinical care: Detecting highly informative events in electronic health records with foundation models](https://doi.org/10.1142/9789819824755_0013),
    Pacific Symposium on Biocomputing 31 (2026), 173–188

[^4]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)

[^5]:
    I. Lee, L. Solo, M. Burkhart, B. Ramadan, W. Parker, & B. Beaulieu-Jones,
    Representation before training: a fixed-budget benchmark for generative
    medical event models,
    [arXiv:2604.16775](https://doi.org/10.48550/arXiv.2604.16775), _to appear in
    ML4H 2026_

[^6]:
    M. Burkhart, L. Solo, I. Lee, S. Charles, Z. Liao, K. Chhikara, D. Therese,
    W.-T. Liao, C. Gao, W. Parker, & B. Beaulieu-Jones, Federated generative
    event models for tokenized electronic health records,
    [arXiv:2608.02939](https://doi.org/10.48550/arXiv.2608.02939)

<!-- --8<-- [end:references] -->

<!--

Run in tmux:
```
tmux new -s co || tmux a -t co
```

Format:
```sh
ruff format .
ruff check . --fix
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "output" \
 --exclude "wandb" \
 --exclude ".venv" \
 --exclude ".idea" \
 ~/Documents/chicago/cotorra \
 bbj-lab1:~
```

```
for d in data-raw processed; do ln -s /mnt/bbj-lab/users/burkh4rt/$d $d; done
ds='mimic-icu'
cotorra train \
		--training-config src/cotorra/config/training.yaml \
		--processed-data-home processed/mimic-icu \
		--output-home output/test \
    --resume-from-checkpoint \
    --verbose
```

Send to randi:
```
for d in data-raw processed; do ln -s /gpfs/data/bbj-lab/users/burkh4rt/$d $d done
```
```
rsync -avh \
 --exclude "output" \
 --exclude "processed" \
 --exclude "data-raw" \
 --exclude "logs" \
 --exclude "wandb" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cotorra \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

```
srun -p gpuq \
 --gres=gpu:1 \
 --time=8:00:00 \
 --job-name=adhoc \
 --pty bash -i
. .venv/bin/activate
```

Make docs:
```
zensical build
zensical serve --dev-addr 127.0.0.1:8002
```

Make tag and send to pypi and testpypi (pushing a `v*` tag runs
`.github/workflows/publish.yml`, which builds and uploads to both):
```
git tag -s v26.11.0 -m "optimized optuna"
git push origin v26.11.0
git push upstream v26.11.0
```

-->
