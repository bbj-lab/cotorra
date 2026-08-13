# k=5, large architecture — basis blended tokens

## Setup

Same full uncapped Optuna tuning pipeline (`cotorra tune`, 10 trials + best-config
final run, wandb-logged) used for every prior basis-blended run, but with two
changes on top of the best config found so far (plain `k=8`, trainable
`alpha`/`beta`, importance scaling off):

- **Architecture**: `hidden_size` 256→1024, `intermediate_size` 1024→2048,
  `num_hidden_layers` 6→8 (all other runs so far used the small architecture).
  ~69.2M params vs. ~8.4M for every prior run (~8.3x).
- **`k`**: 8→5 basis elements per numeric category.

`train_beta_params: true`, `train_importance_scale: false` (unchanged from the
best config so far). Scored with `--estimator logistic-CV`, outcomes with
AUC<0.55 in either compared model excluded (none dropped here).

See `training.yaml` / `extraction.yaml` in this directory for the exact config.

## Results

15/15 outcomes scored above the 0.55 floor.

| Comparison | Mean AUC delta | Win rate |
| --- | --- | --- |
| vs. small-architecture baseline (`mdl-baseline-mimic-tune`) | **+0.0252** | 15/15 |
| vs. previous best (`k=8`, small architecture, trainable alpha/beta) | **+0.0175** | 15/15 |

This is by a wide margin the largest improvement seen in this experiment
series so far — more than 2x the previous best basis-blended result's
own margin over baseline (`k=8` was +0.0077 / 14 of 15 vs. baseline). Every
single outcome improved, including the two that had been near the AUC floor
before (`LABEL//sepsis_onset`: 0.560→0.562 vs. k=8, still weak in absolute
terms but no longer regressing).

Largest per-outcome gains vs. k=8 (small arch): `LABEL//hypona_init` (+0.036),
`LABEL//anemia_init` (+0.034), `RESP//imv` (+0.022), `DSCG//expired` (+0.021).
Smallest: `LABEL//sepsis_onset` (+0.002), `LABEL//hypertension_init` (+0.007).

## Caveat — this is not yet an apples-to-apples test of the method

This run changes **two things at once** relative to the previous best (k=8,
small arch): the basis-blended config (`k`) *and* the base architecture
(~8.3x more parameters). The comparison above is against the *small*-architecture
baseline, so it cannot separate "the basis-blended mechanism got better at k=5"
from "the model just got bigger." Because this result clearly improved over
current best, the next step (per the standing instructions for this series) is
training a same-architecture baseline (`../baseline_mimic_tune_large/`) so the
large-arch basis-blended model has a fair, matched-capacity comparison point.

## Bug found and fixed along the way (unrelated to this experiment's result)

While setting this up, reloading the original `mdl-basis-blended-mimic-tune`
(k=8) checkpoint via `AutoModelForCausalLM.from_pretrained` was found to leave
`log_importance` as uninitialized garbage memory rather than zero, because
that checkpoint predates the `log_importance` parameter and HF's fast-init
path only reinitializes params *missing* from a checkpoint, at *module*
granularity — and `log_alpha`/`log_beta`/`log_importance` are three separate
params on the same module. Fixed in `BasisBlendedCausalLM._init_weights` by
checking each param's own `_is_hf_initialized` flag individually (see
`src/cotorra/basis_blended.py`, with a new regression test in its `__main__`
self-test block). This silently affected the notebook's Beta-components/PCA/
interactive-slider sections for the k=8 model specifically (not the AUC
comparisons, which only read scored parquet files off disk); those sections
will be regenerated once the notebook is re-executed after this experiment
series wraps up.
