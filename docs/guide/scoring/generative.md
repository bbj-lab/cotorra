# Generative scoring

**Generative scoring** (`cotorra generative-score`) Monte Carlo samples future
trajectories directly from the model. It:

1. Loads the trained model and held-out inference data.
2. Samples future trajectories for each target token.
3. Computes MC, SCOPE, and REACH scores [^1] per target token.

It samples with the [quick-sco-re](https://pypi.org/project/quick-sco-re/)
package, which comes with cotorra, through SGLang, which cotorra's `gen` extra
installs (`pip install 'cotorra[gen]>=26.11.1'`) on Linux with an NVIDIA GPU and
Python 3.11 or 3.12, or 3.13 once a Rust compiler and a few build tools are in
place; see the MIMIC-IV demo's
[generative step](../../recipes/mimic-iv-demo.md#6-predict-outcomes-generatively-linux-with-an-nvidia-gpu).

SGLang can't load a model trained with [secondary heads](../secondary-heads.md),
which cotorra saves as a `cotorra` model, so quick-sco-re refuses one. Score its
backbone instead, which has the same next-token distribution: quick-sco-re's
`export_backbone` writes it on its own, as a model trained without heads is
saved:

```sh
python -c 'from quick_sco_re import export_backbone; export_backbone("<model-home>", "<model-home>-backbone")'
cotorra generative-score -p <processed-data-home> -m <model-home>-backbone
```

## Configuration

Beyond the outcomes to score (see [Configuration](index.md#configuration)),
generative scoring reads the rest of `scoring.yaml`:

- **score**:
    - **max_len**: Maximum input length (tokens) during scoring.
    - **n_samp**: Number of Monte Carlo samples per input per trajectory type.
    - **target_tokens**: Every matching vocabulary token becomes an outcome that
      `generative-score` samples trajectories for.
    - **end_tokens**: Tokens that naturally terminate a generated sequence (e.g.
      `EOS`).
    - **suppressed_tokens**: Tokens to suppress via logit bias during generation
      (e.g. `PAD`).
    - **batch_size**: Batch size for inference.
- **generation** _(modeled on the `pipeline_config.yaml` used by
  [quick-sco-re](https://pypi.org/project/quick-sco-re/))_:
    - **temperature**: Sampling temperature (1.0 for proper Monte Carlo
      estimation).
    - **score_inline**: If true, tracks every `score.target_tokens` outcome's
      SCOPE/REACH estimate in a single generation pass instead of a separate pass
      per outcome.
    - **methods**: Which trajectory types to run — some subset of `[M1, M2]`.
    - **end_tokens.prefixes**: Additional end-token prefixes (e.g. `DSCG`), on
      top of `score.end_tokens`.
    - **time_stopping** _(optional time-based truncation)_:
        - **enabled**: Turns on simulated-time-horizon stopping.
        - **trunc_token**: Token forced once the time horizon is exceeded.
        - **max_time_minutes**: Maximum simulated time horizon, in minutes.
        - **time_check_interval**: Tokens generated between time-horizon checks.
        - **time_token_bounds**: Map of time-bin token name to `[lo, hi]` minute
          bounds (e.g. `"TIME//1h-2h": [60, 120]`), used to estimate elapsed time
          from generated tokens.
- **engine**:
    - **mem_fraction**: Fraction of GPU memory reserved for the inference engine.
    - **patient_chunk_size**: Number of patients generated per batch.
- **prompt_overflow**: How to handle `tokens_past` longer than `score.max_len` —
  `"drop"` skips the patient, `"truncate_left"` keeps the last `max_len` tokens.

## Outputs

- `scores-generative-<model_name>.parquet` — held-out scores, with a
  `<TOKEN>_mc_score`, `<TOKEN>_scope_score`, and `<TOKEN>_reach_score` column for
  each target token.

[^1]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)
