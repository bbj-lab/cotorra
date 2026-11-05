# Generative scoring

**Generative scoring** (`cotorra generative-score`) Monte Carlo samples future
trajectories directly from the model. It:

1. Loads the trained model and held-out inference data.
2. Samples future trajectories for each target token.
3. Computes MC, SCOPE, and REACH scores [^1] per target token.

Note this depends on the
[quick-sco-re](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git)
package and on SGLang, which cotorra's `gen` extra installs, on Linux with an
NVIDIA GPU; see the MIMIC-IV demo's
[generative step](../../recipes/mimic-iv-demo.md#6-predict-outcomes-generatively-linux-with-an-nvidia-gpu).

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
  [quick-sco-re](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git))_:
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
