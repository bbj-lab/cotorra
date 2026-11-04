# Scoring

Scoring uses a trained model to produce outcome scores for the tokens of
interest. Three complementary approaches are provided, and
[baselines](#baseline-scoring) that need no trained model to set them against:

**Generative scoring** (`cotorra generative-score`) Monte Carlo samples future
trajectories directly from the model. It:

1. Loads the trained model and held-out inference data.
2. Samples future trajectories for each target token.
3. Computes MC, SCOPE, and REACH scores [^4] per target token.

Note this depends on the
[quick-sco-re](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git)
package.

**Representation-based scoring** (`cotorra rep-based-score`) fits a lightweight
estimator on extracted features (run `cotorra extract` first). It:

1. Loads the extracted features and label columns.
2. Fits the chosen estimator on the training split.
3. Predicts outcome probabilities for the held-out split.

We support two types of model transfer for rep-based scoring:

1. You can apply a model trained on one dataset to a second dataset to obtain
   train, tuning, and held-out features, and then pair these features with labels
   from the second dataset to train supervised classifiers that can be applied to
   the held-out set of the second set.

2. You can also apply a model trained on one dataset to obtain train and tuning
   features for that first dataset and learn supervised classifiers using the
   labels from this first dataset. You can then apply the model to a second
   dataset to extract features and then apply the supervised classifier to the
   held-out features of this second set. For this second type of transfer, use
   the `--training-home` option offered by this command.

**Scoring via supervised finetuning** (`cotorra train`, then `cotorra extract`)
trains the model to predict an outcome directly, through a head beside its
language-modelling head, and reads the scores off that head: no estimator is fit
afterward, and no trajectories are sampled. It needs no scoring command, nor
`scoring.yaml`:

1. **Train with a supervised head.** Add a `disposition_objective` block to the
   training config to predict how each record ends (e.g. the chance of
   `DSCG//expired`), or a `tte_objective` block to predict the hours until its
   end time, and train as usual; see
   [Training secondary heads](secondary-heads.md). The heads train jointly with
   next-token prediction from the first step: `cotorra train` always builds a
   fresh model, so it can't add a head to a model already trained without one.

2. **Extract the heads' predictions** for every split's inference contexts:

    ```sh
    cotorra extract -p processed/ -m output/mdl-<run_name> -d -t
    ```

    `-d` (`--discharge-disposition`) adds a `<class>_prob` column for each
    disposition class (`DSCG//expired_prob`, say), and `-t` (`--time-to-event`) a
    `time_to_event` column in hours; see [Extraction](extraction.md). Both are
    read at a context's final token, so set `extract.max_len` to cover the
    longest `tokens_past`, or a longer context is scored where it was cut off,
    short of the prediction point.

3. **Evaluate on the held-out split.** `features-held_out-mdl-<run_name>.parquet`
   has a row per row of `held_out_for_inference.parquet`, in its order, so its
   scores line up with that table's label columns. For instance, to evaluate
   `DSCG//expired_prob` as the other scorers' `--verbose` summaries do, over the
   contexts in which the patient hasn't already expired:

    ```python
    import polars as pl

    from cotorra.util import bootstrap_ci

    labels = pl.read_parquet("processed/held_out_for_inference.parquet")
    scores = pl.read_parquet("processed/features-held_out-mdl-<run_name>.parquet")
    qual = ~labels["DSCG//expired_past"]
    bootstrap_ci(
        labels["DSCG//expired_future"].filter(qual).to_numpy(),
        scores["DSCG//expired_prob"].filter(qual).to_numpy(),
    )  # 95% intervals for roc_auc, pr_auc, and brier
    ```

Generative, representation-based, and baseline scoring are all driven by a YAML
config (the package ships a default; see
[`./src/cotorra/config/scoring.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/scoring.yaml))
that specifies:

- **run_name**: Name for the current run, used to label output files.
- **tokens_of_interest**: List of token-based outcomes of interest. Supports
  patterns specified with fnmatch. (Referenced by target tokens.)
- **score**:
    - **max_len**: Maximum input length (tokens) during scoring.
    - **n_samp**: Number of Monte Carlo samples per input per trajectory type.
    - **target_tokens**: Token-based outcomes of interest to score. Supports
      patterns specified with fnmatch. Every matching vocabulary token becomes an
      outcome that `generative-score` samples trajectories for.
    - **end_tokens**: Tokens that naturally terminate a generated sequence (e.g.
      `EOS`).
    - **suppressed_tokens**: Tokens to suppress via logit bias during generation
      (e.g. `PAD`).
    - **batch_size**: Batch size for inference.
- **generation** _(generative-score only; modeled on the `pipeline_config.yaml`
  used by
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
- **engine** _(generative-score only)_:
    - **mem_fraction**: Fraction of GPU memory reserved for the inference engine.
    - **patient_chunk_size**: Number of patients generated per batch.
- **prompt_overflow** _(generative-score only)_: How to handle `tokens_past`
  longer than `score.max_len` — `"drop"` skips the patient, `"truncate_left"`
  keeps the last `max_len` tokens.

## Baseline scoring

**Baseline scoring** (`cotorra baseline-score`) scores the same outcomes with no
trained model, for the scores above to be set against. It fits an estimator on
features of each timeline's past alone, over the train split's timelines whose
past doesn't already hold the outcome (as `rep-based-score` does), and scores the
held-out split's. `--baseline` picks the features:

- `counts` or `one-hot`: a bag of the tokens in each timeline's `tokens_past`,
  how many times each token of the vocabulary appears there or whether it does at
  all, keeping none of their order or timing.
- `tsfresh`: [tsfresh](https://tsfresh.readthedocs.io)'s minimal features of the
  timeline's rows of cocoa's `meds.parquet` up to the time of the last token in
  its `tokens_past` (where the threshold is a time, the rows that past was
  tokenized from). For each code with a value in the train split's pasts, these
  are ten statistics of its values (their count, sum, mean, median, standard
  deviation, variance, root mean square, maximum, largest magnitude and minimum);
  for every other code (an order, a device, a transfer, a demographic, an
  assessment's text result, fused to its code as cocoa's tokenizer fuses them),
  how many times it comes up. A code a past lacks is missing.

`--estimator` picks LightGBM (the default) or XGBoost, set as for
`rep-based-score`, or logistic regression, which scales each feature by its
largest magnitude in the training rows (filling a missing tsfresh feature with 0
beside an indicator of its absence): fit on unscaled counts, which run into the
hundreds, it does far worse. `--training-home` fits on another directory's train
and tuning splits, as for `rep-based-score`; for a bag of tokens, both
directories must be tokenized by one tokenizer (as in the
[transfer-learning recipe](../recipes/transfer-learning.md)), since a column
stands for a token id.

tsfresh extracts its features in `--n-jobs` processes, half the CPUs by default.
Where Python spawns them (macOS, and Linux from Python 3.14), each imports
cotorra again as it starts, which takes a few seconds, so on a small dataset
`--n-jobs 0`, extracting in the one process, can finish sooner. From Python
(`TsfreshScorer(n_jobs=...)`), they import the calling script again too, which so
has to keep its work under `if __name__ == "__main__":`.

## Outputs

- `scores-generative-<model_name>.parquet` — held-out scores from
  `generative-score`, with a `<TOKEN>_mc_score`, `<TOKEN>_scope_score`, and
  `<TOKEN>_reach_score` column for each target token.
- `scores-rep-based-<model_name>.parquet` — held-out scores from
  `rep-based-score`, with a `<TOKEN>_rep_score` column for each target token.
- `scores-bag-of-tokens-<featurization>-<estimator>.parquet` — held-out scores
  from `baseline-score --baseline counts` or `one-hot`, with a
  `<TOKEN>_counts_score` or `<TOKEN>_one-hot_score` column for each target token.
- `scores-tsfresh-<estimator>.parquet` — held-out scores from
  `baseline-score --baseline tsfresh`, with a `<TOKEN>_tsfresh_score` column for
  each target token.

[^4]:
    L. Solo, M. McDermott, W. Parker, B. Ramadan, M. Burkhart, & B.
    Beaulieu-Jones, Efficient generative prediction for EHR foundation models:
    the SCOPE and REACH estimators,
    [arXiv:2602.03730](https://doi.org/10.48550/arXiv.2602.03730)
