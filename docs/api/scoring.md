# Scoring

Scoring turns a trained model into per-timeline outcome predictions. Cotorra
offers two complementary approaches, both driven by a `scoring.yaml`
configuration and both operating on the held-out split: generation-based scoring,
which lets the model simulate the future directly, and representation-based
scoring, which fits a classifier on extracted features. In each case the
`target_tokens` glob patterns in the configuration select which vocabulary tokens
count as outcomes of interest.

## `GenerativeScorer`

`GenerativeScorer` predicts outcomes by _generating_ them. Using the
[`quick_sco_re`](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git)
implementation of the SCORE and REACH algorithms, it autoregressively samples
many possible continuations of each timeline and estimates the probability that a
target outcome token occurs. For every outcome it reports three Monte-Carlo
estimates — a raw occurrence score (`mc`), a SCOPE score, and a REACH score —
computed only for subjects who have not already experienced the outcome.
Generation runs asynchronously in batches and the scores are written to a parquet
file.

::: cotorra.scorer_generative.GenerativeScorer

## `RepBasedScorer`

`RepBasedScorer` predicts outcomes from the representations dumped by the
[`Extractor`](extraction.md#extractor). It loads the extracted feature vectors
for the train, tuning, and held-out splits, fits a supervised classifier per
outcome token to predict whether that outcome occurs, and writes the held-out
predicted probabilities to a parquet file. The classifier family is chosen with
[`EstimatorType`](#estimatortype); it errors with a helpful message if the
features are missing, prompting you to run `cotorra extract` first.

::: cotorra.scorer_rep_based.RepBasedScorer

## `EstimatorType`

`EstimatorType` enumerates the classifier families available to
[`RepBasedScorer`](#repbasedscorer): k-nearest-neighbors, LightGBM (the default),
XGBoost, and several logistic-regression variants (plain, standardized/z-scored,
and cross-validated). It exists so the choice of estimator can be passed as a
plain string on the CLI or in configuration.

::: cotorra.scorer_rep_based.EstimatorType
