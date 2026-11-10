# Scoring

Scoring turns a trained model into per-timeline outcome predictions on the
held-out split. Two approaches have scorers of their own, both driven by a
`scoring.yaml` configuration: representation-based scoring, which fits a
classifier on extracted features, and generation-based scoring, which lets the
model simulate the future directly. In each case the `target_tokens` glob
patterns in the configuration select which vocabulary tokens count as outcomes of
interest. A third, supervised finetuning, has none: a head trained beside the
language-modelling head predicts the outcome, and the
[`Extractor`](extraction.md#extractor) reads its predictions off.

Against those predictions, Cotorra also scores [baselines](#baselinescorer) that
need no trained model: classifiers fit on features of the timelines alone, from
the same configuration. The [scoring guide](../guide/scoring/index.md) sets the
four side by side.

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

## `GenerativeScorer`

`GenerativeScorer` predicts outcomes by _generating_ them. Using the
[`quick_sco_re`](https://pypi.org/project/quick-sco-re/) implementation of the
SCORE and REACH algorithms, it autoregressively samples many possible
continuations of each timeline and estimates the probability that a target
outcome token occurs. For every outcome it reports three Monte-Carlo estimates —
a raw occurrence score (`mc`), a SCOPE score, and a REACH score — computed only
for subjects who have not already experienced the outcome. Generation runs
asynchronously in batches and the scores are written to a parquet file.

::: cotorra.scorer_generative.GenerativeScorer

## `BaselineScorer`

`BaselineScorer` holds what the baselines share, with each other and with
[`RepBasedScorer`](#repbasedscorer): the outcomes `target_tokens` names, the
train split's rows whose past doesn't already hold an outcome to fit on, the
held-out rows to score, the estimators (see
[`BaselineEstimator`](#baselineestimator)) with the tuning split for the boosters
to evaluate on, transfer through `training_home`, and the scores' output. Each
baseline builds its own features, and prepares them for logistic regression.

::: cotorra.baselines.BaselineScorer

## `BagOfTokensScorer`

`BagOfTokensScorer` fits on a bag of the tokens before each timeline's threshold
(its `tokens_past`): how many times each token of the vocabulary appears there,
or whether it does at all (see [`Featurization`](#featurization)), as a sparse
matrix. Its logistic regression scales each token by its largest value in the
training rows. Transfer needs both directories tokenized by one tokenizer, which
it checks.

::: cotorra.baselines.BagOfTokensScorer

## `TsfreshScorer`

`TsfreshScorer` fits on [tsfresh](https://tsfresh.readthedocs.io)'s minimal
features of each timeline's rows of cocoa's `meds.parquet` up to its threshold
(see [`past_meds`](#past_meds)): ten statistics of the values of each code
measured in the train split's pasts, and how many times each other code comes up
(see [`tsfresh_points`](#tsfresh_points)). A code a past lacks is missing: the
boosters take it as missing, and its logistic regression fills it with 0 beside
an indicator of its absence, then scales each feature by its largest magnitude.

::: cotorra.baselines.TsfreshScorer

## `Baseline`

`Baseline` enumerates the baselines `cotorra baseline-score --baseline` offers:
`counts` and `one-hot`, the two featurizations of
[`BagOfTokensScorer`](#bagoftokensscorer), and `tsfresh`, for
[`TsfreshScorer`](#tsfreshscorer). Each is also the method its score columns are
named for, `<TOKEN>_<baseline>_score`.

::: cotorra.baselines.Baseline

## `Featurization`

`Featurization` enumerates the bags of tokens
[`BagOfTokensScorer`](#bagoftokensscorer) fits on: counts, or one-hot indicators.

::: cotorra.baselines.Featurization

## `BaselineEstimator`

`BaselineEstimator` enumerates the estimators the baselines fit: LightGBM (the
default), logistic regression, and XGBoost, under the names
[`EstimatorType`](#estimatortype) gives them.

::: cotorra.baselines.BaselineEstimator

## Feature helpers

The functions the baselines build their features with.

### `bag_of_tokens`

::: cotorra.baselines.bag_of_tokens

### `past_meds`

::: cotorra.baselines.past_meds

### `tsfresh_points`

::: cotorra.baselines.tsfresh_points

### `tsfresh_features`

::: cotorra.baselines.tsfresh_features
