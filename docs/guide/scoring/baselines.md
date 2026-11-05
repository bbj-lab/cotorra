# Baseline scoring

**Baseline scoring** (`cotorra baseline-score`) scores the same outcomes with no
trained model, for a model's scores to be set against. It fits an estimator on
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
and tuning splits, as for `rep-based-score` (see
[Transfer](rep-based.md#transfer)); for a bag of tokens, both directories must be
tokenized by one tokenizer (as in the
[transfer-learning recipe](../../recipes/transfer-learning.md)), since a column
stands for a token id. Of `scoring.yaml`, it reads only the outcomes to score
(see [Configuration](index.md#configuration)).

tsfresh extracts its features in `--n-jobs` processes, half the CPUs by default.
Where Python spawns them (macOS, and Linux from Python 3.14), each imports
cotorra again as it starts, which takes a few seconds, so on a small dataset
`--n-jobs 0`, extracting in the one process, can finish sooner. From Python
(`TsfreshScorer(n_jobs=...)`), they import the calling script again too, which so
has to keep its work under `if __name__ == "__main__":`.

## Outputs

- `scores-bag-of-tokens-<featurization>-<estimator>.parquet` — held-out scores
  from `--baseline counts` or `one-hot`, with a `<TOKEN>_counts_score` or
  `<TOKEN>_one-hot_score` column for each target token.
- `scores-tsfresh-<estimator>.parquet` — held-out scores from
  `--baseline tsfresh`, with a `<TOKEN>_tsfresh_score` column for each target
  token.
