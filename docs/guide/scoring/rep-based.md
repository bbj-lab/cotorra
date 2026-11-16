# Representation-based scoring

<!-- prettier-ignore-start -->
> [!EXAMPLE] `cotorra rep-based-score`
>
> Reads the features that [`cotorra extract`](../extraction.md) wrote to the
> processed data directory `-p` for the model `-m`, and writes
> `scores-rep-based-<model_name>.parquet` there, or to `-o` if given (see the
> [CLI reference](../../api/cli.md#cotorra-rep-based-score) for every option):
>
> ```sh
> cotorra extract -p processed/ -m output/mdl-<run_name>
> cotorra rep-based-score -p processed/ -m output/mdl-<run_name>
> ```
<!-- prettier-ignore-end -->

**Representation-based scoring** (`cotorra rep-based-score`) fits a lightweight
estimator on extracted features (run [`cotorra extract`](../extraction.md)
first). It:

1. Loads the extracted features and label columns.
2. Fits the chosen estimator on the training split.
3. Predicts outcome probabilities for the held-out split.

`--estimator` picks the estimator, LightGBM by default; see
[`EstimatorType`](../../api/scoring.md#estimatortype) for the others. Of
`scoring.yaml`, it reads only the outcomes to score (see
[Configuration](index.md#configuration)).

## Transfer

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

## Outputs

- `scores-rep-based-<model_name>.parquet` — held-out scores, with a
  `<TOKEN>_rep_score` column for each target token.
