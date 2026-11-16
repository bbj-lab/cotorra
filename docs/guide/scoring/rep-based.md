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

A model trained on one dataset, A, can score another, B, as long as B is
tokenized with A's tokenizer (as in the
[transfer-learning recipe](../../recipes/transfer-learning.md)), so that its
tokens mean what the model learned. Either way, A's model extracts B's features;
what differs is whose labels the estimator is fit on:

1. **Fit on B.** The estimator learns from B's train split and labels, and scores
   B's held-out split:

    ```sh
    cotorra extract -p processed/B -m output/A/mdl-<run_name>
    cotorra rep-based-score -p processed/B -m output/A/mdl-<run_name>
    ```

2. **Fit on A.** The estimator learns from A's train split and labels, named by
   `--training-home`, and scores B's held-out split. A's features have to be
   extracted with the same model:

    ```sh
    cotorra extract -p processed/A -m output/A/mdl-<run_name>
    cotorra extract -p processed/B -m output/A/mdl-<run_name>
    cotorra rep-based-score -p processed/B -m output/A/mdl-<run_name> \
        --training-home processed/A
    ```

## Outputs

- `scores-rep-based-<model_name>.parquet` — held-out scores, with a
  `<TOKEN>_rep_score` column for each target token.
