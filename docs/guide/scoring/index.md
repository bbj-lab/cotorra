# Scoring

Scoring produces a score for each outcome of interest on the held-out split.
Three approaches read it off a trained model, and baselines that need no model
give those scores something to be set against:

| Approach                                          | Run                                                    | Needs                                                       | Scores in                                    |
| ------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------- | -------------------------------------------- |
| [Representation-based](rep-based.md)              | `cotorra extract`, then `rep-based-score`              | a trained model                                             | `scores-rep-based-…`                         |
| [Generative](generative.md)                       | `cotorra generative-score`                             | a trained model, and the `gen` extra (Linux, an NVIDIA GPU) | `scores-generative-…`                        |
| [Supervised finetuning](supervised-finetuning.md) | `cotorra train` with a head, then `extract -d` or `-t` | a model trained with a supervised head                      | `features-held_out-…`                        |
| [Baselines](baselines.md)                         | `cotorra baseline-score`                               | no model (cocoa's `meds.parquet`, for tsfresh)              | `scores-bag-of-tokens-…`, `scores-tsfresh-…` |

- **Representation-based scoring** fits a lightweight estimator on the hidden
  states a trained model extracts.
- **Generative scoring** Monte Carlo samples future trajectories from the model
  and counts how often each outcome comes up.
- **Supervised finetuning** trains a head beside the language-modelling head to
  predict an outcome directly, and reads the scores off it.
- **Baselines** fit the same estimators on features of the timelines alone: a bag
  of their tokens, or tsfresh's statistics of their raw rows.

<figure markdown="span">
  ![Three panels. Representation-based: the model represents the training and tuning timelines, a classifier for each outcome is fit on those representations, and each classifier predicts its outcome from a held-out timeline's representation. Generative: the model samples N continuations of a held-out timeline, and an outcome's score is the fraction of them it appears in. Supervised finetuning: a model with discharge, time-to-event, and next-token heads trains on the training and tuning timelines, then predicts a held-out timeline's discharge and time to event.](../../img/scoring-approaches.svg)
  <figcaption>The three approaches that score with a trained model:
  representation-based (top), generative (middle), and supervised finetuning
  (bottom). Click to enlarge.</figcaption>
</figure>

Each scores an outcome only on the held-out timelines whose past doesn't already
hold it (`<TOKEN>_past` false in `held_out_for_inference.parquet`), against
whether it comes in their future (`<TOKEN>_future`); the representation-based and
baseline estimators are fit on the train split's timelines chosen the same way.
With `--verbose`, each scoring command logs bootstrapped 95% intervals for every
outcome's ROC AUC, PR AUC, and Brier score over those timelines; the
[supervised finetuning](supervised-finetuning.md) page shows how to compute the
same for a head's predictions.

## Configuration

Representation-based, generative, and baseline scoring are all driven by a YAML
config (the package ships a default; see
[`./src/cotorra/config/scoring.yaml`](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/scoring.yaml)),
passed with `--scoring-config`; supervised finetuning reads none. For all three,
it specifies:

- **run_name**: Name for the current run, used to label output files.
- **tokens_of_interest**: List of token-based outcomes of interest. Supports
  patterns specified with fnmatch. (Referenced by target tokens.)
- **score.target_tokens**: Token-based outcomes of interest to score. Supports
  patterns specified with fnmatch. Every matching vocabulary token becomes an
  outcome, which needs a `<TOKEN>_past` and a `<TOKEN>_future` column in the
  inference tables.

The rest of the `score` block, and the `generation`, `engine`, and
`prompt_overflow` keys, are generative scoring's alone; see
[Generative scoring](generative.md#configuration).
