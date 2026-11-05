# Scoring via supervised finetuning

**Scoring via supervised finetuning** (`cotorra train`, then `cotorra extract`)
trains the model to predict an outcome directly, through a head beside its
language-modelling head, and reads the scores off that head: no estimator is fit
afterward, and no trajectories are sampled. It needs no scoring command, nor
`scoring.yaml`:

1. **Train with a supervised head.** Add a `disposition_objective` block to the
   training config to predict how each record ends (e.g. the chance of
   `DSCG//expired`), or a `tte_objective` block to predict the hours until its
   end time, and train as usual; see
   [Training secondary heads](../secondary-heads.md). The heads train jointly
   with next-token prediction from the first step: `cotorra train` always builds
   a fresh model, so it can't add a head to a model already trained without one.

2. **Extract the heads' predictions** for every split's inference contexts:

    ```sh
    cotorra extract -p processed/ -m output/mdl-<run_name> -d -t
    ```

    `-d` (`--discharge-disposition`) adds a `<class>_prob` column for each
    disposition class (`DSCG//expired_prob`, say), and `-t` (`--time-to-event`) a
    `time_to_event` column in hours; see [Extraction](../extraction.md). Both are
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
