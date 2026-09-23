## Transfer learning

This recipe trains a model on one dataset and then applies it — extracting
representations and scoring outcomes — on other datasets. The trick that makes
this work is a _shared vocabulary_: we learn a single tokenizer on the first
dataset and apply it to all of them, so every dataset speaks the same token
language and a model trained on one can be run over the rest.

0. Point at your configs and list the datasets to work with. The first entry
   (`dsets[0]`) is the one we train on; the rest are the transfer targets.

    ```sh
    config_home=config
    dsets=(mimic-icu eicu hirid)
    ```

1. Collate each dataset's raw data into the processed layout (in parallel):

    ```sh
    parallel --bar cocoa collate \
        --collation-config ${config_home}/collation.yaml \
        --raw-data-home data-raw/{} \
        --processed-data-home processed/{} \
        ::: "${dsets[@]}"
    ```

2. Learn the tokenizer on the first dataset, then apply that same tokenizer to
   the others. Reusing one tokenizer is what gives every dataset a common
   vocabulary:

    ```sh
    # learn tokenizer on first dataset
    cocoa tokenize \
        --tokenization-config ${config_home}/tokenization.yaml \
        --processed-data-home ./processed/${dsets[0]}

    # apply tokenizer to other datasets
    parallel --bar cocoa tokenize \
        --tokenization-config ${config_home}/tokenization.yaml \
        --tokenizer-home ./processed/${dsets[0]}/tokenizer.yaml \
        --processed-data-home ./processed/{} \
        ::: "${dsets[@]:1}"
    ```

3. Winnow every dataset to produce the split-specific inference tables
   (`train_for_inference.parquet`, etc.) needed for extraction and scoring:

    ```sh
    parallel --bar cocoa winnow \
        --winnowing-config ${config_home}/winnowing.yaml \
        --processed-data-home ./processed/{} \
        ::: "${dsets[@]}"
    ```

4. Train a model on the first dataset only:

    ```sh
    cotorra train \
        --training-config ${config_home}/training.yaml \
        --processed-data-home ./processed/${dsets[0]} \
        --output-home ./output/${dsets[0]} \
        --verbose
    ```

    The trained model is saved to `./output/${dsets[0]}/mdl-<run_name>`. Capture
    that path so the transfer steps can point at it:

    ```sh
    model_home=$(ls -d ./output/${dsets[0]}/mdl-*)
    ```

5. Transfer: apply the model trained on the first dataset to every other dataset,
   extracting hidden-state representations for each split:

    ```sh
    parallel --bar cotorra extract \
        --extraction-config ${config_home}/extraction.yaml \
        --processed-data-home ./processed/{} \
        --model-home ${model_home} \
        ::: "${dsets[@]:1}"
    ```

6. Fit a lightweight classifier on those extracted features and score held-out
   outcomes on each transfer dataset:

    ```sh
    parallel --bar cotorra rep-based-score \
        --scoring-config ${config_home}/scoring.yaml \
        --processed-data-home ./processed/{} \
        --model-home ${model_home} \
        --verbose \
        ::: "${dsets[@]:1}"
    ```

    This writes `scores-rep-based-<model_name>.parquet` under each transfer
    dataset's `processed` directory, letting you compare how well the model
    learned on `dsets[0]` transfers to each of the others.

<!-- prettier-ignore-start -->
> [!TIP]
> Extraction and scoring find the trained model by `--model-home`, so the
> transfer targets never need their own trained model — only the shared
> tokenizer from step 2 and the winnowed inference tables from step 3.
<!-- prettier-ignore-end -->
