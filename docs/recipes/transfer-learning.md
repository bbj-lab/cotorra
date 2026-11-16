# Transfer learning

This recipe trains a model on one dataset and then applies it — extracting
representations and scoring outcomes — on other datasets. The trick that makes
this work is a _shared vocabulary_: we learn a single tokenizer on the first
dataset and apply it to all of them, so every dataset speaks the same token
language and a model trained on one can be run over the rest.

0. Point at a directory of your configs: copies of cocoa's shipped
   `collation.yaml`, `tokenization.yaml`, and `winnowing.yaml`, and of cotorra's
   `training.yaml`, `extraction.yaml`, and `scoring.yaml`, edited as you need.
   Then name the dataset to train on, and the datasets to transfer to:

    ```sh
    config_home=config
    train_dset=mimic-icu
    transfer_dsets=(eicu hirid)
    ```

1. Collate each dataset's raw data into the processed layout (in parallel):

    ```sh
    parallel --bar cocoa collate \
        --collation-config ${config_home}/collation.yaml \
        --raw-data-home data-raw/{} \
        --processed-data-home processed/{} \
        ::: "$train_dset" "${transfer_dsets[@]}"
    ```

2. Learn the tokenizer on the first dataset, then apply that same tokenizer to
   the others. Reusing one tokenizer is what gives every dataset a common
   vocabulary. Its `tokenizer.yaml` carries the config it was learned with, which
   `--tokenizer-home` applies as written, so the second command takes no
   `--tokenization-config`:

    ```sh
    # learn tokenizer on first dataset
    cocoa tokenize \
        --tokenization-config ${config_home}/tokenization.yaml \
        --processed-data-home ./processed/${train_dset}

    # apply tokenizer to other datasets
    parallel --bar cocoa tokenize \
        --tokenizer-home ./processed/${train_dset}/tokenizer.yaml \
        --processed-data-home ./processed/{} \
        ::: "${transfer_dsets[@]}"
    ```

3. Winnow every dataset to produce the split-specific inference tables
   (`train_for_inference.parquet`, etc.) needed for extraction and scoring:

    ```sh
    parallel --bar cocoa winnow \
        --winnowing-config ${config_home}/winnowing.yaml \
        --processed-data-home ./processed/{} \
        ::: "$train_dset" "${transfer_dsets[@]}"
    ```

4. Train a model on the first dataset only:

    ```sh
    cotorra train \
        --training-config ${config_home}/training.yaml \
        --processed-data-home ./processed/${train_dset} \
        --output-home ./output/${train_dset} \
        --verbose
    ```

    The trained model is saved to `./output/${train_dset}/mdl-<run_name>/`,
    beside the config it trained with, `mdl-<run_name>-training.yaml`. Capture
    the model's path so the transfer steps can point at it (the trailing `/`
    matches only the directory):

    ```sh
    model_home=$(ls -d ./output/${train_dset}/mdl-*/)
    ```

5. Transfer: apply the model trained on the first dataset to every other dataset,
   extracting hidden-state representations for each split:

    ```sh
    parallel --bar cotorra extract \
        --extraction-config ${config_home}/extraction.yaml \
        --processed-data-home ./processed/{} \
        --model-home ${model_home} \
        ::: "${transfer_dsets[@]}"
    ```

6. Fit a lightweight classifier on those extracted features and score held-out
   outcomes on each transfer dataset:

    ```sh
    parallel --bar cotorra rep-based-score \
        --scoring-config ${config_home}/scoring.yaml \
        --processed-data-home ./processed/{} \
        --model-home ${model_home} \
        --verbose \
        ::: "${transfer_dsets[@]}"
    ```

    This writes `scores-rep-based-<model_name>.parquet` under each transfer
    dataset's `processed` directory, letting you compare how well the model
    trained on `$train_dset` transfers to each of the others.

<!-- prettier-ignore-start -->
> [!TIP]
> Extraction and scoring find the trained model by `--model-home`, so the
> transfer targets never need their own trained model — only the shared
> tokenizer from step 2 and the winnowed inference tables from step 3.
<!-- prettier-ignore-end -->
