## Development workflow

1. We start with [MIMIC data](https://mimic.mit.edu) that's been converted to the
   [CLIF standard](https://clif-icu.com):
   https://physionet.org/content/mimic-iv-ext-clif. We first collate and tokenize
   it with the [cocoa package](https://github.com/bbj-lab/cocoa).

    ```sh
    cocoa pipeline \
        --raw-data-home /path/to/raw \
        --processed-data-home processed/dev \
        --verbose
    ```

2. Next we train a model on this data (with hyperparameter tuning):

    ```sh
    cotorra tune \
        --processed-data-home processed/dev \
        --output-home output/dev/ \
        --verbose
    ```

3. You can get generative predictions with:

    ```sh
    cotorra generative-score \
        --processed-data-home processed/dev \
        --model-home output/dev/mdl-cotorra-tuning \
        --verbose
    ```

4. You can get representations of the initial parts of the sequences and
   rep-based predictions with:

    ```sh
    cotorra extract \
        --processed-data-home processed/dev \
        --model-home output/dev/mdl-cotorra-tuning

    cotorra rep-based-score \
        --processed-data-home processed/dev \
        --model-home output/dev/mdl-cotorra-tuning \
        --verbose
    ```

<!-- prettier-ignore-start -->
> [!TIP]
> For this example, we used a small fraction of the whole dataset, allowing commands to
> complete in a timely manner. For serious use cases, consider using a terminal
> multiplexer like [tmux](https://github.com/tmux/tmux/wiki) or
> [screen](https://www.gnu.org/software/screen/) so that commands  will continue to
> run if your connection is interrupted.
<!-- prettier-ignore-end -->
