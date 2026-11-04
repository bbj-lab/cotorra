# Training secondary heads

Every model learns to predict the next token. Alongside that, it can train any
combination of three secondary heads, each with its own config block:

| block                   | head                     | predicts, at each token           | target                          |
| ----------------------- | ------------------------ | --------------------------------- | ------------------------------- |
| `tte_objective`         | time to event (TTE)      | hours until the record's end time | `hours_to_end_time`, from cocoa |
| `tnt_objective`         | time to next token (TNT) | hours until the next token        | derived from `times`            |
| `disposition_objective` | discharge disposition    | how the record ends               | the record's `DSCG//*` token    |

With any of them set, training writes a `cotorra` model: the selected preset plus
those heads, each trained jointly with next-token prediction.

The TNT head turns the model into a marked point process: the language-modelling
head predicts _what_ the next token is, and the TNT head _when_ it arrives. It
comes in two kinds:

- **A point head**, the default, predicts a single value at each token: the
  log1p-hours until the next token, trained by squared error. It predicts the
  same gap whatever token comes next, so it can't time a token sampled from the
  model.
- **A mixture head**, with `mixture_components` set, predicts a distribution over
  the gap instead, given which token comes next. That is what sampling realistic
  times needs.

The mixture head's distribution has two parts:

- **The gap is exactly 0** with some probability: the next token shares this
  one's timestamp, as the tokens of a single event, or of results recorded at
  once, do. A continuous distribution can't put any probability on exactly 0.
- **Otherwise the gap is log-normal**, its log-hours a mixture of Gaussians, each
  with its own weight, mean and width, so that the gap can have several typical
  scales at once. `mixture_components` sets how many (at least 1); the examples
  here use 8.

The head reads the hidden state at each token together with the embedding of the
token after it, so the gap can depend on what arrives: a quantile token
completing an event comes at once, a new event later. It's trained by the
negative log-likelihood of the gap that came, the positive part's density taken
on the log-hours scale, where the mixture lives. With `custom_loss: true` the
term is logged as `tnt_loss` (`eval_tnt_loss` in evaluation). It isn't comparable
with a point head's squared error, and it can go below 0. Besides each record's
last token, the mixture head leaves out each row's last token, since the token
after it lies in the next row.

Choose the mixture head to sample timelines with their times, or to score how
likely a gap is. For a single predicted gap, either kind works:
`cotorra extract --time-to-next-token` writes the same quantity for both, the
`expm1` of the mean log1p-hours until the next token, given the history alone.
That's a typical gap rather than the average one, which a few long waits pull
higher. A mixture head gets it by averaging over every possible next token,
weighted by the model's own next-token probabilities, so it costs more to extract
(see [Extraction](extraction.md)).

The disposition head is supervised. At each token it predicts how the record will
end, as a probability for each of its `classes`: for example, the chance that the
patient dies before discharge (`DSCG//expired`). Next-token prediction gives a
disposition's probability only where it's the very next token; the head gives one
at every token, from the first.

- **Its target** is the record's own disposition token, which cocoa's default
  config writes at the discharge time, from the hospitalization's
  `discharge_category`, among the record's last tokens. Every token before it is
  scored against that disposition, by cross-entropy, and the tokens from it on
  aren't, since by then the disposition is known. A record holding more than one
  token from `classes` is scored against the last.
- **Its classes**, token labels or fnmatch patterns, are resolved against the
  vocabulary when training starts, and training stops with an error if they match
  no token at all. The model keeps them in token-id order, whatever order the
  config lists them in, as `model.heads["disposition"].classes`, and its
  predictions follow that order.
- **A record whose disposition matches none of `classes` isn't scored at all**,
  rather than counted under some other class. Narrowing the classes narrows the
  question: with `classes: ["DSCG//expired", "DSCG//home"]`, the head learns only
  from records ending in one of the two, so its `DSCG//expired` probability is
  the chance of death among those records, not among all of them. `DSCG//*` keeps
  every record, `DSCG//missing` included; to leave that one out, list the classes
  you want instead.

To train one or more:

1. **Configure training.** A config passed with `--training-config` replaces the
   shipped default rather than merging into it, so start from a full copy of
   [training.yaml](https://github.com/burkh4rt/cotorra/blob/master/src/cotorra/config/training.yaml)
   and add (or uncomment) the blocks for the heads you want:

    ```yaml
    tte_objective:
        weight: !!float 1.0
        per_record: !!bool false # optional: true weights each record alike
    tnt_objective:
        weight: !!float 1.0
        mixture_components: !!int 8 # optional: a distribution over the gap
    disposition_objective:
        weight: !!float 1.0
        classes: ["DSCG//*"] # optional: fnmatch patterns
        per_record: !!bool false # optional: true weights each record alike
    ```

    Each weight defaults to `1.0`, so empty blocks work too. With
    `custom_loss: true` (as shipped), the weighted terms are added to the custom
    next-token loss; otherwise the model adds them to HuggingFace's standard loss
    itself, each term in training counting its batch's share of the step's
    tokens, as HuggingFace's own does, so that a weight means the same whatever
    `gradient_accumulation_steps` and however many processes train. (Through
    v26.10.1, training multiplied those weights by both.)

2. **Tokenize with end times (TTE head only).** The TTE head trains on an
   `hours_to_end_time` column, which cocoa writes only when asked. Copy cocoa's
   [default tokenization config](https://github.com/bbj-lab/cocoa/blob/master/src/cocoa/config/tokenization.yaml),
   set `include_hours_to_end_time: !!bool true`, and rerun tokenization and
   winnowing. Winnowing carries the column into the `*_for_inference.parquet`
   tables as `hours_to_end_time_past`, which cotorra never reads: it is the
   outcome itself, so it reaches neither the model nor the feature tables, and
   `extract --time-to-event` needs no end times.

    ```sh
    cocoa tokenize -c tokenization.yaml -p processed/
    cocoa winnow -p processed/
    ```

    Skip this step for the other heads: cotorra derives their targets from
    `tokens_times.parquet`.

3. **Train** as usual:

    ```sh
    cotorra train -t training.yaml -p processed/ -o output/
    ```

    The startup log names the heads, e.g.
    `<model_name> (with tte, tnt, disposition heads)`, and the run writes
    `mdl-<run_name>/` as a `cotorra` model.

4. **Use the trained model.** Importing `cotorra.model` registers the `cotorra`
   model type, so `AutoModelForCausalLM.from_pretrained` loads it. The time heads
   predict non-negative values on the log1p-hours scale, so `expm1` converts them
   back to hours, never fewer than 0; the disposition head gives logits over its
   classes:

    ```python
    import torch as t
    from transformers import AutoModelForCausalLM

    import cotorra.model  # noqa: F401 -- registers the `cotorra` model type

    model = AutoModelForCausalLM.from_pretrained("output/mdl-<run_name>").eval()
    input_ids = t.tensor([[1, 5, 7, 9]])
    s_elapsed = t.tensor([[0.0, 0.0, 600.0, 3600.0]])  # seconds since start
    sec_per_pos_id = 300  # as set under `time_based_rope` for training
    position_ids = s_elapsed / sec_per_pos_id + t.arange(input_ids.shape[-1])

    with t.inference_mode():
        out = model(input_ids=input_ids, position_ids=position_ids)
    next_token_logits = out.logits[:, -1]
    if "tte" in model.heads:
        hours_to_end_time = out.tte_pred.expm1()  # (batch, seq_len)
    if "tnt" in model.heads:  # a point head; a mixture head leaves it None
        hours_to_next_token = out.tnt_pred.expm1()  # (batch, seq_len)
    if "disposition" in model.heads:
        classes = model.heads["disposition"].classes  # ["DSCG//expired", ...]
        p_disposition = out.disposition_pred.softmax(-1)  # (batch, seq_len, n)
        p_expired = p_disposition[..., classes.index("DSCG//expired")]
    ```

    Pass `position_ids` only if the model was trained with `time_based_rope`,
    built the way training builds them (above). Each head's prediction at
    position `i` is made having read tokens up to `i`: the TTE head's is the
    hours from token `i` to the end of the record, the TNT head's the wait from
    token `i` to token `i + 1`, and the disposition head's the chance of each way
    the record could end. Read at a prompt's last token (an inference table's
    `tokens_past`), they condition on everything so far.

    A mixture TNT head leaves `out.tnt_pred` empty, since its prediction depends
    on which token comes next; `model.tnt_distribution(hidden_states, next_ids)`
    gives the distribution, with `sample()`, `log_prob(hours)`,
    `point_estimate()` and `mean_log1p_hours()`. For a prediction from the
    history alone, which is what `cotorra extract --time-to-next-token` writes,
    `model.predict_hours_to_next_token(hidden_states)` averages the mean
    log1p-hours given each possible next token over the model's own next-token
    probabilities, then converts it to hours; for a point head it gives
    `out.tnt_pred.float().expm1()`. Both methods take the last hidden states,
    `out.hidden_states[-1]` from a forward pass with `output_hidden_states=True`.

Of the other stages, `cotorra extract` can add the heads' predictions to the
feature tables it writes (see [Extraction](extraction.md)); `rep-based-score`
fits on the features alone, and `generative-score` hasn't been tested with a
`cotorra` model.
