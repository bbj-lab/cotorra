---
hide:
    - navigation
---

--8<-- "README.md:intro"

<div class="grid cards" markdown>

- :material-school:{ .lg .middle } **Train**

    ***

    Fit a causal LM to predict the next token in each subject's timeline —
    optionally with custom losses, time-aware RoPE, or hyperparameter tuning.

    [:octicons-arrow-right-24: Training](guide/training.md)

- :material-table-arrow-right:{ .lg .middle } **Extract**

    ***

    Run a trained model over inference contexts and write hidden-state feature
    tables for downstream representation-based scoring.

    [:octicons-arrow-right-24: Extraction](guide/extraction.md)

- :material-target:{ .lg .middle } **Score**

    ***

    Produce outcome scores for tokens of interest — Monte-Carlo generative
    scoring (MC / SCOPE / REACH) or a lightweight estimator on extracted
    features.

    [:octicons-arrow-right-24: Scoring](guide/scoring.md)

- :material-book-open-variant:{ .lg .middle } **Recipes**

    ***

    Task-oriented guides for common workflows, from an end-to-end run on the
    public MIMIC-IV demo to the development workflow and transfer learning.

    [:octicons-arrow-right-24: Recipes](recipes/index.md)

</div>

--8<-- "README.md:install"

--8<-- "README.md:references"
