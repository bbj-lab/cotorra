---
hide:
    - navigation
---

{%
   include-markdown "../README.md"
   heading-offset=0
   end="<!-- cards-anchor -->"
%}

<div class="grid cards" markdown>

- :material-school:{ .lg .middle } **Train**

    ***

    Fit a causal LM to predict the next token in each subject's timeline —
    optionally with custom losses, time-aware RoPE, differential privacy, or
    hyperparameter tuning.

    [:octicons-arrow-right-24: Training](#1-training)

- :material-table-arrow-right:{ .lg .middle } **Extract**

    ***

    Run a trained model over inference contexts and write hidden-state feature
    tables for downstream representation-based scoring.

    [:octicons-arrow-right-24: Extraction](#2-extraction)

- :material-target:{ .lg .middle } **Score**

    ***

    Produce outcome scores for tokens of interest — Monte-Carlo generative
    scoring (MC / SCOPE / REACH) or a lightweight estimator on extracted
    features.

    [:octicons-arrow-right-24: Scoring](#3-scoring)

- :material-book-open-variant:{ .lg .middle } **Recipes**

    ***

    Task-oriented guides for common workflows, from the development workflow to
    transfer learning.

    [:octicons-arrow-right-24: Recipes](recipes/index.md)

</div>

{%
   include-markdown "../README.md"
   heading-offset=0
   start="<!-- cards-anchor -->"
%}
