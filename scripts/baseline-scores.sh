#!/usr/bin/env bash
# fits every baseline scorer (counts, one-hot, tsfresh) with every estimator
# (lightGBM, logistic, XGBoost) on a processed-data home, processed/mimic-icu by
# default, then tabulates their held-out AUROCs; on bbj-lab1, in tmux:
#   tmux new -s co || tmux a -t co
#   bash ~/cotorra/scripts/baseline-scores.sh [processed-data-home]

set -uo pipefail

cd "$(dirname "$0")/.."
data=${1:-processed/mimic-icu}
out=output/baselines/$(basename "$data")

# micromamba's base env holds cotorra 26.6.1 and transformers 4.45, too old for
# this tree, so it gets a venv of its own, which rsync's --exclude ".venv"
# keeps; rerun the pip install after pyproject.toml raises a floor
if [[ ! -x .venv/bin/cotorra ]]; then
    python3 -m venv .venv
    .venv/bin/pip install -e . \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple || exit 1
fi
. .venv/bin/activate

baselines=(counts one-hot tsfresh)
if [[ ! -f $data/meds.parquet ]]; then
    echo "no meds.parquet in $data: skipping tsfresh" >&2
    baselines=(counts one-hot)
fi
estimators=(lightGBM logistic XGBoost)

# the outcomes every split's inference table labels, a <token>_past and
# <token>_future each: scoring.yaml's patterns also match tokens of the vocab
# that winnow labeled no columns for (mimic-icu's ASMT//cam_total_yes), and a
# scorer fails on the first of those it reaches; of those, the ones whose train
# rows to score (their past lacking it) hold both classes, as logistic
# regression and XGBoost refuse one (mimic-icu's XFR-IN//icu, which every such
# row reaches)
targets=$(
    python - "$data" <<'EOF'
import json
import pathlib
import sys

import polars as pl
from omegaconf import OmegaConf

home = pathlib.Path(sys.argv[1])
vocab = OmegaConf.load(home / "tokenizer.yaml").lookup.keys()
cols = set.intersection(
    *(
        set(pl.scan_parquet(home / f"{s}_for_inference.parquet").collect_schema())
        for s in ("train", "tuning", "held_out")
    )
)
labeled = [t for t in vocab if {f"{t}_past", f"{t}_future"} <= cols]
n_classes = (
    pl.scan_parquet(home / "train_for_inference.parquet")
    .select(
        pl.col(f"{t}_future").filter(~pl.col(f"{t}_past")).n_unique().alias(t)
        for t in labeled
    )
    .collect()
    .row(0, named=True)
)
if skipped := [t for t in labeled if n_classes[t] < 2]:
    print(f"skipping, one class in train: {skipped}", file=sys.stderr)
print(json.dumps([t for t in labeled if n_classes[t] == 2]))
EOF
) || exit 1
echo "outcomes: $targets"

mkdir -p "$out"
failed=()
for b in "${baselines[@]}"; do
    for e in "${estimators[@]}"; do
        echo "=== $b × $e ==="
        cotorra baseline-score -p "$data" -o "$out" -b "$b" -e "$e" --verbose \
            "score.target_tokens=$targets" 2>&1 |
            tee "$out/$b-$e.log" || failed+=("$b-$e")
    done
done

python - "$out" <<'EOF'
"""held-out AUROC of each run's scores, over the rows whose past lacks the token"""

import pathlib
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

out = pathlib.Path(sys.argv[1])
aucs = {}
for f in sorted(out.glob("scores-*.parquet"), key=lambda f: f.stat().st_mtime):
    run = f.stem.removeprefix("scores-").removeprefix("bag-of-tokens-")
    lf = pl.scan_parquet(f)
    for col in lf.collect_schema().names():
        if col.endswith("_score"):
            tt = col.rsplit("_", 2)[0]  # <token>_<method>_score
            y, s = (
                lf.filter(~pl.col(f"{tt}_past"))
                .select(f"{tt}_future", col)
                .collect()
                .to_numpy()
                .T
            )
            aucs.setdefault(tt, {})[run] = (
                roc_auc_score(y, np.nan_to_num(s)) if 0 < y.mean() < 1 else np.nan
            )

table = pl.DataFrame([{"outcome": tt, **runs} for tt, runs in aucs.items()])
table.write_csv(out / "auroc.csv")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=400, float_precision=3):
    print(table)
EOF

echo "scores, logs, and auroc.csv: $PWD/$out"
if ((${#failed[@]})); then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
