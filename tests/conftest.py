#!/usr/bin/env python3

"""
shared fixtures: a synthetic processed dataset (produced by the real cocoa
collate -> tokenize -> winnow pipeline, cocoa-tokenizer being a hard
dependency of cotorra) plus tiny/fast configs for exercising cotorra's
training, extraction, and scoring stages without downloading model weights or
running full-size training
"""

import collections.abc
import fnmatch
import logging
import pathlib
import subprocess
import sys
import typing

import joblib as jl
import numpy as np
import polars as pl
import pytest
import synth
from helpers import (
    TINY_MODEL_ARGS,
    base_extraction_cfg,
    base_scoring_cfg,
    base_training_cfg,
    tiny_model_dir,
    write_cfg,
)
from omegaconf import OmegaConf
from rich.logging import RichHandler

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer
from cocoa.winnower import Winnower

if typing.TYPE_CHECKING:
    from cotorra.trainer import Trainer

N_PATIENTS = 100


# keyed by `EstimatorType` value; both boosting libraries ship their own
# `libomp.dylib` and hit the clash described in `helpers.LIBOMP_HINT`
_BOOSTER_PROBES = {
    "lightGBM": """
import numpy as np, torch, lightgbm as lgb
X, y = np.random.randn(40, 4), (np.random.rand(40) > 0.5).astype(int)
lgb.LGBMClassifier(n_estimators=10, n_jobs=-1, verbose=-1).fit(X, y)
""",
    "XGBoost": """
import numpy as np, torch, xgboost as xgb
X, y = np.random.randn(40, 4), (np.random.rand(40) > 0.5).astype(int)
xgb.XGBClassifier(n_estimators=10, n_jobs=-1).fit(X, y)
""",
}


@pytest.fixture(scope="session")
def boosters_usable() -> dict[str, bool]:
    """
    whether torch and each boosting library can be fit in one process here;
    tests that drive one skip with `helpers.LIBOMP_HINT` when it cannot, since
    the failure mode is a segfault that would end the whole session. Probe
    both: guarding only LightGBM left `--estimator XGBoost` to take the suite
    down as soon as its arm stopped raising before reaching `fit`.
    """
    return {
        name: subprocess.run(
            [sys.executable, "-c", probe], capture_output=True
        ).returncode
        == 0
        for name, probe in _BOOSTER_PROBES.items()
    }


@pytest.fixture(scope="session", autouse=True)
def quiet_third_party_logging() -> collections.abc.Iterator[None]:
    """
    restore the root log level `cotorra.logger` asks for. It calls
    `logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])` at
    import, but pytest's logging plugin then resets the *root* logger to
    NOTSET so that it can capture everything -- and that `RichHandler` carries
    no level of its own, so every fsspec/filelock DEBUG record gets printed.
    Left alone it buries failure reports (and any assertion on a command's
    output) under thousands of lines of lock-acquired chatter. Setting the
    handler rather than the logger leaves pytest's own capture handlers on
    `log_level = INFO`, so `caplog` still sees INFO records.
    """
    import cotorra.logger  # noqa: F401 -- installs the root handler on import

    handlers = [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]
    restore = [(h, h.level) for h in handlers]
    for handler in handlers:
        handler.setLevel(logging.WARNING)
    yield
    for handler, level in restore:
        handler.setLevel(level)


@pytest.fixture(scope="session", autouse=True)
def joblib_threads() -> collections.abc.Iterator[None]:
    """
    run `bootstrap_ci`'s `joblib.Parallel(n_jobs=-1)` on threads rather than
    loky's forked workers: once any test has put a torch model on the mps (or
    cuda) device, forking the interpreter segfaults, and the bootstrap-backed
    summaries in `Logger.summarize_preds` are reached from several modules
    """
    with jl.parallel_config(backend="threading"):
        yield


@pytest.fixture(scope="session")
def raw_data(tmp_path_factory) -> synth.Manifest:
    """synthetic raw clif-like tables covering the default collation config"""
    return synth.write_raw_dataset(
        tmp_path_factory.mktemp("raw"), n_patients=N_PATIENTS
    )


@pytest.fixture(scope="session")
def processed(tmp_path_factory, raw_data) -> pathlib.Path:
    """collate, tokenize, and winnow `raw_data` with cocoa's shipped defaults"""
    dest = tmp_path_factory.mktemp("processed")
    Collator(raw_data_home=raw_data.root, processed_data_home=dest).save_all()
    Tokenizer(processed_data_home=dest).save_all()
    Winnower(processed_data_home=dest).save_all()
    return dest


@pytest.fixture(scope="session")
def processed_with_end_times(processed, tmp_path_factory) -> pathlib.Path:
    """
    `processed` re-tokenized with cocoa's `include_hours_to_end_time` and
    re-winnowed, so `tokens_times.parquet` carries cocoa's own
    `hours_to_end_time` and the inference tables its `hours_to_end_time_past`
    -- the data a `tte_objective` trains on
    """
    import shutil

    home = tmp_path_factory.mktemp("processed-end-times") / "processed"
    shutil.copytree(processed, home)
    Tokenizer(processed_data_home=home, include_hours_to_end_time=True).save_all()
    Winnower(processed_data_home=home).save_all()
    return home


@pytest.fixture(scope="session")
def tokenizer_cfg(processed):
    return OmegaConf.load(processed / "tokenizer.yaml")


# a label needs enough support for every estimator `RepBasedScorer` offers:
# k-NN asks train for `max(25, ...)` neighbors, `LogisticRegressionCV`
# stratifies train over 5 folds, and lightGBM/XGBoost evaluate AUC on tuning
MIN_TRAIN_ROWS = 25
MIN_TRAIN_MINORITY = 5


def _class_counts(df: pl.DataFrame, token: str) -> np.ndarray | None:
    """(negative, positive) counts among rows not already past the threshold"""
    if f"{token}_past" not in df.columns:
        return None
    valid = ~df[f"{token}_past"].to_numpy().astype(bool)
    future = df[f"{token}_future"].to_numpy().astype(int)[valid]
    return np.bincount(future, minlength=2)


@pytest.fixture(scope="session")
def target_token(processed) -> str:
    """
    an outcome token (from the shipped default scoring config) that every
    estimator can actually be fit and evaluated on, over the subset
    `RepBasedScorer` uses -- rows not already past the winnowing threshold,
    which is a good deal smaller than the raw column
    """
    cfg = base_scoring_cfg()
    tkzr = OmegaConf.load(processed / "tokenizer.yaml")
    candidates = [
        t
        for t in tkzr.lookup.keys()
        if any(fnmatch.fnmatch(t, p) for p in cfg["score"]["target_tokens"])
    ]
    dfs = {
        split: pl.read_parquet(processed / f"{split}_for_inference.parquet")
        for split in ("train", "tuning", "held_out")
    }
    for token in candidates:
        counts = {s: _class_counts(df, token) for s, df in dfs.items()}
        if any(c is None for c in counts.values()):
            continue
        if (
            counts["train"].sum() >= MIN_TRAIN_ROWS
            and counts["train"].min() >= MIN_TRAIN_MINORITY
            and counts["tuning"].min() >= 1
            and counts["held_out"].min() >= 1
        ):
            return token
    # a hard failure, not a skip: silently skipping here once hid every
    # rep-based scoring test in the suite
    pytest.fail(
        "no outcome token in the synthetic dataset has enough per-class support "
        f"to fit and evaluate on; counts were "
        f"{ {t: {s: _class_counts(d, t) for s, d in dfs.items()} for t in candidates} }"
    )


@pytest.fixture(scope="session")
def session_training_cfg_path(tmp_path_factory) -> pathlib.Path:
    d = tmp_path_factory.mktemp("session-cfg")
    return write_cfg(d / "training.yaml", base_training_cfg())


@pytest.fixture(scope="session")
def built_trainer(processed, session_training_cfg_path, tmp_path_factory) -> "Trainer":
    """
    a constructed (but never trained) tiny Trainer, shared read-only across
    tests that only inspect wiring (model_init, collate_fn, ...) and never
    call `.train()`; building this pays the cost of materializing the loader's
    batched training/tuning datasets exactly once for the whole session
    """
    from cotorra.trainer import Trainer

    out = tmp_path_factory.mktemp("built-trainer-output")
    return Trainer(
        training_cfg=session_training_cfg_path,
        processed_data_home=processed,
        output_home=out,
    )


@pytest.fixture(scope="session")
def fake_model_home(built_trainer, tmp_path_factory) -> pathlib.Path:
    """
    a valid, randomly-initialized (untrained) tiny model checkpoint matching
    the synthetic tokenizer's vocabulary; cheap to build and sufficient for
    extraction/scoring tests that don't care about training quality
    """
    model_home = tmp_path_factory.mktemp("fake-model") / "mdl-fake"
    built_trainer.model.save_pretrained(model_home)
    return model_home


@pytest.fixture(scope="session")
def model_with_heads(
    tokenizer_cfg, tmp_path_factory
) -> collections.abc.Callable[..., pathlib.Path]:
    """
    saves a tiny, randomly-initialized checkpoint carrying the secondary heads
    named -- any of `tte`, `tnt` (a mixture with `mixture=True`), `disposition`
    (over `classes` patterns, every `DSCG//*` token by default) -- and returns
    its home. Built from the synthetic tokenizer's lookup as
    `Trainer.model_init` builds a model from objective blocks, and initialized
    wide (`initializer_range=0.5`, against the usual 0.02) so that
    neighbouring positions give values far enough apart to tell apart. Each
    combination is built once a session
    """
    import torch as t
    from transformers import AutoConfig, AutoModelForCausalLM

    from cotorra.model import CotorraConfig, head_options

    built = {}

    def make(*names, mixture=False, classes=None, dtype="float32") -> pathlib.Path:
        key = (names, mixture, None if classes is None else tuple(classes), dtype)
        if key not in built:
            objectives = {f"{name}_objective": {} for name in names}
            if mixture:
                objectives["tnt_objective"]["mixture_components"] = 3
            if classes is not None:
                objectives["disposition_objective"]["classes"] = list(classes)
            lookup = tokenizer_cfg.lookup
            config = AutoConfig.from_pretrained(
                tiny_model_dir(),
                vocab_size=len(lookup),
                bos_token_id=lookup.BOS,
                eos_token_id=lookup.EOS,
                **TINY_MODEL_ARGS,
                initializer_range=0.5,
                dtype=dtype,
            )
            if heads := head_options(OmegaConf.create(objectives), lookup):
                config = CotorraConfig(text_config=config, heads=heads)
            t.manual_seed(0)
            home = tmp_path_factory.mktemp("model-with-heads") / "mdl-{}{}-{}".format(
                "-".join(names) or "stock", "-mixture" if mixture else "", dtype
            )
            AutoModelForCausalLM.from_config(config).save_pretrained(home)
            built[key] = home
        return built[key]

    return make


@pytest.fixture
def extraction_cfg_path(tmp_path) -> pathlib.Path:
    return write_cfg(tmp_path / "extraction.yaml", base_extraction_cfg())


@pytest.fixture
def scoring_cfg_path(tmp_path) -> pathlib.Path:
    return write_cfg(tmp_path / "scoring.yaml", base_scoring_cfg())
