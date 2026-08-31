#!/usr/bin/env python3

"""tests for cotorra.logger.Logger"""

import logging

import polars as pl
import pytest
from rich.logging import RichHandler

from cotorra.logger import Logger


class _Capture(logging.Handler):
    """collects formatted records; `Logger` instances aren't fetched via
    `logging.getLogger()` and set `propagate = False`, so pytest's `caplog`
    (which attaches to the registry/root logger) can't see their records --
    attach this directly to the instance under test instead"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def logger():
    return Logger()


@pytest.fixture
def capture(logger):
    cap = _Capture()
    logger.addHandler(cap)
    return cap


def test_logger_is_configured_for_info_level_rich_output(logger):
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)


def test_info_and_warning_do_not_raise(logger):
    logger.info("hello")
    logger.warning("uh oh")


def test_summarize_trained_model_generates_samples(built_trainer, logger, capture):
    tkzr_cfg = built_trainer.tkzr_cfg
    reverse = {v: k for k, v in tkzr_cfg.lookup.items()}

    logger.summarize_trained_model(
        built_trainer.model,
        bos_token_id=tkzr_cfg.lookup["BOS"],
        reverse=reverse,
        n_samp=1,
        max_len=8,
    )
    assert any("Sample 1" in m for m in capture.messages)


@pytest.fixture
def preds_df():
    n = 20
    future = [i % 2 == 0 for i in range(n)]
    score = [0.9 if f else 0.1 for f in future]
    return pl.LazyFrame(
        {
            "foo//bar_past": [False] * n,
            "foo//bar_future": future,
            "foo//bar_rep_score": score,
        }
    )


def test_summarize_preds_handles_a_single_method_column(preds_df, logger, capture):
    logger.summarize_preds(preds_df, ["foo//bar"])
    assert any("method='rep'" in m for m in capture.messages)


def test_summarize_preds_warns_on_non_finite_scores(logger, capture):
    n = 20
    future = [i % 2 == 0 for i in range(n)]
    score = [
        float("nan") if i == 0 else (0.9 if f else 0.1) for i, f in enumerate(future)
    ]
    df = pl.LazyFrame(
        {
            "foo//bar_past": [False] * n,
            "foo//bar_future": future,
            "foo//bar_rep_score": score,
        }
    )
    logger.summarize_preds(df, ["foo//bar"])
    assert any("non-finite" in m for m in capture.messages)
