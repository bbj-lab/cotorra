#!/usr/bin/env python3

"""
tests for cotorra.scorer_generative.GenerativeScorer, run against quick-sco-re
itself, which imports without sglang (the `gen` extra's); every test here fakes
sglang's part, the engine and the generation
"""

import dataclasses
import datetime as dt
import math
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest
from helpers import base_scoring_cfg, write_cfg
from omegaconf import OmegaConf
from quick_sco_re import (
    GeneratedTrajectory,
    GenerationConfig,
    ScoredTrajectory,
    TrajectoryType,
    silence_restriction_bias,
)

import cotorra.logger
from cotorra import scorer_generative

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

VOCAB = {
    "UNK": 0,
    "BOS": 1,
    "EOS": 2,
    "TRUNC": 3,
    "PAD": 4,
    "DSCG//expired": 5,
    "DSCG//home": 6,
    "TIME//1h-2h": 7,
    "RESP//imv": 8,
}
TRACKED = ["DSCG//expired", "RESP//imv"]
SCORES = ("mc", "scope", "reach")


def test_neither_the_cli_nor_the_scorer_imports_sglang():
    """
    quick-sco-re, which every install has, imports sglang only once
    `create_engine` starts an engine, so the CLI imports `GenerativeScorer` as it
    does every stage; sglang, which only the `gen` extra installs (Linux, an
    NVIDIA GPU), has to stay out of both, or a base install runs no command
    """
    probe = textwrap.dedent("""
        import sys
        from typer.testing import CliRunner
        from cotorra.cli import app
        import cotorra.scorer_generative

        assert CliRunner().invoke(app, ["--help"]).exit_code == 0
        assert "sglang" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------- the generation's configuration


def build_from(cfg: dict):
    """the `GenerationConfig` `cfg` gives, tracking `TRACKED` over `VOCAB`"""
    return scorer_generative.build_generation_config(
        OmegaConf.create(cfg), VOCAB, TRACKED, [VOCAB[name] for name in TRACKED]
    )


def build(**overrides):
    return build_from(base_scoring_cfg(**overrides))


def test_end_tokens_resolve_to_ids():
    gen_cfg = build()
    assert gen_cfg.end_token_ids == {VOCAB["EOS"]}


def test_end_token_prefixes_expand_over_the_vocabulary():
    gen_cfg = build(generation={"end_tokens": {"prefixes": ["DSCG"]}})
    assert gen_cfg.end_token_ids == {
        VOCAB["EOS"],
        VOCAB["DSCG//expired"],
        VOCAB["DSCG//home"],
    }


def test_unknown_token_names_are_ignored_rather_than_crashing():
    gen_cfg = build(
        score={"end_tokens": ["EOS", "NOT//A//TOKEN"], "suppressed_tokens": ["PAD"]}
    )
    assert gen_cfg.end_token_ids == {VOCAB["EOS"]}
    assert gen_cfg.suppressed_ids == [VOCAB["PAD"]]


def test_the_score_block_sizes_each_generation():
    """`score.max_len` caps a prompt and its continuation together, and
    `score.n_samp` counts the trajectories sampled per patient; the first
    tracked outcome is the generation's own target"""
    gen_cfg = build(score={"max_len": 48, "n_samp": 7})
    assert (gen_cfg.max_len, gen_cfg.n_samp) == (48, 7)
    assert gen_cfg.tracked_names == TRACKED
    assert gen_cfg.tracked_ids == [VOCAB[name] for name in TRACKED]
    assert gen_cfg.target_event_id == VOCAB[TRACKED[0]]


def test_the_temperature_is_the_generation_blocks_and_one_without_it():
    """1.0 being what Monte Carlo estimation needs"""
    assert build(generation={"temperature": 0.7}).temperature == 0.7
    cfg = base_scoring_cfg()
    del cfg["generation"]["temperature"]
    assert build_from(cfg).temperature == 1.0


def test_time_stopping_is_off_by_default():
    gen_cfg = build()
    assert gen_cfg.trunc_id is None
    assert gen_cfg.max_time is None
    assert gen_cfg.token_id_to_minutes == {}


def test_time_stopping_maps_token_bounds_to_their_geometric_mean():
    gen_cfg = build(
        generation={
            "time_stopping": {
                "enabled": True,
                "trunc_token": "TRUNC",
                "max_time_minutes": 1440,
                "time_token_bounds": {"TIME//1h-2h": [60, 120]},
            }
        }
    )
    assert gen_cfg.trunc_id == VOCAB["TRUNC"]
    assert gen_cfg.max_time == 1440
    assert gen_cfg.token_id_to_minutes[VOCAB["TIME//1h-2h"]] == pytest.approx(
        (60 * 120) ** 0.5
    )


def test_time_token_bounds_skip_tokens_the_vocabulary_lacks():
    """as `score.end_tokens` skips them, rather than failing on the lookup"""
    gen_cfg = build(
        generation={
            "time_stopping": {
                "enabled": True,
                "max_time_minutes": 1440,
                "time_token_bounds": {
                    "TIME//1h-2h": [60, 120],
                    "TIME//2h-4h": [120, 240],
                },
            }
        }
    )
    assert gen_cfg.token_id_to_minutes == pytest.approx(
        {VOCAB["TIME//1h-2h"]: math.sqrt(60 * 120)}
    )


def test_time_stopping_checks_the_clock_as_often_as_it_is_told():
    """every `time_check_interval` generated tokens, 100 without it"""
    stopping = {"enabled": True, "trunc_token": "TRUNC", "max_time_minutes": 60}
    every_25 = build(
        generation={"time_stopping": stopping | {"time_check_interval": 25}}
    )
    assert every_25.time_check_interval == 25
    cfg = base_scoring_cfg(generation={"time_stopping": stopping})
    del cfg["generation"]["time_stopping"]["time_check_interval"]
    assert build_from(cfg).time_check_interval == 100


def test_time_stopping_truncates_with_trunc_unless_told_otherwise():
    stopping = {"enabled": True, "max_time_minutes": 60, "trunc_token": "PAD"}
    assert build(generation={"time_stopping": stopping}).trunc_id == VOCAB["PAD"]
    cfg = base_scoring_cfg(generation={"time_stopping": stopping})
    del cfg["generation"]["time_stopping"]["trunc_token"]
    assert build_from(cfg).trunc_id == VOCAB["TRUNC"]


def test_time_stopping_disabled_reads_none_of_its_settings():
    """not even a trunc token the vocabulary lacks, which enabled time stopping
    refuses; and a suppressed TRUNC stays suppressed, there being no time
    horizon for it to mark"""
    gen_cfg = build(
        score={"suppressed_tokens": ["TRUNC"]},
        generation={
            "time_stopping": {
                "enabled": False,
                "trunc_token": "NOT//A//TOKEN",
                "max_time_minutes": 60,
                "time_token_bounds": {"TIME//1h-2h": [60, 120]},
            }
        },
    )
    assert (gen_cfg.trunc_id, gen_cfg.max_time) == (None, None)
    assert gen_cfg.token_id_to_minutes == {}
    assert gen_cfg.suppressed_ids == [VOCAB["TRUNC"]]


def test_time_stopping_rejects_a_trunc_token_absent_from_the_vocabulary():
    with pytest.raises(ValueError, match="not found in vocabulary"):
        build(
            generation={
                "time_stopping": {"enabled": True, "trunc_token": "NOT//A//TOKEN"}
            }
        )


def test_time_stopping_unsuppresses_the_trunc_token():
    """a suppressed TRUNC could never be forced, defeating the time horizon"""
    gen_cfg = build(
        score={"suppressed_tokens": ["PAD", "TRUNC"]},
        generation={
            "time_stopping": {
                "enabled": True,
                "trunc_token": "TRUNC",
                "max_time_minutes": 1440,
            }
        },
    )
    assert VOCAB["TRUNC"] not in gen_cfg.suppressed_ids
    assert VOCAB["PAD"] in gen_cfg.suppressed_ids


@pytest.mark.parametrize(
    "path",
    [
        ("generation",),
        ("generation", "end_tokens"),
        ("generation", "end_tokens", "prefixes"),
        ("generation", "time_stopping"),
        ("generation", "time_stopping", "time_token_bounds"),
        ("score", "end_tokens"),
        ("score", "suppressed_tokens"),
    ],
    ids="/".join,
)
def test_a_block_or_list_left_empty_reads_as_if_absent(path):
    """an entry left empty in the yaml parses to None, which `.get(key, default)`
    returns in place of its default, so each of these used to fail on `None.get`
    or on iterating None; time stopping is on, so that its bounds are read"""
    built = {}
    for how in ("empty", "absent"):
        cfg = base_scoring_cfg(
            generation={
                "time_stopping": {
                    "enabled": True,
                    "trunc_token": "TRUNC",
                    "time_token_bounds": {"TIME//1h-2h": [60, 120]},
                }
            }
        )
        *parents, leaf = path
        parent = cfg
        for key in parents:
            parent = parent[key]
        if how == "empty":
            parent[leaf] = None
        else:
            del parent[leaf]
        built[how] = build_from(cfg)
    assert built["empty"] == built["absent"]


def test_aggregation_reads_each_trajectorys_estimates_for_its_outcome():
    """
    an M1 trajectory gives a patient an M0 sample -- its occurrence flag for
    the outcome, else whether its timeline ended on it -- and an M1 sample, its
    inline SCOPE estimate, else 0; an M2 one an M2 sample, its inline REACH
    estimate, else 0. One that tracks no outcome inline, or not this one, gives
    nothing
    """
    config = GenerationConfig(
        max_len=16, n_samp=1, target_event_id=8, end_token_ids={2}
    )
    M1, M2 = TrajectoryType.M1, TrajectoryType.M2

    def trajectory(patient, kind, **fields):
        return GeneratedTrajectory(
            patient_idx=patient,
            sample_idx=0,
            traj_type=kind,
            prompt_len=4,
            output_ids=[],
            **{"timeline_terminating_id": None} | fields,
        )

    results = scorer_generative.aggregate_inline_results(
        [
            trajectory(
                0,
                M1,
                inline_tracked_ids=[5, 8],
                occurred_flag=np.array([True, False]),
                scope_estimates=np.array([0.1, 0.4]),
            ),
            trajectory(
                0, M2, inline_tracked_ids=[5, 8], reach_estimates=np.array([0.2, 0.3])
            ),
            trajectory(1, M1, inline_tracked_ids=[8, 5], timeline_terminating_id=8),
            trajectory(1, M2, inline_tracked_ids=[8]),
            trajectory(2, M1, scope_estimates=np.array([0.9, 0.9])),
            trajectory(2, M1, inline_tracked_ids=[5], scope_estimates=np.array([0.9])),
        ],
        num_patients=4,
        config=config,
    )
    assert [r.m0_samples for r in results] == [[False], [True], [], []]
    assert [r.m1_samples for r in results] == [[0.4], [0.0], [], []]
    assert [r.m2_samples for r in results] == [[0.3], [0.0], [], []]


# ----------------------------------------------------- scoring, sglang faked


def mark(prompt) -> int:
    """a number read off a prompt, which tells patients apart by what they were
    prompted with, so that scores landing on another patient's row show"""
    return sum(prompt) + 7 * len(prompt)


def scope(prompt, k: int, s: int) -> float:
    """sample s's inline SCOPE estimate for the k-th tracked outcome"""
    return (mark(prompt) % 11 + 1) / 16 + k / 64 + s / 256


def reach(prompt, k: int, s: int) -> float:
    """its inline REACH estimate"""
    return (mark(prompt) % 5 + 1) / 8 + k / 64 + s / 256


def occurs(prompt, k: int, s: int) -> bool:
    """whether the outcome was generated in sample s: in about a third of them,
    the first two outcomes together in some"""
    return (mark(prompt) * (k + 1) + s) % 3 == 0


def rescored(prompt, k: int, s: int) -> float:
    """what scoring sample s regenerated without the outcome gives"""
    return 1 - (mark(prompt) % 3 + 1) / 8 - k / 64 - s / 256


class FakeSGLang:
    """
    sglang's part, behind quick_sco_re, faked: an engine that records how it was
    started, and generation and scoring that record what they're asked for and
    answer deterministically -- each M1 trajectory's estimates read off the
    prompt it continues, by `scope`, `reach` and `occurs`, and each regenerated
    trajectory scored by `rescored` -- so that a test can work out every
    patient's scores on its own. `tweak` edits each M1 trajectory, given its
    prompt, before it's handed back
    """

    def __init__(self, tweak=None):
        self.tweak = tweak
        self.started = None
        self.passes = []  # the prompts of each M1 pass, and how it was asked for
        self.regenerated = []  # (outcome id, prompt, sample) per M2 regeneration

    def install(self, monkeypatch) -> "FakeSGLang":
        for name in (
            "create_engine",
            "generate_trajectories",
            "generate_m2_from_m1_trajectory",
            "score_trajectory",
        ):
            monkeypatch.setattr(scorer_generative, name, getattr(self, name))
        return self

    def create_engine(self, **kwargs) -> "FakeSGLang":
        self.started = kwargs
        return self

    async def generate_trajectories(
        self,
        engine,
        config,
        patient_tokens,
        methods,
        stop_at_tracked_events=True,
        first_token_biases=None,
    ):
        assert engine is self
        self.passes.append(
            {
                "prompts": [list(prompt) for prompt in patient_tokens],
                "methods": list(methods),
                "stop_at_tracked_events": stop_at_tracked_events,
                "first_token_biases": first_token_biases,
            }
        )
        trajectories = []
        for i, prompt in enumerate(patient_tokens):
            for s in range(config.n_samp):
                ks = range(len(config.tracked_ids))
                occurred = [occurs(prompt, k, s) for k in ks]
                trajectory = GeneratedTrajectory(
                    patient_idx=i,
                    sample_idx=s,
                    traj_type=TrajectoryType.M1,
                    prompt_len=len(prompt),
                    output_ids=[
                        t for t, hit in zip(config.tracked_ids, occurred) if hit
                    ],
                    timeline_terminating_id=None,
                    scope_estimates=np.array([scope(prompt, k, s) for k in ks]),
                    reach_estimates=np.array([reach(prompt, k, s) for k in ks]),
                    occurred_flag=np.array(occurred),
                    inline_tracked_ids=list(config.tracked_ids),
                )
                if self.tweak is not None:
                    trajectory = self.tweak(trajectory, prompt)
                trajectories.append(trajectory)
        return trajectories

    async def generate_m2_from_m1_trajectory(self, engine, config, m1_traj, prompt):
        assert engine is self
        self.regenerated.append(
            (config.target_event_id, list(prompt), m1_traj.sample_idx)
        )
        return dataclasses.replace(m1_traj, traj_type=TrajectoryType.M2, n_new_tokens=5)

    async def score_trajectory(
        self, engine, config, traj, prompt, truncate_again=False
    ):
        assert engine is self and traj.traj_type == TrajectoryType.M2
        k = config.tracked_ids.index(config.target_event_id)
        score = rescored(prompt, k, traj.sample_idx)
        return ScoredTrajectory(trajectory=traj, score=score)


def scorer_with(scoring_cfg: dict, home, tmp_path, monkeypatch):
    """a `GenerativeScorer` built on `home` under `scoring_cfg`, writing to
    `tmp_path`, with sglang's part faked, and the fake"""
    fake = FakeSGLang().install(monkeypatch)
    scorer = scorer_generative.GenerativeScorer(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", scoring_cfg),
        processed_data_home=home,
        model_home=tmp_path / "mdl-stub",
        output_home=tmp_path,
    )
    return scorer, fake


@pytest.fixture(scope="module")
def held_out(processed) -> pl.DataFrame:
    return pl.read_parquet(processed / "held_out_for_inference.parquet")


@pytest.fixture(scope="module")
def outcomes(held_out, tokenizer_cfg) -> list[str]:
    """
    two outcomes the held-out split labels, in vocabulary order, as the scorer
    tracks them: one no patient has had yet, and one some already have -- whose
    scores those patients go without -- and others haven't
    """
    labeled = [
        c.removesuffix("_past")
        for c in held_out.columns
        if c.endswith("_past") and held_out[c].dtype == pl.Boolean
    ]
    past = {name: held_out[f"{name}_past"].sum() for name in labeled}
    unhad = [name for name in labeled if past[name] == 0]
    partly = [name for name in labeled if 0 < past[name] < held_out.height]
    if not (unhad and partly):  # a hard failure, as `target_token` fails
        pytest.fail(f"the held-out split lacks the outcomes these tests need: {past}")
    return sorted([unhad[0], partly[0]], key=tokenizer_cfg.lookup.get)


def expected_scores(
    held_out: pl.DataFrame,
    outcomes: list[str],
    max_len: int,
    n_samp: int,
    overflow: str = "truncate_left",
    methods=("M1", "M2"),
    samples=None,
    occurred=occurs,
    ended=occurs,
) -> dict[str, np.ndarray]:
    """
    each outcome's scores on each held-out row, worked out from `FakeSGLang`'s
    estimates as the generative scorer defines them: mc, the share of a
    patient's samples in which the outcome occurred (`ended`: in which its
    timeline ended on it, where the trajectories carry no occurrence flags);
    scope, their mean inline SCOPE estimate; and under M2, reach, the mean of
    the regenerated trajectory's score where the outcome `occurred` and the
    inline REACH estimate where it didn't. NaN for an outcome a patient has had
    already, and with `overflow="drop"` for a patient prompted past `max_len`.
    `samples(k, kind)` picks the samples that count towards outcome k's scores
    of a kind, every one by default
    """
    scores = {
        f"{name}_{kind}_score": np.full(held_out.height, np.nan)
        for name in outcomes
        for kind in SCORES
    }
    for row, prompt in enumerate(held_out["tokens_past"].to_list()):
        if len(prompt) > max_len:
            if overflow == "drop":
                continue
            prompt = prompt[-max_len:]
        for k, name in enumerate(outcomes):
            if f"{name}_past" in held_out.columns and held_out[f"{name}_past"][row]:
                continue
            counted = {
                kind: list(samples(k, kind) if samples else range(n_samp))
                for kind in SCORES
            }
            scores[f"{name}_mc_score"][row] = np.mean(
                [ended(prompt, k, s) for s in counted["mc"]]
            )
            scores[f"{name}_scope_score"][row] = np.mean(
                [scope(prompt, k, s) for s in counted["scope"]]
            )
            if "M2" in methods and counted["reach"]:
                scores[f"{name}_reach_score"][row] = np.mean(
                    [
                        rescored(prompt, k, s)
                        if occurred(prompt, k, s)
                        else reach(prompt, k, s)
                        for s in counted["reach"]
                    ]
                )
    return scores


def written_scores(scorer, held_out, expected) -> pl.DataFrame:
    """the table `save_all` writes, checked against `expected`: the held-out
    table as it was, then each outcome's three scores"""
    scorer.save_all()
    table = pl.read_parquet(scorer.output_home)
    assert table.columns == held_out.columns + list(expected)
    assert table.select(held_out.columns).equals(held_out)
    for column, values in expected.items():
        np.testing.assert_allclose(table[column].to_numpy(), values, err_msg=column)
    return table


def test_each_patients_scores_are_the_means_over_its_samples(
    processed, held_out, outcomes, tmp_path, monkeypatch
):
    """mc, scope and reach for every outcome `score.target_tokens` names, over
    the `score.n_samp` trajectories of each patient, the prompts cut to their
    last `score.max_len` tokens -- NaN where the patient has had the outcome"""
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 3})
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    expected = expected_scores(held_out, outcomes, max_len=16, n_samp=3)
    table = written_scores(scorer, held_out, expected)
    # both kinds of reach sample came up, regenerated and inline
    prompts = [p[-16:] for p in held_out["tokens_past"].to_list()]
    hits = [occurs(p, k, s) for p in prompts for k in range(2) for s in range(3)]
    assert any(hits) and not all(hits) and fake.regenerated
    for name in outcomes:
        assert (
            table[f"{name}_reach_score"]
            .is_nan()
            .equals(held_out[f"{name}_past"], check_names=False)
        )


def test_an_outcome_is_regenerated_from_its_patients_prompt_where_it_occurred(
    processed, held_out, outcomes, tokenizer_cfg, tmp_path, monkeypatch
):
    """
    under M2, each trajectory in which an outcome was generated is regenerated
    without it, from that patient's own (cut) prompt, and nowhere else -- not
    for an outcome the patient has already had, though it was generated there
    too
    """
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 3})
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    scorer.save_all()
    lookup = tokenizer_cfg.lookup
    expected, skipped = [], 0
    for row, prompt in enumerate(held_out["tokens_past"].to_list()):
        for k, name in enumerate(outcomes):
            for s in range(3):
                if occurs(prompt[-16:], k, s):
                    if held_out[f"{name}_past"][row]:
                        skipped += 1
                    else:
                        expected.append((lookup[name], prompt[-16:], s))
    assert expected and skipped
    assert sorted(fake.regenerated) == sorted(expected)


def test_with_m1_alone_nothing_is_regenerated_and_reach_goes_unscored(
    processed, held_out, outcomes, tmp_path, monkeypatch
):
    cfg = base_scoring_cfg(
        score={"target_tokens": outcomes, "n_samp": 3}, generation={"methods": ["M1"]}
    )
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    expected = expected_scores(held_out, outcomes, 16, 3, methods=("M1",))
    table = written_scores(scorer, held_out, expected)
    assert fake.regenerated == []
    for name in outcomes:
        assert table[f"{name}_reach_score"].is_nan().all()
        assert not table[f"{name}_scope_score"].is_nan().all()


@pytest.mark.parametrize("methods", [["M1"], ["M1", "M2"]], ids="+".join)
@pytest.mark.parametrize("chunk", [5, 64])
def test_patients_are_generated_a_chunk_at_a_time_each_scored_on_its_own_row(
    chunk, methods, processed, held_out, outcomes, tmp_path, monkeypatch
):
    """
    `engine.patient_chunk_size` patients to an M1 pass, in order, each pass
    running every timeline to its end -- M1 alone, whatever `generation.methods`
    asks for, an M2 trajectory being regenerated from an M1 one -- and every
    patient's scores, regenerated ones included, on its own row
    """
    cfg = base_scoring_cfg(
        score={"target_tokens": outcomes, "n_samp": 2},
        generation={"methods": methods},
        engine={"patient_chunk_size": chunk},
    )
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    expected = expected_scores(held_out, outcomes, 16, 2, methods=methods)
    written_scores(scorer, held_out, expected)
    n = held_out.height
    assert [len(p["prompts"]) for p in fake.passes] == [
        min(chunk, n - start) for start in range(0, n, chunk)
    ]
    assert [prompt for p in fake.passes for prompt in p["prompts"]] == [
        prompt[-16:] for prompt in held_out["tokens_past"].to_list()
    ]
    assert all(p["methods"] == ["M1"] for p in fake.passes)
    assert not any(p["stop_at_tracked_events"] for p in fake.passes)


def test_a_trajectory_lacking_an_estimate_adds_no_sample_for_it(
    processed, held_out, outcomes, tmp_path, monkeypatch
):
    """
    a trajectory without REACH estimates adds no reach sample for any outcome,
    and isn't regenerated; one that doesn't track an outcome inline adds no
    sample of any kind for it
    """

    def tweak(trajectory, prompt):
        if trajectory.sample_idx == 0:
            return dataclasses.replace(trajectory, reach_estimates=None)
        if trajectory.sample_idx == 1:  # tracking the first outcome alone
            first = slice(0, 1)
            return dataclasses.replace(
                trajectory,
                inline_tracked_ids=trajectory.inline_tracked_ids[first],
                scope_estimates=trajectory.scope_estimates[first],
                reach_estimates=trajectory.reach_estimates[first],
                occurred_flag=trajectory.occurred_flag[first],
            )
        return trajectory

    counted = {
        (0, "mc"): [0, 1, 2],
        (0, "scope"): [0, 1, 2],
        (0, "reach"): [1, 2],
        (1, "mc"): [0, 2],
        (1, "scope"): [0, 2],
        (1, "reach"): [2],
    }
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 3})
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    fake.tweak = tweak
    expected = expected_scores(
        held_out, outcomes, 16, 3, samples=lambda k, kind: counted[k, kind]
    )
    written_scores(scorer, held_out, expected)
    # sample 0 would have been regenerated somewhere, had it its estimates
    prompts = [p[-16:] for p in held_out["tokens_past"].to_list()]
    assert any(occurs(p, k, 0) for p in prompts for k in range(2))
    assert fake.regenerated and all(s != 0 for _, _, s in fake.regenerated)


def test_without_occurrence_flags_the_generated_tokens_say_what_occurred(
    processed, held_out, outcomes, tmp_path, monkeypatch
):
    """
    a trajectory carrying no occurrence flags gives an M0 sample for the
    outcome its timeline ended on, and under M2 is regenerated for each outcome
    among its generated tokens
    """

    def tweak(trajectory, prompt):
        ended = trajectory.output_ids[0] if trajectory.output_ids else None
        return dataclasses.replace(
            trajectory, occurred_flag=None, timeline_terminating_id=ended
        )

    def ended(prompt, k, s) -> bool:  # the first outcome generated ends it
        return occurs(prompt, k, s) and not any(occurs(prompt, j, s) for j in range(k))

    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 4})
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    fake.tweak = tweak
    expected = expected_scores(held_out, outcomes, 16, 4, ended=ended)
    written_scores(scorer, held_out, expected)
    # an outcome generated after another isn't the one its timeline ended on
    assert any(
        occurs(p[-16:], 1, s) and not ended(p[-16:], 1, s)
        for p in held_out["tokens_past"].to_list()
        for s in range(4)
    )


def test_where_no_outcome_was_generated_reach_is_the_inline_estimate(
    processed, held_out, outcomes, tmp_path, monkeypatch
):
    """nothing to regenerate under M2, so every reach sample is the trajectory's
    own inline REACH estimate, and mc is 0 wherever the patient goes scored"""

    def tweak(trajectory, prompt):
        return dataclasses.replace(
            trajectory,
            output_ids=[],
            occurred_flag=np.zeros_like(trajectory.occurred_flag),
        )

    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 3})
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    fake.tweak = tweak
    never = lambda prompt, k, s: False
    expected = expected_scores(held_out, outcomes, 16, 3, occurred=never, ended=never)
    table = written_scores(scorer, held_out, expected)
    assert fake.regenerated == []
    assert table[f"{outcomes[0]}_mc_score"].drop_nans().unique().to_list() == [0.0]


@pytest.fixture
def logged(monkeypatch) -> list[str]:
    """what every `Logger` logs at info, the scorer's included, which logs as
    it's built"""
    messages, info = [], cotorra.logger.Logger.info

    def record(self, msg, *args, **kwargs):
        messages.append(str(msg))
        info(self, msg, *args, **kwargs)

    monkeypatch.setattr(cotorra.logger.Logger, "info", record)
    return messages


@pytest.mark.parametrize(
    "overflow", ["drop", "truncate_left", None], ids=["drop", "truncate", "absent"]
)
def test_a_prompt_past_max_len_is_dropped_or_cut_to_its_last_tokens(
    overflow, processed, held_out, outcomes, logged, tmp_path, monkeypatch
):
    """
    `prompt_overflow`, `truncate_left` without it: a prompt longer than
    `score.max_len` keeps its last `max_len` tokens, or under `drop` its
    patient goes ungenerated, every score on its row NaN and the other rows as
    they'd be anyway
    """
    prompts = held_out["tokens_past"].to_list()
    max_len = sorted(len(p) for p in prompts)[len(prompts) // 2]
    long = [len(p) > max_len for p in prompts]
    assert any(long) and not all(long)
    cfg = base_scoring_cfg(
        score={"max_len": max_len, "n_samp": 2, "target_tokens": outcomes}
    )
    if overflow is None:
        del cfg["prompt_overflow"]
    else:
        cfg["prompt_overflow"] = overflow
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    expected = expected_scores(
        held_out, outcomes, max_len, 2, overflow=overflow or "truncate_left"
    )
    table = written_scores(scorer, held_out, expected)
    generated = [prompt for p in fake.passes for prompt in p["prompts"]]
    if overflow == "drop":
        assert generated == [p for p, cut in zip(prompts, long) if not cut]
        message = f"Dropped {sum(long)} patients with prompts > {max_len} tokens"
        scores = [f"{name}_{kind}_score" for name in outcomes for kind in SCORES]
        dropped = table.filter(pl.Series(long)).select(scores)
        assert all(dropped[c].is_nan().all() for c in scores)
    else:
        assert generated == [p[-max_len:] for p in prompts]
        message = f"Left-truncated {sum(long)} prompts to {max_len} tokens"
    assert message in logged


@pytest.mark.parametrize(
    "blocks, started",
    [
        ({}, {"use_time_horizon": False, "mem_fraction": 0.85}),
        ({"engine": {"mem_fraction": 0.5}}, {"mem_fraction": 0.5}),
        ({"engine": None}, {"mem_fraction": 0.85}),
        (
            {"generation": {"time_stopping": {"enabled": True, "trunc_token": "UNK"}}},
            {"use_time_horizon": False},
        ),
        (
            {
                "generation": {
                    "time_stopping": {
                        "enabled": True,
                        "trunc_token": "UNK",
                        "max_time_minutes": 1440,
                    }
                }
            },
            {"use_time_horizon": True},
        ),
    ],
    ids=["packaged", "mem-fraction", "empty-engine", "no-horizon", "time-horizon"],
)
def test_the_engine_is_started_as_the_score_engine_and_time_stopping_blocks_ask(
    blocks, started, processed, outcomes, tmp_path, monkeypatch
):
    """
    on the model at `model_home`, with a context of `score.max_len`, reserving
    `engine.mem_fraction` of the gpu's memory (0.85 without it), and with the
    time horizon on only where time stopping both is enabled and has a
    `max_time_minutes` to stop at
    """
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "max_len": 48})
    for key, block in blocks.items():
        cfg[key] = None if block is None else cfg[key] | block
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    assert fake.started == {
        "model_path": str((tmp_path / "mdl-stub").resolve()),
        "max_len": 48,
        "use_time_horizon": False,
        "mem_fraction": 0.85,
        **started,
    }
    assert scorer.engine is fake


def test_target_tokens_resolve_against_the_vocabulary_in_its_order(
    processed, tokenizer_cfg, tmp_path, monkeypatch
):
    """fnmatch patterns, every match tracked in the one pass and in vocabulary
    order, whatever order the patterns came in, the first the generation's own
    target"""
    lookup = tokenizer_cfg.lookup
    cfg = base_scoring_cfg(score={"target_tokens": ["XFR-IN//icu", "DSCG//exp*"]})
    scorer, _ = scorer_with(cfg, processed, tmp_path, monkeypatch)
    names = ["DSCG//expired", "XFR-IN//icu"]
    assert lookup[names[0]] < lookup[names[1]]
    assert scorer.grokked_outcome_tokens == scorer.gen_config.tracked_names == names
    assert scorer.gen_config.tracked_ids == [lookup[name] for name in names]
    assert scorer.gen_config.target_event_id == lookup[names[0]]


def test_target_tokens_matching_no_token_are_refused_naming_them(
    processed, tmp_path, monkeypatch
):
    cfg = base_scoring_cfg(score={"target_tokens": ["NOPE//*", "NONE//*"]})
    message = "No vocabulary tokens matched score.target_tokens=['NOPE//*', 'NONE//*']"
    with pytest.raises(ValueError, match=re.escape(message)):
        scorer_with(cfg, processed, tmp_path, monkeypatch)


def test_an_outcome_the_inference_tables_dont_label_is_scored_for_every_patient(
    processed, held_out, tmp_path, monkeypatch
):
    """
    a target token with no `_past` column -- `DSCG//home`, which the winnower
    doesn't label -- is one no patient has had yet, as far as the scorer can
    tell, so every row gets its scores
    """
    assert "DSCG//home_past" not in held_out.columns
    cfg = base_scoring_cfg(score={"target_tokens": ["DSCG//home"], "n_samp": 2})
    scorer, _ = scorer_with(cfg, processed, tmp_path, monkeypatch)
    expected = expected_scores(held_out, ["DSCG//home"], 16, 2)
    table = written_scores(scorer, held_out, expected)
    assert not table["DSCG//home_scope_score"].is_nan().any()


def test_the_scores_are_written_beside_the_inference_tables_unless_told_otherwise(
    processed, outcomes, tmp_path, monkeypatch
):
    home = tmp_path / "processed"
    shutil.copytree(processed, home)
    FakeSGLang().install(monkeypatch)
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 1})
    stage = dict(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", cfg),
        processed_data_home=home,
        model_home=tmp_path / "mdl-stub",
    )
    beside = scorer_generative.GenerativeScorer(**stage)
    assert beside.output_home == home.resolve() / "scores-generative-mdl-stub.parquet"
    beside.save_all()
    assert beside.output_home.is_file()
    elsewhere = scorer_generative.GenerativeScorer(
        **stage, output_home=tmp_path / "scores"
    )
    assert elsewhere.output_home == (
        (tmp_path / "scores").resolve() / "scores-generative-mdl-stub.parquet"
    )


def test_verbose_save_all_summarizes_the_scores_it_writes(
    processed, target_token, tmp_path, monkeypatch
):
    """
    `save_all` assembles its table as a `pl.DataFrame`, but
    `Logger.summarize_preds` collects what it selects, which only a
    `pl.LazyFrame` can, so `-v` used to crash at the very end of a generation
    run, its scores already written. Built on the synthetic data, with the
    SGLang engine and the generation itself stubbed out
    """
    monkeypatch.setattr(scorer_generative, "create_engine", lambda **_: None)
    cfg = base_scoring_cfg(score={"target_tokens": [target_token]})
    scorer = scorer_generative.GenerativeScorer(
        scoring_cfg=write_cfg(tmp_path / "scoring.yaml", cfg),
        processed_data_home=processed,
        model_home=tmp_path / "mdl-stub",
        output_home=tmp_path,
    )
    columns = [f"{target_token}_{m}_score" for m in SCORES]
    future = scorer.ds[f"{target_token}_future"].to_numpy().astype(float)

    async def score():
        return {column: future for column in columns}

    scorer.score = score
    scorer.save_all(verbose=True)

    assert set(columns) <= set(pl.read_parquet(scorer.output_home).columns)


@pytest.mark.parametrize(
    "blocks, methods",
    [
        ({"generation": None, "engine": None}, ["M1", "M2"]),
        ({"generation": {"methods": None}}, ["M1", "M2"]),
        ({"generation": {"methods": []}}, []),  # explicit, so kept
    ],
    ids=["empty-blocks", "empty-methods", "no-methods"],
)
def test_empty_generation_and_engine_blocks_take_their_defaults(
    blocks, methods, processed, target_token, tmp_path, monkeypatch
):
    """as in `build_generation_config`, the scorer read `generation` and `engine`
    with `.get(key, {})`, which returns a block left empty as None, and an empty
    `methods:` likewise, which `"M2" in self.methods` then failed on"""
    cfg = base_scoring_cfg(score={"target_tokens": [target_token]}, **blocks)
    scorer, _ = scorer_with(cfg, processed, tmp_path, monkeypatch)
    assert scorer.methods == methods
    assert scorer.chunk_size == 64


def test_a_generative_scorer_without_its_directories_is_refused_by_name(tmp_path):
    """both default to None, a holdover from when the config named them, and
    leaving one out used to fail inside `pathlib` without saying which"""
    GenerativeScorer = scorer_generative.GenerativeScorer
    with pytest.raises(TypeError, match=r"^GenerativeScorer needs `model_home`$"):
        GenerativeScorer(processed_data_home=tmp_path)
    with pytest.raises(
        TypeError,
        match=r"^GenerativeScorer needs `processed_data_home` and `model_home`$",
    ):
        GenerativeScorer()


# ------------------------------------ the first token, after the cut's silence

FORBID = -10_000.0  # quick-sco-re's bias for a token it rules out
HOUR = dt.timedelta(hours=1)


def test_spacer_bins_are_cocoas_right_closed_bins_between_its_boundaries():
    """the last running on without end, and none without spacers configured"""
    tkzr = OmegaConf.create(
        {"cfg": {"spacers": {"5m-15m": 5, "15m-1h": 15, "1h-2h": 60}}}
    )
    assert scorer_generative.spacer_bins(tkzr) == {
        "TIME//5m-15m": (5.0, 15.0),
        "TIME//15m-1h": (15.0, 60.0),
        "TIME//1h-2h": (60.0, math.inf),
    }
    for cfg in ({}, {"cfg": {}}, {"cfg": {"spacers": None}}):
        assert scorer_generative.spacer_bins(OmegaConf.create(cfg)) == {}


def test_each_spacer_cocoa_inserted_stands_for_a_gap_its_bin_holds(
    processed_with_spacers,
):
    """read off the timelines cocoa tokenized: the gap, in whole minutes, from
    the instant before each spacer to the spacer's own, which it opens; cocoa
    gives the smallest boundary itself the first spacer"""
    tkzr = OmegaConf.load(processed_with_spacers / "tokenizer.yaml")
    bins = scorer_generative.spacer_bins(tkzr)
    names = {tid: name for name, tid in tkzr.lookup.items()}
    smallest = min(lo for lo, _ in bins.values())
    seen = set()
    timelines = pl.read_parquet(processed_with_spacers / "tokens_times.parquet")
    for tokens, times in timelines.select("tokens", "times").iter_rows():
        for i, token in enumerate(tokens):
            if (name := names[token]) in bins:
                gap = (times[i] - times[i - 1]).total_seconds() // 60
                lo, hi = bins[name]
                assert lo < gap <= hi or gap == lo == smallest, (name, gap)
                seen.add(name)
    assert seen  # spacers enough to check


def cut_frame() -> pl.DataFrame:
    """three timelines cut four hours in, two of them counting from their first
    token or their subject's start_time alike, the other's start_time an hour
    before its first token"""
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "times": [
                [t0, t0 + HOUR / 6, t0 + 5 * HOUR],
                [t0 + HOUR, t0 + 2 * HOUR, t0 + 2 * HOUR, t0 + 6 * HOUR],
                [t0, t0 + 4 * HOUR],
            ],
            "tokens_past": [[1, 5], [1, 5, 6], [1]],
            "start_time": [t0, t0, t0],
        },
        schema={
            "times": pl.List(pl.Datetime("ms", "UTC")),
            "tokens_past": pl.List(pl.Int64),
            "start_time": pl.Datetime("ms", "UTC"),
        },
    )


def test_silences_run_from_the_last_past_token_to_either_cut():
    """the last past token on the cut's eve, the next one on the cut itself"""
    silences = scorer_generative.silences
    np.testing.assert_allclose(
        silences(cut_frame(), "duration_s", 4 * 3600), [230, 180, 240]
    )
    np.testing.assert_allclose(
        silences(cut_frame(), "after_start_time_s", 4 * 3600), [230, 120, 240]
    )


@pytest.mark.parametrize("hours", [1, 5])
def test_a_cut_the_timelines_werent_cut_at_is_refused(hours):
    """an hour in, the second timeline's past runs past the cut; five hours in,
    the third's future starts before it"""
    with pytest.raises(ValueError, match=r"of only 2 of 3 timelines; set it as"):
        scorer_generative.silences(cut_frame(), "duration_s", hours * 3600)
    with pytest.raises(ValueError, match=r"needs a start_time column"):
        scorer_generative.silences(
            cut_frame().drop("start_time"), "after_start_time_s", 4 * 3600
        )


def restricted_cfg(outcomes: list[str], **restriction) -> dict:
    """the default scoring config tracking `outcomes`, each trajectory's first
    token restricted after its timeline's silence at `restriction`'s cut"""
    return base_scoring_cfg(
        score={"target_tokens": outcomes, "n_samp": 2},
        generation={"silence_restriction": restriction},
    )


def silences_by_hand(home, origin: str, seconds: int) -> np.ndarray:
    """each held-out timeline's silence in minutes, from its last past token to
    `seconds` after its first token (`origin` "times") or its start_time"""
    held_out = pl.read_parquet(home / "held_out_for_inference.parquet")
    return np.array(
        [
            (
                (row["times"][0] if origin == "times" else row["start_time"])
                + dt.timedelta(seconds=seconds)
                - row["times"][len(row["tokens_past"]) - 1]
            ).total_seconds()
            / 60
            for row in held_out.iter_rows(named=True)
        ]
    )


def possible(bias: dict | None, vocab: dict[str, int]) -> set[int]:
    """the token ids a first-token bias leaves possible"""
    return {t for t in vocab.values() if (bias or {}).get(t, 0.0) > FORBID / 2}


@pytest.mark.parametrize("mode", ["hard", "prop"])
def test_every_first_token_is_a_spacer_long_enough_for_its_timelines_silence(
    mode, processed_with_spacers, outcomes, tmp_path, monkeypatch
):
    """each patient's bias, passed with its prompt, is quick-sco-re's for its
    silence at the cut cocoa's default threshold made, a day after each
    timeline's first token, and leaves possible only the spacers whose bins
    reach that far"""
    cfg = restricted_cfg(outcomes, duration_s=86_400, mode=mode)
    scorer, fake = scorer_with(cfg, processed_with_spacers, tmp_path, monkeypatch)
    scorer.save_all()
    vocab = scorer.vocab
    bins = {
        vocab[name]: bounds
        for name, bounds in scorer_generative.spacer_bins(scorer.tkzr_cfg).items()
        if name in vocab
    }
    minutes = silences_by_hand(processed_with_spacers, "times", 86_400)
    biases = [b for p in fake.passes for b in p["first_token_biases"]]
    assert biases == [
        silence_restriction_bias(m, bins, vocab.values(), mode=mode, min_minutes=5)
        for m in minutes
    ]
    for bias, m in zip(biases, minutes):
        assert possible(bias, vocab) == {t for t, (_, hi) in bins.items() if hi >= m}
        assert possible(bias, vocab)  # and some spacer reaches it


def test_each_patient_keeps_its_own_first_token_bias_chunk_after_chunk(
    processed_with_spacers, outcomes, tmp_path, monkeypatch
):
    """five patients a pass, each with a silence of its own, so that a bias
    handed to another patient would show"""
    cfg = restricted_cfg(outcomes, duration_s=86_400)
    cfg["engine"]["patient_chunk_size"] = 5
    scorer, fake = scorer_with(cfg, processed_with_spacers, tmp_path, monkeypatch)
    scorer.silence_minutes = 1.0 + 30 * np.arange(len(scorer.final_tokens))
    scorer.save_all()
    assert len(fake.passes) > 1
    assert [len(p["first_token_biases"]) for p in fake.passes] == [
        len(p["prompts"]) for p in fake.passes
    ]
    assert [b for p in fake.passes for b in p["first_token_biases"]] == [
        silence_restriction_bias(
            m, scorer.spacer_bounds, scorer.vocab.values(), min_minutes=5
        )
        for m in scorer.silence_minutes
    ]
    assert fake.passes[0]["first_token_biases"][0] is None  # silent a minute


@pytest.mark.parametrize("overflow", ["drop", "truncate_left"])
def test_the_silences_are_the_kept_patients(
    overflow, processed_with_spacers, outcomes, tmp_path, monkeypatch
):
    """a prompt cut to its last tokens keeps its last token's time; one dropped
    takes its silence with it. `max_len` is the median prompt's, so that only
    some are"""
    held_out = pl.read_parquet(
        processed_with_spacers / "held_out_for_inference.parquet"
    )
    lengths = held_out["tokens_past"].list.len()
    cfg = restricted_cfg(outcomes, duration_s=86_400)
    cfg["score"]["max_len"] = int(lengths.median())
    cfg["prompt_overflow"] = overflow
    scorer, _ = scorer_with(cfg, processed_with_spacers, tmp_path, monkeypatch)
    kept = np.array(scorer.keep_mask)
    assert (lengths > cfg["score"]["max_len"]).any()
    assert kept.all() if overflow == "truncate_left" else 0 < kept.sum() < kept.size
    np.testing.assert_allclose(
        scorer.silence_minutes,
        silences_by_hand(processed_with_spacers, "times", 86_400)[kept],
    )


def test_a_cut_after_each_subjects_start_counts_from_its_start_time(
    processed_with_spacers_after_start, outcomes, tmp_path, monkeypatch
):
    """as cocoa's after_start_time_s cuts, its silences unrestricted past the
    longest spacer the vocabulary has"""
    home = processed_with_spacers_after_start
    cfg = restricted_cfg(outcomes, after_start_time_s=86_400)
    scorer, _ = scorer_with(cfg, home, tmp_path, monkeypatch)
    expected = silences_by_hand(home, "start_time", 86_400)
    reach = max(hi for _, hi in scorer.spacer_bounds.values())
    expected[np.floor(expected + 1e-9) > reach] = np.nan
    np.testing.assert_allclose(scorer.silence_minutes, expected)


def test_a_silence_past_every_spacer_goes_unrestricted_with_a_warning(
    processed_with_spacers, outcomes, tmp_path, monkeypatch
):
    """the bias would rule out every token there is, spacers included"""
    warned = []
    monkeypatch.setattr(
        cotorra.logger.Logger, "warning", lambda self, msg, *a, **k: warned.append(msg)
    )
    real = scorer_generative.silences
    monkeypatch.setattr(
        scorer_generative,
        "silences",
        lambda *args: np.where(np.arange(len(real(*args))) % 2, 10_000.0, 240.0),
    )
    cfg = restricted_cfg(outcomes, duration_s=86_400)
    scorer, fake = scorer_with(cfg, processed_with_spacers, tmp_path, monkeypatch)
    scorer.save_all()
    biases = [b for p in fake.passes for b in p["first_token_biases"]]
    assert [b is None for b in biases] == [bool(i % 2) for i in range(len(biases))]
    (message,) = warned
    assert message.startswith(f"{len(biases) // 2} patients were silent for longer")


@pytest.mark.parametrize(
    "restriction, error",
    [
        (None, "needs one cut"),
        ({"mode": "hard"}, "needs one cut"),
        ({"duration_s": 86_400, "after_start_time_s": 86_400}, "needs one cut"),
        ({"duration_s": 86_400, "mode": "soft"}, "is hard or prop, not 'soft'"),
        ({"duration_s": 3_600}, "of only"),
    ],
    ids=["empty", "no-cut", "two-cuts", "unknown-mode", "another-cut"],
)
def test_a_silence_restriction_it_cant_honor_is_refused_before_the_engine_starts(
    restriction, error, processed_with_spacers, outcomes, tmp_path, monkeypatch
):
    """the block's presence turns it on, even left empty"""
    cfg = base_scoring_cfg(
        score={"target_tokens": outcomes},
        generation={"silence_restriction": restriction},
    )
    fake = FakeSGLang().install(monkeypatch)
    with pytest.raises(ValueError, match=re.escape(error)):
        scorer_generative.GenerativeScorer(
            scoring_cfg=write_cfg(tmp_path / "scoring.yaml", cfg),
            processed_data_home=processed_with_spacers,
            model_home=tmp_path / "mdl-stub",
        )
    assert fake.started is None


@pytest.mark.parametrize(
    "fixture, targets, error",
    [
        ("processed", [], "cocoa's tokenizer inserts with insert_spacers"),
        ("processed_with_spacers", ["TIME//*"], "can't track the time spacers"),
    ],
    ids=["no-spacers", "spacers-tracked"],
)
def test_a_vocabulary_the_restriction_cant_restrict_is_refused(
    fixture, targets, error, outcomes, tmp_path, monkeypatch, request
):
    """without a spacer to restrict to, or with one an outcome, which quick-sco-re
    refuses to let a first token be"""
    cfg = restricted_cfg(targets + outcomes, duration_s=86_400)
    with pytest.raises(ValueError, match=re.escape(error)):
        scorer_with(cfg, request.getfixturevalue(fixture), tmp_path, monkeypatch)


def test_without_a_silence_restriction_no_first_token_is_biased(
    processed, outcomes, tmp_path, monkeypatch
):
    """the block absent, as from the shipped config and from every config before
    it, each pass is asked for as it was"""
    cfg = base_scoring_cfg(score={"target_tokens": outcomes, "n_samp": 2})
    assert "silence_restriction" not in cfg["generation"]
    scorer, fake = scorer_with(cfg, processed, tmp_path, monkeypatch)
    scorer.save_all()
    assert scorer.silence_minutes is None
    assert fake.passes and all(p["first_token_biases"] is None for p in fake.passes)
