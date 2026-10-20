#!/usr/bin/env python3

"""
a stand-in for `quick_sco_re`, which `cotorra.scorer_generative` imports at
load: the `[gen]` extra, installed from git, which itself imports sglang at load
and so can't be installed where CI runs. The data structures are copied from
its `structures.py` (github.com/lukesolo-ml/SCOPE_REACH_optimized_inference at
1b0af76), and `test_scorer_generative` holds them, and the signatures below, to
the real package's wherever that's installed; the engine and the generation are
placeholders, for a test to swap for fakes of its own
"""

import dataclasses
import enum

import numpy as np


class TrajectoryType(enum.Enum):
    """M1 lets the target event be generated, for the M0 and M1 (SCOPE)
    estimators; M2 forbids it, for M2 (REACH)"""

    M1 = "m1"
    M2 = "m2"


@dataclasses.dataclass
class GenerationConfig:
    max_len: int
    n_samp: int
    target_event_id: int
    end_token_ids: set[int]
    suppressed_ids: list[int] = dataclasses.field(default_factory=list)
    temperature: float = 1.0
    trunc_id: int | None = None
    token_id_to_minutes: dict[int, float] = dataclasses.field(default_factory=dict)
    max_time: float | None = None
    time_check_interval: int = 100
    tracked_ids: list[int] | None = None
    tracked_names: list[str] | None = None


@dataclasses.dataclass
class GeneratedTrajectory:
    patient_idx: int
    sample_idx: int
    traj_type: TrajectoryType
    prompt_len: int
    output_ids: list[int]
    timeline_terminating_id: int | None
    was_time_truncated: bool = False
    truncation_idx: int | None = None
    scope_estimates: np.ndarray | None = None
    reach_estimates: np.ndarray | None = None
    occurred_flag: np.ndarray | None = None
    occurred_index: np.ndarray | None = None
    inline_tracked_ids: list[int] | None = None
    inline_tracked_name: list[str] | None = None
    n_new_tokens: int | None = None


@dataclasses.dataclass
class ScoredTrajectory:
    trajectory: GeneratedTrajectory
    score: float


@dataclasses.dataclass
class PatientResults:
    m0_samples: list[bool] = dataclasses.field(default_factory=list)
    m1_samples: list[float] = dataclasses.field(default_factory=list)
    m2_samples: list[float] = dataclasses.field(default_factory=list)


class NeedsSGLang(RuntimeError):
    """what a placeholder raises, a test having left it unswapped"""


def create_engine(model_path, max_len, use_time_horizon=False, mem_fraction=0.8):
    raise NeedsSGLang("`create_engine` starts an sglang engine; swap in a fake")


async def generate_trajectories(
    engine, config, patient_tokens, methods=["M1", "M2"], stop_at_tracked_events=True
):
    raise NeedsSGLang("`generate_trajectories` samples with sglang; swap in a fake")


async def generate_m2_from_m1_trajectory(engine, config, m1_traj, prompt_tokens):
    raise NeedsSGLang("`generate_m2_from_m1_trajectory` samples with sglang")


async def score_trajectory(engine, config, traj, prompt_tokens, truncate_again=False):
    raise NeedsSGLang("`score_trajectory` scores with sglang; swap in a fake")
