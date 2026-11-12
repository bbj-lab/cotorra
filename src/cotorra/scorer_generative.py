#!/usr/bin/env python3

"""
use @lukesolo-ml's implementation of SCORE and REACH to make generative predictions
"""

import asyncio
import dataclasses
import fnmatch
import math
import pathlib

import numpy as np
import polars as pl
import tqdm
from omegaconf import DictConfig
from quick_sco_re import (
    GeneratedTrajectory,
    GenerationConfig,
    PatientResults,
    TrajectoryType,
    create_engine,
    generate_m2_from_m1_trajectory,
    generate_trajectories,
    score_trajectory,
    silence_restriction_bias,
)

from cotorra.configurable import Configurable, load_yaml

# the cuts cocoa's winnower makes at a time, by the `threshold` key that sets
# each, and what each counts its seconds from: a timeline's first token, or its
# subject's start_time
CUTS = {"duration_s": "first token", "after_start_time_s": "start_time"}


def spacer_bins(tkzr_cfg: DictConfig) -> dict[str, tuple[float, float]]:
    """
    each time spacer cocoa's tokenizer config sets (`tokenizer.yaml`'s
    `cfg.spacers`, whatever `insert_spacers` says) and the gaps between instants,
    in minutes, that it stands for, `(lo, hi]`: cocoa gives a gap of at least the
    smallest boundary the spacer after as many of the boundaries but the first as
    the gap exceeds, so the last spacer's bin runs on without end
    """
    spacers = dict((tkzr_cfg.get("cfg") or {}).get("spacers") or {})
    if not spacers:
        return {}
    names, edges = list(spacers), list(spacers.values())
    breaks = sorted(edges[1:])
    return {
        f"TIME//{name}": (float(lo), float(hi))
        for name, lo, hi in zip(names, [min(edges), *breaks], [*breaks, math.inf])
    }


def silences(ds: pl.DataFrame, cut: str, seconds: float) -> np.ndarray:
    """
    each of `ds`'s timelines' silence at its cut, in minutes: from the last token
    of its `tokens_past` to the cut, `seconds` after its first token or its
    subject's start_time (`cut`, a key of `CUTS`); NaN for a timeline without
    tokens. Refuses a cut that doesn't fall between a timeline's past and future,
    as one the timelines weren't winnowed at wouldn't, cocoa's winnower leaving
    no past empty
    """
    if cut == "after_start_time_s" and "start_time" not in ds.columns:
        raise ValueError(f"a cut at {cut} needs a start_time column, which ds lacks")
    origin, dtype = (
        (pl.col("times").list.first(), ds.schema["times"].inner)
        if cut == "duration_s"
        else (pl.col("start_time"), ds.schema["start_time"])
    )
    n_past = pl.col("tokens_past").list.len()
    frame = ds.select(
        last=pl.col("times").list.head(n_past).list.last(),
        next=pl.col("times").list.slice(n_past, 1).list.first(),
        cut=origin
        + pl.duration(
            milliseconds=round(float(seconds) * 1000), time_unit=dtype.time_unit
        ),
    )
    astray = frame.select(
        ((pl.col("last") >= pl.col("cut")) | (pl.col("next") < pl.col("cut")))
        .fill_null(False)
        .sum()
    ).item()
    if astray:
        raise ValueError(
            f"a cut {seconds} seconds after each timeline's {CUTS[cut]} ({cut}) "
            f"falls between the past and the future of only {ds.height - astray} "
            f"of {ds.height} timelines; set it as the winnowing config's threshold"
        )
    return (
        frame.select((pl.col("cut") - pl.col("last")).dt.total_milliseconds() / 60_000)
        .to_series()
        .fill_null(math.nan)
        .to_numpy()
    )


def build_generation_config(
    cfg: DictConfig,
    vocab: dict[str, int],
    tracked_names: list[str],
    tracked_ids: list[int],
) -> GenerationConfig:
    """Build a GenerationConfig for inline multi-outcome SCOPE/REACH scoring.

    `tracked_names`/`tracked_ids` are the outcome tokens discovered from
    `score.target_tokens`; all other generation settings come from `cfg.generation`
    (and the shared `cfg.score.end_tokens`/`suppressed_tokens`).
    """
    score_cfg = cfg.score
    # a block or list left empty in the yaml parses to None, which `.get` returns
    # in place of its default, hence the `or`s
    gen_cfg = cfg.get("generation") or {}

    end_ids: set[int] = {
        tid
        for name in (score_cfg.get("end_tokens") or [])
        if (tid := vocab.get(name)) is not None
    }
    for prefix in (gen_cfg.get("end_tokens") or {}).get("prefixes") or []:
        end_ids |= {tid for name, tid in vocab.items() if name.startswith(prefix)}

    suppressed_ids = [
        tid
        for name in (score_cfg.get("suppressed_tokens") or [])
        if (tid := vocab.get(name)) is not None
    ]

    ts = gen_cfg.get("time_stopping") or {}
    trunc_id: int | None = None
    token_id_to_minutes: dict[int, float] = {}
    max_time = None
    time_check_interval = ts.get("time_check_interval", 100)

    if ts.get("enabled", False):
        trunc_name = ts.get("trunc_token", "TRUNC")
        trunc_id = vocab.get(trunc_name)
        if trunc_id is None:
            raise ValueError(
                f"time_stopping.trunc_token '{trunc_name}' not found in vocabulary"
            )
        max_time = ts.get("max_time_minutes")
        for tok_name, bounds in (ts.get("time_token_bounds") or {}).items():
            if (tid := vocab.get(tok_name)) is not None:
                lo, hi = bounds
                token_id_to_minutes[tid] = math.sqrt(lo * hi)
        if trunc_id in suppressed_ids:
            suppressed_ids.remove(trunc_id)

    return GenerationConfig(
        max_len=score_cfg.max_len,
        n_samp=score_cfg.n_samp,
        target_event_id=tracked_ids[0],
        end_token_ids=end_ids,
        suppressed_ids=suppressed_ids,
        temperature=gen_cfg.get("temperature", 1.0),
        trunc_id=trunc_id,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
        time_check_interval=time_check_interval,
        tracked_ids=tracked_ids,
        tracked_names=tracked_names,
    )


def aggregate_inline_results(
    trajectories: list[GeneratedTrajectory], num_patients: int, config: GenerationConfig
) -> list[PatientResults]:
    """Build per-patient results from inline SCOPE/REACH estimates on trajectories.

    Looks up the target_event_id in each trajectory's inline_tracked_ids to
    locate the SCOPE/REACH index. M0 is derived from timeline_terminating_id.
    """
    target_id = config.target_event_id
    results = {i: PatientResults() for i in range(num_patients)}

    for traj in trajectories:
        if traj.inline_tracked_ids is None:
            continue
        try:
            k = traj.inline_tracked_ids.index(target_id)
        except ValueError:
            continue

        if traj.traj_type == TrajectoryType.M1:
            if traj.occurred_flag is not None:
                m0 = bool(traj.occurred_flag[k])
            else:
                m0 = traj.timeline_terminating_id == target_id
            results[traj.patient_idx].m0_samples.append(m0)
            scope = (
                float(traj.scope_estimates[k])
                if traj.scope_estimates is not None
                else 0.0
            )
            results[traj.patient_idx].m1_samples.append(scope)
        else:
            reach = (
                float(traj.reach_estimates[k])
                if traj.reach_estimates is not None
                else 0.0
            )
            results[traj.patient_idx].m2_samples.append(reach)

    return [results[i] for i in range(num_patients)]


class GenerativeScorer(Configurable):
    default_file = "scoring.yaml"

    def __init__(
        self,
        scoring_cfg: pathlib.Path | str | None = None,
        processed_data_home: pathlib.Path | str | None = None,
        model_home: pathlib.Path | str | None = None,
        output_home: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        self.require(processed_data_home=processed_data_home, model_home=model_home)
        super().__init__(scoring_cfg, **kwargs)
        self.processed_data_home, self.model_home = map(
            lambda x: pathlib.Path(x).expanduser().resolve(),
            (processed_data_home, model_home),
        )
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        ) / f"scores-generative-{self.model_home.name}.parquet"

        self.tkzr_cfg = load_yaml(self.processed_data_home / "tokenizer.yaml")
        self.vocab: dict[str, int] = {
            str(name): int(tid) for name, tid in self.tkzr_cfg.lookup.items()
        }

        self.grokked_outcome_tokens = [
            x
            for x in self.tkzr_cfg.lookup.keys()
            if any(fnmatch.fnmatch(x, p) for p in self.cfg.score.target_tokens)
        ]
        if not self.grokked_outcome_tokens:
            raise ValueError(
                "No vocabulary tokens matched score.target_tokens="
                f"{list(self.cfg.score.target_tokens)!r}"
            )
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )
        tracked_ids = [self.vocab[name] for name in self.grokked_outcome_tokens]

        self.gen_config = build_generation_config(
            self.cfg, self.vocab, self.grokked_outcome_tokens, tracked_ids
        )
        gen_cfg = self.cfg.get("generation") or {}
        # an empty `methods:` means the default, but an explicit `[]` stays empty
        methods = gen_cfg.get("methods")
        self.methods = ["M1", "M2"] if methods is None else methods
        engine_cfg = self.cfg.get("engine") or {}
        self.chunk_size = engine_cfg.get("patient_chunk_size", 64)

        self.ds = pl.read_parquet(
            self.processed_data_home / "held_out_for_inference.parquet"
        )
        subject_ids = self.ds.select("subject_id").to_series().to_list()
        self.tokens_past = self.ds.select("tokens_past").to_series().to_list()

        self.overflow = self.cfg.get("prompt_overflow", "truncate_left")
        max_len = self.gen_config.max_len
        self.final_tokens = []
        self.final_ids = []
        self.keep_mask = []
        n_dropped = 0
        n_truncated = 0
        for sid, toks in zip(subject_ids, self.tokens_past):
            if len(toks) > max_len:
                if self.overflow == "drop":
                    n_dropped += 1
                    self.keep_mask.append(False)
                    continue
                elif self.overflow == "truncate_left":
                    toks = toks[-max_len:]
                    n_truncated += 1
            self.final_tokens.append(toks)
            self.final_ids.append(sid)
            self.keep_mask.append(True)
        if n_dropped:
            self.logger.info(
                f"Dropped {n_dropped} patients with prompts > {max_len} tokens"
            )
        if n_truncated:
            self.logger.info(
                f"Left-truncated {n_truncated} prompts to {max_len} tokens"
            )

        kept = np.array(self.keep_mask)
        self.outcome_past_masks: list[np.ndarray] = []
        for evt_name in self.grokked_outcome_tokens:
            past_col = f"{evt_name}_past"
            if past_col in self.ds.columns:
                mask = self.ds[past_col].to_numpy().astype(bool)[kept]
            else:
                mask = np.zeros(len(self.final_ids), dtype=bool)
            self.outcome_past_masks.append(mask)

        # each kept patient's silence at the cut, in minutes, which the first
        # token of its trajectories has to cover; None leaves them all free
        self.silence_minutes: np.ndarray | None = None
        if "silence_restriction" in gen_cfg:  # its presence turns it on
            self.restrict_silences(gen_cfg.silence_restriction, kept)

        # started last, once the config and the data have been checked
        self.engine = create_engine(
            model_path=str(self.model_home),
            max_len=self.gen_config.max_len,
            use_time_horizon=self.gen_config.max_time is not None
            and self.gen_config.trunc_id is not None,
            mem_fraction=engine_cfg.get("mem_fraction", 0.85),
        )

    def restrict_silences(
        self, restriction: DictConfig | None, kept: np.ndarray
    ) -> None:
        """
        readies `generation.silence_restriction`: a timeline cut at a time ends at
        its last event before the cut, so its next event comes at least as long
        after that as the cut does, and a trajectory's first token has to be a
        time spacer whose bin reaches that far, as quick-sco-re's first-token
        restriction has it
        """
        restriction = restriction or {}
        cuts = [cut for cut in CUTS if cut in restriction]
        if len(cuts) != 1:
            raise ValueError(
                "generation.silence_restriction needs one cut, as the winnowing "
                f"config's threshold sets it, from {list(CUTS)}; it has {cuts}"
            )
        (cut,) = cuts
        self.silence_mode = restriction.get("mode", "hard")
        if self.silence_mode not in ("hard", "prop"):
            raise ValueError(
                "generation.silence_restriction.mode is hard or prop, "
                f"not {self.silence_mode!r}"
            )
        bins = spacer_bins(self.tkzr_cfg)
        self.spacer_bounds = {
            self.vocab[name]: bounds
            for name, bounds in bins.items()
            if name in self.vocab
        }
        if not self.spacer_bounds:
            raise ValueError(
                "generation.silence_restriction needs time spacers in the "
                "vocabulary, which cocoa's tokenizer inserts with insert_spacers"
            )
        # quick-sco-re refuses a first token that may be a tracked outcome
        if spacers := [t for t in self.grokked_outcome_tokens if t in bins]:
            raise ValueError(
                f"generation.silence_restriction can't track the time spacers {spacers}"
            )
        # cocoa inserts no spacer for a shorter gap, so nothing is restricted below
        self.min_spacer_minutes = min(lo for lo, _ in bins.values())
        self.silence_minutes = silences(self.ds, cut, restriction[cut])[kept]
        # past the longest spacer the vocabulary has, the restriction would
        # forbid every token, so those patients go unrestricted, as without it
        # (quick-sco-re floors a silence to whole minutes)
        reach = max(hi for _, hi in self.spacer_bounds.values())
        floored = np.floor(self.silence_minutes + 1e-9)
        if (beyond := floored > reach).any():
            self.logger.warning(
                f"{beyond.sum()} patients were silent for longer than the longest "
                f"time spacer in the vocabulary reaches ({reach:.0f} minutes), so "
                "the first tokens of their trajectories go unrestricted"
            )
            self.silence_minutes[beyond] = math.nan
        restricted = ~beyond & (floored >= self.min_spacer_minutes)
        median = np.median(self.silence_minutes[restricted]) if restricted.any() else 0
        self.logger.info(
            f"Restricting the first token of {restricted.sum()} of "
            f"{len(self.silence_minutes)} patients' trajectories to time spacers "
            f"covering their silence at the cut ({cut}={restriction[cut]}, "
            f"mode={self.silence_mode}), a median {median:.0f} minutes among them"
        )

    def first_token_biases(self, start: int, end: int) -> list | None:
        """
        the first-token logit bias `generation.silence_restriction` sets for each
        kept patient from `start` to `end`, None for one it leaves free (silent
        for less than the shortest spacer's gap), or None when it's off; made a
        chunk at a time, as each forbids nearly the whole vocabulary
        """
        if self.silence_minutes is None:
            return None
        return [
            silence_restriction_bias(
                float(minutes),
                self.spacer_bounds,
                self.vocab.values(),
                mode=self.silence_mode,
                min_minutes=self.min_spacer_minutes,
            )
            for minutes in self.silence_minutes[start:end]
        ]

    async def sco_re(self) -> list[list[PatientResults]]:
        """Run one inline generation pass over the whole cohort, tracking every
        outcome in `self.grokked_outcome_tokens` simultaneously."""
        outcome_ids = self.gen_config.tracked_ids
        outcome_configs = [
            dataclasses.replace(self.gen_config, target_event_id=evt_id)
            for evt_id in outcome_ids
        ]
        n_patients = len(self.final_ids)
        n_outcomes = len(self.grokked_outcome_tokens)
        all_results: list[list[PatientResults]] = [[] for _ in range(n_outcomes)]
        per_outcome_m2_tokens: list[int] = [0] * n_outcomes
        per_outcome_m2_count: list[int] = [0] * n_outcomes

        with tqdm.tqdm(total=n_patients, desc="Generating") as pbar:
            for chunk_start in range(0, n_patients, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, n_patients)
                chunk_tokens = self.final_tokens[chunk_start:chunk_end]

                # Generate the initial M1 chunk; M2's regeneration below keeps
                # each trajectory's restricted first token, before any outcome
                chunk_m1 = await generate_trajectories(
                    self.engine,
                    self.gen_config,
                    chunk_tokens,
                    ["M1"],
                    stop_at_tracked_events=False,
                    first_token_biases=self.first_token_biases(chunk_start, chunk_end),
                )

                outcome_m1_results_list = [
                    aggregate_inline_results(chunk_m1, len(chunk_tokens), oc)
                    for oc in outcome_configs
                ]
                for k, om1r in enumerate(outcome_m1_results_list):
                    all_results[k].extend(om1r)

                # M2 regen within this batch — avoids accumulating all regen
                # requests and triggering them in one giant post-loop gather.
                if "M2" in self.methods:
                    chunk_regen: list[tuple] = []
                    for traj in chunk_m1:
                        global_idx = chunk_start + traj.patient_idx
                        if (
                            traj.inline_tracked_ids is None
                            or traj.reach_estimates is None
                        ):
                            continue
                        for k, (evt_id, oc) in enumerate(
                            zip(outcome_ids, outcome_configs)
                        ):
                            # skip regenning this outcome if it occurred in the prefix
                            if self.outcome_past_masks[k][global_idx]:
                                continue
                            try:
                                ki = traj.inline_tracked_ids.index(evt_id)
                            except ValueError:
                                continue
                            # Prefer occurred_flag (O(1), correct for time-truncated
                            # trajectories where output_ids is already trimmed).
                            if traj.occurred_flag is not None:
                                evt_occurred = bool(traj.occurred_flag[ki])
                            else:
                                evt_occurred = evt_id in traj.output_ids
                            if evt_occurred:
                                chunk_regen.append((traj, global_idx, k, oc))
                            else:
                                all_results[k][global_idx].m2_samples.append(
                                    float(traj.reach_estimates[ki])
                                )

                    if chunk_regen:
                        m2_trajs = list(
                            await asyncio.gather(
                                *[
                                    generate_m2_from_m1_trajectory(
                                        self.engine,
                                        oc,
                                        traj,
                                        self.final_tokens[global_idx],
                                    )
                                    for traj, global_idx, _, oc in chunk_regen
                                ]
                            )
                        )
                        scored_m2 = list(
                            await asyncio.gather(
                                *[
                                    score_trajectory(
                                        self.engine,
                                        oc,
                                        m2_traj,
                                        self.final_tokens[global_idx],
                                    )
                                    for (_, global_idx, _, oc), m2_traj in zip(
                                        chunk_regen, m2_trajs
                                    )
                                ]
                            )
                        )
                        for (traj, global_idx, k, oc), st in zip(
                            chunk_regen, scored_m2
                        ):
                            all_results[k][global_idx].m2_samples.append(st.score)
                            per_outcome_m2_tokens[k] += st.trajectory.n_new_tokens or 0
                            per_outcome_m2_count[k] += 1

                pbar.update(len(chunk_tokens))

        return all_results

    async def score(self) -> dict[str, np.ndarray]:
        all_results = await self.sco_re()
        orig_idx = np.flatnonzero(self.keep_mask)
        n = len(self.tokens_past)
        res = {}

        for k, tt in enumerate(tqdm.tqdm(self.grokked_outcome_tokens)):
            results_k = all_results[k]
            # Patients where this outcome already occurred in the past are
            # excluded from that outcome's scores (NaN).
            for i, is_past in enumerate(self.outcome_past_masks[k]):
                if is_past:
                    results_k[i] = PatientResults()

            m0 = np.nan * np.ones(n)
            m1 = np.nan * np.ones(n)
            m2 = np.nan * np.ones(n)
            m0[orig_idx] = [
                np.mean(r.m0_samples) if r.m0_samples else np.nan for r in results_k
            ]
            m1[orig_idx] = [
                np.mean(r.m1_samples) if r.m1_samples else np.nan for r in results_k
            ]
            m2[orig_idx] = [
                np.mean(r.m2_samples) if r.m2_samples else np.nan for r in results_k
            ]

            res[f"{tt}_mc_score"] = m0
            res[f"{tt}_scope_score"] = m1
            res[f"{tt}_reach_score"] = m2

        return res

    def save_all(self, verbose: bool = False) -> None:
        res = asyncio.run(self.score())
        (df_res := self.ds.with_columns(pl.from_dict(res))).write_parquet(
            self.output_home
        )

        if verbose:  # lazy, since `summarize_preds` collects what it selects
            self.logger.summarize_preds(df_res.lazy(), self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = GenerativeScorer()
    self.save_all(verbose=True)
    # breakpoint()
