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
from omegaconf import OmegaConf
from quick_sco_re import (
    GenerationConfig,
    PatientResults,
    TrajectoryType,
    create_engine,
    generate_m2_from_m1_trajectory,
    generate_trajectories,
    score_trajectory,
)

from cotorra.configurable import Configurable


def build_generation_config(
    cfg, vocab: dict[str, int], tracked_names: list[str], tracked_ids: list[int]
) -> GenerationConfig:
    """Build a GenerationConfig for inline multi-outcome SCOPE/REACH scoring.

    `tracked_names`/`tracked_ids` are the outcome tokens discovered from
    `score.target_tokens`; all other generation settings come from `cfg.generation`
    (and the shared `cfg.score.end_tokens`/`suppressed_tokens`).
    """
    score_cfg = cfg.score
    gen_cfg = cfg.get("generation", {})

    end_ids: set[int] = {
        tid
        for name in score_cfg.get("end_tokens", [])
        if (tid := vocab.get(name)) is not None
    }
    for prefix in gen_cfg.get("end_tokens", {}).get("prefixes", []):
        end_ids |= {tid for name, tid in vocab.items() if name.startswith(prefix)}

    suppressed_ids = [
        tid
        for name in score_cfg.get("suppressed_tokens", [])
        if (tid := vocab.get(name)) is not None
    ]

    ts = gen_cfg.get("time_stopping", {})
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
        for tok_name, bounds in ts.get("time_token_bounds", {}).items():
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
    trajectories: list, num_patients: int, config: GenerationConfig
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
        scoring_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        model_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
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

        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
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
        gen_cfg = self.cfg.get("generation", {})
        self.methods = gen_cfg.get("methods", ["M1", "M2"])
        engine_cfg = self.cfg.get("engine", {})
        self.chunk_size = engine_cfg.get("patient_chunk_size", 64)

        self.engine = create_engine(
            model_path=str(self.model_home),
            max_len=self.gen_config.max_len,
            use_time_horizon=self.gen_config.max_time is not None
            and self.gen_config.trunc_id is not None,
            mem_fraction=engine_cfg.get("mem_fraction", 0.85),
        )

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

                # Generate the initial M1 chunk
                chunk_m1 = await generate_trajectories(
                    self.engine,
                    self.gen_config,
                    chunk_tokens,
                    ["M1"],
                    stop_at_tracked_events=False,
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

    async def score(self):
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

    def save_all(self, verbose: bool = False):
        res = asyncio.run(self.score())
        (df_res := self.ds.with_columns(pl.from_dict(res))).write_parquet(
            self.output_home
        )

        if verbose:
            self.logger.summarize_preds(df_res, self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = GenerativeScorer()
    self.save_all(verbose=True)
    # breakpoint()
