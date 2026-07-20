#!/usr/bin/env python3

"""
use @lukesolo-ml's implementation of SCORE and REACH to make generative predictions
"""

import asyncio
import collections
import fnmatch
import pathlib
import dataclasses
import numpy as np
import math
import polars as pl
import yaml
import tqdm
from omegaconf import OmegaConf
from quick_sco_re import (
    GenerationConfig, 
    PatientResults,
    TrajectoryType,
    create_engine, 
    generate_trajectories, 
    generate_m2_from_m1_trajectory,
    score_trajectory
)

from cotorra.util import batched

def load_config(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for s in ["cocoa_outputs", "model_path", "output_dir", "generation"]:
        if s not in cfg:
            raise ValueError(f"Config missing required section: '{s}'")

    for k in ["held_out_for_inference", "tokenizer_yaml"]:
        if k not in cfg["cocoa_outputs"]:
            raise ValueError(f"cocoa_outputs missing required key: '{k}'")

    for k in ["max_len", "n_samp"]:
        if k not in cfg["generation"]:
            raise ValueError(f"generation missing required key: '{k}'")
    if "tracked_events" not in cfg["generation"] and "target_event" not in cfg["generation"]:
        raise ValueError("generation config must specify 'tracked_events' (list) or 'target_event' (single)")

    return cfg

def load_vocab(path: str | pathlib.Path) -> dict[str, int]:
    """Load name→id mapping from a cocoa tokenizer.yaml."""
    path = pathlib.Path(path).expanduser().resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    return {str(name): int(tid) for name, tid in data["lookup"].items()}


def ids_with_prefix(vocab: dict[str, int], prefix: str) -> set[int]:
    return {tid for name, tid in vocab.items() if name.startswith(prefix)}



def build_generation_config(
    cfg: dict, vocab: dict[str, int]
) -> tuple[GenerationConfig, bool]:
    gen = cfg["generation"]

    def lookup_required(name: str, role: str) -> int:
        tid = vocab.get(name)
        if tid is None:
            raise ValueError(f"{role} '{name}' not found in vocabulary")
        return tid

    def lookup_optional(name: str, role: str) -> int | None:
        tid = vocab.get(name)
        return tid

    # Tracked events — support both new 'tracked_events' list and legacy 'target_event'
    if "tracked_events" in gen:
        event_names = list(gen["tracked_events"])
        if not event_names:
            raise ValueError("tracked_events cannot be empty")
    else:
        event_names = [gen["target_event"]]

    tracked_event_ids = [lookup_required(n, "tracked_event") for n in event_names]
    primary_id = tracked_event_ids[0]

    # End tokens
    end_ids: set[int] = set()
    end_cfg = gen.get("end_tokens", {})
    for prefix in end_cfg.get("prefixes", []):
        ids = ids_with_prefix(vocab, prefix)
        end_ids |= ids
    for name in end_cfg.get("names", []):
        if (tid := lookup_optional(name, "end token")) is not None:
            end_ids.add(tid)

    # Suppressed tokens
    suppressed_ids = [
        tid
        for name in gen.get("suppressed_tokens", [])
        if (tid := lookup_optional(name, "suppressed token")) is not None
    ]

    # Time stopping
    ts = gen.get("time_stopping") or {}
    trunc_id: int | None = None
    token_id_to_minutes: dict[int, float] = {}
    max_time = None
    time_check_interval = 100

    if ts.get("enabled", False):
        trunc_name = ts.get("trunc_token", "TRUNC")
        trunc_id = lookup_required(trunc_name, "trunc_token")

        max_time = ts.get("max_time_minutes")
        time_check_interval = ts.get("time_check_interval", 100)

        for tok_name, (lo, hi) in ts.get("time_token_bounds", {}).items():
            if (tid := vocab.get(tok_name)) is not None:
                token_id_to_minutes[tid] = math.sqrt(lo * hi)

        if trunc_id in suppressed_ids:
            suppressed_ids.remove(trunc_id)
    score_inline = gen.get("score_inline", False)
    inline_tracked_ids = tracked_event_ids if score_inline else None

    config = GenerationConfig(
        max_len=gen["max_len"],
        n_samp=gen["n_samp"],
        target_event_id=primary_id,
        end_token_ids=end_ids,
        suppressed_ids=suppressed_ids,
        temperature=gen.get("temperature", 1.0),
        trunc_id=trunc_id,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
        time_check_interval=time_check_interval,
        tracked_ids=inline_tracked_ids,
        tracked_names=event_names,
    )
    return config, score_inline

def aggregate_inline_results(
    trajectories: list,
    num_patients: int,
    config: GenerationConfig,
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
            scope = float(traj.scope_estimates[k]) if traj.scope_estimates is not None else 0.0
            results[traj.patient_idx].m1_samples.append(scope)
        else:
            reach = float(traj.reach_estimates[k]) if traj.reach_estimates is not None else 0.0
            results[traj.patient_idx].m2_samples.append(reach)

    return [results[i] for i in range(num_patients)]

class GenerativeScorer():
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
        self.cfg = load_config(scoring_cfg) 
        self.vocab = load_vocab(self.cfg["cocoa_outputs"]["tokenizer_yaml"])
        self.gen_config, score_inline = build_generation_config(self.cfg, self.vocab)
        self.methods = self.cfg["generation"].get("methods", ["M1", "M2"])
        self.output_home = (
            pathlib.Path(output_home).expanduser().resolve()
            if output_home is not None
            else self.processed_data_home
        ) / f"scores-generative-{self.model_home.name}.parquet"
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

        self.engine = create_engine(
            model_path=str(self.model_home),
            max_len=self.cfg.score.max_len,
            use_time_horizon="max_time" in self.cfg.score,  # use if max_time configured
        )

        self.ds = pl.read_parquet(
            self.processed_data_home / "held_out_for_inference.parquet"
        )
        subject_ids = self.ds.select("subject_id").to_series().to_list()
        self.tokens_past = self.ds.select("tokens_past").to_series().to_list()
        max_len = self.cfg.score.max_len
        self.final_tokens = []
        self.final_ids = []
        self.keep_mask = []
        for i, (sid, toks) in enumerate(zip(subject_ids, self.tokens_past)):
            if len(toks) > max_len:
                n_dropped += 1
                self.keep_mask.append(False)
                continue
            self.final_tokens.append(toks)
            self.final_ids.append(sid)
            self.keep_mask.append(True)
        self.grokked_outcome_tokens = [
            x
            for x in self.tkzr_cfg.lookup.keys()
            if any(fnmatch.fnmatch(x, p) for p in self.cfg.score.target_tokens)
        ]
        self.logger.info(
            f"Processed expressions to generate {self.grokked_outcome_tokens=}"
        )
        self.outcome_past_masks: list[np.ndarray] = []
        for evt_name in self.grokked_outcome_tokens:
            past_col = f"{evt_name}_past"
            if past_col in self.ds.columns:
                mask = self.ds[past_col].to_numpy().astype(bool)
            else:
                mask = np.zeros(len(self.final_ids), dtype=bool)
            self.outcome_past_masks.append(mask)

    def aggregate_inline_results(
        trajectories: list,
        num_patients: int,
        config: GenerationConfig,
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
                scope = float(traj.scope_estimates[k]) if traj.scope_estimates is not None else 0.0
                results[traj.patient_idx].m1_samples.append(scope)
            else:
                reach = float(traj.reach_estimates[k]) if traj.reach_estimates is not None else 0.0
                results[traj.patient_idx].m2_samples.append(reach)

        return [results[i] for i in range(num_patients)]
    async def sco_re(self, target_token: str, to_score_tokens: list[int]):
        tid = self.tkzr_cfg.lookup[target_token]
        sco_re_config = GenerationConfig(
            max_len=self.cfg.score.max_len,
            n_samp=self.cfg.score.n_samp,
            target_event_id=tid,
            end_token_ids=set(map(self.tkzr_cfg.lookup.get, self.cfg.score.end_tokens)),
            suppressed_ids=list(
                map(self.tkzr_cfg.lookup.get, self.cfg.score.suppressed_tokens)
            ),
            trunc_id=self.tkzr_cfg.lookup.get(self.cfg.score.trunc_id, -1),
            max_time=self.cfg.score.get("max_time", None),
        )
        outcome_configs = [
            dataclasses.replace(self.gen_config, target_event_id=evt_id)
            for evt_id in self.grokked_outcome_tokens
        ] 
        n_patients = len(self.final_ids)
        # TODO: Make this configurable
        chunk_size = 64
        m1_trajectories: list = []
        n_outcomes = len(self.grokked_outcome_tokens)
        all_results: list[list[PatientResults]] = [[] for _ in range(n_outcomes)]
        per_outcome_m2_tokens: list[int] = [0] * n_outcomes
        per_outcome_m2_count: list[int] = [0] * n_outcomes
        for chunk_start in range(0, n_patients, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_patients)
            chunk_tokens = self.final_tokens[chunk_start:chunk_end]\
            #Generate the intial M1 Chunk
            chunk_m1 = await generate_trajectories(self.engine, self.gen_config, chunk_tokens, ["M1"],
                                                                stop_at_tracked_events=False)
            #Add it to the existing trajectories
            m1_trajectories.extend(chunk_m1)
            # Aggregate results
            outcome_m1_results_list = [
                aggregate_inline_results(chunk_m1, len(chunk_tokens), oc)
                for oc in outcome_configs
            ]
            for k, om1r in enumerate(outcome_m1_results_list):
                all_results[k].extend(om1r)
                # M2 regen within this batch — avoids accumulating all regen
                # requests and triggering them in one giant post-loop gather.
                if "M2" in self.methods and self.grokked_outcome_tokens:
                    chunk_regen: list[tuple] = []
                    for traj in chunk_m1:
                        global_idx = chunk_start + traj.patient_idx
                        if traj.inline_tracked_ids is None or traj.reach_estimates is None:
                            continue
                        # Accumulate timelines that need to be regenerated for each task
                        for k, (evt_id, oc) in enumerate(zip(self.grokked_outcome_tokens, outcome_configs)):
                            # skip regenning this outcome if it ocurred in the pre-fix
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
                        m2_trajs = list(await asyncio.gather(*[
                            generate_m2_from_m1_trajectory(self.engine, oc, traj, self.final_tokens[global_idx])
                            for traj, global_idx, _, oc in chunk_regen
                        ]))
                        scored_m2 = list(await asyncio.gather(*[
                            score_trajectory(self.engine, oc, m2_traj, self.final_tokens[global_idx])
                            for (_, global_idx, _, oc), m2_traj in zip(chunk_regen, m2_trajs)
                        ]))
                        for (traj, global_idx, k, oc), st in zip(chunk_regen, scored_m2):
                            all_results[k][global_idx].m2_samples.append(st.score)
                            per_outcome_m2_tokens[k] += st.trajectory.n_new_tokens or 0
                            per_outcome_m2_count[k] += 1
        return all_results
        
    async def score(self):
        res = collections.defaultdict(lambda: np.nan * np.ones(len(self.tokens_past)))

        for tt in tqdm.tqdm(self.grokked_outcome_tokens, position=0):
            to_score = (
                self.ds.select(~pl.col(f"{tt}_past")).collect().to_series().to_numpy()
            )
            to_score_idx = np.flatnonzero(to_score)
            to_score_tokens = [
                x[
                    -self.cfg.score.max_len + 100 :
                ]  # allow some extra room for generation
                for x, flag in zip(self.tokens_past, to_score)
                if flag
            ]
            for idx_tks in tqdm.tqdm(
                batched(enumerate(to_score_tokens), self.cfg.score.batch_size),
                position=1,
                leave=False,
                total=np.ceil(len(to_score_tokens) / self.cfg.score.batch_size),
            ):
                idx, tks = zip(*idx_tks)
                _, results = await self.sco_re(tt, tks)
                rows = to_score_idx[np.array(idx).ravel()]
                res[f"{tt}_mc_score"][rows] = np.array(
                    [np.mean(r.m0_samples) for r in results]
                )
                res[f"{tt}_scope_score"][rows] = np.array(
                    [np.mean(r.m1_samples) for r in results]
                )
                res[f"{tt}_reach_score"][rows] = np.array(
                    [np.mean(r.m2_samples) for r in results]
                )
        return res

    def save_all(self, verbose: bool = False):
        res = asyncio.run(self.score())
        (df_res := self.ds.with_columns(pl.from_dict(res))).sink_parquet(
            self.output_home
        )

        if verbose:
            self.logger.summarize_preds(df_res, self.grokked_outcome_tokens)


if __name__ == "__main__":
    self = GenerativeScorer()
    self.save_all(verbose=True)
    # breakpoint()
