#!/usr/bin/env python3

"""
train a model
"""

import fnmatch
import json
import os
import pathlib

import torch as t
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer as t_Trainer

from cotorra.basis_blended import (
    BasisBlendedCausalLM,
    BasisBlendedConfig,
    build_basis_vocab,
    build_gaussian_vocab,
)
from cotorra.configurable import Configurable
from cotorra.loader import Loader
from cotorra.loss import Loss


class TrainerWithCustomLoss(t_Trainer):
    # the value-side batch entries: collated for the loss, and accepted by
    # BasisBlendedCausalLM.forward but by no plain HF model.
    VALUE_KEYS = ("category_ids", "ranks", "rank_widths")

    def __init__(self, compute_loss_func=None, value_kwargs_to_model=True, **kwargs):
        super().__init__(**kwargs)
        self.compute_loss_func = compute_loss_func
        # False for the decile baseline under baseline_exact_ranks, where
        # `ranks` is collated for the LOSS only -- a plain LlamaForCausalLM
        # would raise TypeError on the unexpected kwarg.
        self.value_kwargs_to_model = value_kwargs_to_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.compute_loss_func is not None:
            labels = inputs.get("labels")
            # rank_widths goes to BOTH sides now: the model needs it under
            # interval_mixture_weights (the input blend) and the loss under
            # interval_nll_loss (scoring). forward() ignores it otherwise,
            # so this is safe when only one -- or neither -- is enabled.
            extra = {k: inputs[k] for k in self.VALUE_KEYS if k in inputs}
            model_inputs = (
                inputs
                if self.value_kwargs_to_model
                else {k: v for k, v in inputs.items() if k not in self.VALUE_KEYS}
            )
            outputs = model(**model_inputs)
            # the loss drops its EXTRA numeric term at eval (see
            # Loss.custom_loss): eval_loss then measures the primary
            # objective alone, so it is comparable across arms and cannot be
            # gamed by choosing a smaller numeric_loss_weight.
            loss = self.compute_loss_func(
                outputs, labels, training=model.training, **extra
            )
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)


class Trainer(Configurable):
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    default_file = "training.yaml"

    def __init__(
        self,
        training_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        **kwargs,
    ):
        super().__init__(training_cfg, **kwargs)

        self.processed_data_home, self.output_home = map(
            lambda p: pathlib.Path(p).expanduser().resolve(),
            [processed_data_home, output_home],
        )

        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

        # single source of truth for the collapsed basis vocabulary -- built once
        # here and reused by both model_init (below) and Loss, so the model's
        # embedding table and the loss's target category/basis-id assignment can
        # never disagree (see fuzzy_token_planning.md, "The collapsed basis
        # vocabulary")
        self.basis_vocab = (
            (
                build_gaussian_vocab(self.tkzr_cfg)
                # film_value_embed addresses one slot per category too: its
                # curve is gamma_c * MLP(v) + beta_c, with no anchors to
                # address, so a k-wide vocabulary would leave most slots
                # unreachable.
                if self.cfg.basis_blended_tokens.get("numerical_basis_model", False)
                or self.cfg.basis_blended_tokens.get("film_value_embed", False)
                else build_basis_vocab(self.tkzr_cfg, self.cfg.basis_blended_tokens.k)
            )
            if "basis_blended_tokens" in self.cfg
            else None
        )
        if self.basis_vocab is not None:
            self._raw_to_category_t = t.tensor(
                self.basis_vocab["raw_to_category"], dtype=t.long
            )
            self._raw_to_collapsed_t = t.tensor(
                self.basis_vocab["raw_to_collapsed"], dtype=t.long
            )

        self.loss = (
            Loss(self.cfg, self.tkzr_cfg, basis_vocab=self.basis_vocab).custom_loss
            if self.cfg.custom_loss
            else None
        )
        self.run_name = self.cfg.get("run_name", self.cfg.wandb.get("run_name", ""))
        self.loader = Loader(training_cfg, self.processed_data_home)
        # contemporaneous_shuffle: seeded so a run is reproducible, but drawn
        # fresh per batch so the same chunk gets a different within-timestamp
        # order each time it is presented. See _contemporaneous_perm.
        self._shuffle_rng = t.Generator().manual_seed(
            int(self.cfg.get("contemporaneous_shuffle_seed", 0))
        )

        self.trainer = TrainerWithCustomLoss(
            model_init=self.model_init,
            data_collator=self.collate_fn,
            compute_loss_func=self.loss,
            value_kwargs_to_model=self.basis_vocab is not None,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_home), **self.cfg.training_args
            ),
        )
        self.model = self.trainer.model

        os.environ["WANDB_PROJECT"] = self.cfg.get("wandb", {}).get(
            "project", "cotorra"
        )
        os.environ["WANDB_NAME"] = self.cfg.get("wandb", {}).get("run_name", "cotorra")

    def _k_per_category(self):
        """
        Per-category slot budget k_c, from
        `basis_blended_tokens.k_per_category_file` -- a JSON mapping category
        name (as it appears in basis_vocab["categories"]) to k_c. Absent =
        None = every category uses all k, the uniform-k behaviour.

        Categories missing from the file keep the full k, so a file only has
        to name the ones it wants to cap. Values are clamped to [1, k]: the
        file is generated from a fixed k_max (see
        experiments/*/make_k_per_category.py) and clamping means a config
        that lowers k afterwards stays valid instead of failing an assert
        deep in the model constructor.
        """
        cfg_bb = self.cfg.get("basis_blended_tokens", {})
        path = cfg_bb.get("k_per_category_file", None)
        if not path or self.basis_vocab is None:
            return None
        with open(pathlib.Path(path).expanduser()) as f:
            by_name = json.load(f)
        k = int(self.basis_vocab["k"])
        cats = self.basis_vocab["categories"]
        if cfg_bb.get("fixed_uniform_component", False):
            # slot 0 is the pinned uniform, so a category with m unique
            # realizations gets m tunable slots + 1 uniform = m+1, and the
            # tunable budget is capped at k-1 rather than k.
            out = [min(max(int(by_name.get(c, k)), 1), k - 1) + 1 for c in cats]
        else:
            out = [min(max(int(by_name.get(c, k)), 1), k) for c in cats]
        capped = sum(1 for kc in out if kc < k)
        self.logger.info(
            f"k_per_category: {capped}/{len(out)} categories capped below "
            f"k={k}; total slots {sum(out)} vs {k * len(out)} uniform"
        )
        return out

    def model_init(self):
        if self.basis_vocab is not None:
            inner_cfg = AutoConfig.from_pretrained(
                self.cfg.model.model_name,
                vocab_size=self.basis_vocab["vocab_size"],
                bos_token_id=self.basis_vocab["bos_token_id"],
                eos_token_id=self.basis_vocab["eos_token_id"],
                tie_word_embeddings=True,  # required -- see fuzzy_token_planning.md
                **self.cfg.model.model_args,
            )
            basis_cfg = BasisBlendedConfig(
                base_model_type=inner_cfg.model_type,
                base_config=inner_cfg.to_dict(),
                train_beta_params=self.cfg.basis_blended_tokens.get(
                    "train_beta_params", True
                ),
                train_importance_scale=self.cfg.basis_blended_tokens.get(
                    "train_importance_scale", True
                ),
                train_category_embed=self.cfg.basis_blended_tokens.get(
                    "train_category_embed", True
                ),
                numerical_basis_model=self.cfg.basis_blended_tokens.get(
                    "numerical_basis_model", False
                ),
                interval_mixture_weights=self.cfg.basis_blended_tokens.get(
                    "interval_mixture_weights", False
                ),
                k_per_category=self._k_per_category(),
                share_basis_embed_init=self.cfg.basis_blended_tokens.get(
                    "share_basis_embed_init", False
                ),
                # MUST be forwarded: Loss reads component_family straight off
                # the YAML, so omitting it here builds a Beta model whose
                # parameters the loss then reads as (mu, sigma) -- a silent
                # mismatch that trains without error and produces nonsense.
                component_family=self.cfg.basis_blended_tokens.get(
                    "component_family", "beta"
                ),
                untie_output_basis=self.cfg.basis_blended_tokens.get(
                    "untie_output_basis", False
                ),
                # MUST be forwarded, for the same reason component_family
                # must: omitting it silently trains the CH ablation as plain
                # tied-head TCH, and nothing in the logs would differ.
                untie_continuous_head=self.cfg.basis_blended_tokens.get(
                    "untie_continuous_head", False
                ),
                # same MUST-forward reasoning: omitting this trains level 1
                # from the discrete anchor sum, i.e. plain TCH, and the only
                # visible difference would be the loss value.
                continuous_category_logits=self.cfg.basis_blended_tokens.get(
                    "continuous_category_logits", False
                ),
                # MUST-forward: without it the arm silently trains as plain
                # TCH, scoring the quadrature instead of the value vocabulary.
                value_vocab_curve=self.cfg.basis_blended_tokens.get(
                    "value_vocab_curve", False
                ),
                # same MUST-forward reasoning: without these the xVal arm
                # silently trains as the per-category Gaussian-head
                # reproduction it is meant to be compared against.
                xval_head=self.cfg.basis_blended_tokens.get("xval_head", False),
                film_value_embed=self.cfg.basis_blended_tokens.get(
                    "film_value_embed", False
                ),
                film_hidden=self.cfg.basis_blended_tokens.get("film_hidden", 64),
                xval_value_scale=self.cfg.basis_blended_tokens.get(
                    "xval_value_scale", 1.0
                ),
                beta_mu_kappa_param=self.cfg.basis_blended_tokens.get(
                    "beta_mu_kappa_param", False
                ),
                fixed_uniform_component=self.cfg.basis_blended_tokens.get(
                    "fixed_uniform_component", False
                ),
                value_vocab_file=self.cfg.basis_blended_tokens.get(
                    "value_vocab_file", None
                ),
                decoupled_magnitude=self.cfg.basis_blended_tokens.get(
                    "decoupled_magnitude", False
                ),
                magnitude_target=self.cfg.basis_blended_tokens.get(
                    "magnitude_target", None
                ),
                tied_continuous_head=self.cfg.basis_blended_tokens.get(
                    "tied_continuous_head", False
                ),
                # derived, not a separate key: the tied head's CRPS exists only
                # to be added at interval_crps_weight (see loss.py).
                #
                # NB this is gated on the weight's VALUE, which is why
                # interval_crps_weight cannot go in an Optuna search space:
                # transformers builds the model (call_model_init) before it
                # runs hp_space (_hp_search_setup), so a searched weight is
                # not yet known here and every trial would get a model that
                # never computed the CRPS. numeric_loss_weight below is the
                # tunable knob and is gated on PRESENCE instead.
                continuous_crps=bool(
                    self.cfg.basis_blended_tokens.get("tied_continuous_head", False)
                )
                and (
                    float(
                        self.cfg.basis_blended_tokens.get("interval_crps_weight", 0.0)
                    )
                    > 0
                    or self.cfg.get("numeric_loss", None) == "crps"
                ),
                # presence-gated, so the statistic exists for every trial of a
                # numeric_loss_weight search. Costs two extra reductions per
                # forward when numeric_loss is set and nothing at all when it
                # is not.
                continuous_moments=bool(
                    self.cfg.basis_blended_tokens.get("tied_continuous_head", False)
                )
                and self.cfg.get("numeric_loss", None) in ("was", "mse"),
                continuous_crps_z=bool(
                    self.cfg.basis_blended_tokens.get("tied_continuous_head", False)
                )
                and self.cfg.get("numeric_loss", None) == "crps_z",
                value_quantile_file=self.cfg.basis_blended_tokens.get(
                    "value_quantile_file", None
                ),
                value_quantile_points=self.cfg.basis_blended_tokens.get(
                    "value_quantile_points", 0
                ),
                poly_curve_basis=self.cfg.basis_blended_tokens.get(
                    "poly_curve_basis", False
                ),
                bspline_weights=self.cfg.basis_blended_tokens.get(
                    "bspline_weights", False
                ),
                bspline_degree=self.cfg.basis_blended_tokens.get("bspline_degree", 3),
                continuous_quad_panels=self.cfg.basis_blended_tokens.get(
                    "continuous_quad_panels", 16
                ),
                continuous_quad_nodes=self.cfg.basis_blended_tokens.get(
                    "continuous_quad_nodes", 8
                ),
                continuous_interval_panels=self.cfg.basis_blended_tokens.get(
                    "continuous_interval_panels", 8
                ),
                continuous_logit_bound=self.cfg.basis_blended_tokens.get(
                    "continuous_logit_bound", 8.0
                ),
                continuous_interval_nodes=self.cfg.basis_blended_tokens.get(
                    "continuous_interval_nodes", 8
                ),
                **self.basis_vocab,
            )
            mdl = BasisBlendedCausalLM(basis_cfg)
        else:
            conf_param = dict(
                vocab_size=len(self.tkzr_cfg.lookup),
                bos_token_id=self.tkzr_cfg.lookup.BOS,
                eos_token_id=self.tkzr_cfg.lookup.EOS,
            )
            config = AutoConfig.from_pretrained(
                self.cfg.model.model_name, **conf_param, **self.cfg.model.model_args
            )
            mdl = AutoModelForCausalLM.from_config(config)
        self.logger.info(
            "Loaded model {name} with {num} params ({dtype}).".format(
                name=self.cfg.model.model_name,
                num=sum(p.numel() for p in mdl.parameters()),
                dtype=next(mdl.parameters()).dtype,
            )
        )

        return mdl

    def _order_keys(self):
        """
        Per-token-id (priority, is_outcome), indexed by RAW cocoa token id.

        `priority` is what cocoa's tokenizer sorts contemporaneous events by:
        Tokenizer.get_priority ranks the code TYPE -- the part of the code
        before "//" -- against cfg.ordering, so it is a pure function of the
        token string and can be rebuilt here from the saved lookup rather
        than carried through the processed data (which drops it).

        `is_outcome` marks the tokens an outcome is read off, matched by the
        same fnmatch patterns the winnower uses (contemporaneous_shuffle_
        outcome_tokens, default = cocoa's outcome_tokens list).

        Cached in __dict__ rather than registered, so it never becomes part
        of any state_dict or moves with the module.
        """
        cached = self.__dict__.get("_order_keys_cache")
        if cached is not None:
            return cached
        lookup = self.tkzr_cfg.lookup
        ordering = [str(x) for x in self.tkzr_cfg.cfg.ordering]
        prio_of = {ct: i for i, ct in enumerate(ordering)}
        unknown = len(ordering)  # e.g. UNK, whose prefix is in no ordering
        n = max(int(v) for v in lookup.values()) + 1
        prio = t.full((n,), unknown, dtype=t.long)
        outc = t.zeros(n, dtype=t.bool)
        pats = [
            str(x)
            for x in (self.cfg.get("contemporaneous_shuffle_outcome_tokens") or [])
        ]
        for name, tid in lookup.items():
            name = str(name)
            prio[int(tid)] = prio_of.get(name.split("//")[0], unknown)
            if any(fnmatch.fnmatch(name, pat) for pat in pats):
                outc[int(tid)] = True
        self.logger.info(
            f"order keys: {int(outc.sum())} outcome token ids over {len(pats)} "
            f"pattern(s); {len(ordering)} priority levels"
        )
        self.__dict__["_order_keys_cache"] = (prio, outc)
        return prio, outc

    def _contemporaneous_perm(self, input_ids, batch):
        """
        A permutation that reshuffles simultaneous events within each
        timestamp, independently every time a batch is collated.

        WHY. Events sharing a timestamp have no true order, but the tokenizer
        must emit one. Emitting a DETERMINISTIC one (cocoa sorts by
        time, priority, code, ...) hands the model a monotone rule it can
        learn for free: measured on iv3, that is worth 1.12 nats/token of
        permutation entropy, and it made eval_loss incomparable with every
        earlier build. Emitting a nondeterministic one is worse -- it was the
        original bug, moving held-out AUC by ~0.0008 between rebuilds.

        Reshuffling per presentation gets both: the tokenized file stays
        deterministic and reproducible, while the model sees a different
        order each epoch and so cannot learn one. Each panel entry appears in
        every slot, so it must predict the first entry of a block from
        history alone AND later entries given earlier ones, rather than
        relying on position. It also pushes the representation toward
        order-invariance, which is the thing that made extraction sensitive
        to the tokenizer's ordering choice in the first place.

        Blocks are runs of equal s_elapsed (`times` is not carried into the
        dataset; see Loader's select_columns). batched_iter concatenates
        ACROSS subjects, and s_elapsed resets to 0 per subject, so a
        single-timestamp patient could otherwise merge with the next
        patient's time-0 events -- BOS therefore always opens a new block.
        """
        se = t.stack([x["s_elapsed"] for x in batch]).to(t.float64)
        new_blk = se != se.roll(1, dims=1)
        new_blk[:, 0] = True
        # BOS/EOS are structural, not observations: they must stay at their
        # subject's boundary, so give each its OWN block. Opening a block at
        # them is not enough -- a BOS that merely starts a block is still
        # inside it and can be shuffled behind its own subject's first event.
        # Closing the block after them too makes each a singleton, which no
        # permutation can move. This also fixes the cross-subject hazard:
        # batched_iter concatenates subjects and s_elapsed resets to 0, so a
        # single-timestamp patient would otherwise merge with the next
        # patient's time-0 events.
        struct = (input_ids == self.bos_token_id) | (input_ids == self.eos_token_id)
        new_blk |= struct
        new_blk |= struct.roll(1, dims=1)
        new_blk[:, 0] = True
        blk = new_blk.cumsum(1)

        # Rank WITHIN a time block. Two refinements over shuffling the whole
        # block uniformly, which is what the first version did:
        #
        # respect_priority -- cocoa sorts contemporaneous events by
        #   (time, priority, code, numeric_value, text_value). Only the tail
        #   of that key is arbitrary: `priority` ranks the code TYPE against
        #   cfg.ordering and is real clinical structure, while code/
        #   numeric_value/text_value is alphabetical tie-breaking with no
        #   meaning. Shuffling the whole block destroys both. Keeping
        #   priority as a rank and randomising only inside it removes exactly
        #   the arbitrary part.
        #
        # outcomes_first -- put the tokens an outcome is read off at the
        #   FRONT of their time block, permuted among themselves, ahead of
        #   every priority level. Otherwise LABEL (priority 22 of 24) is
        #   predicted last, i.e. after the model has already seen every
        #   contemporaneous event, and the gradient teaches "predict
        #   LABEL//pressor_init given the pressor you just saw at this same
        #   timestamp" -- trivial, and it puts none of the predictive work
        #   into the representation of history. Front-loading them forces
        #   that work into the history representation, which is the thing
        #   the downstream probe actually reads.
        #
        # ord_key 0 is reserved for outcome tokens so they sort ahead of
        # priority 0; everything else is 1 + priority.
        prio, outc = self._order_keys()
        dev = input_ids.device
        if self.cfg.get("contemporaneous_shuffle_respect_priority", True):
            ord_key = 1 + prio.to(dev)[input_ids]
        else:
            ord_key = t.ones_like(input_ids)
        if self.cfg.get("contemporaneous_shuffle_outcomes_first", True):
            ord_key = t.where(outc.to(dev)[input_ids], 0, ord_key)

        # block dominates ord_key dominates noise: with m = max(ord_key) + 1,
        # block b spans [b*m, b*m + m) since ord_key <= m-1 and noise < 1, so
        # no token can cross a block or an ord_key boundary.
        m = int(ord_key.max()) + 1
        key = (
            blk.to(t.float64) * m
            + ord_key.to(t.float64)
            + t.rand(se.shape, generator=self._shuffle_rng, dtype=t.float64).to(dev)
        )
        return key.argsort(dim=1)

    def collate_fn(self, batch):
        input_ids = t.stack([x["input_ids"] for x in batch])
        perm = None
        if self.cfg.get("contemporaneous_shuffle", False):
            perm = self._contemporaneous_perm(input_ids, batch)
            input_ids = input_ids.gather(1, perm)
        labels = input_ids
        extra = {}

        if self.basis_vocab is not None:
            # must read category_ids/ranks off the *raw* cocoa token ids before
            # input_ids gets remapped into collapsed-vocab space below
            extra["category_ids"] = self._raw_to_category_t[input_ids]
            rank_column = self.cfg.basis_blended_tokens.get(
                "rank_column", "exact_ranks"
            )
            _ranks = t.stack([x[rank_column] for x in batch]).to(t.float32)
            extra["ranks"] = _ranks if perm is None else _ranks.gather(1, perm)
            if (
                self.cfg.basis_blended_tokens.get("interval_nll_loss", False)
                or self.cfg.basis_blended_tokens.get("interval_mixture_weights", False)
                # the FiLM arm scores the same rank interval through
                # Loss.film_continuous_loss, which is not gated on
                # interval_nll_loss (that flag selects a branch of the ANCHOR
                # loss it does not use)
                or self.cfg.basis_blended_tokens.get("film_value_embed", False)
            ):
                _w = t.stack([x["exact_rank_widths"] for x in batch]).to(t.float32)
                extra["rank_widths"] = _w if perm is None else _w.gather(1, perm)
            input_ids = self._raw_to_collapsed_t[input_ids]
            labels = input_ids
        elif self.cfg.get("baseline_exact_ranks", False):
            # the decile baseline has no basis vocabulary, so it normally gets
            # no ranks at all and its numeric losses fall back to the bin
            # MIDPOINT. This collates the exact rank so those losses score the
            # same target the tied head's do, leaving backbone as the only
            # difference between the two arms. See Loss._bin_terms.
            _r = t.stack([x["exact_ranks"] for x in batch]).to(t.float32)
            extra["ranks"] = _r if perm is None else _r.gather(1, perm)

        if "time_based_rope" in self.cfg:
            _se = t.stack([x["s_elapsed"] for x in batch])
            if perm is not None:
                _se = _se.gather(1, perm)  # constant within a block, but keep aligned
            p_ids = _se / self.cfg.time_based_rope.sec_per_pos_id
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
            extra["position_ids"] = p_ids

        return {"input_ids": input_ids, "labels": labels, **extra}

    @property
    def bos_token_id(self) -> int:
        return (
            self.basis_vocab["bos_token_id"]
            if self.basis_vocab is not None
            else self.tkzr_cfg.lookup["BOS"]
        )

    @property
    def eos_token_id(self) -> int:
        return (
            self.basis_vocab["eos_token_id"]
            if self.basis_vocab is not None
            else self.tkzr_cfg.lookup["EOS"]
        )

    @property
    def reverse_lookup(self) -> dict:
        lookup = (
            self.basis_vocab["basis_lookup"]
            if self.basis_vocab is not None
            else self.tkzr_cfg.lookup
        )
        return {v: k for k, v in lookup.items()}

    def train(self, resume_from_checkpoint: bool = False, verbose: bool = False):
        if resume_from_checkpoint:
            try:
                self.trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                self.logger.warning(f"Encountered {e} on resume from checkpoint.")
                self.trainer.train()
        else:
            self.trainer.train()

        self.trainer.model.save_pretrained(self.output_home / f"mdl-{self.run_name}")

        with open(self.output_home / f"mdl-{self.run_name}-training.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if verbose:
            self.logger.summarize_trained_model(
                model=self.trainer.model,
                bos_token_id=self.bos_token_id,
                reverse=self.reverse_lookup,
            )


if __name__ == "__main__":
    self = Trainer(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    self.train(verbose=True)
    # breakpoint()
