#!/usr/bin/env python3

"""
train a model with hyperparameter tuning
"""

from omegaconf import OmegaConf

from cotorra.trainer import Trainer

# Adam second-moment coupling (see Tuner.coupled_adam_beta2). Anchor is the
# config beta2's default was implicitly fine for in this series: effective
# batch 16 (per_device 16 x grad_accum 1, the fixed setting used by every
# retune through basis_blended_mimic_tune_k10_large_nll_retuned20) at
# torch/HF's stock beta2 = 0.999.
ADAM_BETA2_REF = 0.999
EFF_BATCH_REF = 16


class Tuner(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def coupled_adam_beta2(per_device_train_batch_size, gradient_accumulation_steps):
        """
        Adam's v_t (second moment) is an EMA of the squared gradient with
        half-life ~ln2/(1-beta2) *optimizer steps*. What it's actually
        estimating is the true squared gradient, and each step contributes
        one minibatch's worth of data toward that estimate -- so the
        estimate's quality depends on the tokens in the window,
        ~(steps averaged) x (tokens per step), not on the step count alone.

        Holding beta2 fixed while shrinking the batch therefore silently
        shrinks the averaging window in token terms (drop effective batch
        16x and v_t is built from 16x less data), leaving a noisier sqrt(v)
        dividing every update -- the usual explanation for small batches
        "being unstable" for LM training. arXiv 2507.07101 (NeurIPS 2025)
        shows that's an un-retuned-beta2 artifact rather than a batch-size
        one, and prescribes holding the half-life fixed in *tokens*.

        Since seq_len is constant across trials here, tokens/step is
        proportional to the effective batch (per_device x grad_accum), so
        fixing the token half-life means (1-beta2) proportional to
        effective batch:

            1 - beta2 = (1 - ADAM_BETA2_REF) * eff_batch / EFF_BATCH_REF

        e.g. eff batch 4 -> 0.99975, eff batch 1 -> ~0.99994, and the
        anchor eff batch 16 -> 0.999 unchanged. Derived rather than
        searched, so it costs no extra trials and stays consistent between
        the Optuna trials and the final best-config retrain (see train()).
        """
        eff_batch = per_device_train_batch_size * gradient_accumulation_steps
        one_minus = (1.0 - ADAM_BETA2_REF) * eff_batch / EFF_BATCH_REF
        return float(min(max(1.0 - one_minus, 0.9), 0.9999999))

    def optuna_hp_space(self, trial):
        """
        The search space is config-driven via tuning_args.search_space, so
        comparing a new loss against an earlier result doesn't require
        editing this file (and silently changing what every other
        experiment searches). Every key is optional and falls back to the
        default below:

            tuning_args:
              search_space:
                learning_rate: [1.0e-4, 1.0e-3]        # log-uniform range
                num_train_epochs: [1, 2]               # int range
                gradient_accumulation_steps: [1, 4]    # int range; omit to
                                                       #   pin at the
                                                       #   training_args value
                per_device_train_batch_size: [1, 2, 4] # categorical choices;
                                                       #   omit to pin
                couple_adam_beta2: true                # see coupled_adam_beta2
                numeric_loss_weight: [1.0e-3, 1.0e1]   # log-uniform; w2 on
                                                       #   the extra numeric
                                                       #   term. Requires
                                                       #   numeric_loss to be
                                                       #   set in the config

        Defaults reproduce the batch-size-searching, beta2-coupled space:
        learning_rate 1e-4..1e-3, num_train_epochs 1..2,
        per_device_train_batch_size [1,2,4,8,16,32], grad_accum pinned,
        adam_beta2 coupled.

        On gradient_accumulation_steps: searching it *alongside*
        per_device_train_batch_size makes the space degenerate. What
        actually shapes training is the effective batch (per_device x
        grad_accum), so searching both makes the space degenerate --
        (per_device 2, accum 4) and (per_device 8, accum 1) are the same
        effective batch and, with fixed-length 1024-token sequences and
        the per-token loss normalization in Loss.custom_loss, the same
        gradient -- burning trials on duplicate points. Splitting the
        product across accumulation steps only costs memory and wall time
        for an identical update (arXiv 2507.07101, "gradient accumulation
        is wasteful"), and grad_accum=1 is what the winning trials kept
        selecting anyway. So accumulation stays pinned at the config value
        (1) and per_device_train_batch_size *is* the effective batch,
        searched directly over a range wide enough to cover what the split
        search used to reach.
        """
        space = self.cfg.get("tuning_args", {}).get("search_space", {}) or {}
        params = {}

        bs_choices = space.get("per_device_train_batch_size", [1, 2, 4, 8, 16, 32])
        if len(bs_choices) > 1:
            batch_size = trial.suggest_categorical(
                "per_device_train_batch_size", list(bs_choices)
            )
            params["per_device_train_batch_size"] = batch_size
        else:
            # pinned: take the single choice if given, else training_args
            batch_size = (
                int(bs_choices[0])
                if len(bs_choices) == 1
                else self.cfg.training_args.get("per_device_train_batch_size", 16)
            )
            if len(bs_choices) == 1:
                params["per_device_train_batch_size"] = batch_size

        if "gradient_accumulation_steps" in space:
            lo, hi = space["gradient_accumulation_steps"]
            grad_accum = trial.suggest_int(
                "gradient_accumulation_steps", int(lo), int(hi)
            )
            params["gradient_accumulation_steps"] = grad_accum
        else:
            grad_accum = self.cfg.training_args.get("gradient_accumulation_steps", 1)

        lr_lo, lr_hi = space.get("learning_rate", [1e-4, 1e-3])
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", float(lr_lo), float(lr_hi), log=True
        )
        ep_lo, ep_hi = space.get("num_train_epochs", [1, 2])
        params["num_train_epochs"] = trial.suggest_int(
            "num_train_epochs", int(ep_lo), int(ep_hi)
        )

        if "numeric_loss_weight" in space:
            # w2, the weight on the extra numeric term (see
            # Loss._numeric_loss). Deliberately NOT returned in `params`:
            # TrainingArguments has no such field, so transformers would warn
            # and drop it. It is written straight into self.cfg instead --
            # Loss holds the SAME OmegaConf node (see Trainer.__init__) and
            # re-reads the weight on every call, so the trial's value takes
            # effect immediately.
            assert self.cfg.get("numeric_loss", None) is not None, (
                "searching numeric_loss_weight requires numeric_loss to be "
                "set in the config: transformers builds the MODEL (via "
                "call_model_init) before it runs this search space, so the "
                "model-side continuous_moments/continuous_crps flag is "
                "derived from numeric_loss's PRESENCE, not from the searched "
                "weight. Without it every trial would run a model that never "
                "computed the statistic its loss then asks for."
            )
            lo, hi = space["numeric_loss_weight"]
            nlw = trial.suggest_float(
                "numeric_loss_weight", float(lo), float(hi), log=True
            )
            self.cfg.numeric_loss_weight = nlw
            trial.set_user_attr("numeric_loss_weight", nlw)
            # logged when SAMPLED, not when the trial finishes: optuna only
            # prints a trial's parameters on completion, the study is
            # in-memory so nothing can be queried mid-run, and a tune is
            # hours per trial -- leaving the one knob under test invisible
            # for the whole of it.
            self.logger.info(
                f"trial {trial.number}: numeric_loss="
                f"{self.cfg.get('numeric_loss')} weight={nlw:.6g}"
            )

        if space.get("couple_adam_beta2", True):
            beta2 = self.coupled_adam_beta2(batch_size, grad_accum)
            params["adam_beta2"] = beta2
            trial.set_user_attr("adam_beta2", beta2)
        trial.set_user_attr("effective_batch", batch_size * grad_accum)
        return params

    def train(self, verbose=False):
        tuning_args = {
            k: v for k, v in self.cfg.tuning_args.items() if k != "search_space"
        }
        best_trial = self.trainer.hyperparameter_search(
            hp_space=self.optuna_hp_space, **tuning_args
        )
        for n, v in best_trial.hyperparameters.items():
            setattr(self.trainer.args, n, v)
        # numeric_loss_weight is not a TrainingArguments field, so the setattr
        # above just hangs a dead attribute on args: without this write-back
        # the final saved model would retrain at the BASE config's weight
        # (typically 0.0, i.e. with no numeric term at all) and the dumped
        # mdl-<run>-tuning.yaml would record that instead of the winner. Same
        # failure mode as adam_beta2 below, same fix.
        if "numeric_loss_weight" in best_trial.hyperparameters:
            self.cfg.numeric_loss_weight = float(
                best_trial.hyperparameters["numeric_loss_weight"]
            )
            self.logger.info(
                f"final run: numeric_loss={self.cfg.get('numeric_loss')} at "
                f"weight {self.cfg.numeric_loss_weight:.6g}"
            )
        # adam_beta2 is derived from the batch/grad_accum pair rather than
        # searched, so it isn't in best_trial.hyperparameters -- recompute it
        # here or this final run silently reverts to the stock 0.999 and
        # trains the actually-saved model under a different second-moment
        # setting than the trial that won.
        if (self.cfg.get("tuning_args", {}).get("search_space", {}) or {}).get(
            "couple_adam_beta2", True
        ):
            self.trainer.args.adam_beta2 = self.coupled_adam_beta2(
                self.trainer.args.per_device_train_batch_size,
                self.trainer.args.gradient_accumulation_steps,
            )
        eff_batch = (
            self.trainer.args.per_device_train_batch_size
            * self.trainer.args.gradient_accumulation_steps
        )
        self.logger.info(
            f"final run: {eff_batch=}, adam_beta2={self.trainer.args.adam_beta2:.6g}"
        )
        self.trainer.train()
        self.trainer.model.save_pretrained(self.output_home / f"mdl-{self.run_name}")

        with open(self.output_home / f"mdl-{self.run_name}-tuning.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if verbose:
            self.logger.summarize_trained_model(
                model=self.trainer.model,
                bos_token_id=self.bos_token_id,
                reverse=self.reverse_lookup,
            )


if __name__ == "__main__":
    self = Tuner()
    self.train(verbose=True)
    # breakpoint()
