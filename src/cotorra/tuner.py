#!/usr/bin/env python3

"""
train a model with hyperparameter tuning
"""

import functools

import optuna
from omegaconf import OmegaConf
from transformers import TrainingArguments

from cotorra.trainer import Trainer

# the search space when `tuning_args` sets no `hp_space`, as before it could
DEFAULT_HP_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-4, "log": True},
    "gradient_accumulation_steps": {"type": "int", "low": 1, "high": 3},
}


class Tuner(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def optuna_hp_space(
        trial: "optuna.trial.BaseTrial", hp_space: dict | None = None
    ) -> dict[str, float | int | str | bool | None]:
        """samples each `training_args` field `hp_space` names with optuna's
        `trial.suggest_<type>`, passing it the field's other keys"""
        return {
            name: getattr(trial, f"suggest_{spec['type']}")(
                name, **{k: v for k, v in spec.items() if k != "type"}
            )
            for name, spec in (hp_space or DEFAULT_HP_SPACE).items()
        }

    @staticmethod
    def compute_objective(metrics: dict[str, float]) -> float:
        # hf's default sums every metric once there's more than `eval_loss`, and
        # evaluation reports each of the custom loss's terms beside it
        return metrics["eval_loss"]

    def train(self, verbose: bool = False) -> None:
        tuning_args = OmegaConf.to_container(self.cfg.tuning_args, resolve=True)
        hp_space = tuning_args.pop("hp_space", None)
        # hf skips a field `TrainingArguments` lacks with a warning, so a typo would
        # run every trial alike
        if unknown := set(hp_space or {}) - set(TrainingArguments.__dataclass_fields__):
            raise ValueError(
                f"tuning_args.hp_space names {sorted(unknown)}, which aren't "
                "`TrainingArguments` fields"
            )
        best_trial = self.trainer.hyperparameter_search(
            hp_space=functools.partial(self.optuna_hp_space, hp_space=hp_space),
            compute_objective=self.compute_objective,
            **tuning_args,
        )
        for n, v in best_trial.hyperparameters.items():
            setattr(self.trainer.args, n, v)
        self.trainer.train()
        self.trainer.model.save_pretrained(self.output_home / f"mdl-{self.run_name}")

        with open(self.output_home / f"mdl-{self.run_name}-tuning.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if verbose:
            self.logger.summarize_trained_model(
                model=self.trainer.model,
                bos_token_id=self.tkzr_cfg.lookup["BOS"],
                reverse={v: k for k, v in self.tkzr_cfg.lookup.items()},
            )


if __name__ == "__main__":
    self = Tuner()
    self.train(verbose=True)
    # breakpoint()
