#!/usr/bin/env python3

"""
train a model with hyperparameter tuning
"""

from omegaconf import OmegaConf

from cotorra.trainer import Trainer


class Tuner(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps", 1, 3
            ),
        }

    def train(self, verbose=False):
        best_trial = self.trainer.hyperparameter_search(
            hp_space=self.optuna_hp_space, **self.cfg.tuning_args
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
                bos_token_id=self.bos_token_id,
                reverse=self.reverse_lookup,
            )


if __name__ == "__main__":
    self = Tuner()
    self.train(verbose=True)
    # breakpoint()
