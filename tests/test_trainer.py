#!/usr/bin/env python3

"""tests for cotorra.trainer.Trainer"""

import math
import os
import pathlib
import shutil

import pytest
import torch as t
from helpers import TINY_MODEL_ARGS, base_training_cfg, default_cfg, write_cfg
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from cotorra.model import HEADS
from cotorra.trainer import Trainer
from cotorra.tuner import Tuner


def test_model_init_wires_vocab_and_special_tokens_from_tokenizer(built_trainer):
    cfg = built_trainer.model.config
    tkzr = built_trainer.tkzr_cfg
    assert cfg.vocab_size == len(tkzr.lookup)
    assert cfg.bos_token_id == tkzr.lookup["BOS"]
    assert cfg.eos_token_id == tkzr.lookup["EOS"]
    assert cfg.hidden_size == 32  # from the tiny test model preset


def test_run_name_comes_from_cfg(built_trainer):
    assert built_trainer.run_name == "test-run"


def test_custom_loss_is_wired_into_the_hf_trainer_by_default(built_trainer):
    assert built_trainer.loss is not None
    assert built_trainer.trainer.compute_loss_func is built_trainer.loss


def test_custom_loss_is_disabled_when_cfg_says_so(processed, tmp_path_factory):
    cfg = base_training_cfg(custom_loss=False)
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("no-custom-loss") / "training.yaml", cfg
    )
    out = tmp_path_factory.mktemp("no-custom-loss-output")
    trainer = Trainer(
        training_cfg=cfg_path, processed_data_home=processed, output_home=out
    )
    assert trainer.loss is None
    assert trainer.trainer.compute_loss_func is None


def test_collate_fn_adds_time_based_position_ids_when_configured(built_trainer):
    batch = [
        {"input_ids": t.arange(4), "s_elapsed": t.tensor([0.0, 300.0, 600.0, 900.0])},
        {
            "input_ids": t.arange(4, 8),
            "s_elapsed": t.tensor([0.0, 150.0, 300.0, 450.0]),
        },
    ]
    out = built_trainer.collate_fn(batch)
    assert set(out.keys()) == {"input_ids", "labels", "position_ids"}
    assert t.equal(out["labels"], out["input_ids"])

    sec_per_pos_id = built_trainer.cfg.time_based_rope.sec_per_pos_id
    expected = t.stack([b["s_elapsed"] for b in batch]) / sec_per_pos_id
    expected += t.arange(4)
    assert t.allclose(out["position_ids"], expected)


def test_collate_fn_omits_position_ids_without_time_based_rope(
    processed, tmp_path_factory
):
    cfg = base_training_cfg()
    del cfg["time_based_rope"]
    cfg_path = write_cfg(tmp_path_factory.mktemp("no-rope") / "training.yaml", cfg)
    out_home = tmp_path_factory.mktemp("no-rope-output")
    trainer = Trainer(
        training_cfg=cfg_path, processed_data_home=processed, output_home=out_home
    )

    batch = [{"input_ids": t.arange(4)}, {"input_ids": t.arange(4, 8)}]
    out = trainer.collate_fn(batch)
    assert set(out.keys()) == {"input_ids", "labels"}
    assert t.equal(out["labels"], out["input_ids"])


def test_model_init_builds_a_fresh_model_each_call(built_trainer):
    """`model_init` is handed to HF Trainer, which re-calls it per tuning trial"""
    first, second = built_trainer.model_init(), built_trainer.model_init()
    assert first is not second
    assert first.config.vocab_size == second.config.vocab_size


def test_constructor_kwargs_reach_the_loader(processed, tmp_path_factory):
    """
    `Trainer.__init__` hands `Loader` its own merged `self.cfg`, so a
    `max_seq_len` (or any other loader-relevant) override passed as a keyword
    argument applies to the data it trains on too. It used to hand over the
    config *file*, and the override silently applied to the trainer alone
    """
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("kwarg-split") / "training.yaml",
        base_training_cfg(max_seq_len=16),
    )
    out = tmp_path_factory.mktemp("kwarg-split-output")
    trainer = Trainer(
        training_cfg=cfg_path,
        processed_data_home=processed,
        output_home=out,
        max_seq_len=8,
    )

    assert trainer.cfg.max_seq_len == 8
    assert trainer.loader.cfg.max_seq_len == 8
    assert trainer.trainer.train_dataset[0]["input_ids"].shape == (8,)


@pytest.mark.slow
def test_train_saves_a_reloadable_model_and_training_config(
    processed, tmp_path_factory
):
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("train-cfg") / "training.yaml", base_training_cfg()
    )
    out = tmp_path_factory.mktemp("train-output")
    trainer = Trainer(
        training_cfg=cfg_path, processed_data_home=processed, output_home=out
    )

    trainer.train()

    model_dir = out / f"mdl-{trainer.run_name}"
    assert model_dir.is_dir()
    assert (model_dir / "config.json").is_file()

    saved_cfg = OmegaConf.load(out / f"mdl-{trainer.run_name}-training.yaml")
    assert saved_cfg.max_seq_len == 16

    reloaded = AutoModelForCausalLM.from_pretrained(model_dir)
    sample = t.tensor(
        [[trainer.tkzr_cfg.lookup["BOS"], trainer.tkzr_cfg.lookup["EOS"]]]
    )
    logits = reloaded(sample).logits
    assert logits.shape == (1, 2, len(trainer.tkzr_cfg.lookup))
    assert t.isfinite(logits).all()


@pytest.mark.slow
def test_train_resume_from_checkpoint_falls_back_when_none_exists(
    processed, tmp_path_factory
):
    """training_args disables checkpointing in the fast test config, so
    resume_from_checkpoint=True should hit Trainer's except-and-retry path
    rather than raising -- pinning the "safe to pass unconditionally" claim"""
    cfg_path = write_cfg(
        tmp_path_factory.mktemp("resume-cfg") / "training.yaml", base_training_cfg()
    )
    out = tmp_path_factory.mktemp("resume-output")
    trainer = Trainer(
        training_cfg=cfg_path, processed_data_home=processed, output_home=out
    )

    trainer.train(resume_from_checkpoint=True)

    assert (out / f"mdl-{trainer.run_name}").is_dir()


def test_shipped_default_config_keeps_unused_columns():
    """
    HF's `Trainer` drops any dataset column its model's `forward` does not
    name, which would take `s_elapsed` with it; the shipped default has to
    turn that off or time-based RoPE silently loses its position ids
    """
    training_args = default_cfg("cotorra", "training")["training_args"]
    assert training_args["remove_unused_columns"] is False


def test_train_dataloader_batches_carry_time_based_position_ids(built_trainer):
    """
    the end-to-end version of the above: what the model actually receives once
    HF has built the dataloader around `collate_fn`
    """
    assert built_trainer.trainer.args.remove_unused_columns is False
    batch = next(iter(built_trainer.trainer.get_train_dataloader()))
    assert set(batch.keys()) == {"input_ids", "labels", "position_ids"}
    assert batch["position_ids"].shape == batch["input_ids"].shape


def test_removing_unused_columns_starves_the_collator_of_s_elapsed(
    processed, tmp_path_factory
):
    """why the default has to be `false`, demonstrated rather than asserted"""
    cfg = base_training_cfg()
    cfg["training_args"]["remove_unused_columns"] = True
    cfg_path = write_cfg(tmp_path_factory.mktemp("drop-columns") / "training.yaml", cfg)
    trainer = Trainer(
        training_cfg=cfg_path,
        processed_data_home=processed,
        output_home=tmp_path_factory.mktemp("drop-columns-output"),
    )

    with pytest.raises(KeyError, match="s_elapsed"):
        next(iter(trainer.trainer.get_train_dataloader()))


def test_balanced_toi_loss_runs_through_the_trainers_custom_loss(
    processed, tmp_path_factory
):
    """as the shipped default configures it, fed the trainer's own batches"""
    cfg = base_training_cfg()
    assert "balanced_toi_loss" in cfg
    home = tmp_path_factory.mktemp("balanced-toi")
    trainer = Trainer(
        training_cfg=write_cfg(home / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=home / "out",
    )
    mdl = trainer.model_init()
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(2)])
    loss = trainer.trainer.compute_loss(mdl, batch)
    loss.backward()
    assert t.isfinite(loss)
    assert all(p.grad is not None for p in mdl.parameters() if p.requires_grad)


@pytest.mark.parametrize("custom_loss", [True, False])
def test_label_weighted_loss_is_deprecated_but_still_honored(
    processed, tmp_path, custom_loss
):
    """configs that use it train as they always did, with a warning pointing at
    `balanced_toi_loss`; a FutureWarning, so a `cotorra train` run shows it"""
    cfg = base_training_cfg(custom_loss=custom_loss)
    cfg["label_weighted_loss"] = {
        "tokens_of_interest": cfg["tokens_of_interest"],
        "toi_weight": 20.0,
    }
    with pytest.warns(FutureWarning, match="`label_weighted_loss` is deprecated"):
        trainer = Trainer(
            training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
            processed_data_home=processed,
            output_home=tmp_path / "out",
        )
    if custom_loss:
        assert trainer.loss.__self__.weights.max().item() == 20.0


def test_a_config_without_it_draws_no_deprecation_warning(processed, tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        Trainer(
            training_cfg=write_cfg(tmp_path / "training.yaml", base_training_cfg()),
            processed_data_home=processed,
            output_home=tmp_path / "out",
        )


def test_overrides_reach_the_loader_as_well_as_the_trainer(
    processed, session_training_cfg_path, tmp_path
):
    """
    the loader does the chunking and picks the columns the config asks for, so
    it gets the trainer's merged config rather than re-reading the file, which
    would miss the overrides: a shorter `max_seq_len` would go unused, and the
    collate function would look for no `s_elapsed` the loader still kept
    """
    trainer = Trainer(
        training_cfg=session_training_cfg_path,
        processed_data_home=processed,
        output_home=tmp_path / "out",
        overrides=["max_seq_len=8", "~time_based_rope"],
    )
    assert trainer.loader.cfg == trainer.cfg
    assert len(trainer.trainer.train_dataset[0]["input_ids"]) == 8
    assert "s_elapsed" not in trainer.loader.dataset["train"].column_names
    assert "position_ids" not in trainer.collate_fn([trainer.trainer.train_dataset[0]])


def test_every_objective_trains_on_what_cocoa_and_the_loader_produce(
    processed_with_end_times, tmp_path
):
    """
    next token plus all three secondary heads -- time to event, a mixture
    time-to-next-token head, and discharge disposition -- on the columns cocoa
    and the loader actually produce: same-timestamp tokens arrive as exact zero
    gaps, every packed chunk carries disposition targets, and the trainer's own
    batches reach every head
    """
    cfg = base_training_cfg()
    del cfg["time_based_rope"]
    cfg["tte_objective"] = {"weight": 1.0}
    cfg["tnt_objective"] = {"weight": 1.0, "mixture_components": 3}
    cfg["disposition_objective"] = {}
    trainer = Trainer(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed_with_end_times,
        output_home=tmp_path / "out",
    )
    assert trainer.loader.dataset["train"].column_names == [
        "input_ids",
        "hours_to_end_time",
        "hours_to_next_token",
        "disposition",
    ]
    # the inference tables carry the end times too, but only the context is read
    for ds_ in trainer.loader.for_inference.values():
        assert ds_.column_names == ["subject_id", "input_ids"]

    mdl = trainer.model_init()
    assert list(mdl.heads) == ["tte", "tnt", "disposition"]
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(8)])
    gaps = batch["hours_to_next_token"]
    assert (gaps == 0).any() and (gaps > 0).any()  # contiguous events, and not
    # a record's disposition token sits near its end, so a few short chunks may
    # all fall before it; across the packed set, both kinds of target occur
    dispositions = t.cat([eg["disposition"] for eg in trainer.trainer.train_dataset])
    assert (dispositions >= 0).any() and (dispositions == -100).any()

    loss = trainer.trainer.compute_loss(mdl, batch)
    loss.backward()
    assert t.isfinite(loss)
    assert all(p.grad is not None for p in mdl.parameters() if p.requires_grad)
    for head in mdl.heads.values():
        assert max(p.grad.abs().max().item() for p in head.parameters()) > 0


@pytest.mark.parametrize(
    "name", [name for name, head in HEADS.items() if head.target_weight]
)
def test_a_per_record_objective_trains_and_evaluates_on_what_cocoa_produces(
    name, processed_with_end_times, tmp_path
):
    """
    `per_record` set by an override, as the cli passes one: the loader weights
    each record, the collator batches the weights beside the target, and both
    training's loss and evaluation's reach the head
    """
    trainer = Trainer(
        training_cfg=write_cfg(
            tmp_path / "training.yaml", base_training_cfg(**{f"{name}_objective": {}})
        ),
        processed_data_home=processed_with_end_times,
        output_home=tmp_path / "out",
        overrides=[f"{name}_objective.per_record=true"],
    )
    assert trainer.head_options[name]["per_record"] is True
    assert trainer.loader.dataset["train"].column_names[-1] == (
        HEADS[name].target_weight
    )

    mdl = trainer.model_init()
    assert mdl.heads[name].per_record
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(8)])
    loss = trainer.trainer.compute_loss(mdl, batch)
    loss.backward()
    assert t.isfinite(loss)
    assert max(p.grad.abs().max().item() for p in mdl.heads[name].parameters()) > 0
    assert math.isfinite(trainer.trainer.evaluate()[f"eval_{name}_loss"])


def test_the_custom_loss_is_averaged_over_gradient_accumulation(
    built_trainer, monkeypatch
):
    """hf leaves that to a custom loss, and a per-batch mean doesn't do it; only
    in training, so evaluation's loss is the batch's own"""
    trainer = built_trainer.trainer
    mdl = built_trainer.model_init()
    batch = built_trainer.collate_fn([trainer.train_dataset[i] for i in range(2)])
    monkeypatch.setattr(
        trainer, "current_gradient_accumulation_steps", 3, raising=False
    )
    with t.no_grad():
        training = trainer.compute_loss(mdl.train(), batch)
        evaluating = trainer.compute_loss(mdl.eval(), batch)
    assert training.item() == pytest.approx(evaluating.item() / 3, rel=1e-5)


def test_training_logs_each_terms_mean_since_the_last_log(built_trainer):
    """beside hf's `loss`, which averages the steps since its last log; each
    batch weighted by its rows, and none of evaluation's mixed in"""
    trainer = built_trainer.trainer
    mdl = built_trainer.model_init()
    batches = [
        built_trainer.collate_fn([trainer.train_dataset[i] for i in rows])
        for rows in ([0, 1], [2, 3, 4])
    ]
    trainer.reset_terms("train")
    x_ents = list()
    with t.no_grad():
        trainer.compute_loss(mdl.eval(), batches[0])
        for batch in batches:
            _, outputs = trainer.compute_loss(mdl.train(), batch, return_outputs=True)
            x_ents.append(
                built_trainer.loss.__self__.x_ent_loss(outputs, batch["labels"])
            )
    trainer.log({"loss": 0.0})
    logged = trainer.state.log_history[-1]
    assert {"x_ent_loss", "balanced_toi_loss", "quantile_token_loss"} <= set(logged)
    assert logged["x_ent_loss"] == pytest.approx(
        ((2 * x_ents[0] + 3 * x_ents[1]) / 5).item(), rel=1e-5
    )
    trainer.log({"loss": 0.0})  # reported, and so reset
    assert "x_ent_loss" not in trainer.state.log_history[-1]


def test_evaluation_reports_each_terms_mean_over_the_whole_eval_set(
    processed, tmp_path_factory
):
    """weighted by rows as hf weights `eval_loss` -- here over a short last
    batch -- so the weighted terms add up to it"""
    cfg = base_training_cfg(training_args={"per_device_eval_batch_size": 5})
    trainer = Trainer(
        training_cfg=write_cfg(
            tmp_path_factory.mktemp("eval-terms") / "training.yaml", cfg
        ),
        processed_data_home=processed,
        output_home=tmp_path_factory.mktemp("eval-terms-output"),
    )
    assert len(trainer.trainer.eval_dataset) % 5
    metrics = trainer.trainer.evaluate()
    assert metrics["eval_loss"] == pytest.approx(
        metrics["eval_x_ent_loss"]
        + cfg["balanced_toi_loss"]["bce_weight"] * metrics["eval_balanced_toi_loss"]
        + cfg["quantile_token_loss"]["qt_weight"] * metrics["eval_quantile_token_loss"],
        rel=1e-5,
    )
    assert "x_ent_loss" not in trainer.trainer.state.log_history[-1]
    assert (
        trainer.trainer.state.log_history[-1]["eval_x_ent_loss"]
        == (metrics["eval_x_ent_loss"])
    )


@pytest.mark.slow
def test_accumulated_batches_step_as_one_batch_of_their_size(tmp_path):
    """
    two accumulated batches of two take the same optimizer step as one batch of
    four -- with the custom loss as with hf's own. Plain sgd, since adam's first
    step barely depends on the gradient's scale and would hide a sum
    """
    import copy

    from helpers import TINY_MODEL_ARGS
    from transformers import AutoConfig, TrainingArguments

    from cotorra.trainer import TrainerWithCustomLoss

    t.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        AutoConfig.for_model("llama", vocab_size=64, **TINY_MODEL_ARGS)
    )
    data = [{"input_ids": t.randint(3, 63, (8,))} for _ in range(4)]

    def collate(rows):
        ids = t.stack([r["input_ids"] for r in rows])
        return {"input_ids": ids, "labels": ids}

    def x_ent(outputs, labels, **kwargs):
        logits = outputs.get("logits")[:, :-1]
        return t.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
        )

    def one_step(batch_size, accumulate, loss_func):
        mdl = copy.deepcopy(model)
        TrainerWithCustomLoss(
            model=mdl,
            data_collator=collate,
            compute_loss_func=loss_func,
            train_dataset=data,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=1,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=accumulate,
                optim="sgd",
                learning_rate=0.5,
                lr_scheduler_type="constant",
                max_grad_norm=1e9,
                report_to="none",
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                remove_unused_columns=False,
            ),
        ).train()
        return t.cat([p.detach().flatten() for p in mdl.parameters()])

    for loss_func in (x_ent, None):
        whole, accumulated = one_step(4, 1, loss_func), one_step(2, 2, loss_func)
        assert t.allclose(whole, accumulated, atol=1e-6), loss_func


@pytest.mark.parametrize("stage", [Trainer, Tuner])
def test_a_stage_without_its_directories_is_refused_by_name(stage, tmp_path):
    """both directories default to None, a holdover from when the config named
    them, and leaving one out used to fail inside `pathlib` without saying which"""
    with pytest.raises(TypeError, match=rf"^{stage.__name__} needs `output_home`$"):
        stage(processed_data_home=tmp_path)
    with pytest.raises(
        TypeError, match="needs `processed_data_home` and `output_home`$"
    ):
        stage()


@pytest.mark.parametrize("block", ["absent", "empty"])
def test_the_wandb_block_is_optional(processed, tmp_path, monkeypatch, block):
    """`run_name` fell back on `self.cfg.wandb.get(...)`, evaluated whether or not
    it was needed, so a config without a `wandb` block raised `Missing key
    wandb`, and one left empty -- parsed to None -- failed on `None.get`"""
    cfg = base_training_cfg()
    if block == "absent":
        del cfg["wandb"]
    else:
        cfg["wandb"] = None
    # set, so that monkeypatch restores their absence over what Trainer writes
    monkeypatch.setenv("WANDB_PROJECT", "unset")
    monkeypatch.setenv("WANDB_NAME", "unset")
    trainer = Trainer(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=tmp_path,
    )
    assert trainer.run_name == "test-run"
    assert os.environ["WANDB_PROJECT"] == os.environ["WANDB_NAME"] == "cotorra"


def test_custom_loss_is_off_when_the_config_omits_it(processed, tmp_path):
    """the README gives `custom_loss` a default of false, but it was read as
    `self.cfg.custom_loss`, which a config without it raised on"""
    cfg = base_training_cfg()
    del cfg["custom_loss"]
    trainer = Trainer(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=tmp_path,
    )
    assert trainer.loss is None
    assert trainer.trainer.compute_loss_func is None


def test_a_custom_loss_can_score_a_batch_without_labels(built_trainer, monkeypatch):
    """the rows weighting the logged terms used to be counted as `len(labels)`,
    which failed on a batch without `labels`, though hf's own custom-loss path
    hands such a batch's loss function `labels=None`"""
    trainer = built_trainer.trainer
    mdl = built_trainer.model_init()
    batch = built_trainer.collate_fn([trainer.train_dataset[i] for i in range(3)])
    del batch["labels"]

    def loss_func(outputs, labels, terms):
        assert labels is None
        terms["mean_logit"] = outputs.logits.mean().detach()
        return outputs.logits.mean()

    monkeypatch.setattr(trainer, "compute_loss_func", loss_func)
    trainer.reset_terms("eval")
    with t.no_grad():
        trainer.compute_loss(mdl.eval(), batch)
    assert trainer.term_rows["eval"] == 3
    assert set(trainer.term_sums["eval"]) == {"mean_logit"}
    trainer.reset_terms("eval")


def trainer_with(cfg: dict, processed, tmp_path, **kwargs) -> Trainer:
    """a `Trainer` built from `cfg`, writing under `tmp_path / "out"`"""
    return Trainer(
        training_cfg=write_cfg(tmp_path / "training.yaml", cfg),
        processed_data_home=processed,
        output_home=tmp_path / "out",
        **kwargs,
    )


@pytest.mark.parametrize(
    "block, project, name",
    [
        (
            {"project": "elsewhere", "run_name": "a-wandb-run"},
            "elsewhere",
            "a-wandb-run",
        ),
        ({"project": "elsewhere"}, "elsewhere", "cotorra"),
        ({"run_name": "a-wandb-run"}, "cotorra", "a-wandb-run"),
    ],
    ids=["both", "project-only", "run-name-only"],
)
def test_the_wandb_block_names_the_project_and_run_wandb_records(
    block, project, name, processed, tmp_path, monkeypatch
):
    """through the environment, which hf's wandb callback reads as training
    starts; each key `cotorra` without it"""
    monkeypatch.setenv("WANDB_PROJECT", "unset")
    monkeypatch.setenv("WANDB_NAME", "unset")
    cfg = base_training_cfg()
    cfg["wandb"] = block
    trainer_with(cfg, processed, tmp_path)
    assert (os.environ["WANDB_PROJECT"], os.environ["WANDB_NAME"]) == (project, name)


@pytest.mark.parametrize(
    "names, run_name",
    [
        ({"run_name": "the-run", "wandb": {"run_name": "the-wandb-run"}}, "the-run"),
        ({"wandb": {"run_name": "the-wandb-run"}}, "the-wandb-run"),
    ],
    ids=["run-name", "wandb-run-name"],
)
def test_the_model_and_its_config_are_saved_under_the_run_name(
    names, run_name, processed, tmp_path, monkeypatch
):
    """the top-level `run_name`, or without it the wandb block's"""
    cfg = base_training_cfg()
    del cfg["run_name"]
    cfg["wandb"] = {"project": "cotorra-tests"} | names.get("wandb", {})
    cfg |= {k: v for k, v in names.items() if k != "wandb"}
    trainer = trainer_with(cfg, processed, tmp_path)
    assert trainer.run_name == run_name
    monkeypatch.setattr(trainer.trainer, "train", lambda **kwargs: None)
    trainer.train()
    assert (tmp_path / "out" / f"mdl-{run_name}" / "config.json").is_file()
    assert (tmp_path / "out" / f"mdl-{run_name}-training.yaml").is_file()


def test_every_model_arg_lands_on_the_built_models_config(processed, tmp_path):
    """over what the named model's config says, beside the vocabulary size and
    special tokens the tokenizer gives"""
    args = TINY_MODEL_ARGS | {"num_hidden_layers": 2, "intermediate_size": 48}
    args |= {"rms_norm_eps": 1e-3}
    trainer = trainer_with(
        base_training_cfg(model={"model_args": args}), processed, tmp_path
    )
    model = trainer.model_init()
    for key, value in args.items():
        assert getattr(model.config, key) == value, key
    assert len(model.model.layers) == 2
    assert model.model.layers[0].mlp.up_proj.out_features == 48
    lookup = trainer.tkzr_cfg.lookup
    assert model.config.vocab_size == len(lookup)
    assert (model.config.bos_token_id, model.config.eos_token_id) == (
        lookup.BOS,
        lookup.EOS,
    )


PACKAGED_TRAINING_ARGS = default_cfg("cotorra", "training")["training_args"]


@pytest.fixture(scope="module")
def shipped_args_trainer(processed, tmp_path_factory) -> Trainer:
    """a trainer given the packaged `training_args` as they ship, `report_to:
    wandb` included, which only installs hf's callback until training starts"""
    cfg = base_training_cfg()
    cfg["training_args"] = dict(PACKAGED_TRAINING_ARGS)
    return trainer_with(cfg, processed, tmp_path_factory.mktemp("shipped-args"))


@pytest.mark.parametrize("key", list(PACKAGED_TRAINING_ARGS))
def test_every_packaged_training_arg_reaches_hfs_training_arguments(
    key, shipped_args_trainer
):
    """as it's written -- hf only wraps `report_to` in a list, and takes the
    strategies as enums that compare equal to their names -- rather than
    dropped along the way or overridden"""
    value = PACKAGED_TRAINING_ARGS[key]
    expected = [value] if key == "report_to" else value
    assert getattr(shipped_args_trainer.trainer.args, key) == expected


def test_the_batch_sizes_set_the_batches_training_and_evaluation_take(
    processed, tmp_path
):
    cfg = base_training_cfg(
        training_args={
            "per_device_train_batch_size": 3,
            "per_device_eval_batch_size": 5,
        }
    )
    hf = trainer_with(cfg, processed, tmp_path).trainer
    assert next(iter(hf.get_train_dataloader()))["input_ids"].shape == (3, 16)
    assert next(iter(hf.get_eval_dataloader()))["input_ids"].shape == (5, 16)


@pytest.mark.slow
def test_the_packaged_checkpointing_keeps_and_saves_the_best_model(
    processed, tmp_path, monkeypatch
):
    """
    the packaged scheme: evaluating and saving every tenth of the run, keeping
    the best checkpoint by `eval_loss`, lowest best -- by the end, it alone --
    and loading it then, which makes it the model saved as mdl-<run_name>. Its
    evaluations report losses scripted so that the best comes mid-run, which
    real ones at this scale wouldn't reliably do, and each checkpoint is
    copied aside as it's saved, to tell them apart afterwards
    """
    from safetensors.torch import load_file
    from transformers import TrainerCallback

    class CopiedAside(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            name = f"checkpoint-{state.global_step}"
            shutil.copytree(pathlib.Path(args.output_dir) / name, tmp_path / name)

    scheme = [
        "eval_strategy",
        "eval_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "prediction_loss_only",
    ]
    args = {key: PACKAGED_TRAINING_ARGS[key] for key in scheme} | {"max_steps": 10}
    trainer = trainer_with(base_training_cfg(training_args=args), processed, tmp_path)
    hf = trainer.trainer
    losses = iter([5.0, 4.0, 3.0, 2.0, 2.5, 3.5, 4.5, 4.0, 3.0, 2.2])
    evaluate, reported = hf.evaluate, []

    def scripted(*args, **kwargs):
        metrics = evaluate(*args, **kwargs)
        assert math.isfinite(metrics["eval_loss"])  # the real one, from the custom loss
        reported.append(metrics["eval_loss"])
        return metrics | {"eval_loss": next(losses)}

    monkeypatch.setattr(hf, "evaluate", scripted)
    hf.add_callback(CopiedAside())
    trainer.train()
    assert len(reported) == 10  # every tenth of the run
    assert hf.state.best_metric == 2.0
    assert pathlib.Path(hf.state.best_model_checkpoint).name == "checkpoint-4"
    assert [p.name for p in (tmp_path / "out").glob("checkpoint-*")] == ["checkpoint-4"]
    saved = load_file(tmp_path / "out" / "mdl-test-run" / "model.safetensors")
    best = load_file(tmp_path / "checkpoint-4" / "model.safetensors")
    last = load_file(tmp_path / "checkpoint-10" / "model.safetensors")
    assert set(saved) == set(best) == set(last)
    assert all(t.equal(saved[k], best[k]) for k in saved)
    assert not all(t.equal(saved[k], last[k]) for k in saved)


@pytest.mark.parametrize("loss_only", [True, False])
def test_prediction_loss_only_leaves_evaluation_its_loss_alone(
    loss_only, processed, tmp_path
):
    """the packaged `true` keeps evaluation from gathering every position's
    logits, which only the loss needs"""
    cfg = base_training_cfg(training_args={"prediction_loss_only": loss_only})
    hf = trainer_with(cfg, processed, tmp_path).trainer
    out = hf.predict(hf.eval_dataset)
    assert math.isfinite(out.metrics["test_loss"])
    if loss_only:
        assert out.predictions is None
    else:
        assert out.predictions.shape[:2] == (len(hf.eval_dataset), 16)


@pytest.fixture
def untrained(processed, tmp_path, monkeypatch) -> Trainer:
    """a trainer whose hf training is a no-op that records how it was called, so
    that `Trainer.train` goes straight on to saving"""
    trainer = trainer_with(base_training_cfg(), processed, tmp_path)
    trainer.calls = []
    monkeypatch.setattr(
        trainer.trainer, "train", lambda **kwargs: trainer.calls.append(kwargs)
    )
    return trainer


def test_training_verbosely_summarizes_the_trained_model(untrained, monkeypatch):
    """by sampling from it, starting at the tokenizer's BOS, and naming what it
    samples through the tokenizer's lookup turned around"""
    summaries = []
    monkeypatch.setattr(
        untrained.logger,
        "summarize_trained_model",
        lambda **kwargs: summaries.append(kwargs),
    )
    untrained.train()
    assert summaries == []
    untrained.train(verbose=True)
    [summary] = summaries
    lookup = untrained.tkzr_cfg.lookup
    assert summary["model"] is untrained.trainer.model
    assert summary["bos_token_id"] == lookup["BOS"]
    assert summary["reverse"] == {v: k for k, v in lookup.items()}


def test_resuming_asks_hf_for_the_latest_checkpoint_and_starts_afresh_failing_that(
    untrained, monkeypatch
):
    warned = []
    monkeypatch.setattr(untrained.logger, "warning", warned.append)
    untrained.train(resume_from_checkpoint=True)
    assert untrained.calls == [{"resume_from_checkpoint": True}]
    assert warned == []

    def train(**kwargs):
        untrained.calls.append(kwargs)
        if kwargs:
            raise ValueError("No valid checkpoint found in output directory")

    untrained.calls.clear()
    monkeypatch.setattr(untrained.trainer, "train", train)
    untrained.train(resume_from_checkpoint=True)
    assert untrained.calls == [{"resume_from_checkpoint": True}, {}]
    [warning] = warned
    assert "No valid checkpoint found" in warning
    untrained.calls.clear()
    untrained.train()
    assert untrained.calls == [{}]


def test_the_config_saved_beside_the_model_is_the_merged_one(
    processed, tmp_path, monkeypatch
):
    """the file, then the overrides, then the constructor's keyword arguments,
    as the run used them"""
    trainer = trainer_with(
        base_training_cfg(n_epochs=1),
        processed,
        tmp_path,
        overrides=["n_epochs=2", "~quantile_token_loss", "training_args.seed=7"],
        max_seq_len=8,
    )
    monkeypatch.setattr(trainer.trainer, "train", lambda **kwargs: None)
    trainer.train()
    saved = OmegaConf.load(tmp_path / "out" / "mdl-test-run-training.yaml")
    assert saved == trainer.cfg
    assert (saved.n_epochs, saved.max_seq_len, saved.training_args.seed) == (2, 8, 7)
    assert "quantile_token_loss" not in saved
