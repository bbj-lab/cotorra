#!/usr/bin/env python3

"""tests for cotorra.trainer.Trainer"""

import pytest
import torch as t
from helpers import base_training_cfg, default_cfg, write_cfg
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from cotorra.trainer import Trainer


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
    for ds_ in trainer.loader.for_inference.values():
        assert "hours_to_end_time_past" in ds_.column_names

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
