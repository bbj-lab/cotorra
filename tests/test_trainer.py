#!/usr/bin/env python3

"""tests for cotorra.trainer.Trainer"""

import pytest
import torch as t
from helpers import base_training_cfg, write_cfg
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


def test_constructor_kwargs_do_not_reach_the_loader(processed, tmp_path_factory):
    """
    `Trainer.__init__` hands `Loader` the config *file* rather than its own
    merged `self.cfg`, so a `max_seq_len` (or any other loader-relevant)
    override passed as a keyword argument silently applies to the trainer but
    not to the data it trains on -- pinning the current behavior so a fix is
    a deliberate, visible change
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
    assert trainer.loader.cfg.max_seq_len == 16
    assert trainer.trainer.train_dataset[0]["input_ids"].shape == (16,)


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
