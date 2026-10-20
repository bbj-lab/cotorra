#!/usr/bin/env python3

"""
tests for the packaged configs in cotorra.config: each is the default of the
stages that read it, its model presets build what they name, the values its
yaml anchors tie together agree, and its commented-out options work once
uncommented, as the file invites
"""

import importlib
import importlib.resources as resources
import json
import re
import types

import pytest
import torch as t
from helpers import base_training_cfg, default_cfg, write_cfg
from omegaconf import OmegaConf
from transformers import AutoConfig, TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available

from cotorra.configurable import Configurable
from cotorra.logger import Logger
from cotorra.model import CotorraForCausalLM, TntMixtureHead, TteHead, head_options
from cotorra.trainer import Trainer

PACKAGED = {name: default_cfg("cotorra", name) for name in ("training", "extraction")}
PACKAGED["scoring"] = default_cfg("cotorra", "scoring")
PRESETS = PACKAGED["training"]["model_presets"]

# the architecture behind each preset's `model_name`, whose config the hub
# serves -- unreachable here, so a local config naming just the architecture
# stands in for it
MODEL_TYPES = {
    "meta-llama/Llama-3.2-1B": "llama",
    "Qwen/Qwen3-1.7B-Base": "qwen3",
    "google/gemma-3-1b-pt": "gemma3_text",
}


def packaged_text(name: str) -> str:
    return (resources.files("cotorra.config") / f"{name}.yaml").read_text()


@pytest.mark.parametrize(
    "stage, name",
    [
        ("cotorra.trainer.Trainer", "training"),
        ("cotorra.tuner.Tuner", "training"),
        ("cotorra.loader.Loader", "training"),
        ("cotorra.extractor.Extractor", "extraction"),
        ("cotorra.scorer_rep_based.RepBasedScorer", "scoring"),
    ],
    ids=lambda x: x.rpartition(".")[2],
)
def test_each_stage_defaults_to_its_packaged_config(stage, name):
    """what a stage reads when no config file is passed, as the README says"""
    module, _, cls = stage.rpartition(".")
    assert getattr(importlib.import_module(module), cls).default_file == f"{name}.yaml"


def test_the_generative_scorer_defaults_to_the_packaged_scoring_config(
    scorer_generative,
):
    assert scorer_generative.GenerativeScorer.default_file == "scoring.yaml"


@pytest.mark.parametrize("stage", ["loader", "extractor"])
def test_a_stage_given_no_config_file_reads_the_packaged_one(
    stage, processed, fake_model_home
):
    """whole, and as it is: the loader the training config, the extractor the
    extraction config, which its own loader then reads in turn"""
    if stage == "loader":
        from cotorra.loader import Loader

        built, name = Loader(processed_data_home=processed), "training"
    else:
        from cotorra.extractor import Extractor

        built = Extractor(processed_data_home=processed, model_home=fake_model_home)
        name = "extraction"
        assert built.loader.cfg == built.cfg
    assert OmegaConf.to_container(built.cfg, resolve=True) == PACKAGED[name]


# ------------------------------------------------------------- model presets


def test_the_packaged_model_is_the_llama_32_preset():
    assert PACKAGED["training"]["model"] == PRESETS["llama_32"]
    assert re.search(r"^model: \*llama_32\b", packaged_text("training"), re.M)


def test_every_preset_names_a_model_these_tests_know_the_architecture_of():
    """so that a preset added to the yaml is checked below, rather than skipped"""
    assert {preset["model_name"] for preset in PRESETS.values()} == set(MODEL_TYPES)


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_preset_can_be_swapped_in_by_an_override(name):
    """as the README shows: `model=${model_presets.<name>}`"""

    class _Training(Configurable):
        default_file = "training.yaml"

    cfg = _Training(overrides=[f"model=${{model_presets.{name}}}"]).cfg
    assert OmegaConf.to_container(cfg.model, resolve=True) == PRESETS[name]


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_presets_model_args_are_fields_of_its_architectures_config(name):
    """
    `Trainer.model_init` hands them to `AutoConfig.from_pretrained`, which sets
    a keyword it doesn't recognize as one more attribute rather than refusing
    it, so a misspelled one would be ignored without a word -- a
    `num_hiden_layers` would leave the preset its full depth
    """
    preset = PRESETS[name]
    fields = AutoConfig.for_model(MODEL_TYPES[preset["model_name"]]).to_dict()
    assert [key for key in preset["model_args"] if key not in fields] == []


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_preset_builds_a_model_with_its_model_args(name, tmp_path):
    """
    along `Trainer.model_init`'s own path, from a stand-in for the hub's config
    that names just the architecture, with the tokenizer's vocabulary and
    special tokens, and on the meta device, so that nothing is allocated: every
    `model_args` value ends up on the built model's config
    """
    preset = PRESETS[name]
    hub = tmp_path / "hub-config"
    hub.mkdir()
    (hub / "config.json").write_text(
        json.dumps({"model_type": MODEL_TYPES[preset["model_name"]]})
    )
    lookup = {"BOS": 0, "EOS": 1} | {f"TOKEN//{i}": i for i in range(2, 1340)}
    stage = types.SimpleNamespace(
        cfg=OmegaConf.create(
            {"model": {"model_name": str(hub), "model_args": preset["model_args"]}}
        ),
        tkzr_cfg=OmegaConf.create({"lookup": lookup}),
        head_options={},
        logger=Logger(),
    )
    with t.device("meta"):
        model = Trainer.model_init(stage)
    for key, value in preset["model_args"].items():
        assert getattr(model.config, key) == value, key
    assert (model.config.vocab_size, model.config.bos_token_id) == (1340, 0)
    assert model.config.eos_token_id == 1
    assert model.get_input_embeddings().weight.shape == (
        1340,
        preset["model_args"]["hidden_size"],
    )


# ------------------------------------------- the values the anchors tie together


@pytest.mark.parametrize(
    "name, paths",
    [
        ("training", ["run_name", "wandb.run_name", "training_args.run_name"]),
        ("training", ["tokens_of_interest", "balanced_toi_loss.tokens_of_interest"]),
        ("extraction", ["max_seq_len", "extract.max_len"]),
        ("scoring", ["max_seq_len", "score.max_len"]),
        ("scoring", ["tokens_of_interest", "score.target_tokens"]),
    ],
    ids=lambda x: x if isinstance(x, str) else "=".join(x),
)
def test_the_values_a_packaged_yaml_ties_together_agree(name, paths):
    """
    each set an anchor and its aliases in the yaml: the run is named alike for
    the output, wandb and hf; the training loss emphasizes the tokens of
    interest, and scoring scores them; and extraction lays its `--all-times`
    features out over as many positions as it reads, which it has to, since a
    row read past `max_seq_len` doesn't fit
    """
    cfg = OmegaConf.create(PACKAGED[name])
    values = [OmegaConf.select(cfg, path) for path in paths]
    assert all(value == values[0] for value in values[1:]), dict(zip(paths, values))


# ---------------------------------------------------- the commented-out options

# the `*_objective` blocks training.yaml ships commented out, their keys
# commented under them; and the `training_args` it does likewise, indented
OBJECTIVES = r"^()# ((?:\w+_objective:|  \w+:) .*)$"
COMMENTED_TRAINING_ARGS = r"^(  )# (\w+: .*)$"


def uncommented(pattern: str) -> dict:
    """the packaged training.yaml with the lines `pattern` matches uncommented,
    each keeping the indent its first group holds"""
    text, n = re.subn(pattern, r"\1\2", packaged_text("training"), flags=re.M)
    assert n > 0, pattern
    return OmegaConf.to_container(OmegaConf.create(text), resolve=True)


def test_uncommenting_the_objective_blocks_asks_for_each_head_with_its_defaults(
    tokenizer_cfg,
):
    """
    every secondary head, each weighted 1.0, neither weighting per record, the
    time-to-next-token head a mixture of 8, and the disposition head over every
    `DSCG//*` token -- and nothing else changes
    """
    cfg = uncommented(OBJECTIVES)
    objectives = {"tte_objective", "tnt_objective", "disposition_objective"}
    assert set(cfg) - set(PACKAGED["training"]) == objectives
    assert {k: v for k, v in cfg.items() if k not in objectives} == PACKAGED["training"]
    lookup = tokenizer_cfg.lookup
    dispositions = sorted((k for k in lookup if k.startswith("DSCG//")), key=lookup.get)
    assert head_options(cfg, lookup) == {
        "tte": {"weight": 1.0, "per_record": False},
        "tnt": {"weight": 1.0, "mixture_components": 8},
        "disposition": {"weight": 1.0, "classes": dispositions, "per_record": False},
    }


def test_uncommented_the_objective_blocks_train_every_head(
    processed_with_end_times, tmp_path
):
    """a trainer given them, on data tokenized with end times, builds a model
    carrying the three heads, and its loss reaches each"""
    blocks = {k: v for k, v in uncommented(OBJECTIVES).items() if "_objective" in k}
    trainer = Trainer(
        training_cfg=write_cfg(
            tmp_path / "training.yaml", base_training_cfg() | blocks
        ),
        processed_data_home=processed_with_end_times,
        output_home=tmp_path / "out",
    )
    model = trainer.model_init()
    assert type(model) is CotorraForCausalLM
    assert list(model.heads) == ["tte", "tnt", "disposition"]
    assert type(model.heads["tte"]) is TteHead and not model.heads["tte"].per_record
    assert isinstance(model.heads["tnt"], TntMixtureHead)
    assert model.heads["tnt"].proj[-1].out_features == 1 + 3 * 8
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(4)])
    loss = trainer.trainer.compute_loss(model, batch)
    loss.backward()
    assert t.isfinite(loss)
    for head in model.heads.values():
        assert max(p.grad.abs().max().item() for p in head.parameters()) > 0


COMMENTED = uncommented(COMMENTED_TRAINING_ARGS)["training_args"]
COMMENTED = {
    k: v for k, v in COMMENTED.items() if k not in PACKAGED["training"]["training_args"]
}


def test_the_commented_training_args_are_the_ones_the_yaml_documents():
    assert COMMENTED == {
        "neftune_noise_alpha": 5.0,
        "bf16": True,
        "max_steps": 10,
        "gradient_checkpointing": True,
    }


# hf refuses bf16 without a gpu that supports it, unless told to train on the cpu
NEEDS_BF16_GPU = pytest.mark.skipif(
    not is_torch_bf16_gpu_available(), reason="needs a gpu that supports bf16"
)


@pytest.mark.parametrize(
    "key",
    [pytest.param(k, marks=NEEDS_BF16_GPU) if k == "bf16" else k for k in COMMENTED],
)
def test_each_commented_training_arg_is_valid_once_uncommented(key, tmp_path):
    """hf takes each, alongside the packaged arguments, as it's written"""
    shipped = PACKAGED["training"]["training_args"]
    args = TrainingArguments(
        output_dir=str(tmp_path), **shipped | {key: COMMENTED[key]}
    )
    assert getattr(args, key) == COMMENTED[key]
