#!/usr/bin/env python3

"""
huggingface causal lms with extra heads for the time-to-event and time-to-next-token
"""

import dataclasses

import torch as t
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    Cache,
    GenerationMixin,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import can_return_tuple

# keys generic huggingface code (and cotorra's own stages) expect to read off a
# top-level config, so they get mirrored up out of the wrapped backbone's config;
# `dtype` matters most: `_from_config` builds under the *top-level* dtype and
# overwrites the backbone's with it, so leaving it unset would quietly build in
# float32 a preset that the stock path builds in bfloat16
MIRRORED_KEYS = (
    "vocab_size",
    "hidden_size",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "tie_word_embeddings",
    "use_cache",
    "dtype",
)


def log1p_hours_mse(preds: t.Tensor, hours: t.Tensor) -> t.Tensor:
    """mean squared error between `preds` and `log1p(hours)`, taken only over the
    positions whose target is usable -- finite and non-negative. Summed and then
    normalized rather than masked-then-averaged so that a batch holding no usable
    target still yields a differentiable 0.0, keeping the head that made `preds`
    off the list of parameters distributed training considers unused"""
    preds = preds.to(dtype=t.float32)
    target = hours.to(device=preds.device, dtype=t.float32)
    keep = t.isfinite(target) & (target >= 0)
    return t.nn.functional.mse_loss(
        preds[keep], t.log1p(target[keep]), reduction="sum"
    ) / keep.sum().clamp(min=1)


class TteAwareConfig(PreTrainedConfig):
    """the configuration of any causal-lm backbone, nested under `text_config` and
    resolved through `AutoConfig`, plus the keys governing the time-to-event head"""

    model_type = "tte_aware"
    sub_configs = {"text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values", "tte_loss"]
    has_no_defaults_at_init = True

    text_config: dict | PreTrainedConfig | None = None
    tte_weight: float = 1.0
    vocab_size: int | None = None
    hidden_size: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool | None = None
    use_cache: bool | None = None

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            raise ValueError(
                f"a `{self.model_type}` config wraps a causal-lm backbone and so "
                "cannot be built without a `text_config`"
            )
        if isinstance(self.text_config, dict):
            self.text_config = AutoConfig.for_model(**self.text_config)
        for key in MIRRORED_KEYS:
            if getattr(self, key) is None:
                setattr(self, key, getattr(self.text_config, key, None))
        super().__post_init__(**kwargs)


@dataclasses.dataclass
class TteAwareCausalLMOutputWithPast(CausalLMOutputWithPast):
    """`CausalLMOutputWithPast` plus the time-to-event head's prediction (on the
    log1p-hours scale, so `expm1` recovers hours) and the loss it incurred; `loss`
    holds the two terms summed, `tte_loss` the time-to-event one on its own"""

    tte_pred: t.FloatTensor | None = None
    tte_loss: t.FloatTensor | None = None


class TteAwarePreTrainedModel(PreTrainedModel):
    """what the models below share: the backbone they wrap, with its
    language-modelling head, and the time-to-event loss"""

    config: TteAwareConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_backbone(self, config: TteAwareConfig):
        """the backbone and its language-modelling head, laid out as a
        llama-family `*ForCausalLM` is; to be called before `post_init`"""
        self.model = AutoModel.from_config(config.text_config)
        self.lm_head = t.nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        # every llama-family backbone calls its embedding `embed_tokens`, which is
        # what the equivalent class attribute hardcodes upstream, but the backbone
        # here is whatever `model_name` resolved to -- older architectures name it
        # `wte`/`word_embeddings`/`embed_in` and would fail tying on that guess
        embeddings = self.model.get_input_embeddings()
        self._tied_weights_keys = {
            "lm_head.weight": "model.{}.weight".format(
                next(n for n, mod in self.model.named_modules() if mod is embeddings)
            )
        }

    def tte_loss_function(self, tte_pred, hours_to_end_time, **kwargs):
        """`log1p_hours_mse`, left unshifted -- position i is scored against the
        hours remaining once token i has been read, which also keeps a packed
        sequence from scoring one record's last token against the next record's
        first target. The targets it masks do turn up: cocoa emits tokens
        recorded after the reference end time, and a missing end time arrives
        here as a nan"""
        return log1p_hours_mse(tte_pred, hours_to_end_time)


class TteAwareForCausalLM(TteAwarePreTrainedModel, GenerationMixin):
    """a causal lm carrying a second, scalar head that predicts the log1p-hours
    remaining until the end of the record as of the token just read (where the
    language-modelling head looks one token ahead, this one does not); laid out as
    a llama-family `*ForCausalLM` is, so for those
    backbones -- every one cotorra ships a preset for -- dropping `tte_head.*`
    leaves a state dict the stock class loads unchanged"""

    def __init__(self, config: TteAwareConfig):
        super().__init__(config)
        self._init_backbone(config)
        self.tte_head = t.nn.Linear(config.text_config.hidden_size, 1)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: t.LongTensor | None = None,
        attention_mask: t.Tensor | None = None,
        position_ids: t.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: t.FloatTensor | None = None,
        labels: t.LongTensor | None = None,
        hours_to_end_time: t.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | t.Tensor = 0,
        **kwargs,
    ) -> TteAwareCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        keep = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = outputs.last_hidden_state[:, keep, :]
        logits = self.lm_head(hidden_states)
        tte_pred = self.tte_head(hidden_states).squeeze(-1)

        loss = (
            self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
            if labels is not None
            else None
        )
        tte_loss = (
            self.tte_loss_function(tte_pred, hours_to_end_time, **kwargs)
            if hours_to_end_time is not None
            else None
        )
        if loss is not None and tte_loss is not None:
            loss = loss + self.config.tte_weight * tte_loss.to(dtype=loss.dtype)

        return TteAwareCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            tte_pred=tte_pred,
            tte_loss=tte_loss,
        )


class MppConfig(TteAwareConfig):
    """a `tte_aware` config plus the key governing the time-to-next-token head
    that makes the model a marked point process; here the time-to-event head is
    optional, built only when `tte_weight` is set"""

    model_type = "mpp"
    keys_to_ignore_at_inference = ["past_key_values", "tte_loss", "ttnt_loss"]

    tte_weight: float | None = None
    ttnt_weight: float = 1.0


@dataclasses.dataclass
class MppCausalLMOutputWithPast(TteAwareCausalLMOutputWithPast):
    """`TteAwareCausalLMOutputWithPast` plus the time-to-next-token head's
    prediction (also on the log1p-hours scale) and the loss it incurred; `loss`
    holds every term summed, `ttnt_loss` the time-to-next-token one on its own,
    and the `tte_*` fields stay empty for a model without that head"""

    ttnt_pred: t.FloatTensor | None = None
    ttnt_loss: t.FloatTensor | None = None


class MppForCausalLM(TteAwarePreTrainedModel, GenerationMixin):
    """a marked point process: a causal lm carrying a scalar head that predicts
    the log1p-hours from the token just read until the next one, so that where
    the language-modelling head predicts the next event's mark, this one predicts
    when it arrives -- plus, when the config sets a `tte_weight`, the time-to-event
    head `TteAwareForCausalLM` carries. Laid out as that class is, so dropping
    `ttnt_head.*` (and `tte_head.*`) leaves a state dict the stock class loads"""

    config: MppConfig

    def __init__(self, config: MppConfig):
        super().__init__(config)
        self._init_backbone(config)
        self.ttnt_head = t.nn.Linear(config.text_config.hidden_size, 1)
        # left out entirely rather than built and left untrained, which
        # `ddp_find_unused_parameters: false` would refuse
        self.tte_head = (
            t.nn.Linear(config.text_config.hidden_size, 1)
            if config.tte_weight is not None
            else None
        )
        self.post_init()

    def ttnt_loss_function(self, ttnt_pred, hours_to_next_token, **kwargs):
        """`log1p_hours_mse`, left unshifted like the time-to-event term: the
        target at position i already looks ahead, holding the hours from token i
        to token i+1 -- the token the language-modelling head at i predicts. A
        record's last token has no successor and arrives here as a nan, so a
        packed sequence never scores the gap across a record boundary"""
        return log1p_hours_mse(ttnt_pred, hours_to_next_token)

    @can_return_tuple
    def forward(
        self,
        input_ids: t.LongTensor | None = None,
        attention_mask: t.Tensor | None = None,
        position_ids: t.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: t.FloatTensor | None = None,
        labels: t.LongTensor | None = None,
        hours_to_end_time: t.Tensor | None = None,
        hours_to_next_token: t.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | t.Tensor = 0,
        **kwargs,
    ) -> MppCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        keep = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = outputs.last_hidden_state[:, keep, :]
        logits = self.lm_head(hidden_states)
        ttnt_pred = self.ttnt_head(hidden_states).squeeze(-1)
        if self.tte_head is not None:
            tte_pred = self.tte_head(hidden_states).squeeze(-1)
        elif hours_to_end_time is not None:
            raise ValueError(
                "got `hours_to_end_time` but this `mpp` model was built without a "
                "time-to-event head; set `tte_weight` in its config to have one"
            )
        else:
            tte_pred = None

        loss = (
            self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )
            if labels is not None
            else None
        )
        tte_loss = (
            self.tte_loss_function(tte_pred, hours_to_end_time, **kwargs)
            if hours_to_end_time is not None
            else None
        )
        ttnt_loss = (
            self.ttnt_loss_function(ttnt_pred, hours_to_next_token, **kwargs)
            if hours_to_next_token is not None
            else None
        )
        if loss is not None and tte_loss is not None:
            loss = loss + self.config.tte_weight * tte_loss.to(dtype=loss.dtype)
        if loss is not None and ttnt_loss is not None:
            loss = loss + self.config.ttnt_weight * ttnt_loss.to(dtype=loss.dtype)

        return MppCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            tte_pred=tte_pred,
            tte_loss=tte_loss,
            ttnt_pred=ttnt_pred,
            ttnt_loss=ttnt_loss,
        )


# importing this module is what teaches the auto classes to resolve a saved
# `mdl-<run_name>/` back to the pairs defined above
AutoConfig.register(TteAwareConfig.model_type, TteAwareConfig, exist_ok=True)
AutoModelForCausalLM.register(TteAwareConfig, TteAwareForCausalLM, exist_ok=True)
AutoConfig.register(MppConfig.model_type, MppConfig, exist_ok=True)
AutoModelForCausalLM.register(MppConfig, MppForCausalLM, exist_ok=True)


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    # go through `from_config` rather than calling the class: that is what puts the
    # whole model, heads included, under the config's dtype
    self = AutoModelForCausalLM.from_config(
        TteAwareConfig(text_config=trainer.model.config)
    )
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(2)])
    outputs = self(**batch)
    # breakpoint()
