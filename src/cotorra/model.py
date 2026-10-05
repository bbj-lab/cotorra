#!/usr/bin/env python3

"""
huggingface causal lms carrying any combination of cotorra's secondary heads
"""

import copy
import dataclasses
import fnmatch
import math
from typing import ClassVar

import torch as t
from omegaconf import DictConfig, OmegaConf
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


class ZeroInflatedLogNormalMixture:
    """
    a distribution over the hours until the next token: exactly 0 -- the next
    token shares this one's timestamp, as the tokens of a single event and of a
    panel recorded at once do -- with probability `sigmoid(zero_logit)`, and
    otherwise log-normal, its log-hours a mixture of gaussians. Built from the
    raw parameters a `TntMixtureHead` emits, `1 + 3 * n_components` of them along
    the last axis, and computed in float32 at least, whatever the model's dtype
    """

    # a sampled gap is capped at a million hours (~114 years), so that a far tail
    # draw can't overflow the clock it gets added to
    max_log_hours = math.log(1e6)

    def __init__(self, params: t.Tensor):
        params = params.to(dtype=t.promote_types(params.dtype, t.float32))
        k = (params.shape[-1] - 1) // 3
        self.zero_logit = params[..., 0]
        self.log_weights = params[..., 1 : 1 + k].log_softmax(dim=-1)
        self.means = params[..., 1 + k : 1 + 2 * k]
        self.scales = t.nn.functional.softplus(params[..., 1 + 2 * k :]) + 1e-3

    def log_prob(self, hours: t.Tensor) -> t.Tensor:
        """the log-likelihood of `hours`, a positive gap's density taken on the
        log-hours scale, where the mixture lives"""
        hours = hours.to(device=self.means.device, dtype=self.means.dtype)
        # finite even at hours == 0, a branch `where` discards but backward visits
        log_hours = hours.clamp(min=1e-6).log()[..., None]
        log_normal = (
            -0.5 * ((log_hours - self.means) / self.scales) ** 2
            - self.scales.log()
            - 0.5 * math.log(2 * math.pi)
        )
        positive = t.nn.functional.logsigmoid(-self.zero_logit) + (
            self.log_weights + log_normal
        ).logsumexp(dim=-1)
        return t.where(
            hours == 0, t.nn.functional.logsigmoid(self.zero_logit), positive
        )

    def sample(self, generator: t.Generator | None = None) -> t.Tensor:
        shape, device = self.zero_logit.shape, self.zero_logit.device
        zero = t.rand(shape, generator=generator, device=device) < (
            self.zero_logit.sigmoid()
        )
        component = t.multinomial(
            self.log_weights.exp().reshape(-1, self.means.shape[-1]),
            1,
            generator=generator,
        ).reshape(*shape, 1)
        log_hours = self.means.gather(-1, component).squeeze(-1) + self.scales.gather(
            -1, component
        ).squeeze(-1) * t.randn(shape, generator=generator, device=device)
        return t.where(zero, 0.0, log_hours.clamp(max=self.max_log_hours).exp())

    def point_estimate(self) -> t.Tensor:
        """a deterministic stand-in for a draw, for greedy decoding: 0 when that
        is the likelier outcome, else the geometric mean of the positive part"""
        log_hours = (self.log_weights.exp() * self.means).sum(dim=-1)
        return t.where(
            self.zero_logit >= 0, 0.0, log_hours.clamp(max=self.max_log_hours).exp()
        )


# ------------------------------------------------------------------ the heads


class Head(t.nn.Module):
    """
    a secondary objective, trained alongside next-token prediction: a module that
    reads the backbone's last hidden states and is scored, at every position,
    against a per-token target. Where the language-modelling head at position i
    is scored against token i+1, a head is scored against position i's own
    target, which says what the head should know once token i has been read.

    A subclass sets `name` -- which keys its `<name>_objective` training block,
    its entry in `CotorraConfig.heads`, and the `<name>_pred`/`<name>_loss`
    fields of `CotorraCausalLMOutputWithPast` -- and `target`, the batch column
    it is scored against, and implements `predict` and `loss`. A loss summed over
    the usable positions and then normalized leaves a batch with none of them a
    differentiable 0, so the head never goes without a gradient, which
    `ddp_find_unused_parameters: false` would refuse
    """

    name: ClassVar[str]
    target: ClassVar[str]

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @classmethod
    def options_from(cls, block: dict, lookup) -> dict:
        """the options this head's `<name>_objective` training block gives it,
        resolved against the tokenizer's `lookup`; every head takes a `weight`
        on its term, 1.0 unless the block says otherwise"""
        return {"weight": 1.0, **block}

    @classmethod
    def build(cls, hidden_size: int, **options) -> "Head":
        """the head `options` describe, for a backbone of `hidden_size`"""
        return cls(hidden_size, **options)

    def predict(self, hidden_states: t.Tensor) -> t.Tensor | None:
        """the prediction at every position, or None for a head that has no
        single one to give"""
        raise NotImplementedError

    def loss(self, pred: t.Tensor | None, target: t.Tensor, **context) -> t.Tensor:
        """`pred` scored against `target`; `context` carries the `hidden_states`
        it came from, the `input_ids`, and the backbone's `input_embeddings`, for
        a head that needs more than its prediction"""
        raise NotImplementedError

    def forward(
        self, hidden_states: t.Tensor, target: t.Tensor | None = None, **context
    ) -> tuple[t.Tensor | None, t.Tensor | None]:
        """the prediction and, given a target, the loss"""
        pred = self.predict(hidden_states)
        if target is None:
            return pred, None
        return pred, self.loss(pred, target, hidden_states=hidden_states, **context)


class Log1pHoursHead(Head):
    """
    a point estimate of a number of hours: a linear read of the hidden state on
    the log1p-hours scale, kept non-negative -- so `expm1` never gives fewer than
    0 hours -- by a softplus, which unlike a clamp still passes a gradient to a
    head whose raw output has drifted below zero; scored by `log1p_hours_mse`
    """

    def __init__(self, hidden_size: int, weight: float = 1.0):
        super().__init__(weight)
        self.linear = t.nn.Linear(hidden_size, 1)

    def predict(self, hidden_states):
        return t.nn.functional.softplus(self.linear(hidden_states)).squeeze(-1)

    def loss(self, pred, target, **context):
        return log1p_hours_mse(pred, target)


class TteHead(Log1pHoursHead):
    """
    time to event: the hours remaining until the record's end time, as of the
    token just read. Cocoa emits tokens recorded after the end time, with
    negative hours; predicting past the end matters little, so those count as 0
    hours remaining rather than being masked. A missing end time arrives as a
    nan, which stays masked (`clamp` keeps it a nan). Unshifted, which also keeps
    a packed sequence from scoring one record's last token against the next
    record's first target
    """

    name = "tte"
    target = "hours_to_end_time"

    def loss(self, pred, target, **context):
        return super().loss(pred, target.clamp(min=0))


class TntHead(Head):
    """
    time to next token: the hours from the token just read until the next one,
    which makes the model a marked point process -- the language-modelling head
    predicts the next event's mark, this head when it arrives. Position i's
    target already looks ahead, holding the gap from token i to token i+1; a
    record's last token has no successor and arrives as a nan, so a packed
    sequence never scores the gap across a record boundary. Built as a
    `TntPointHead` unless `mixture_components` is set, which makes it the
    `TntMixtureHead` that generation needs
    """

    name = "tnt"
    target = "hours_to_next_token"

    @classmethod
    def build(cls, hidden_size, mixture_components=None, **options):
        if mixture_components is None:
            return TntPointHead(hidden_size, **options)
        return TntMixtureHead(hidden_size, mixture_components, **options)

    def sample_hours(
        self,
        hidden_states: t.Tensor,
        next_embeds: t.Tensor,
        do_sample: bool = True,
        generator: t.Generator | None = None,
    ) -> t.Tensor:
        """the hours from each hidden state's position until the token embedded
        in `next_embeds`: a draw, or with `do_sample=False` a point estimate"""
        raise NotImplementedError


class TntPointHead(TntHead, Log1pHoursHead):
    """a single value per token, whatever token comes next, trained by squared
    error on log1p-hours"""

    def sample_hours(self, hidden_states, next_embeds, do_sample=True, generator=None):
        """the prediction, which is all a point head has to give, whatever the
        next token -- the reason to train a mixture head for generation"""
        log1p_hours = self.predict(hidden_states)
        log1p_hours = log1p_hours.to(
            dtype=t.promote_types(log1p_hours.dtype, t.float32)
        )
        return log1p_hours.expm1()


class TntMixtureHead(TntHead):
    """
    maps a hidden state, together with the embedding of the token that follows
    it, to a `ZeroInflatedLogNormalMixture` over the hours until that token,
    trained by likelihood. Conditioning on the next token is what lets the gap
    depend on what arrives: the quantile token completing an event comes at
    once, a new event later. A hidden layer lets the two inputs interact rather
    than merely add. The embedding is layer-normed on the way in: the hidden
    state arrives through the backbone's final norm at a scale near 1, a raw
    input embedding at its init's ~0.02, and left that much quieter the next
    token went all but unheard -- a head trained alongside the other objectives
    learned the zero-gap rate but not which tokens arrive at once
    """

    def __init__(self, hidden_size: int, mixture_components: int, weight: float = 1.0):
        if mixture_components < 1:
            raise ValueError(
                "`mixture_components` counts the log-normal components of the "
                "time-to-next-token head, so it must be positive; got "
                f"{mixture_components}"
            )
        super().__init__(weight)
        self.next_norm = t.nn.LayerNorm(hidden_size)
        self.proj = t.nn.Sequential(
            t.nn.Linear(2 * hidden_size, hidden_size),
            t.nn.GELU(),
            t.nn.Linear(hidden_size, 1 + 3 * mixture_components),
        )

    def distribution(
        self, hidden_states: t.Tensor, next_embeds: t.Tensor
    ) -> ZeroInflatedLogNormalMixture:
        return ZeroInflatedLogNormalMixture(
            self.proj(t.cat([hidden_states, self.next_norm(next_embeds)], dim=-1))
        )

    def predict(self, hidden_states):
        return None  # what it predicts depends on the next token

    def loss(
        self, pred, target, *, hidden_states, input_ids=None, input_embeddings=None, **_
    ):
        """
        the mean negative log-likelihood, position i's distribution conditioned
        on token i+1 -- known here, sampled first in generation. The last
        position has no next token in the sequence to condition on, so it is
        left out, along with the targets `log1p_hours_mse` masks; summed and
        then normalized for the same reason as there
        """
        if input_ids is None:
            raise ValueError(
                "a mixture time-to-next-token head conditions on the next token, "
                "so scoring `hours_to_next_token` needs `input_ids`"
            )
        dist = self.distribution(
            hidden_states[:, :-1], input_embeddings(input_ids[:, 1:])
        )
        target = target[:, :-1].to(device=dist.means.device, dtype=t.float32)
        keep = t.isfinite(target) & (target >= 0)
        nll = -dist.log_prob(t.where(keep, target, 0.0))
        return (nll * keep).sum() / keep.sum().clamp(min=1)

    def sample_hours(self, hidden_states, next_embeds, do_sample=True, generator=None):
        dist = self.distribution(hidden_states, next_embeds)
        return dist.sample(generator) if do_sample else dist.point_estimate()


class DispositionHead(Head):
    """
    the record's discharge disposition, one of `classes` -- the tokens that mark
    it, `DSCG//home`, `DSCG//expired`, ... -- as logits over them at every
    position, trained by cross-entropy, so that their softmax gives the chance
    of each: a supervised head, for questions like which patients die before
    discharge. Position i is scored against the disposition its record ends
    with, while that is still to come; from the disposition token on, and
    throughout a record whose disposition is none of `classes`, the target is
    -100 and the position goes unscored
    """

    name = "disposition"
    target = "disposition"
    default_classes = ("DSCG//*",)

    def __init__(self, hidden_size: int, classes: list[str], weight: float = 1.0):
        if not classes:
            raise ValueError("a disposition head needs at least one class")
        super().__init__(weight)
        self.classes = list(classes)
        self.linear = t.nn.Linear(hidden_size, len(self.classes))

    @classmethod
    def options_from(cls, block, lookup):
        """`classes` given as fnmatch patterns -- `DSCG//*` unless the block says
        otherwise -- resolved to the tokens they match, in token-id order"""
        options = super().options_from(block, lookup)
        patterns = options.get("classes", cls.default_classes)
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        options["classes"] = [
            tok
            for tok in sorted(lookup, key=lookup.get)
            if any(fnmatch.fnmatch(tok, p) for p in patterns)
        ]
        if not options["classes"]:
            raise ValueError(
                f"`disposition_objective.classes` {patterns!r} matched no token in "
                "the vocabulary"
            )
        return options

    def predict(self, hidden_states):
        return self.linear(hidden_states)

    def loss(self, pred, target, **context):
        target = target.to(device=pred.device, dtype=t.long)
        nll = t.nn.functional.cross_entropy(
            pred.flatten(0, -2).to(dtype=t.float32),
            target.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        return nll / (target != -100).sum().clamp(min=1)


# every secondary objective, by name; adding one takes a `Head` subclass here and
# its `<name>_pred`/`<name>_loss` fields on `CotorraCausalLMOutputWithPast`
HEADS: dict[str, type[Head]] = {
    head.name: head for head in (TteHead, TntHead, DispositionHead)
}


def head_options(cfg, lookup) -> dict[str, dict]:
    """
    the heads a training config asks for -- one per `<name>_objective` block,
    any combination of them -- each with the options its block gives, resolved
    against the tokenizer's `lookup`; a block left empty parses to None and takes
    every default. A block naming no head is refused rather than silently
    training without it
    """
    if unknown := [
        key
        for key in cfg
        if str(key).endswith("_objective")
        and str(key).removesuffix("_objective") not in HEADS
    ]:
        raise ValueError(
            f"no secondary head answers to {unknown}; the objectives are "
            f"{[f'{name}_objective' for name in HEADS]}"
        )
    options = {}
    for name, head in HEADS.items():
        if (key := f"{name}_objective") in cfg:
            block = cfg[key]
            if isinstance(block, DictConfig):
                block = OmegaConf.to_container(block, resolve=True)
            options[name] = head.options_from(dict(block or {}), lookup)
    return options


# ------------------------------------------------------------------ the model


class CotorraConfig(PreTrainedConfig):
    """the configuration of any causal-lm backbone, nested under `text_config` and
    resolved through `AutoConfig`, plus `heads`: the secondary heads to carry,
    each name in `HEADS` mapped to its options, as `head_options` gives them --
    `{"tte": {"weight": 1.0}, "tnt": {"weight": 1.0, "mixture_components": 8}}`"""

    model_type = "cotorra"
    sub_configs = {"text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"] + [f"{n}_loss" for n in HEADS]
    has_no_defaults_at_init = True

    text_config: dict | PreTrainedConfig | None = None
    heads: dict | None = None
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
        self.heads = copy.deepcopy(dict(self.heads or {}))
        if unknown := sorted(set(self.heads) - set(HEADS)):
            raise ValueError(f"no secondary head is named {unknown}; see `HEADS`")
        for key in MIRRORED_KEYS:
            if getattr(self, key) is None:
                setattr(self, key, getattr(self.text_config, key, None))
        super().__post_init__(**kwargs)


@dataclasses.dataclass
class CotorraCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    `CausalLMOutputWithPast` plus each head's prediction and loss, left empty
    for a head the model doesn't carry (and the loss for one given no target);
    `loss` holds the language-modelling term plus every head's weighted one. The
    time heads predict on the log1p-hours scale, non-negative, so `expm1`
    recovers hours, never fewer than 0; a mixture time-to-next-token head has no
    single prediction to give, so it leaves `tnt_pred` empty
    (`CotorraForCausalLM.tnt_distribution` gives what it predicts); and
    `disposition_pred` holds logits over the disposition head's `classes`
    """

    tte_pred: t.FloatTensor | None = None
    tte_loss: t.FloatTensor | None = None
    tnt_pred: t.FloatTensor | None = None
    tnt_loss: t.FloatTensor | None = None
    disposition_pred: t.FloatTensor | None = None
    disposition_loss: t.FloatTensor | None = None


class CotorraForCausalLM(PreTrainedModel, GenerationMixin):
    """
    a causal lm carrying any combination of the secondary heads in `HEADS`, as
    `config.heads` lists them, each trained jointly with next-token prediction.
    Laid out as a llama-family `*ForCausalLM` is, with the heads under
    `heads.<name>`, so for those backbones -- every one cotorra ships a preset
    for -- dropping `heads.*` leaves a state dict the stock class loads unchanged
    """

    config: CotorraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: CotorraConfig):
        super().__init__(config)
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
        # only the heads asked for: one built and left untrained would cost
        # parameters, and `ddp_find_unused_parameters: false` would refuse it
        self.heads = t.nn.ModuleDict(
            {
                name: HEADS[name].build(config.text_config.hidden_size, **options)
                for name, options in config.heads.items()
            }
        )
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
        use_cache: bool | None = None,
        logits_to_keep: int | t.Tensor = 0,
        **kwargs,
    ) -> CotorraCausalLMOutputWithPast:
        """the stock causal-lm forward, plus each head's prediction; a head's
        target, passed under the name its `target` gives (`hours_to_end_time`,
        `hours_to_next_token`, `disposition`), scores it too"""
        # popped before `kwargs` reaches the backbone, whose own `**kwargs` would
        # otherwise swallow a target meant for a head this model doesn't carry
        targets = {h.target: kwargs.pop(h.target, None) for h in HEADS.values()}
        carried = {head.target for head in self.heads.values()}
        if stray := [
            k for k, v in targets.items() if v is not None and k not in carried
        ]:
            raise ValueError(
                f"got `{stray[0]}` but this model carries no head scored against it; "
                "add the head's objective to the training config to build one"
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        keep = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, keep, :])
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

        fields = dict()
        for name, head in self.heads.items():
            # scored at every position, whatever `logits_to_keep` trims from what
            # gets returned
            pred, head_loss = head(
                hidden_states,
                targets[head.target],
                input_ids=input_ids,
                input_embeddings=self.get_input_embeddings(),
            )
            fields[f"{name}_pred"] = None if pred is None else pred[:, keep]
            fields[f"{name}_loss"] = head_loss
            if loss is not None and head_loss is not None:
                loss = loss + head.weight * head_loss.to(dtype=loss.dtype)

        return CotorraCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            **fields,
        )

    # what generation asks of the time-to-next-token head, given token ids rather
    # than the embeddings the head itself reads

    def _tnt_head(self) -> TntHead:
        if "tnt" not in self.heads:
            raise ValueError(
                "this model carries no time-to-next-token head; train it with a "
                "`tnt_objective` to have one"
            )
        return self.heads["tnt"]

    def tnt_distribution(
        self, hidden_states: t.Tensor, next_input_ids: t.LongTensor
    ) -> ZeroInflatedLogNormalMixture:
        """a mixture time-to-next-token head's distribution over the hours from
        each hidden state's position until the token `next_input_ids` holds"""
        if not isinstance(head := self._tnt_head(), TntMixtureHead):
            raise ValueError(
                "this model's time-to-next-token head is a point estimate; set "
                "`mixture_components` in its objective for a distribution"
            )
        return head.distribution(
            hidden_states, self.get_input_embeddings()(next_input_ids)
        )

    def sample_hours_to_next_token(
        self,
        hidden_states: t.Tensor,
        next_input_ids: t.LongTensor,
        do_sample: bool = True,
        generator: t.Generator | None = None,
    ) -> t.Tensor:
        """the hours from each hidden state's position until `next_input_ids`:
        drawn from a mixture head's distribution, or with `do_sample=False` its
        point estimate; a point head gives its prediction either way"""
        return self._tnt_head().sample_hours(
            hidden_states,
            self.get_input_embeddings()(next_input_ids),
            do_sample=do_sample,
            generator=generator,
        )


# importing this module is what teaches the auto classes to resolve a saved
# `mdl-<run_name>/` back to the pair defined above
AutoConfig.register(CotorraConfig.model_type, CotorraConfig, exist_ok=True)
AutoModelForCausalLM.register(CotorraConfig, CotorraForCausalLM, exist_ok=True)


if __name__ == "__main__":
    from cotorra.trainer import Trainer

    trainer = Trainer(
        processed_data_home="./processed/mimic", output_home="./output/mimic"
    )
    # go through `from_config` rather than calling the class: that is what puts the
    # whole model, heads included, under the config's dtype
    self = AutoModelForCausalLM.from_config(
        CotorraConfig(
            text_config=trainer.model.config,
            heads=head_options(
                {"tte_objective": None, "tnt_objective": None}, trainer.tkzr_cfg.lookup
            ),
        )
    )
    batch = trainer.collate_fn([trainer.trainer.train_dataset[i] for i in range(2)])
    outputs = self(**batch)
    # breakpoint()
