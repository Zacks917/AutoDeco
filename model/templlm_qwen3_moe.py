from typing import Optional, Union, Tuple, List, Any

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

# HF Qwen3-MoE classes and utilities
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeModel,
        Qwen3MoePreTrainedModel,
        load_balancing_loss_func,
    )
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
except Exception as e:  # pragma: no cover - provide a clearer import error at runtime
    raise ImportError(
        "Qwen3-MoE components not found. Please install transformers>=4.51 with Qwen3-MoE support."
    ) from e

# Optional decorators used by upstream Qwen implementations
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import auto_docstring, can_return_tuple
except Exception:
    def auto_docstring(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn
        return _wrap

    def can_return_tuple(fn):  # type: ignore
        return fn

# Reuse common kwargs protocol and heads from the non-MoE variant
from .templlm_qwen3 import TempHead, TopPHead  # noqa: E402
from transformers.utils import ModelOutput


logger = logging.get_logger(__name__)

@dataclass
class TempLLMMoECausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    temp_loss: Optional[torch.FloatTensor] = None
    top_p_loss: Optional[torch.FloatTensor] = None
    top_k_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None
    top_k_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

class TempLLMQwen3MoeConfig(Qwen3MoeConfig):
    """Configuration class for TempLLM Qwen3-MoE model with training parameters."""

    def __init__(
        self,
        train_temp: bool = False,
        train_top_p: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.train_temp = train_temp
        self.train_top_p = train_top_p


class TempLLMQwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Temperature and Top-P heads
        hidden_size = config.hidden_size
        self.temp_head = TempHead(hidden_size)
        self.top_p_head = TopPHead(hidden_size)

        # Training toggles
        self.train_temp = getattr(config, "train_temp", False)
        self.train_top_p = getattr(config, "train_top_p", False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> TempLLMMoECausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # Keep original logits for loss calculation to ensure training stability
        with torch.no_grad():
            unscaled_logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Temperature prediction (range ~ [0, 2])
        temp_logits = self.temp_head(hidden_states[:, slice_indices, :])

        # Top-p prediction in (0,1), conditioned on temp and prob stats
        top_p_logits = self.top_p_head(
            hidden_states[:, slice_indices, :],
            temp_logits.detach(),
            unscaled_logits=unscaled_logits,
        )

        # Losses
        loss, lm_loss, temp_loss, top_p_loss = None, None, None, None

        if labels is not None and (self.train_temp or self.train_top_p):
            # Shift for next-token prediction
            unscaled_shift = unscaled_logits[:, :-1, :]
            temp_shift = temp_logits[:, :-1, :]
            top_p_shift = top_p_logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            if self.train_temp:
                valid_mask = shift_labels != -100  # [B, S-1]
                with torch.no_grad():
                    base_probs = torch.softmax(unscaled_shift, dim=-1)
                    pred_ids = base_probs.argmax(dim=-1)
                    correct_positions = (pred_ids == shift_labels.clamp_min(0))
                    rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
                    drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
                    masked_valid_mask = valid_mask & (~drop_mask)

                scaled_shift = unscaled_shift / temp_shift.clamp_min(1e-8)
                scaled_valid = scaled_shift[masked_valid_mask]
                unscaled_valid = unscaled_shift[masked_valid_mask]
                labels_valid = shift_labels[masked_valid_mask]

                if labels_valid.numel() > 0:
                    token_ce = torch.nn.functional.cross_entropy(
                        scaled_valid.view(-1, scaled_valid.size(-1)),
                        labels_valid.view(-1),
                        reduction='none',
                        label_smoothing=0.0,
                    )
                    token_weights = torch.softmax(unscaled_valid, dim=-1).gather(1, labels_valid.unsqueeze(-1)).squeeze(-1).detach()
                    token_ce = token_ce * token_weights
                    temp_loss = token_ce.mean()
                    lm_loss = temp_loss
                    loss = lm_loss

            if self.train_top_p:
                with torch.no_grad():
                    scaled_shift = unscaled_shift / temp_shift.clamp_min(1e-8)
                    probs_sorted, idx_sorted = torch.softmax(scaled_shift, dim=-1).sort(dim=-1, descending=True)
                    rank = (idx_sorted == shift_labels.unsqueeze(-1)).float().argmax(dim=-1)
                    cumsum = probs_sorted.cumsum(dim=-1)
                    inferred_top_p_labels = cumsum.gather(-1, rank.unsqueeze(-1)).squeeze(-1).detach()
                    valid_mask = shift_labels != -100
                    base_probs = torch.softmax(unscaled_shift, dim=-1)
                    pred_ids = base_probs.argmax(dim=-1)
                    correct_positions = (pred_ids == shift_labels.clamp_min(0))
                    rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
                    drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
                    masked_valid_mask = valid_mask & (~drop_mask)

                top_p_pred = top_p_shift.squeeze(-1)
                target = inferred_top_p_labels
                error = top_p_pred - target
                lambda_under = 2.0
                lambda_over = 1.0
                per_token_loss = torch.where(error < 0, lambda_under * error.pow(2), lambda_over * error.pow(2))
                top_p_loss = per_token_loss[masked_valid_mask].mean()
                loss = top_p_loss

        # # Aux/router loss
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return TempLLMMoECausalLMOutputWithPast(
            loss=loss,
            lm_loss=lm_loss,
            temp_loss=temp_loss,
            top_p_loss=top_p_loss,
            aux_loss=aux_loss,
            logits=unscaled_logits,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )



