# coding=utf-8
from typing import Optional, Tuple, Union, Callable, List, Dict, Any

import inspect
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dataclasses import dataclass

from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.utils import GenerateOutput, GenerateNonBeamOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_available
from transformers.modeling_utils import can_return_tuple, auto_docstring
from transformers.generation.utils_base import TransformersKwargs

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssModel,
    GptOssPreTrainedModel,
    load_balancing_loss_func,
)

from transformers.generation.utils import GenerationMixin

logger = logging.get_logger(__name__)


class TempLLMGptOssConfig(GptOssConfig):
    def __init__(self, train_temp: bool = False, train_top_p: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.train_temp = train_temp
        self.train_top_p = train_top_p




class TopPHead(nn.Module):
    def __init__(self, hidden_size, vocab_size=None, use_enhanced_features=True):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        input_dim = hidden_size + 1 + (4 if use_enhanced_features else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.vocab_size = vocab_size

    def compute_prob_stats(self, logits):
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        prob_var = probs.var(dim=-1, keepdim=True)
        top5_probs, _ = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        top5_sum = top5_probs.sum(dim=-1, keepdim=True)
        return torch.cat([max_prob, entropy, prob_var, top5_sum], dim=-1)

    def forward(self, hidden_states, temp_logits, unscaled_logits=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        if self.use_enhanced_features:
            features = [hidden_states, temp_logits]
            if unscaled_logits is not None:
                scaled_logits = unscaled_logits / (temp_logits + 1e-8)
                prob_stats = self.compute_prob_stats(scaled_logits)
                features.append(prob_stats)
            else:
                prob_stats = torch.zeros(batch_size, seq_len, 4, device=hidden_states.device)
                features.append(prob_stats)
            combined_features = torch.cat(features, dim=-1)
        else:
            combined_features = torch.cat([hidden_states, temp_logits], dim=-1)
        return self.mlp(combined_features)


class TempHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states):
        sigmoid_output = self.mlp(hidden_states)
        return sigmoid_output * 2


@dataclass
class TempLLMCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    temp_loss: Optional[torch.FloatTensor] = None
    top_p_loss: Optional[torch.FloatTensor] = None
    top_k_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None
    top_k_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class GptOssBackbone(PreTrainedModel):
    config_class = GptOssConfig


@auto_docstring
class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel"]