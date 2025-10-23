from pickle import FALSE
import re
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.generation import GenerationMixin
import inspect
import transformers
from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList
)
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.generation.utils import GenerateOutput, GenerateNonBeamOutput
from transformers.utils import ModelOutput
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from dataclasses import dataclass
import logging
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TempLLMLlamaConfig(LlamaConfig):
    """Configuration class for TempLLM Llama model with training parameters."""
    
    def __init__(
        self,
        train_temp: bool = False,
        train_top_p: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_temp = train_temp
        self.train_top_p = train_top_p



@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    temps: Optional[List] = None
    top_p: Optional[List] = None

# class TopPHead(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         # self.input_norm = nn.LayerNorm(hidden_size)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, hidden_states):
#         # normalized_hidden_state = self.input_norm(hidden_states)
#         # return self.mlp(normalized_hidden_state)
#         return self.mlp(hidden_states)

class TopPHead(nn.Module):
    def __init__(self, hidden_size, vocab_size=None, use_enhanced_features=False):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        
        if use_enhanced_features:
            # Enhanced features: hidden_states + temp + prob_stats
            # hidden_states: hidden_size
            # temp_logits: 1
            # prob_stats: 4 (max_prob, entropy, variance, top5_sum)
            input_dim = hidden_size + 1 + 4
        else:
            # Original: hidden_states + temp
            input_dim = hidden_size + 1
            
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.vocab_size = vocab_size

    def compute_prob_stats(self, logits):
        """Compute probability distribution statistics"""
        probs = torch.softmax(logits, dim=-1)
        
        # Max probability
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        
        # Entropy
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        
        # Variance of probabilities
        prob_var = probs.var(dim=-1, keepdim=True)
        
        # Top-5 probability sum
        top5_probs, _ = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        top5_sum = top5_probs.sum(dim=-1, keepdim=True)
        
        return torch.cat([max_prob, entropy, prob_var, top5_sum], dim=-1)

    def forward(self, hidden_states, temp_logits, unscaled_logits=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            temp_logits: [batch_size, seq_len, 1]
            unscaled_logits: [batch_size, seq_len, vocab_size] (optional, for prob stats)
            position_ids: [batch_size, seq_len] (optional, for position features)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.use_enhanced_features:
            features = [hidden_states, temp_logits]
            
            # Add probability statistics if unscaled_logits provided
            if unscaled_logits is not None:
                # Use temperature-scaled distribution to better align with the target top-p label
                scaled_logits = unscaled_logits / (temp_logits + 1e-8)
                prob_stats = self.compute_prob_stats(scaled_logits)
                features.append(prob_stats)
            else:
                # Use zeros if not provided
                prob_stats = torch.zeros(batch_size, seq_len, 4, device=hidden_states.device)
                features.append(prob_stats)
            
            # Position features are disabled per design choice
            
            # Concatenate all features
            combined_features = torch.cat(features, dim=-1)
        else:
            # Original implementation
            combined_features = torch.cat([hidden_states, temp_logits], dim=-1)
        
        return self.mlp(combined_features)

class TempHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states):
        sigmoid_output = self.mlp(hidden_states)
        return sigmoid_output * 2

@dataclass
class TempLLMCausalLMOutputWithPast(ModelOutput):
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
    logits: Optional[torch.FloatTensor] = None
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None
    top_k_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class TempLLMLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Training parameters from config
        self.train_temp = getattr(config, 'train_temp', False)
        self.train_top_p = getattr(config, 'train_top_p', False)

        self.temp_head = TempHead(config.hidden_size)
        self.top_p_head = TopPHead(config.hidden_size)

        self.mse_criteria = nn.MSELoss(reduction='none')
        self.ce_criteria = nn.CrossEntropyLoss(reduction='none')
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> TempLLMCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        with torch.no_grad():
            unscaled_logits = self.lm_head(hidden_states[:, slice_indices, :])

        temp_logits = self.temp_head(hidden_states[:, slice_indices, :])
        top_p_logits = self.top_p_head(
            hidden_states[:, slice_indices, :], 
            temp_logits.detach(),
            unscaled_logits=unscaled_logits,
        )

        loss, lm_loss, temp_loss, top_p_loss, top_k_loss = None, None, None, None, None
        
        if labels is not None:
            if self.train_temp:
                ############ mask loss ############## 
                unscaled_shift = unscaled_logits[:, :-1, :]
                temp_shift = temp_logits[:, :-1, :].clamp_min(1e-2)
                shift_labels = labels[:, 1:]
                valid_mask = shift_labels != -100  # [B, S-1]
                # Base-model confidence of GT token (no grad and independent of temp)
                with torch.no_grad():
                    base_probs = torch.softmax(unscaled_shift, dim=-1)
                    # p_gt_base = base_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # [B, S-1]
                    pred_ids = base_probs.argmax(dim=-1)  # [B, S-1]
                    correct_positions = (pred_ids == shift_labels.clamp_min(0))

                # Randomly drop 60% of positions where model argmax == GT
                rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
                drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
                
                masked_valid_mask = valid_mask & (~drop_mask)

                scaled_shift = unscaled_shift / temp_shift
                # Select only valid tokens
                scaled_valid = scaled_shift[masked_valid_mask]
                unscaled_valid = unscaled_shift[masked_valid_mask]
                labels_valid = shift_labels[masked_valid_mask]

                # Per-token CE with label smoothing
                if labels_valid.numel() > 0:
                    token_ce = F.cross_entropy(
                        scaled_valid.view(-1, scaled_valid.size(-1)),
                        labels_valid.view(-1),
                        reduction='none',
                        label_smoothing=0.0,
                    )

                token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(1, labels_valid.unsqueeze(-1)).squeeze(-1).detach()
                temp_loss = token_ce.mean()
                loss = temp_loss
            # if self.train_top_p:
            #     # top-p loss
            #     unscaled_shift = unscaled_logits[:, :-1, :]
            #     temp_shift = temp_logits[:, :-1, :]
            #     shift_labels = labels[:, 1:]
            #     scaled_shift = unscaled_shift / temp_shift
            #     top_p_shift = top_p_logits[:, :-1, :]
            #     with torch.no_grad():
            #         probs_sorted, idx_sorted = torch.softmax(scaled_shift, dim=-1).sort(dim=-1, descending=True)
            #         rank = (idx_sorted == shift_labels.unsqueeze(-1)).float().argmax(dim=-1)
            #         cumsum = probs_sorted.cumsum(dim=-1)                                 
            #         top_p_labels = cumsum.gather(-1, rank.unsqueeze(-1)).squeeze(-1).detach()
                    
            #         valid_mask = shift_labels != -100
                    
            #         base_probs = torch.softmax(unscaled_shift, dim=-1)
            #         pred_ids = base_probs.argmax(dim=-1)
            #         correct_positions = (pred_ids == shift_labels.clamp_min(0))
            #         rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
            #         drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
            #         masked_valid_mask = valid_mask & (~drop_mask)
                
            #     top_p_pred = top_p_shift.squeeze(-1)  # [B, S-1]
            #     target = top_p_labels                # [B, S-1]
            #     error = top_p_pred - target
            #     lambda_under = 2.0
            #     lambda_over = 1.0
            #     per_token_loss = torch.where(error < 0, lambda_under * error.pow(2), lambda_over * error.pow(2))
            #     top_p_loss = per_token_loss[masked_valid_mask].mean()

            #     loss = top_p_loss

            if self.train_top_p:
                steepness = 30.0 # 控制衰减速度，即平滑幅度

                unscaled_shift = unscaled_logits[:, :-1, :]
                labels_shift = labels[:, 1:]
            
                temp_shift = temp_logits[:, :-1, :]
                p_shift = top_p_logits[:, :-1, :]

                scaled_logits = unscaled_shift / temp_shift
                
                probs = torch.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                overage = torch.relu(cumulative_probs - p_shift)
                decay_factor = torch.exp(-steepness * overage)

                mask = torch.zeros_like(probs).scatter_(
                    -1, sorted_indices, decay_factor
                )
                masked_probs = probs * mask
                renormalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)

                log_probs = torch.log(renormalized_probs + 1e-9)

                valid_mask = labels_shift != -100
                log_probs = log_probs[valid_mask]
                labels_shift = labels_shift[valid_mask]

                unscaled_valid = unscaled_shift[valid_mask]

                token_ce = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), labels_shift.view(-1), reduction='none')
                token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(1, labels_shift.unsqueeze(-1)).squeeze(-1).detach()

                top_p_loss = token_ce.mean()
                if loss is None:
                    loss = top_p_loss
                else:
                    loss = loss + top_p_loss
            lm_loss = loss

        # if labels is not None:
        #     if self.train_temp:
        #         ############ mask loss ############## 
        #         unscaled_shift = unscaled_logits[:, :-1, :]
        #         temp_shift = temp_logits[:, :-1, :]
        #         shift_labels = labels[:, 1:]
        #         valid_mask = shift_labels != -100  # [B, S-1]
        #         # Base-model confidence of GT token (no grad and independent of temp)
        #         with torch.no_grad():
        #             base_probs = torch.softmax(unscaled_shift, dim=-1)
        #             # p_gt_base = base_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # [B, S-1]
        #             pred_ids = base_probs.argmax(dim=-1)  # [B, S-1]
        #             correct_positions = (pred_ids == shift_labels.clamp_min(0))

        #         # Randomly drop 60% of positions where model argmax == GT
        #         rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
        #         drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
                
        #         masked_valid_mask = valid_mask & (~drop_mask)

        #         scaled_shift = unscaled_shift / temp_shift
        #         # Select only valid tokens
        #         scaled_valid = scaled_shift[masked_valid_mask]
        #         unscaled_valid = unscaled_shift[masked_valid_mask]
        #         labels_valid = shift_labels[masked_valid_mask]

        #         # Per-token CE with label smoothing
        #         if labels_valid.numel() > 0:
        #             token_ce = F.cross_entropy(
        #                 scaled_valid.view(-1, scaled_valid.size(-1)),
        #                 labels_valid.view(-1),
        #                 reduction='none',
        #                 label_smoothing=0.0,
        #             )

        #         token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(1, labels_valid.unsqueeze(-1)).squeeze(-1).detach()
        #         temp_loss = token_ce.mean()

        #         lm_loss = temp_loss

        #         loss = lm_loss
        #     if self.train_top_p:
        #         # top-p loss
        #         unscaled_shift = unscaled_logits[:, :-1, :]
        #         temp_shift = temp_logits[:, :-1, :]
        #         shift_labels = labels[:, 1:]
        #         scaled_shift = unscaled_shift / temp_shift
        #         top_p_shift = top_p_logits[:, :-1, :]
        #         with torch.no_grad():
        #             probs_sorted, idx_sorted = torch.softmax(scaled_shift, dim=-1).sort(dim=-1, descending=True)
        #             rank = (idx_sorted == shift_labels.unsqueeze(-1)).float().argmax(dim=-1)
        #             cumsum = probs_sorted.cumsum(dim=-1)                                 
        #             top_p_labels = cumsum.gather(-1, rank.unsqueeze(-1)).squeeze(-1).detach()
        #             top_p_mask = shift_labels != -100
                
        #         top_p_loss = self.mse_criteria(
        #             top_p_shift.view(-1), top_p_labels.view(-1) # B, S, 1; B, S, 1
        #         )

        #         top_p_loss = top_p_loss.view_as(shift_labels)
        #         top_p_loss = top_p_loss[top_p_mask]
        #         top_p_loss = top_p_loss.mean()

        #         loss = top_p_loss

               

        return TempLLMCausalLMOutputWithPast(
            loss=loss,
            temp_loss=temp_loss, # FIXME
            lm_loss=lm_loss,
            top_p_loss=top_p_loss,
            top_k_loss=top_k_loss,
            logits=unscaled_logits,
            temp_logits=temp_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]

