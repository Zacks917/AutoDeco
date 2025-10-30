# coding=utf-8
"""
Unified AutoDeco Model for Causal Language Modeling
Supports temperature and top-p prediction heads on top of any AutoModelForCausalLM
"""
from typing import Optional, Tuple, Union, Dict, Any
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.utils import ModelOutput, logging
from transformers.cache_utils import Cache

from transformers.modeling_utils import SpecificPreTrainedModelType

from safetensors.torch import load_file, load_model, save_file

logger = logging.get_logger(__name__)


# AutoDeco Heads
class TopPHead(nn.Module):
    """Top-P prediction head with enhanced features"""
    
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
        """Compute probability distribution statistics"""
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
    """Temperature prediction head"""
    
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
class AutoDecoOutputWithPast(ModelOutput):
    """
    Output class for AutoDeco models with past key values.
    Compatible with both standard and MoE models.
    """
    loss: Optional[torch.FloatTensor] = None
    temp_loss: Optional[torch.FloatTensor] = None
    top_p_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None  # For MoE models
    lm_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None  # For MoE models


class AutoDecoModelForCausalLMConfig(PretrainedConfig):
    
    model_type = "autodeco"  # Class attribute - REQUIRED for transformers registration!
    
    def __init__(
        self,
        enable_temperature_head: bool=True,
        enable_top_p_head: bool=True,
        use_enhanced_features: bool=True,
        base_model_name_or_path: str=None,
        **kwargs  # All base model config parameters
    ):
        super().__init__(**kwargs)
        self.enable_temperature_head = enable_temperature_head
        self.enable_top_p_head = enable_top_p_head
        self.top_p_hidden_size = kwargs.get('hidden_size', None)
        self.temperature_hidden_size = kwargs.get('hidden_size', None)
        self.use_enhanced_features = use_enhanced_features
        self.base_model_name_or_path = base_model_name_or_path
        self._name_or_path = base_model_name_or_path
        self.base_model_type = kwargs.get('model_type', None)
    # @classmethod
    # def from_base_model(
    #     cls, 
    #     pretrained_model_name_or_path: str,
    #     train_temp: bool = False,
    #     train_top_p: bool = False,
    #     use_enhanced_features: bool = True,
    #     **kwargs
    # ):
    #     """
    #     Create AutoDecoConfig from a base model.
        
    #     Args:
    #         pretrained_model_name_or_path: Path or name of the base model
    #         train_temp: Whether to train temperature head
    #         train_top_p: Whether to train top-p head
    #         use_enhanced_features: Whether to use enhanced features
    #         **kwargs: Additional arguments passed to base config loading
    #     """
    #     # Load base model config
    #     base_config = AutoConfig.from_pretrained(
    #         pretrained_model_name_or_path, 
    #         trust_remote_code=kwargs.get('trust_remote_code', False)
    #     )
        
    #     # Convert base config to dict
    #     config_dict = base_config.to_dict()
        
    #     # Create AutoDecoConfig with merged parameters
    #     return cls(
    #         base_model_type=base_config.model_type, 
    #         base_model_name_or_path=pretrained_model_name_or_path,
    #         train_temp=train_temp,
    #         train_top_p=train_top_p,
    #         use_enhanced_features=use_enhanced_features,
    #         **config_dict
    #     )
    
    # def to_dict(self):
    #     """
    #     Convert config to dictionary.
        
    #     Returns a dict containing:
    #     - All base model config parameters (inherited)
    #     - AutoDeco-specific parameters (train_temp, train_top_p, use_enhanced_features)
    #     - Metadata (model_type, base_model_type, base_model_name_or_path, architectures)
    #     """
    #     # Get all base model parameters from parent
    #     output = super().to_dict()
        
    #     # Ensure AutoDeco-specific fields are included
    #     # (Only 3 additional parameters + metadata)
    #     output.update({
    #         'model_type': 'autodeco',  # "AutoDeco"
    #         'architectures': ['AutoDecoModelForCausalLM'],  # Architecture class name
    #     })
    #     return output


class AutoDecoModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Unified AutoDeco model that wraps any AutoModelForCausalLM with 
    temperature and top-p prediction heads.
    
    This eliminates the need for separate model files for each architecture.
    """
    
    supports_gradient_checkpointing = True
    _no_split_modules = []  # Will be set based on base model
    config_class = AutoDecoModelForCausalLMConfig
    def __init__(self, config: AutoDecoModelForCausalLMConfig, **kwargs):
        """
        Initialize AutoDeco model.
        
        Args:
            config: AutoDecoConfig instance with base model information
        """
        super().__init__(config)
        self.config = config
        # Get base model path
        base_model_path = config.base_model_name_or_path
        if base_model_path is None:
            raise ValueError("config.base_model_name_or_path must be specified")
        
        # Load the base causal LM model
        logger.info(f"Loading base model from {base_model_path}")
        logger.info(f"Base model type: {config.base_model_type}")
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_path,
        )
    
        
        # Initialize AutoDeco heads
        self.temp_head = TempHead(config.temperature_hidden_size)
        self.top_p_head = TopPHead(
            config.top_p_hidden_size, 
            use_enhanced_features=config.use_enhanced_features
        )
        
        # Training flags
        self.train_temp = config.enable_temperature_head
        self.train_top_p = config.enable_top_p_head
        
        # TODO: maybe we don't need this check? any llm base model training can be aborted.
        # # Check if base model is MoE
        # self.is_moe = hasattr(config, 'num_local_experts') or \
        #               hasattr(config, 'num_experts')
        
        # if self.is_moe:
        #     self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.01)
        #     self.num_experts = getattr(config, 'num_local_experts', None) or \
        #                       getattr(config, 'num_experts', None)
        #     self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
        
        # # Copy _no_split_modules from base model if available
        # if self.llm is not None and hasattr(self.llm, '_no_split_modules'):
        #     self._no_split_modules = self.llm._no_split_modules
        
        logger.info(f"AutoDeco model initialized:")
        logger.info(f"  - base_model_type={config.base_model_type}, base_model_name_or_path={config.base_model_name_or_path}")
        logger.info(f"  - train_temp={self.train_temp}, train_top_p={self.train_top_p}")
        
        # Log training mode
        if self.train_temp or self.train_top_p:
            heads = []
            if self.train_temp:
                heads.append("temp_head")
            if self.train_top_p:
                heads.append("top_p_head")
            logger.info(f"  - Training mode: AutoDeco heads ({', '.join(heads)})")
        else:
            logger.info(f"  - Training mode: Base LLM (standard language modeling)")
        
        # Set light-weight saving mode
        self._keys_to_ignore_on_save = [k for k in self.state_dict().keys() if k.startswith("llm.")]

    # whole model only
    @classmethod
    def from_pretrained(
            cls: type[SpecificPreTrainedModelType],
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            **kwargs,
    ) -> "AutoDecoModelForCausalLM":
        config = AutoDecoModelForCausalLMConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs
        )
        autodeco_model: AutoDecoModelForCausalLM = cls(config)

        head_state_dict = {}
        for fname in os.listdir(pretrained_model_name_or_path):
            if fname.endswith(".safetensors"):
                state_dict = load_file(filename=os.path.join(pretrained_model_name_or_path, fname))
                head_state_dict.update({
                    k: v for k, v in state_dict.items() if k.startswith("temp_head") or k.startswith("top_p_head")
                })

        if len(head_state_dict) > 0:
            for k in head_state_dict:
                print(f"Load {k}")
            autodeco_model.load_state_dict(state_dict=head_state_dict, strict=False)
        else:
            print("no head state dict found...")
        return autodeco_model
    
    # def save_pretrained(
    #     self,
    #     save_directory: str,
    #     is_main_process: bool = True,
    #     state_dict: Optional[dict] = None,
    #     save_function: callable = torch.save,
    #     push_to_hub: bool = False,
    #     max_shard_size: Union[int, str] = "5GB",
    #     safe_serialization: bool = True,
    #     variant: Optional[str] = None,
    #     token: Optional[Union[str, bool]] = None,
    #     save_peft_format: bool = True,
    #     **kwargs
    # ):
    #     """
    #     Save AutoDeco model config and heads weights only (lightweight).
        
    #     This saves:
    #     1. config.json with AutoDecoConfig (includes base_model_name_or_path)
    #     2. ONLY temp_head and top_p_head weights (NOT the base model weights)
        
    #     The base model will be reloaded from base_model_name_or_path when loading.
    #     This significantly reduces checkpoint size (~5MB vs ~14GB).
        
    #     Use merge_autodeco.py script to create full checkpoint for vLLM deployment.
    #     """
    #     import os
    #     import json
    #     from transformers.modeling_utils import unwrap_model
    #     from huggingface_hub import split_torch_state_dict_into_shards
        
    #     if os.path.isfile(save_directory):
    #         logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
    #         return
        
    #     os.makedirs(save_directory, exist_ok=True)
        
    #     # Unwrap model (handles PEFT wrapping)
    #     model_to_save = unwrap_model(self)
        
    #     # Save config
    #     if is_main_process:
    #         model_to_save.config.save_pretrained(save_directory)
        
    #     # Prepare state_dict (heads only)
    #     logger.info("Saving AutoDeco heads only (lightweight)")
        
    #     if state_dict is None:
    #         # No state_dict provided, get it from model
    #         state_dict = {}
    #         temp_head_state = model_to_save.temp_head.state_dict()
    #         for key, value in temp_head_state.items():
    #             state_dict[f"temp_head.{key}"] = value
            
    #         top_p_head_state = model_to_save.top_p_head.state_dict()
    #         for key, value in top_p_head_state.items():
    #             state_dict[f"top_p_head.{key}"] = value
    #     else:
    #         # state_dict provided (e.g., by Trainer), filter to keep only heads
    #         # The provided state_dict may have full model keys or already-prefixed keys
    #         heads_state_dict = {}
            
    #         for key, value in state_dict.items():
    #             # Keep only temp_head.* and top_p_head.* keys
    #             if key.startswith('temp_head.') or key.startswith('top_p_head.'):
    #                 heads_state_dict[key] = value
            
    #         state_dict = heads_state_dict
            
    #         # If no heads found with prefix, the state_dict might not have prefixes
    #         # In that case, extract from the model itself
    #         if not state_dict:
    #             logger.warning("No heads found in provided state_dict, extracting from model")
    #             temp_head_state = model_to_save.temp_head.state_dict()
    #             for key, value in temp_head_state.items():
    #                 state_dict[f"temp_head.{key}"] = value
                
    #             top_p_head_state = model_to_save.top_p_head.state_dict()
    #             for key, value in top_p_head_state.items():
    #                 state_dict[f"top_p_head.{key}"] = value
        
    #     # Use custom name for heads-only checkpoint
    #     weights_name = "autodeco_heads.safetensors" if safe_serialization else "autodeco_heads.bin"
    #     filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        
    #     # Shard if needed
    #     state_dict_split = split_torch_state_dict_into_shards(
    #         state_dict,
    #         filename_pattern=filename_pattern,
    #         max_shard_size=max_shard_size
    #     )
        
    #     # Prepare index if sharded
    #     index = None
    #     if state_dict_split.is_sharded:
    #         index = {
    #             "metadata": state_dict_split.metadata,
    #             "weight_map": state_dict_split.tensor_to_filename,
    #         }
        
    #     # Save shards
    #     for shard_file, tensors in state_dict_split.filename_to_tensors.items():
    #         shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            
    #         if safe_serialization:
    #             from safetensors.torch import save_file
    #             save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
    #         else:
    #             save_function(shard, os.path.join(save_directory, shard_file))
        
    #     # Save index if sharded
    #     if index is not None:
    #         index_file = "autodeco_heads.safetensors.index.json" if safe_serialization else "autodeco_heads.bin.index.json"
    #         index_path = os.path.join(save_directory, index_file)
    #         with open(index_path, "w", encoding="utf-8") as f:
    #             content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    #             f.write(content)
    #         logger.info(f"Model weights saved in {len(state_dict_split.filename_to_tensors)} shard(s)")
    #     else:
    #         logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
        
    #     logger.info(f"✓ Lightweight checkpoint saved (base model: {self.config.base_model_name_or_path})")
    
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        if hasattr(self.llm, 'set_decoder'):
            self.llm.set_decoder(decoder)

    def get_decoder(self):
        if hasattr(self.llm, 'get_decoder'):
            return self.llm.get_decoder()
        return self.llm.model if hasattr(self.llm, 'model') else self.llm
    
    def _compute_temp_loss(self, unscaled_logits, temp_logits, labels):
        """Compute temperature loss"""
        unscaled_shift = unscaled_logits[:, :-1, :]
        temp_shift = temp_logits[:, :-1, :].clamp_min(1e-2)
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        
        # Base-model confidence of GT token
        with torch.no_grad():
            base_probs = torch.softmax(unscaled_shift, dim=-1)
            pred_ids = base_probs.argmax(dim=-1)
            correct_positions = (pred_ids == shift_labels.clamp_min(0))
            
            # Randomly drop 60% of positions where model argmax == GT
            rand_vals = torch.rand(shift_labels.shape, device=shift_labels.device)
            drop_mask = (rand_vals < 0.6) & valid_mask & correct_positions
            masked_valid_mask = valid_mask & (~drop_mask)
        
        scaled_shift = unscaled_shift / temp_shift
        scaled_valid = scaled_shift[masked_valid_mask]
        unscaled_valid = unscaled_shift[masked_valid_mask]
        labels_valid = shift_labels[masked_valid_mask]
        
        if labels_valid.numel() > 0:
            token_ce = F.cross_entropy(
                scaled_valid.view(-1, scaled_valid.size(-1)),
                labels_valid.view(-1),
                reduction='none',
                label_smoothing=0.0,
            )
            token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(
                1, labels_valid.unsqueeze(-1)
            ).squeeze(-1).detach()
            return token_ce.mean()
        
        return torch.tensor(0.0, device=unscaled_logits.device)
    
    def _compute_top_p_loss(self, unscaled_logits, temp_logits, top_p_logits, labels, method='soft'):
        """
        Compute top-p loss
        
        Args:
            method: 'soft' for soft top-p with exponential decay
        """
        unscaled_shift = unscaled_logits[:, :-1, :]
        temp_shift = temp_logits[:, :-1, :]
        top_p_shift = top_p_logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # Soft top-p loss with exponential decay
        steepness = 30.0
        scaled_logits = unscaled_shift / temp_shift.clamp_min(1e-8)
        probs = torch.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        overage = torch.relu(cumulative_probs - top_p_shift)
        decay_factor = torch.exp(-steepness * overage)
        
        mask = torch.zeros_like(probs).scatter_(-1, sorted_indices, decay_factor)
        masked_probs = probs * mask
        renormalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
        log_probs = torch.log(renormalized_probs + 1e-9)
        
        valid_mask = shift_labels != -100
        log_probs = log_probs[valid_mask]
        labels_shift = shift_labels[valid_mask]
        unscaled_valid = unscaled_shift[valid_mask]

        token_ce = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            labels_shift.view(-1),
            reduction='none'
        )
        token_ce = token_ce * torch.softmax(unscaled_valid, dim=-1).gather(
            1, labels_shift.unsqueeze(-1)
        ).squeeze(-1).detach()
        
        return token_ce.mean()

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
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        top_p_loss_method: str = 'soft',  # 'soft' or 'mse'
        **kwargs,
    ) -> AutoDecoOutputWithPast:
        """
        Forward pass of AutoDeco model.
        
        Args:
            top_p_loss_method: Method for computing top-p loss ('soft' or 'mse')
        """
        # Prepare kwargs for base model
        base_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': True,  # Always need hidden states
            'cache_position': cache_position,
        }
        
        # Add MoE-specific args if applicable
        if output_router_logits is not None:
            base_kwargs['output_router_logits'] = output_router_logits
        
        # Forward through base model
        outputs = self.llm(**base_kwargs, **kwargs)
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Compute logits and predictions
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # Compute unscaled logits
        # If training base model (train_temp=False and train_top_p=False), need gradients
        # Otherwise, can use no_grad for efficiency
        if self.train_temp or self.train_top_p:
            # Training heads only - no gradient needed for base model logits
            with torch.no_grad():
                unscaled_logits = self.llm.lm_head(hidden_states[:, slice_indices, :])
        else:
            # Training base model - need gradients
            unscaled_logits = self.llm.lm_head(hidden_states[:, slice_indices, :])
        
        temp_logits = self.temp_head(hidden_states[:, slice_indices, :])
        top_p_logits = self.top_p_head(
            hidden_states[:, slice_indices, :],
            temp_logits.detach(),
            unscaled_logits=unscaled_logits,
        )
        
        # Compute losses
        loss, lm_loss, temp_loss, top_p_loss = None, None, None, None
        
        if labels is not None:
            if self.train_temp or self.train_top_p:
                # Mode 1: Training AutoDeco heads
                losses = []
                
                if self.train_temp:
                    temp_loss = self._compute_temp_loss(unscaled_logits, temp_logits, labels)
                    losses.append(temp_loss)
                
                if self.train_top_p:
                    top_p_loss = self._compute_top_p_loss(
                        unscaled_logits, temp_logits, top_p_logits, labels,
                        method=top_p_loss_method
                    )
                    losses.append(top_p_loss)
                
                if losses:
                    loss = sum(losses)
            
            else:
                # Mode 2: Training base LLM (when both train_temp and train_top_p are False)
                # Compute standard language modeling loss
                logger.debug("Computing standard LM loss (training base model)")
                
                if labels is not None:
                    lm_loss = self.llm.loss_function(unscaled_logits, labels, self.llm.vocab_size, **kwargs)
                    loss = lm_loss
        
        # Handle MoE auxiliary loss
        aux_loss = None
        # if self.is_moe and hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        #     aux_loss = load_balancing_loss_func(
        #         outputs.router_logits,
        #         self.llm.num_experts,
        #         self.llm.num_experts_per_tok,
        #         attention_mask,
        #     )
        #     if labels is not None:
        #         loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        
        return AutoDecoOutputWithPast(
            loss=loss,
            temp_loss=temp_loss,
            top_p_loss=top_p_loss,
            aux_loss=aux_loss,
            lm_loss=lm_loss,
            logits=unscaled_logits,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            router_logits=outputs.router_logits if hasattr(outputs, 'router_logits') else None,
        )
    
    # TODO: generate with dynamic temperature/top-p
    def generate(self, *args, **kwargs):
        """
        Generate using the base model's generate method.
        Note: This uses the base model's generation, not the AutoDeco heads.
        For generation with dynamic temperature/top-p, you'll need custom generation logic.
        """
        return self.llm.generate(*args, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation"""
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)


# Register the config and model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM as AutoModelForCausalLMClass
# Register config
AutoConfig.register("autodeco", AutoDecoModelForCausalLMConfig)
AutoModel.register(AutoDecoModelForCausalLMConfig, AutoDecoModelForCausalLM)
AutoModelForCausalLMClass.register(AutoDecoModelForCausalLMConfig, AutoDecoModelForCausalLM)
logger.info("AutoDeco model registered with transformers (AutoConfig, AutoModel, AutoModelForCausalLM)")



__all__ = [
    'AutoDecoModelForCausalLM',
    'AutoDecoConfig',
    'AutoDecoOutputWithPast',
    'TempHead',
    'TopPHead',
]
if __name__ == "__main__":
    # automodel = AutoModelForCausalLM.from_pretrained("./autodeco-R1")
    # print(automodel)
    # assert 1==0
    from transformers import AutoTokenizer
    
    base_model_name_or_path = "/apdcephfs_fsgm/share_303846923/user/zackszcwang/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/"
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    autodeco_config = AutoDecoModelForCausalLMConfig(
        base_model_name_or_path=base_model_name_or_path,
        enable_temperature_head=True,
        enable_top_p_head=True,
        use_enhanced_features=True,
        **base_config.to_dict()
    )
    model = AutoDecoModelForCausalLM(autodeco_config)

    print(model)
    model.save_pretrained("./autodeco-R1")
    tokenizer.save_pretrained("./autodeco-R1")
    # assert 1==0

