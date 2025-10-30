# SPDX-License-Identifier: Apache-2.0
"""
Unified AutoDeco Model for vLLM Inference

This model loads a complete AutoDeco checkpoint (base model + heads) and
provides dynamic temperature and top-p sampling for inference.

The checkpoint should contain:
- config.json with model_type="autodeco" and base_model_type
- Complete model weights including:
  - llm.* (base model weights)
  - temp_head.* (temperature head weights)
  - top_p_head.* (top-p head weights)

Usage:
    # 1. First merge heads with base model (before vLLM deployment):
    python merge_autodeco.py --mode lightweight_to_full \\
        --autodeco-checkpoint ./lightweight-checkpoint \\
        --output ./full-checkpoint
    
    # 2. Then load with vLLM:
    for example:
    from vllm import LLM
    llm = LLM(model="./full-checkpoint", trust_remote_code=True)
"""

from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class TopPHead(nn.Module):
    """Top-P prediction head with enhanced features"""
    
    def __init__(self, hidden_size, vocab_size=None, use_enhanced_features=True):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        
        if use_enhanced_features:
            # Enhanced features: hidden_states + temp + prob_stats
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
        
        max_prob = probs.max(dim=-1, keepdim=True)[0]
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        prob_var = probs.var(dim=-1, keepdim=True)
        top5_probs, _ = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        top5_sum = top5_probs.sum(dim=-1, keepdim=True)
        
        return torch.cat([max_prob, entropy, prob_var, top5_sum], dim=-1)

    def forward(self, hidden_states, temp_logits, unscaled_logits=None):
        if self.use_enhanced_features:
            features = [hidden_states, temp_logits]
            
            if unscaled_logits is not None:
                scaled_logits = unscaled_logits / (temp_logits + 1e-8)
                prob_stats = self.compute_prob_stats(scaled_logits)
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
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states):
        """Returns temperature in range [0, 2]"""
        sigmoid_output = self.mlp(hidden_states)
        return sigmoid_output * 2


class AutoDecoModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    Unified AutoDeco model for vLLM inference.
    
    This loads a complete checkpoint containing:
    - Base LLM model (self.llm)
    - Temperature prediction head (self.temp_head)
    - Top-p prediction head (self.top_p_head)
    
    The checkpoint must be created by merging heads with base model using:
        python merge_autodeco.py --mode lightweight_to_full ...
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        # Validate AutoDeco config
        if not hasattr(config, 'base_model_type'):
            raise ValueError(
                "This model requires an AutoDeco config with 'base_model_type'. "
                "Please use merge_autodeco.py to create a complete AutoDeco checkpoint."
            )
        
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        base_model_type = config.base_model_type
        use_enhanced_features = getattr(config, 'use_enhanced_features', True)
        
        logger.info("="*80)
        logger.info("Initializing AutoDeco model for vLLM:")
        logger.info(f"  - base_model_type: {base_model_type}")
        logger.info(f"  - use_enhanced_features: {use_enhanced_features}")
        logger.info(f"  - hidden_size: {config.hidden_size}")
        logger.info("="*80)
        
        # Get base model class
        base_model_class = self._get_base_model_class(base_model_type)
        
        if base_model_class is None:
            raise ValueError(
                f"Unsupported base model type: {base_model_type}. "
                f"Supported types: qwen2, qwen3, qwen2_moe, qwen3_moe, gpt_oss, llama, mistral, mixtral, deepseek_v3"
            )
        
        logger.info(f"  - Loading base model class: {base_model_class.__name__}")
        
        # Create base model (self.llm)
        # Note: We prefix with "llm" so weights are loaded as llm.*
        self.llm = base_model_class(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm")
        )
        
        # Get hidden size
        hidden_size = config.hidden_size
        
        # Initialize AutoDeco heads
        self.temp_head = TempHead(hidden_size)
        self.top_p_head = TopPHead(
            hidden_size,
            vocab_size=config.vocab_size,
            use_enhanced_features=use_enhanced_features
        )
        
        # Initialize logits processor
        self.logits_processor = LogitsProcessor(config.vocab_size)
        
        # Initialize sampler
        self.sampler = get_sampler()
        
        # Copy useful attributes from base model
        if hasattr(self.llm, 'make_empty_intermediate_tensors'):
            self.make_empty_intermediate_tensors = (
                self.llm.make_empty_intermediate_tensors
            )
        
        # For LoRA support
        if hasattr(self.llm, 'packed_modules_mapping'):
            self.packed_modules_mapping = self.llm.packed_modules_mapping
        
        logger.info("✓ AutoDeco model initialized successfully")
        logger.info("="*80)
    
    def _get_base_model_class(self, base_model_type: str):
        """
        Dynamically import and return the base model class.
        
        Args:
            base_model_type: Model type from config (e.g., 'qwen2', 'qwen3', 'gpt_oss')
        
        Returns:
            Base model class or None if not supported
        """
        # Mapping of model types to their module and class names
        MODEL_REGISTRY = {
            'qwen2': ('qwen2', 'Qwen2ForCausalLM'),
            'qwen3': ('qwen3', 'Qwen3ForCausalLM'),
            'qwen2_moe': ('qwen2_moe', 'Qwen2MoeForCausalLM'),
            'qwen3_moe': ('qwen3_moe', 'Qwen3MoeForCausalLM'),
            'gpt_oss': ('gpt_oss', 'GptOssForCausalLM'),
            'llama': ('llama', 'LlamaForCausalLM'),
            'mistral': ('mistral', 'MistralForCausalLM'),
            'mixtral': ('mixtral', 'MixtralForCausalLM'),
            "deepseek_v3": ("deepseek_v2", "DeepseekV3ForCausalLM"),
        }
        
        if base_model_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown base model type: {base_model_type}")
            return None
        
        module_name, class_name = MODEL_REGISTRY[base_model_type]
        
        try:
            # Import the module
            module = __import__(
                f'vllm.model_executor.models.{module_name}',
                fromlist=[class_name]
            )
            
            # Get the class
            model_class = getattr(module, class_name)
            return model_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load {class_name} from {module_name}: {e}")
            return None
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from base model"""
        return self.llm.get_input_embeddings(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass through base model to get hidden states.
        
        This is called during prefill and decode phases.
        """
        hidden_states = self.llm.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute logits with dynamic temperature and top-p.
        
        This applies the AutoDeco heads to predict temperature and top_p
        for each token, then uses these for sampling.
        
        Args:
            hidden_states: Hidden states from forward pass
            sampling_metadata: Sampling parameters
        
        Returns:
            Tuple of (logits, temperatures, top_ps) when using AutoDeco heads,
            or just logits for standard models
        """
        # Use the logits processor with AutoDeco heads
        # This will call temp_head and top_p_head internally and return
        # (logits, temp, top_p) tuple
        result = self.logits_processor(
            lm_head=self.llm.lm_head,
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            temp_head=self.temp_head,
            top_p_head=self.top_p_head,
        )
        return result
    
    def sample(
        self,
        logits: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """
        Sample next tokens from logits with dynamic temperature and top-p.
        
        Args:
            logits: Either just logits tensor, or tuple of (logits, temperatures, top_ps)
            sampling_metadata: Sampling parameters
        
        Returns:
            Sampler output with next tokens
        """
        # Check if we got dynamic temperature and top_p from AutoDeco heads
        if isinstance(logits, tuple):
            logits_tensor, temperatures, top_ps = logits
            # Pass dynamic temperature and top_p to sampler
            next_tokens = self.sampler(
                logits=logits_tensor,
                sampling_metadata=sampling_metadata,
                dynamic_temperatures=temperatures,
                dynamic_top_ps=top_ps,
            )
        else:
            # Standard sampling without dynamic parameters
            next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """
        Load weights for complete AutoDeco model.
        
        Expected weight structure (from merged checkpoint):
        - llm.* : Base model weights
        - temp_head.* : Temperature head weights
        - top_p_head.* : Top-p head weights
        
        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint
        
        Returns:
            Set of loaded parameter names
        """
        logger.info("Loading AutoDeco weights from merged checkpoint...")
        
        # Use AutoWeightsLoader to handle all weights automatically
        # It will match prefixes: llm.*, temp_head.*, top_p_head.*
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        
        loaded_params = loader.load_weights(weights)
        
        logger.info(f"✓ Successfully loaded {len(loaded_params)} parameters")
        
        # Log breakdown by component
        llm_params = sum(1 for p in loaded_params if p.startswith('llm.'))
        temp_head_params = sum(1 for p in loaded_params if p.startswith('temp_head.'))
        top_p_head_params = sum(1 for p in loaded_params if p.startswith('top_p_head.'))
        
        logger.info(f"  - Base model (llm.*): {llm_params} parameters")
        logger.info(f"  - Temperature head (temp_head.*): {temp_head_params} parameters")
        logger.info(f"  - Top-p head (top_p_head.*): {top_p_head_params} parameters")
        
        return loaded_params


# Export for vLLM model registry
__all__ = ['AutoDecoModelForCausalLM', 'TempHead', 'TopPHead']
