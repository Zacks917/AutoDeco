import json
import os
import io
import time
import tempfile

import functools
from functools import partial
import inspect

from dataclasses import dataclass, field, fields, asdict
from typing import (
    Any, Set, Dict, List, Type, Tuple, Union, Optional, Literal, TypedDict, NamedTuple, Iterable, Callable
)

from sympy.logic import true
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers import (
    AutoConfig, PreTrainedTokenizer, PretrainedConfig, PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoTokenizer
)

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import SpecificPreTrainedModelType

from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Config

from safetensors.torch import load_file, load_model, save_file


@dataclass
class TemperatureCausalLMOutputWithPast(CausalLMOutputWithPast):
    temp_logits: Optional[torch.FloatTensor] = None
    top_p_logits: Optional[torch.FloatTensor] = None


class TemperatureModelForCausalLMConfig(PretrainedConfig):
    top_p_hidden_size: int
    temperature_hidden_size: int
    base_model_name_or_path: str
    enable_temperature_head: bool
    enable_top_p_head: bool
    use_enhanced_features: bool = True

    def base_model_config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path=self.base_model_name_or_path)


class TopPHead(nn.Module):
    def __init__(self, hidden_size, use_enhanced_features=True):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        if use_enhanced_features:
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

    @staticmethod
    def compute_prob_stats(logits: torch.Tensor) -> torch.Tensor:
        """Compute probability distribution statistics"""
        probs = torch.softmax(logits, dim=-1)

        # Max probability
        max_prob = probs.max(dim=-1, keepdim=True)[0]

        # Entropy
        log_probs = torch.log_softmax(input=logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        # Variance of probabilities
        prob_var = probs.var(dim=-1, keepdim=True)

        # Top-5 probability sum
        top5_probs, _ = torch.topk(probs, min(5, probs.size(-1)), dim=-1)
        top5_sum = top5_probs.sum(dim=-1, keepdim=True)
        return torch.cat([max_prob, entropy, prob_var, top5_sum], dim=-1)

    def forward(
            self, hidden_states: torch.Tensor, temp_logits: torch.Tensor, unscaled_logits: Optional[torch.Tensor] = None
    ):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            temp_logits: [batch_size, seq_len, 1]
            unscaled_logits: [batch_size, seq_len, vocab_size] (optional, for prob stats)
        """

        if self.use_enhanced_features:
            scaled_logits = unscaled_logits / (temp_logits + 1e-8)
            prob_stats = self.compute_prob_stats(scaled_logits)
            features = torch.cat(tensors=[hidden_states, temp_logits, prob_stats], dim=-1)
        else:
            # Original implementation
            features = torch.cat([hidden_states, temp_logits], dim=-1)
        return self.mlp(features)


class TempHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sigmoid_output = self.mlp(hidden_states)
        return sigmoid_output * 2


class TemperatureModelForCausalLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config: TemperatureModelForCausalLMConfig):
        super().__init__(config=config)
        self.config = config
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.base_model_name_or_path
        )
        self.temp_head = TempHead(hidden_size=self.llm.config.hidden_size)
        self.top_p_head = TopPHead(
            hidden_size=self.llm.config.hidden_size, use_enhanced_features=config.use_enhanced_features
        )
        self._keys_to_ignore_on_save = [k for k in self.state_dict().keys() if k.startswith("llm.")]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Dict[str, Any],
    ) -> TemperatureCausalLMOutputWithPast:
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True, **kwargs
            )

        temp_logits = self.temp_head(hidden_states=outputs.hidden_states[-1])
        top_p_logits = self.top_p_head(
            hidden_states=outputs.hidden_states[-1],
            temp_logits=temp_logits,
            unscaled_logits=outputs.logits
        )
        return TemperatureCausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits
        )

    @classmethod
    def from_pretrained(
            cls: type[SpecificPreTrainedModelType],
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            **kwargs,
    ) -> "TemperatureModelForCausalLM":
        config: TemperatureModelForCausalLMConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        temp_model: TemperatureModelForCausalLM = cls(config)

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
            temp_model.load_state_dict(state_dict=head_state_dict, strict=False)
        else:
            print("no head state dict found...")
        return temp_model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                # logger.warning(
                #     "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                #     "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                # )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())


class TemperatureQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: TemperatureModelForCausalLMConfig):
        base_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config.base_model_name_or_path)
        super().__init__(config=base_model_config)
        self.config: TemperatureModelForCausalLMConfig = config

        self.temp_head = TempHead(hidden_size=config.temperature_hidden_size)
        self.top_p_head = TopPHead(hidden_size=config.top_p_hidden_size, use_enhanced_features=True)
        self._keys_to_ignore_on_save = [
            k for k in self.state_dict().keys()
            if (not k.startswith("temp_head.")) or (not k.startswith("top_p_head."))
        ]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Dict[str, Any],
    ) -> TemperatureCausalLMOutputWithPast:
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=True, **kwargs
            )
        temp_logits = self.temp_head(hidden_states=outputs.hidden_states[-1])
        top_p_logits = self.top_p_head(
            hidden_states=outputs.hidden_states[-1],
            temp_logits=temp_logits,
            unscaled_logits=outputs.logits
        )

        return TemperatureCausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits
        )

    @classmethod
    def from_pretrained(
            cls: type[SpecificPreTrainedModelType],
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            # cache_dir: Optional[Union[str, os.PathLike]] = None,
            # ignore_mismatched_sizes: bool = False,
            # force_download: bool = False,
            # local_files_only: bool = False,
            # token: Optional[Union[str, bool]] = None,
            # revision: str = "main",
            # use_safetensors: Optional[bool] = None,
            # weights_only: bool = True,
            **kwargs,
    ) -> SpecificPreTrainedModelType:
        config: TemperatureModelForCausalLMConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        # config.top_p_hidden_size = config.hidden_size
        # config.temperature_hidden_size = config.hidden_size
        # config.base_model_name_or_path = '/apdcephfs_fsgm/share_303846923/user/zackszcwang/temp/ckpt/R1-Stage2-5e-6LR-1Epochs-12000Tokens-1BS-4GA'
        # config.enable_temperature_head = True
        # config.enable_top_p_head = True
        # config.use_enhanced_features = True
# class TemperatureModelForCausalLMConfig(PretrainedConfig):
    # top_p_hidden_size: int
    # temperature_hidden_size: int
    # base_model_name_or_path: str
    # enable_temperature_head: bool
    # enable_top_p_head: bool
    # use_enhanced_features: Optional[True]

    # def base_model_config(self) -> PretrainedConfig:
    #     return AutoConfig.from_pretrained(pretrained_model_name_or_path=self.base_model_name_or_path)

        model: TemperatureModelForCausalLM = cls(config=config)

        base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.base_model_name_or_path,
            *model_args,
            **kwargs
        )
        model.load_state_dict(state_dict=base_model.state_dict())

        head_state_dict = {}
        for fname in os.listdir(pretrained_model_name_or_path):
            if fname.endswith(".safetensors"):
                state_dict = load_file(filename=os.path.join(pretrained_model_name_or_path, fname))
                head_state_dict.update({
                    k: v for k, v in state_dict.items() if k.startswith("temp_head") or k.startswith("top_p_head")
                })

        if len(head_state_dict) > 0:
            model.load_state_dict(state_dict=head_state_dict, strict=False)
        else:
            print("no head state dict found...")
        return model

    def save_heads_only(self, output_dir: str, tokenizer: Optional[PreTrainedTokenizer] = None):
        os.makedirs(output_dir, exist_ok=True)
        state_dict = {}
        for k, v in self.state_dict().items():
            if k.startswith("temp_head") or k.startswith("top_p_head"):
                state_dict[k] = v.detach().cpu()
        if len(state_dict) == 0:
            print("no head state dict to save...")
            return
        save_file(tensors=state_dict, filename=os.path.join(output_dir, "model.safetensors"))
        # save config and tokenizer files for convenience
        try:
            self.config.save_pretrained(output_dir)
        except Exception:
            pass
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(output_dir)
            except Exception:
                pass
if __name__ == "__main__":
    model_path = '/apdcephfs_fsgm/share_303846923/user/zackszcwang/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    temp_model_path = '/apdcephfs_fsgm/share_303846923/user/zackszcwang/temp/ckpt/R1-Stage2-5e-6LR-1Epochs-12000Tokens-1BS-4GA'
    # from templlm_qwen2_5 import TempLLMQwen2ForCausalLM
    # model = TempLLMQwen2ForCausalLM.from_pretrained(temp_model_path)
    # model.save_heads_only(output_dir='./R1-only-heads')

    temp_config = TemperatureModelForCausalLMConfig(
        top_p_hidden_size=3584,
        temperature_hidden_size=3584,
        base_model_name_or_path=model_path,
        enable_temperature_head=True,
        enable_top_p_head=True,
        use_enhanced_features=True)
    model = TemperatureModelForCausalLM(config=temp_config)
    import os
    from safetensors.torch import load_file

    heads_dir = "/apdcephfs_fsgm/share_303846923/user/zackszcwang/temp/R1-only-heads"
    heads_sd = load_file(os.path.join(heads_dir, "model.safetensors"))

    # 仅筛选 temp/top-p 头参数键
    heads_sd = {k: v for k, v in heads_sd.items()
                if k.startswith("temp_head") or k.startswith("top_p_head")}

    # 假设你已经实例化好模型：model
    missing, unexpected = model.load_state_dict(heads_sd, strict=False)
    model.save_pretrained(save_directory='./ckpt/R1-merged')

