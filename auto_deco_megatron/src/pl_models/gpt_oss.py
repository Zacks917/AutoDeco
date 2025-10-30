import os

import types
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Annotated, Callable, Optional
)

from safetensors.torch import load_file, save_file

import torch
from torch import nn

from megatron.core import parallel_state

from nemo.collections.llm.utils import Config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm import GPTOSSConfig, GPTOSSConfig20B, GPTOSSConfig120B, GPTOSSModel

from nemo.lightning import OptimizerModule, io

# from src.model.temperature_llm import TempHead, TopPHead
# from src.model.temperature_llm_nemo.override_functions import _postprocess
# from src.model.temperature_llm_nemo.hooks import MCTransformerLayerHiddenStateHook
# from src.model.temperature_llm_nemo.loss_utils import end_to_end_temperature_top_p_loss
#
# from src.model.temperature_llm_nemo.pl_models.common import AutoDecoAdapter

from src.override_functions import _postprocess
from src.pl_models.common import AutoDecoAdapter
from src.loss_utils import end_to_end_temperature_top_p_loss


@dataclass
class AutoDecoGPTOSSConfig20B(GPTOSSConfig20B):
    train_temp_head: bool = field(default=True)
    train_top_p_head: bool = field(default=False)
    output_hidden_states: bool = field(default=False)
    model_head_path: str = field(default="")


@dataclass
class AutoDecoGPTOSSConfig120B(GPTOSSConfig120B):
    train_temp_head: bool = field(default=True)
    train_top_p_head: bool = field(default=False)
    output_hidden_states: bool = field(default=False)
    model_head_path: str = field(default="")


class AutoDecoGPTOSSModelNeMo(GPTOSSModel):
    def __init__(
            self,
            config: Annotated[Optional[AutoDecoGPTOSSConfig20B], Config[AutoDecoGPTOSSConfig120B]] = None,
            optim: Optional[OptimizerModule] = None,
            tokenizer: Optional["TokenizerSpec"] = None,
            model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config=config, optim=optim, tokenizer=tokenizer, model_transform=model_transform)

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        first_configure = not hasattr(self, "module")
        super().configure_model(vp_stage=vp_stage)

        print(">>> custom configure...")
        print(">>> is_pipeline_last_stage: ", parallel_state.is_pipeline_last_stage())
        print(">>> is_first_configure: ", first_configure)
        print(">>> vp_stage: ", vp_stage)

        if parallel_state.is_pipeline_last_stage() and first_configure and not hasattr(self.module, "temp_head"):
            vps = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            print(">>> vps: ", vps)
            if vps is None or vp_stage is None or vp_stage == vps - 1:
                adapter = AutoDecoAdapter(hidden_size=self.config.hidden_size, use_enhanced_features=True)

                model_head_path = self.config.model_head_path
                if model_head_path != "":
                    print(f"Load Head Parameters From: {model_head_path}")
                    state_dict: Dict[str, Any] = load_file(filename=model_head_path)
                    adapter.load_state_dict(state_dict={
                        k.replace("adapter.", ""): v
                        for k, v in state_dict.items() if k.startswith("adapter.")
                    })
                setattr(self.module, "adapter", adapter)
                save_file(tensors=adapter.state_dict(), filename="/root/empty_gpt_oss_head.safetensors")

        self.module._postprocess = types.MethodType(_postprocess, self.module)

    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            # labels: Optional[torch.Tensor] = None,
            decoder_input: Optional[torch.Tensor] = None,
            inference_context=None,
            packed_seq_params=None,
    ) -> torch.Tensor:
        """Forward pass through the GPT model.

        Args:
            batch: inputs
            decoder_input: Optional decoder input
            inference_context: Optional parameters for inference
            packed_seq_params: Optional parameters for packed sequence processing

        Returns:
            torch.Tensor: Output tensor from the model
        """
        extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
        outputs = self.module(
            batch["tokens"],
            batch["position_ids"],
            attention_mask=None,
            decoder_input=decoder_input,
            labels=None,
            inference_context=inference_context,
            **extra_kwargs,
        )
        if not parallel_state.is_pipeline_last_stage():
            return outputs

        per_token_loss, loss_mask = end_to_end_temperature_top_p_loss(
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            logits=outputs["logits"],
            temp_logits=outputs["temp_logits"],
            top_p_logits=outputs["top_p_logits"]
        )
        batch["loss_mask"] = loss_mask
        return per_token_loss

    def forward_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.forward(batch=batch)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_step(batch)


@io.model_exporter(AutoDecoGPTOSSModelNeMo, "hf-peft")
class HFAutoDecoGPTOSSModelExporter(io.ModelConnector[AutoDecoGPTOSSModelNeMo, "AutoModelForCausalLM"]):
    # pylint: disable=C0115,C0116
    def init(self, dtype=torch.bfloat16) -> "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, trust_remote_code=True, torch_dtype=dtype)

    def apply(self, output_path: str) -> str:
        source, _ = self.nemo_load(str(self))
        # print(source.state_dict().keys())
        state_dict = {}
        for k, v in source.state_dict().items():
            if k.startswith("module.adapter."):
                state_dict[k[len("module.adapter."):]] = v.to(dtype=torch.bfloat16)
        from safetensors.torch import save_file
        os.system(f"mkdir -p {output_path}")
        save_file(state_dict, os.path.join(output_path, "model.safetensors"))

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "":
        return ""
