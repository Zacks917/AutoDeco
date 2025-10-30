from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from nemo.lightning.pytorch.callbacks import PEFT
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.utils import is_trainer_attached
from lightning.pytorch.trainer.states import TrainerFn


def red_print(x: str):
    print(f"\033[31m{x}\033[0m")


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

        scaled_logits = unscaled_logits / (temp_logits + 1e-8)
        prob_stats = self.compute_prob_stats(scaled_logits)
        features = torch.cat(tensors=[hidden_states, temp_logits, prob_stats], dim=-1)
        return self.mlp(features)


class TempHead(nn.Module):
    def __init__(self, hidden_size: int):
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


class AutoDecoAdapter(nn.Module):
    def __init__(self, hidden_size: int, use_enhanced_features: bool):
        super().__init__()
        self.temp_head = TempHead(hidden_size=hidden_size)
        self.top_p_head = TopPHead(hidden_size=hidden_size, use_enhanced_features=use_enhanced_features)

    def forward(
            self, logits: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temp_logits = self.temp_head(hidden_states=hidden_states)
        top_p_logits = self.top_p_head(
            hidden_states=hidden_states,
            temp_logits=temp_logits,
            unscaled_logits=logits
        )
        return temp_logits, top_p_logits


class FreezeMainLLMParameters(PEFT):
    def __init__(self, train_temp_head: bool, train_top_p_head: bool):
        super().__init__()
        self.train_temp_head = train_temp_head
        self.train_top_p_head = train_top_p_head

    def apply_to_module(self, module: nn.Module):
        for name, param in module.named_parameters():
            if "temp_head" in name and self.train_temp_head:
                red_print(f">>> unfreeze {name}")
                param.requires_grad = True
            elif "top_p_head" in name and self.train_top_p_head:
                red_print(f">>> unfreeze {name}")
                param.requires_grad = True
            else:
                print(f">>> freeze {name}")
                param.requires_grad = False

    def transform(self, module, name=None, prefix=None):
        return module

    def freeze_model(self, model: nn.Module) -> None:
        """Apply a default freeze method to the model.

        This method freezes all the model parameters. This method can be overridden by subclasses to
        implement custom freeze strategies (e.g. freeze only parts of the model)

        Args:
            model (nn.Module): The model to be fine-tuned.

        Returns:
            nn.Module: The transformed model with PEFT applied.
        """
        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                self.apply_to_module(module=model_chunk)
        else:
            self.apply_to_module(module=model)

        if is_trainer_attached(model) and model.trainer.state.fn == TrainerFn.FITTING:
            model.train(mode=True)


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@dataclass
class TemperatureLLMOutput:
    logits: Optional[torch.Tensor]
    temp_logits: Optional[torch.Tensor]
    top_p_logits: Optional[torch.Tensor]
