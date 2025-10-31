from dataclasses import dataclass, field, fields, asdict
from typing import Any, Set, Dict, List, Tuple, Union, Optional, Literal, TypedDict, NamedTuple, Iterable, override

import torch
from deepspeed.compile.util import dtype_to_elem_size
from torch import nn
from torch.nn import functional as F


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


def end_to_end_temperature_top_p_loss(
        logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor, temp_logits: torch.Tensor,
        top_p_logits: torch.Tensor
):
    """loss computation"""

    token_probs = torch.softmax(input=logits, dim=-1)
    with torch.no_grad():
        greedy_prediction_ids = torch.argmax(input=token_probs.detach(), dim=-1)
        # [batch_size, seq_len]
        correct_position = (greedy_prediction_ids == labels)
        drop_greedy_mask = (torch.rand(size=labels.size(), device=labels.device) < 0.6) & correct_position
        loss_mask = loss_mask & (~drop_greedy_mask)

    def _temperature_loss(logits: torch.Tensor, temp_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits / temp_logits
        log_probs = selective_log_softmax(logits=logits, index=labels)
        return -log_probs.exp().detach() * log_probs
        # return F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=labels.view(-1), reduction="none")

    def _top_p_loss(
            logits: torch.Tensor, temp_logits: torch.Tensor, top_p_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        steepness = 30.0
        unscale_token_probs = torch.softmax(input=logits, dim=-1)
        logits = logits / temp_logits
        token_probs = torch.softmax(logits, dim=-1)

        sorted_token_probs, sorted_token_indices = torch.sort(input=token_probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(input=sorted_token_probs, dim=-1)
        overage = torch.relu(input=cumulative_probs - top_p_logits)
        decay_factor = torch.exp(-steepness * overage)

        top_p_mask = torch.zeros_like(input=token_probs).scatter_(-1, sorted_token_indices, decay_factor)

        top_p_masked_probs = token_probs * top_p_mask
        renormalized_probs = top_p_masked_probs / (top_p_masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
        log_probs = torch.log(renormalized_probs + 1e-9)
        top_p_per_token_loss = F.nll_loss(
            input=log_probs.view(-1, log_probs.size(-1)), target=labels.view(-1), reduction="none"
        )
        top_p_per_token_loss = top_p_per_token_loss * unscale_token_probs.gather(-1, labels.unsqueeze(-1)).squeeze(
            -1).detach()
        return top_p_per_token_loss

    temperature_loss = _temperature_loss(logits=logits, temp_logits=temp_logits, labels=labels)
    top_p_loss = _top_p_loss(logits=logits, temp_logits=temp_logits, top_p_logits=top_p_logits, labels=labels)

    loss = temperature_loss + top_p_loss
    return loss, loss_mask
