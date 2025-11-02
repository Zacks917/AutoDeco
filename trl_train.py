from typing import Any, Optional, Union
from dataclasses import dataclass

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy
import os
from safetensors.torch import save_file


import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from model import AutoDecoModelForCausalLM
from trainer.trl_autodeco import AutoDecoLLMTrainer

# import sys
# PATH_TO_TRL="trl/"
# sys.path.append(PATH_TO_TRL)
import numpy as np

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
# from sft_trainer import SFTTrainer

# Custom ScriptArguments to add training head parameters
@dataclass
class AutoDecoLLMScriptArguments(ScriptArguments):
    """Script arguments for AutoDecoLLM training."""
    train_temp: bool = False
    train_top_p: bool = False

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly. The attention mask will be set to 1 for all tokens.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]

        # Check if we have meaningful seq_lengths from packing (restarting sequences)
        has_packed_position_ids = self.return_position_ids and "seq_lengths" in examples[0] and self.padding_free

        # For packing with position_ids, we should NOT create attention_mask as it causes
        # flash attention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask
        if not has_packed_position_ids:
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "seq_lengths" in examples[0]:
                position_ids = self._convert_seq_lengths_to_position_ids(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if "temp_labels" in examples[0]:
            temp_labels = [torch.tensor(example["temp_labels"]) for example in examples]
            if "top_p_labels" in examples[0]:
                top_p = [torch.tensor(example["top_p_labels"]) for example in examples]
            if "top_k_labels" in examples[0]:
                top_k = [torch.tensor(example["top_k_labels"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # Pad
        output = {}
        if self.padding_free:
            output["input_ids"] = torch.cat(input_ids, dim=0).unsqueeze(0)
            if not has_packed_position_ids:
                output["attention_mask"] = torch.cat(attention_mask, dim=0).unsqueeze(0)
            if self.return_position_ids:
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
            output["labels"] = torch.cat(labels, dim=0).unsqueeze(0)
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = torch.cat(completion_mask, dim=0).unsqueeze(0)
                output["labels"][completion_mask == 0] = -100
            if "assistant_masks" in examples[0]:
                assistant_masks = torch.cat(assistant_masks, dim=0).unsqueeze(0)
                output["labels"][assistant_masks == 0] = -100
        else:
            output["input_ids"] = pad(
                input_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.return_position_ids:
                output["position_ids"] = pad(
                    position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
            output["labels"] = pad(
                labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if "temp_labels" in examples[0]:
                temp_labels = pad(
                    temp_labels, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["temp_labels"] = temp_labels
            if "top_p_labels" in examples[0]:
                top_p = pad(
                    top_p, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["top_p_labels"] = top_p
            if "top_k_labels" in examples[0]:
                top_k = pad(
                    top_k, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["top_k_labels"] = top_k
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = pad(
                    completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

            if "assistant_masks" in examples[0]:
                assistant_masks = pad(
                    assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][assistant_masks == 0] = -100
        return output


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )

    # If we only train heads, avoid saving full model checkpoints during training
    if script_args.train_temp or script_args.train_top_p:
        try:
            training_args.save_strategy = "no"
        except Exception:
            pass

    model = AutoDecoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # Configure which heads to train based on script arguments
    if script_args.train_temp or script_args.train_top_p:
        print(f"[!] Training configuration: train_temp={script_args.train_temp}, train_top_p={script_args.train_top_p}")
    else:
        print(f"[!] Training the LLM model itself. No AutoDeco training.")
    for name, param in model.named_parameters():
        if 'temp_head' in name:
            param.requires_grad = script_args.train_temp
            status = "Training" if script_args.train_temp else "Freezing"
            print(f"[!] {status} parameter: {name}")
            continue
        elif 'top_p_head' in name:
            param.requires_grad = script_args.train_top_p
            status = "Training" if script_args.train_top_p else "Freezing"
            print(f"[!] {status} parameter: {name}")
            continue
        else:
            if not script_args.train_temp and not script_args.train_top_p:
                param.requires_grad = True
                print(f"[!] Training parameter: {name}")
            else:
                param.requires_grad = False
                print(f"[!] Freezing parameter: {name}")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True, 
    )
    # Set default chat template if needed
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # Ensure PAD token exists (some decoder-only models like LLaMA don't define one)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id


    try:
        dataset = load_dataset("json", data_files=f"data/{script_args.dataset_name}")
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        raise ValueError(f"Dataset {script_args.dataset_name} not found")

    # FOR DEBUGGING
    # dataset['train'] = dataset['train'].select(range(50))

    if script_args.train_temp or script_args.train_top_p:
        print(f"[!] AutoDecoLLM Training")
        trainer = AutoDecoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        data_collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),
        peft_config=get_peft_config(model_args),
        )
    else:
        print(f"[!] Normal SFT Training")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
        )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (AutoDecoLLMScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
