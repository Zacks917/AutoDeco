#!/usr/bin/env python3
"""
AutoDeco Model Merge Script

Merge trained AutoDeco heads with base model to create a complete checkpoint for vLLM deployment.

Usage:
    python merge_autodeco.py \\
        --autodeco-checkpoint ./trained-autodeco \\
        --base-model /path/to/base-model \\
        --output ./autodeco-full
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import shutil
import fnmatch
import json
import copy
# Add project root (parent of this script directory) to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.templlm_auto import AutoDecoModelForCausalLMConfig
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import split_torch_state_dict_into_shards
from typing import Any, NoReturn, Set, Dict, List, Tuple, Union, Optional, Literal, TypedDict, NamedTuple, Iterable
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import os
import orjson

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json(fp: str) -> Any:
    with open(fp, 'rb') as f:
        return orjson.loads(f.read())


def write_json(fp: str, obj: Any):
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=4))


def load_state_dict(dir_path: str) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    return [
        (name, load_file(filename=os.path.join(dir_path, name)))
        for name in tqdm(os.listdir(dir_path), desc=f"Load {dir_path} safetensors")
        if name.endswith(".safetensors")
    ]

def merge_autodeco(
    autodeco_path: str,
    base_model_path: str,
    output_dir: str,
):
    """
    Merge AutoDeco heads with base model to create complete checkpoint.
    
    Args:
        autodeco_path: Path to AutoDeco checkpoint (config + heads weights)
        base_model_checkpoint: Path to base model checkpoint
        output_dir: Output directory for merged checkpoint
    """
    logger.info("=" * 80)
    logger.info("AutoDeco Model Merge")
    logger.info("=" * 80)
    
    base_state_dicts = load_state_dict(dir_path=base_model_path)
    weight_index = load_json(fp=os.path.join(base_model_path, "model.safetensors.index.json"))

    c_base_state_dicts = copy.deepcopy(base_state_dicts)
    c_weight_index = copy.deepcopy(weight_index)

    # Add llm. prefix to base model weights (not starting with temp_head or top_p_head)
    for fname, state_dict in c_base_state_dicts:
        new_state_dict = {}
        for k, v in state_dict.items():
            # If key doesn't start with temp_head. or top_p_head., add llm. prefix
            if not k.startswith("temp_head.") and not k.startswith("top_p_head."):
                new_key = f"llm.{k}"
                new_state_dict[new_key] = v
                # Update weight_map
                if k in c_weight_index["weight_map"]:
                    c_weight_index["weight_map"][new_key] = c_weight_index["weight_map"].pop(k)
            else:
                new_state_dict[k] = v
        state_dict.clear()
        state_dict.update(new_state_dict)

    # Load heads state dict
    head_state_dict = {k: v for i in load_state_dict(dir_path=autodeco_path) for k, v in i[1].items()}
    
    # Try to merge heads into existing shards
    merged = False
    for fname, state_dict in c_base_state_dicts:
        for k, v in head_state_dict.items():
            if k in state_dict:
                print(f"[!] merge {k} to {fname}")
                state_dict[k] = v
                merged = True
    if not merged:
        for k, v in head_state_dict.items():
            print(f"[!] force merge {k} to {c_base_state_dicts[-1][0]}")
            c_base_state_dicts[-1][1][k] = v
            c_weight_index["weight_map"][k] = c_base_state_dicts[-1][0] # fine name

    os.system(f"mkdir -p {output_dir}")
    os.system(f"cp -r {os.path.join(base_model_path, '*.jinja')} {output_dir}")
    # Copy all json files from base_model_path (including config.json)
    os.system(f"cp -r {os.path.join(base_model_path, '*.json')} {output_dir}")
    # Overwrite config.json with autodeco_path's config.json
    os.system(f"cp -r {os.path.join(autodeco_path, 'config.json')} {output_dir}")

    write_json(fp=os.path.join(output_dir, "model.safetensors.index.json"), obj=c_weight_index)
    for name, state_dict in c_base_state_dicts:
        print(f"[!] save {name}")
        save_file(tensors=state_dict, filename=os.path.join(output_dir, name))
    logger.info(f"✓ Merge completed successfully!")
    logger.info(f"Output checkpoint: {output_dir}")
    logger.info(f"Checkpoint contents:")
    logger.info(f"  - {len(c_base_state_dicts)} model weights")
    logger.info(f"  - {len(c_weight_index['weight_map'])} weight map")

def _get_size(path: Path) -> str:
    """Get human-readable file size."""
    size = path.stat().st_size
    return _get_size_str(size)


def _get_size_str(size: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def split_autodeco(
    full_checkpoint_path: str,
    output_dir: str,
):
    state_dicts = load_state_dict(dir_path=full_checkpoint_path)

    head_state_dicts = {k: v for i in state_dicts for k, v in i[1].items() if k.startswith("temp_head") or k.startswith("top_p_head")}
    os.system(f"mkdir -p {output_dir}")
    os.system(f"cp -r {os.path.join(full_checkpoint_path, '*.jinja')} {output_dir}")
    os.system(f"cp -r {os.path.join(full_checkpoint_path, '*.json')} {output_dir}")
    save_file(tensors=head_state_dicts, filename=os.path.join(output_dir, "model.safetensors"))
    logger.info(f"✓ Split completed successfully!")





def main():
    parser = argparse.ArgumentParser(
        description="AutoDeco checkpoint merge/split utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Merge (heads + base model → full checkpoint for vLLM):
    python merge_autodeco.py merge \
        --autodeco-path  \
        --base-model-path  \
        --output 

2. Split (full checkpoint → heads-only checkpoint):
    python merge_autodeco.py split \
        --full-checkpoint  \
        --output 
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    subparsers.required = True
    
    # Merge subcommand
    merge_parser = subparsers.add_parser('merge', help='Merge heads with base model')
    merge_parser.add_argument(
        "--autodeco-path",
        type=str,
        required=True,
        help="Path to AutoDeco checkpoint (config.json + autodeco_heads.*)"
    )
    merge_parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to base model checkpoint"
    )
    merge_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged checkpoint"
    )
    
    # Split subcommand
    split_parser = subparsers.add_parser('split', help='Extract heads from full checkpoint')
    split_parser.add_argument(
        "--full-checkpoint",
        type=str,
        required=True,
        help="Path to full AutoDeco checkpoint"
    )
    split_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for heads-only checkpoint"
    )
    
    args = parser.parse_args()
    
    # Run appropriate operation
    if args.mode == 'merge':
        merge_autodeco(
            args.autodeco_path,
            args.base_model_path,
            args.output,
        )
    elif args.mode == 'split':
        split_autodeco(
            args.full_checkpoint,
            args.output,
        )


if __name__ == "__main__":
    main()