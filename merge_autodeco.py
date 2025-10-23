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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model.templlm_auto import AutoDecoConfig, TempHead, TopPHead
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import split_torch_state_dict_into_shards

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_autodeco(
    autodeco_path: str,
    base_model_path: str,
    output_dir: str,
    save_format: str = "safetensors",
    max_shard_size: str = "5GB"
):
    """
    Merge AutoDeco heads with base model to create complete checkpoint.
    
    Args:
        autodeco_path: Path to AutoDeco checkpoint (config + heads weights)
        base_model_checkpoint: Path to base model checkpoint
        output_dir: Output directory for merged checkpoint
        save_format: Save format ('safetensors' or 'pytorch')
        max_shard_size: Maximum size per shard file (e.g., '5GB', '10GB')
    """
    logger.info("=" * 80)
    logger.info("AutoDeco Model Merge")
    logger.info("=" * 80)
    
    # Step 1: Load AutoDeco config and heads
    logger.info(f"\nStep 1: Loading AutoDeco config and heads from: {autodeco_path}")
    autodeco_config = AutoDecoConfig.from_pretrained(autodeco_path)
    
    logger.info(f"  - model_type: {autodeco_config.model_type}")
    logger.info(f"  - base_model_type: {autodeco_config.base_model_type}")
    logger.info(f"  - train_temp: {autodeco_config.train_temp}")
    logger.info(f"  - train_top_p: {autodeco_config.train_top_p}")
    logger.info(f"  - use_enhanced_features: {autodeco_config.use_enhanced_features}")
    
    # Load heads weights
    checkpoint_path = Path(autodeco_path)
    heads_safetensors = checkpoint_path / "autodeco_heads.safetensors"
    heads_bin = checkpoint_path / "autodeco_heads.bin"
    
    heads_state_dict = {}  # Will store flat dict with full keys (temp_head.*, top_p_head.*)
    
    if heads_safetensors.exists():
        logger.info(f"  - Loading heads from: {heads_safetensors.name}")
        from safetensors.torch import load_file
        heads_state_dict = load_file(heads_safetensors)

    elif heads_bin.exists():
        logger.info(f"  - Loading heads from: {heads_bin.name}")
        loaded_dict = torch.load(heads_bin, map_location='cpu')
        
        # If it's nested dict format, flatten it
        if isinstance(loaded_dict, dict) and 'temp_head' in loaded_dict:
            # Flatten the nested structure
            for head_name, head_state in loaded_dict.items():
                for key, value in head_state.items():
                    heads_state_dict[f"{head_name}.{key}"] = value
        else:
            # Already flat
            heads_state_dict = loaded_dict
    else:
        raise FileNotFoundError(
            f"No heads weights found in {autodeco_path}. "
            f"Expected autodeco_heads.safetensors or autodeco_heads.bin"
        )
    
    # Count heads parameters
    temp_head_keys = [k for k in heads_state_dict.keys() if k.startswith('temp_head.')]
    top_p_head_keys = [k for k in heads_state_dict.keys() if k.startswith('top_p_head.')]
    
    logger.info(f"  ✓ Loaded heads weights")
    logger.info(f"    - temp_head parameters: {len(temp_head_keys)}")
    logger.info(f"    - top_p_head parameters: {len(top_p_head_keys)}")
    
    # Debug: print sample keys and shapes
    logger.info("  - Sample temp_head keys:")
    for key in temp_head_keys[:3]:
        logger.info(f"      {key}: shape {heads_state_dict[key].shape}")
    logger.info("  - Sample top_p_head keys:")
    for key in top_p_head_keys[:3]:
        logger.info(f"      {key}: shape {heads_state_dict[key].shape}")
    
    # Step 2: Load base model
    logger.info(f"\nStep 2: Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use fp16 to save memory
    )
    base_config = base_model.config
    
    logger.info(f"  ✓ Loaded base model")
    logger.info(f"    - model_type: {base_config.model_type}")
    logger.info(f"    - hidden_size: {base_config.hidden_size}")
    logger.info(f"    - num_parameters: {sum(p.numel() for p in base_model.parameters())}")
    
    # Step 3: Merge weights with proper prefixes
    logger.info(f"\nStep 3: Merging weights...")
    full_state_dict = {}
    
    # Add base model weights with 'llm.' prefix
    logger.info("  - Adding base model weights (llm.*)...")
    base_model_state = base_model.state_dict()
    for key, value in base_model_state.items():
        full_state_dict[f"llm.{key}"] = value
    logger.info(f"    Added {len(base_model_state)} parameters")
    
    # Add heads weights directly (keys already have temp_head./top_p_head. prefix)
    logger.info("  - Adding heads weights (temp_head.* and top_p_head.*)...")
    heads_added = 0
    for key, value in heads_state_dict.items():
        if key.startswith('temp_head.') or key.startswith('top_p_head.'):
            full_state_dict[key] = value
            heads_added += 1
    logger.info(f"    Added {heads_added} head parameters")
    
    logger.info(f"  ✓ Total merged parameters: {len(full_state_dict)}")
    
    # Debug: Show sample merged keys for heads
    logger.info("  - Sample merged temp_head keys:")
    temp_keys = [k for k in full_state_dict.keys() if k.startswith('temp_head.')][:5]
    for key in temp_keys:
        logger.info(f"      {key}: shape {full_state_dict[key].shape}")
    
    logger.info("  - Sample merged top_p_head keys:")
    topp_keys = [k for k in full_state_dict.keys() if k.startswith('top_p_head.')][:5]
    for key in topp_keys:
        logger.info(f"      {key}: shape {full_state_dict[key].shape}")
    
    # Step 4: Copy auxiliary files from base model
    logger.info(f"\nStep 4: Copying auxiliary files from base model...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_model_path = Path(base_model_path)
    
    # Files to exclude (model weights and config)
    exclude_patterns = {
        'config.json',  # Will use AutoDeco config instead
        'pytorch_model.bin',
        'model.safetensors',
        'training_args.bin',
        # Sharded model files
        'pytorch_model-*.bin',
        'model-*.safetensors',
        'pytorch_model.bin.index.json',
        'model.safetensors.index.json',
    }
    
    import shutil
    import fnmatch
    
    copied_files = []
    for src_file in base_model_path.glob('*'):
        if src_file.is_file():
            filename = src_file.name
            
            # Check if file should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    should_exclude = True
                    break
            
            if not should_exclude:
                dst_file = output_path / filename
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
    
    if copied_files:
        logger.info(f"  ✓ Copied {len(copied_files)} auxiliary files:")
        for filename in sorted(copied_files)[:10]:  # Show first 10
            logger.info(f"    - {filename}")
        if len(copied_files) > 10:
            logger.info(f"    ... and {len(copied_files) - 10} more")
    else:
        logger.info(f"  ℹ️  No auxiliary files to copy")
    
    # Step 5: Save AutoDeco config (overwrite any copied config)
    logger.info(f"\nStep 5: Saving AutoDeco config...")
    autodeco_config.save_pretrained(output_dir)
    logger.info(f"  ✓ Saved config.json (from AutoDeco checkpoint)")
    
    # Step 6: Save merged weights with sharding (using HF standard method)
    logger.info(f"\nStep 6: Saving merged model weights (with sharding)...")
    logger.info(f"  - Max shard size: {max_shard_size}")
    
    # Determine weights name and filename pattern
    if save_format == "safetensors":
        weights_name = "model.safetensors"
    else:
        weights_name = "pytorch_model.bin"
    
    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
    

    # Use HuggingFace's standard sharding function
    state_dict_split = split_torch_state_dict_into_shards(
        full_state_dict,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size
    )
    
    # Prepare index if sharded
    index = None
    if state_dict_split.is_sharded:
        # Calculate total parameters
        total_params = sum(tensor.numel() for tensor in full_state_dict.values())
        index = {
            "metadata": {
                "total_size": state_dict_split.metadata.get("total_size", 0),
                "total_parameters": total_params
            },
            "weight_map": state_dict_split.tensor_to_filename,
        }
    
    logger.info(f"  - Split into {len(state_dict_split.filename_to_tensors)} shard(s)")
    
    # Save each shard
    for shard_file, tensor_keys in state_dict_split.filename_to_tensors.items():
        shard_path = output_path / shard_file
        shard = {key: full_state_dict[key] for key in tensor_keys}
        
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(shard, shard_path)
                logger.info(f"    ✓ Saved {shard_file} ({_get_size(shard_path)})")
            except ImportError:
                logger.error("  ✗ safetensors not available. Please install: pip install safetensors")
                raise
        else:
            torch.save(shard, shard_path)
            logger.info(f"    ✓ Saved {shard_file} ({_get_size(shard_path)})")
    
    # Save index file if sharded
    if index is not None:
        if save_format == "safetensors":
            index_path = output_path / "model.safetensors.index.json"
        else:
            index_path = output_path / "pytorch_model.bin.index.json"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(f"    ✓ Saved {index_path.name}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    logger.info(f"  ✓ Total checkpoint size: {_get_size_str(total_size)}")
    
    # Success message
    logger.info("\n" + "=" * 80)
    logger.info("✓ Merge completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nOutput checkpoint: {output_dir}")
    logger.info(f"\nCheckpoint contents:")
    
    # List all files in output directory
    all_files = sorted([f for f in output_path.glob("*") if f.is_file()])
    
    # Group files by category
    config_files = [f for f in all_files if 'config' in f.name.lower()]
    tokenizer_files = [f for f in all_files if 'tokenizer' in f.name.lower() or 'vocab' in f.name.lower() or 'merges' in f.name.lower()]
    model_files = [f for f in all_files if 'model' in f.name.lower() or 'safetensors' in f.name.lower() or '.bin' in f.name]
    other_files = [f for f in all_files if f not in config_files and f not in tokenizer_files and f not in model_files]
    
    if config_files:
        logger.info("  Config files:")
        for f in config_files:
            logger.info(f"    ✓ {f.name}")
    
    if model_files:
        logger.info("  Model weights:")
        for f in model_files:
            logger.info(f"    ✓ {f.name} ({_get_size(f)})")
    
    if tokenizer_files:
        logger.info("  Tokenizer files:")
        for f in tokenizer_files:
            logger.info(f"    ✓ {f.name}")
    
    if other_files:
        logger.info("  Other files:")
        for f in other_files:
            logger.info(f"    ✓ {f.name}")
    
    logger.info(f"\n✓ This checkpoint is ready for vLLM deployment:")
    logger.info(f"  from vllm import LLM")
    logger.info(f"  llm = LLM(model='{output_dir}', trust_remote_code=True)")
    logger.info("=" * 80)


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
    save_format: str = "safetensors",
    copy_tokenizer: bool = True
):
    """
    Split full AutoDeco checkpoint to extract only heads weights.
    
    This creates a lightweight checkpoint with only config + heads weights,
    which can be loaded by referencing the original base model.
    
    Args:
        full_checkpoint_path: Path to full AutoDeco checkpoint (with llm.*, temp_head.*, top_p_head.*)
        output_dir: Output directory for heads-only checkpoint
        save_format: Save format ('safetensors' or 'pytorch')
        copy_tokenizer: Whether to copy tokenizer files from full checkpoint
    """
    logger.info("=" * 80)
    logger.info("AutoDeco Model Split (Extract Heads Only)")
    logger.info("=" * 80)
    
    # Step 1: Load config
    logger.info(f"\nStep 1: Loading config from: {full_checkpoint_path}")
    config = AutoDecoConfig.from_pretrained(full_checkpoint_path)
    
    if config.model_type != "autodeco":
        raise ValueError(f"Expected AutoDeco checkpoint, got model_type={config.model_type}")
    
    logger.info(f"  ✓ Loaded AutoDeco config")
    logger.info(f"    - base_model_type: {config.base_model_type}")
    logger.info(f"    - base_model_name_or_path: {config.base_model_name_or_path}")
    logger.info(f"    - train_temp: {config.train_temp}")
    logger.info(f"    - train_top_p: {config.train_top_p}")
    
    # Step 2: Load full checkpoint
    logger.info(f"\nStep 2: Loading full checkpoint weights...")
    checkpoint_path = Path(full_checkpoint_path)
    
    # Check for different checkpoint formats
    full_state_dict = {}
    
    # Try loading from sharded safetensors
    index_safetensors = checkpoint_path / "model.safetensors.index.json"
    index_pytorch = checkpoint_path / "pytorch_model.bin.index.json"
    single_safetensors = checkpoint_path / "model.safetensors"
    single_pytorch = checkpoint_path / "pytorch_model.bin"
    
    if index_safetensors.exists():
        logger.info("  - Loading from sharded safetensors...")
        from safetensors.torch import load_file
        
        with open(index_safetensors, 'r') as f:
            index = json.load(f)
        
        weight_map = index['weight_map']
        shard_files = set(weight_map.values())
        
        for shard_file in shard_files:
            shard_path = checkpoint_path / shard_file
            logger.info(f"    Loading {shard_file}...")
            shard_dict = load_file(shard_path)
            full_state_dict.update(shard_dict)
        
        logger.info(f"  ✓ Loaded {len(full_state_dict)} parameters from {len(shard_files)} shards")
        
    elif index_pytorch.exists():
        logger.info("  - Loading from sharded pytorch...")
        
        with open(index_pytorch, 'r') as f:
            index = json.load(f)
        
        weight_map = index['weight_map']
        shard_files = set(weight_map.values())
        
        for shard_file in shard_files:
            shard_path = checkpoint_path / shard_file
            logger.info(f"    Loading {shard_file}...")
            shard_dict = torch.load(shard_path, map_location='cpu')
            full_state_dict.update(shard_dict)
        
        logger.info(f"  ✓ Loaded {len(full_state_dict)} parameters from {len(shard_files)} shards")
        
    elif single_safetensors.exists():
        logger.info("  - Loading from single safetensors file...")
        from safetensors.torch import load_file
        full_state_dict = load_file(single_safetensors)
        logger.info(f"  ✓ Loaded {len(full_state_dict)} parameters")
        
    elif single_pytorch.exists():
        logger.info("  - Loading from single pytorch file...")
        full_state_dict = torch.load(single_pytorch, map_location='cpu')
        logger.info(f"  ✓ Loaded {len(full_state_dict)} parameters")
        
    else:
        raise FileNotFoundError(
            f"No model weights found in {full_checkpoint_path}. "
            f"Expected model.safetensors, pytorch_model.bin, or their sharded versions."
        )
    
    # Step 3: Extract heads weights only
    logger.info(f"\nStep 3: Extracting heads weights...")
    heads_state_dict = {}
    
    for key, value in full_state_dict.items():
        if key.startswith('temp_head.') or key.startswith('top_p_head.'):
            heads_state_dict[key] = value
    
    if not heads_state_dict:
        raise ValueError(
            f"No heads weights found in checkpoint! "
            f"Expected keys starting with 'temp_head.' or 'top_p_head.'"
        )
    
    temp_head_params = sum(1 for k in heads_state_dict.keys() if k.startswith('temp_head.'))
    top_p_head_params = sum(1 for k in heads_state_dict.keys() if k.startswith('top_p_head.'))
    
    logger.info(f"  ✓ Extracted {len(heads_state_dict)} head parameters")
    logger.info(f"    - temp_head: {temp_head_params} parameters")
    logger.info(f"    - top_p_head: {top_p_head_params} parameters")
    
    # Calculate size reduction
    original_params = len(full_state_dict)
    heads_params = len(heads_state_dict)
    reduction_ratio = (1 - heads_params / original_params) * 100
    logger.info(f"    - Size reduction: {reduction_ratio:.1f}% ({original_params} → {heads_params} params)")
    
    # Step 4: Create output directory and save config
    logger.info(f"\nStep 4: Saving heads-only checkpoint...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save_pretrained(output_dir)
    logger.info(f"  ✓ Saved config.json")
    
    # Step 5: Save heads weights
    if save_format == "safetensors":
        heads_file = output_path / "autodeco_heads.safetensors"
        from safetensors.torch import save_file
        save_file(heads_state_dict, heads_file, metadata={"format": "pt"})
        logger.info(f"  ✓ Saved autodeco_heads.safetensors ({_get_size(heads_file)})")
    else:
        heads_file = output_path / "autodeco_heads.bin"
        torch.save(heads_state_dict, heads_file)
        logger.info(f"  ✓ Saved autodeco_heads.bin ({_get_size(heads_file)})")
    
    # Step 6: Optionally copy tokenizer files
    if copy_tokenizer:
        logger.info(f"\nStep 5: Copying tokenizer files...")
        tokenizer_patterns = [
            'tokenizer.json',
            'tokenizer.model',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'vocab.json',
            'merges.txt',
            'vocab.txt',
            'added_tokens.json',
        ]
        
        copied_files = []
        for pattern in tokenizer_patterns:
            src_file = checkpoint_path / pattern
            if src_file.exists():
                dst_file = output_path / pattern
                shutil.copy2(src_file, dst_file)
                copied_files.append(pattern)
        
        if copied_files:
            logger.info(f"  ✓ Copied {len(copied_files)} tokenizer files:")
            for filename in copied_files:
                logger.info(f"    - {filename}")
        else:
            logger.info(f"  ℹ️  No tokenizer files found to copy")
    
    # Calculate total output size
    total_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    
    # Success message
    logger.info("\n" + "=" * 80)
    logger.info("✓ Split completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nOutput checkpoint: {output_dir}")
    logger.info(f"Total size: {_get_size_str(total_size)}")
    logger.info(f"\nCheckpoint contents:")
    
    all_files = sorted([f for f in output_path.glob("*") if f.is_file()])
    for f in all_files:
        logger.info(f"  ✓ {f.name} ({_get_size(f)})")
    
    logger.info(f"\n✓ This lightweight checkpoint can be loaded with:")
    logger.info(f"  from model.templlm_auto import AutoDecoModelForCausalLM")
    logger.info(f"  model = AutoDecoModelForCausalLM.from_pretrained('{output_dir}')")
    logger.info(f"\n  (Base model will be loaded from: {config.base_model_name_or_path})")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="AutoDeco checkpoint merge/split utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Merge (heads + base model → full checkpoint for vLLM):
    python merge_autodeco.py merge \\
        --autodeco-path ./trained-autodeco \\
        --base-model-path /path/to/Qwen2.5-7B \\
        --output ./autodeco-for-vllm

2. Split (full checkpoint → heads-only checkpoint):
    python merge_autodeco.py split \\
        --full-checkpoint ./ckpt/R1-no-DFT-End2End-5e-6LR-1Epochs-12000Tokens-1BS-4 \\
        --output ./R1-autodeco

Merge output structure:
    autodeco-for-vllm/
    ├── config.json
    ├── model-00001-of-00003.safetensors    (~5GB)
    ├── model-00002-of-00003.safetensors    (~5GB)
    ├── model-00003-of-00003.safetensors    (~4GB)
    ├── model.safetensors.index.json
    └── tokenizer files...

Split output structure:
    autodeco-heads-only/
    ├── config.json
    ├── autodeco_heads.safetensors          (~5MB)
    └── tokenizer files...
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
    merge_parser.add_argument(
        "--save-format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch"],
        help="Save format (default: safetensors)"
    )
    merge_parser.add_argument(
        "--max-shard-size",
        type=str,
        default="4GB",
        help="Maximum shard size (default: 4GB)"
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
    split_parser.add_argument(
        "--save-format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch"],
        help="Save format (default: safetensors)"
    )
    split_parser.add_argument(
        "--no-copy-tokenizer",
        action="store_true",
        help="Don't copy tokenizer files"
    )
    
    args = parser.parse_args()
    
    # Run appropriate operation
    if args.mode == 'merge':
        merge_autodeco(
            args.autodeco_path,
            args.base_model_path,
            args.output,
            args.save_format,
            args.max_shard_size
        )
    elif args.mode == 'split':
        split_autodeco(
            args.full_checkpoint,
            args.output,
            args.save_format,
            copy_tokenizer=not args.no_copy_tokenizer
        )


if __name__ == "__main__":
    main()

# Example usage:
# 
# Merge mode (heads + base → full checkpoint):
# python merge_autodeco.py merge \
#     --autodeco-path ./ckpt/R1-no-DFT-End2End-1-5e-6LR-1Epochs-12000Tokens-1BS-4 \
#     --base-model-path ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --output ./ckpt/R1-no-DFT-End2End-1-full
#
# Split mode (full checkpoint → heads only):
# python merge_autodeco.py split \
#     --full-checkpoint ./ckpt/R1-no-DFT-End2End-1-full \
#     --output ./ckpt/R1-no-DFT-End2End-1-heads