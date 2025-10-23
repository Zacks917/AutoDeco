import os
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file


@dataclass
class Args:
    input: str  # path to full HF checkpoint folder
    output: str  # where to write head-only folder
    temp_key: str = "temp_head"
    top_p_key: str = "top_p_head"


def _list_files(folder: str, suffixes: Tuple[str, ...]) -> List[str]:
    return [f for f in os.listdir(folder) if f.endswith(suffixes)]


def _copy_if_exists(src_dir: str, dst_dir: str, name: str):
    src = os.path.join(src_dir, name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(dst_dir, name))


def _collect_from_index(folder: str, patterns: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    """Use model.safetensors.index.json to only load shards that contain heads."""
    index_path = os.path.join(folder, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index.get("weight_map", {})

    # find all keys we care about and the shards that contain them
    target_keys = [k for k in weight_map.keys() if any(p in k for p in patterns)]
    needed_shards = sorted({weight_map[k] for k in target_keys})

    result: Dict[str, torch.Tensor] = {}
    for shard_name in needed_shards:
        shard_path = os.path.join(folder, shard_name)
        if not os.path.exists(shard_path):
            continue
        sd = load_file(shard_path)
        for k, v in sd.items():
            if any(p in k for p in patterns):
                result[k] = v
    return result


def _collect_from_safetensors(folder: str, patterns: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    st_files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
    for name in st_files:
        if name == "model.safetensors.index.json":
            continue
        path = os.path.join(folder, name)
        try:
            sd = load_file(path)
        except Exception:
            continue
        for k, v in sd.items():
            if any(p in k for p in patterns):
                result[k] = v
    return result


def _collect_from_bin(folder: str, patterns: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    bin_files = [f for f in os.listdir(folder) if f.startswith("pytorch_model") and f.endswith(".bin")]
    for name in bin_files:
        path = os.path.join(folder, name)
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if isinstance(obj, dict):
            # try common wrappers
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                sd = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"]
            else:
                sd = obj
            for k, v in sd.items():
                if any(p in k for p in patterns):
                    result[k] = v.detach().cpu() if torch.is_tensor(v) else torch.tensor(v)
    return result


def extract_heads(input_dir: str, output_dir: str, temp_key: str, top_p_key: str):
    os.makedirs(output_dir, exist_ok=True)
    patterns = (temp_key, top_p_key)

    # Strategy 1: use index for safetensors
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        head_state = _collect_from_index(input_dir, patterns)
    else:
        # Strategy 2: scan safetensors files
        head_state = _collect_from_safetensors(input_dir, patterns)
        if not head_state:
            # Strategy 3: scan pytorch .bin files
            head_state = _collect_from_bin(input_dir, patterns)

    if not head_state:
        raise RuntimeError("No head parameters found. Check that the checkpoint contains 'temp_head'/'top_p_head'.")

    # Save as model.safetensors in output
    out_path = os.path.join(output_dir, "model.safetensors")
    save_file(head_state, out_path)
    print(f"[+] Saved head-only weights to {out_path} ({len(head_state)} tensors)")

    # Copy metadata files if available
    meta_files = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",  # for sentencepiece-based tokenizers
        "vocab.json",       # for BPE
        "merges.txt",       # for BPE
        "chat_template.jinja",
    ]
    for fname in meta_files:
        _copy_if_exists(input_dir, output_dir, fname)

    print(f"[+] Copied metadata files to {output_dir}")


if __name__ == "__main__":
    try:
        import tyro  # type: ignore
        args = tyro.cli(Args)
    except Exception:
        # Simple argparse fallback if tyro is not installed
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--input", required=True, help="Path to full checkpoint folder")
        p.add_argument("--output", required=True, help="Output folder for head-only checkpoint")
        p.add_argument("--temp_key", default="temp_head")
        p.add_argument("--top_p_key", default="top_p_head")
        ns = p.parse_args()

        class Simple:
            pass

        args = Simple()
        args.input = ns.input
        args.output = ns.output
        args.temp_key = ns.temp_key
        args.top_p_key = ns.top_p_key

    extract_heads(args.input, args.output, args.temp_key, args.top_p_key)

