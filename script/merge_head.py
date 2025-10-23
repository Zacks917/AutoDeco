import copy
import os
import json
import orjson
import tyro
from tqdm import tqdm
from itertools import zip_longest

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Set, Dict, List, Tuple, Union, Optional, Literal, TypedDict, NamedTuple, Iterable

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, PreTrainedTokenizer
import shutil
from glob import glob


@dataclass
class Args:
    input: str
    output: str
    base_dir: str = field(default="/apdcephfs_fsgm/share_303846923/user/zackszcwang/temp/ckpt/R1-Stage2-5e-6LR-1Epochs-12000Tokens-1BS-4GA")

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


def copy_metadata_files(base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    patterns = ["*.json", "*.jinja"]
    copied = 0
    for pat in patterns:
        for src in glob(os.path.join(base_dir, pat)):
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(output_dir, os.path.basename(src)))
                copied += 1
    print(f"[!] Copied {copied} metadata files from base to {output_dir}")


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    checkpoints = [name for name in os.listdir(args.input) if name.startswith("checkpoint") and os.path.isdir(os.path.join(args.input, name))]
    checkpoints.sort(key=lambda i: int(i.split("-")[-1]))

    base_state_dicts = load_state_dict(dir_path=args.base_dir)
    weight_index = load_json(fp=os.path.join(args.base_dir, "model.safetensors.index.json"))

    # If no checkpoint subfolders, treat input as a single heads folder
    if len(checkpoints) == 0:
        print(f"[!] No 'checkpoint*' subfolders found under {args.input}. Treating it as a single heads folder.")
        c_base_state_dicts = copy.deepcopy(base_state_dicts)
        c_weight_index = copy.deepcopy(weight_index)

        head_state_dict = {k: v for i in load_state_dict(dir_path=args.input) for k, v in i[1].items()}
        if len(head_state_dict) == 0:
            raise RuntimeError(f"No .safetensors found in {args.input}. Ensure it contains model.safetensors with head weights.")

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
                c_weight_index["weight_map"][k] = c_base_state_dicts[-1][0]

        os.system(f"mkdir -p {args.output}")
        copy_metadata_files(args.base_dir, args.output)

        write_json(fp=os.path.join(args.output, "model.safetensors.index.json"), obj=c_weight_index)
        for name, state_dict in c_base_state_dicts:
            print(f"[!] save {name}")
            save_file(tensors=state_dict, filename=os.path.join(args.output, name))
        print(f"[!] Merge completed → {args.output}")
    else:
        pbr = tqdm(total=len(checkpoints))
        for ckpt in checkpoints:
            input_dir = os.path.join(args.input, ckpt)

            output = os.path.join(args.output, ckpt)
            # if os.path.exists(output):
            #     continue

            pbr.set_description(desc=f"Merge {ckpt}: ")
            c_base_state_dicts = copy.deepcopy(base_state_dicts)
            c_weight_index = copy.deepcopy(weight_index)

            merged = False
            head_state_dict = {k: v for i in load_state_dict(dir_path=input_dir) for k, v in i[1].items()}
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
                    c_weight_index["weight_map"][k] = c_base_state_dicts[-1][0]

            os.system(f"mkdir -p {output}")
            copy_metadata_files(args.base_dir, output)

            write_json(fp=os.path.join(output, "model.safetensors.index.json"), obj=c_weight_index)
            for name, state_dict in c_base_state_dicts:
                print(f"[!] save {name}")
                save_file(tensors=state_dict, filename=os.path.join(output, name))
            pbr.update(n=1)
