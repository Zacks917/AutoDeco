from functools import lru_cache
from typing import Dict, List

import numpy as np
import orjson
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from nemo.collections.llm.gpt.data import FineTuningDataModule


class AutoDecoDataset(Dataset):
    def __init__(self, fp: str, max_length: int, limit: int = -1):
        self.fp = fp
        self.max_length = max_length
        self.raw_dataset: List[Dict[str, torch.Tensor]] = []
        self.limit = limit

    def load_dataset(self):
        raw_dataset = []
        with open(self.fp, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Load {self.fp}"):
                if 0 < self.limit < i + 1:
                    break
                raw_dataset.append(orjson.loads(line))

        for r in tqdm(raw_dataset, desc=f"Make Dataset: "):
            p_ids, r_ids = r["inputs"]["p_ids"], r["inputs"]["r_ids"]
            if len(p_ids) + len(r_ids) >= self.max_length:
                continue

            input_ids = p_ids + r_ids
            # base_input_ids
            loss_mask = [False] * len(p_ids) + [True] * len(r_ids)

            self.raw_dataset.append({
                "tokens": torch.from_numpy(np.array(input_ids[:-1], dtype=np.int64)).unsqueeze(dim=0),
                "labels": torch.from_numpy(np.array(input_ids[1:], dtype=np.int64)).unsqueeze(dim=0),
                "loss_mask": torch.from_numpy(np.array(loss_mask[1:], dtype=bool)).unsqueeze(dim=0),
            })
        print("len dataset: ", len(self.raw_dataset))

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.raw_dataset[idx]

    def collate_fn(self, batch, pad_token_id: int = 0, pad_seq_len_divisible=None):
        # if len(batch) == 1:
        #     batch[0]["position_ids"] = torch.arange(
        #         start=0, end=batch[0]["tokens"].size(-1), dtype=torch.int64, device=batch[0]["tokens"].device
        #     ).unsqueeze(dim=0)
        #     return batch[0]

        batch_size = len(batch)
        max_length = max([b['tokens'].size(-1) for b in batch])
        if max_length % 8 != 0:
            max_length = (max_length // 8 + 1) * 8

        for i in range(0, len(batch)):
            pad_length = max_length - batch[i]["tokens"].size(-1)
            if pad_length == 0:
                continue
            batch[i]["tokens"] = F.pad(
                input=batch[i]["tokens"], pad=[0, pad_length], value=0
            )
            batch[i]["labels"] = F.pad(input=batch[i]["labels"], pad=[0, pad_length], value=0)
            batch[i]["loss_mask"] = F.pad(input=batch[i]["loss_mask"], pad=[0, pad_length], value=False)

        batch = {k: torch.cat(tensors=[b[k] for b in batch], dim=0).contiguous() for k in batch[0].keys()}

        # 在这附加一个 position_ids
        batch["position_ids"] = torch.arange(start=0, end=max_length, dtype=torch.int64, device=batch["tokens"].device) \
            .unsqueeze(dim=0) \
            .repeat([batch_size, 1])
        return batch


class AutoDecoDataModule(FineTuningDataModule):
    def __init__(
            self,
            fp: str,
            max_length: int,
            hf_model_path: str,
            dataset_root="",
            *args,
            **kwargs,
    ):
        super().__init__(dataset_root="", *args, **kwargs)

        self.fp = fp
        self.max_length = max_length
        self.hf_model_path = hf_model_path

    def prepare_data(self):
        return

    @lru_cache
    def _create_dataset(self, path, pack_metadata_path=None, is_test=False, **kwargs):
        dataset = AutoDecoDataset(fp=self.fp, max_length=self.max_length, limit=-1)
        dataset.load_dataset()
        return dataset
