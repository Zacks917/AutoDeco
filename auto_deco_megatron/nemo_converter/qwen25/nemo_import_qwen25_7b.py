import tyro

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from nemo.collections import llm
# from nemo.collections.llm.gpt.model.gpt_oss import GPTOSSConfig20B, GPTOSSConfig120B, GPTOSSModel
# from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV3Config
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Model, Qwen25Config7B


@dataclass
class Args:
    input: str
    output: str
    overwrite: bool = field(default=True)


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    llm.import_ckpt(
        model=Qwen2Model(Qwen25Config7B()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
