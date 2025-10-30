import tyro
from dataclasses import dataclass, field

from nemo.collections import llm
from nemo.collections.llm.gpt.model.qwen3 import Qwen3Config235B_A22B, Qwen3Model


@dataclass
class Args:
    input: str
    output: str
    overwrite: bool = field(default=True)


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    llm.import_ckpt(
        model=Qwen3Model(Qwen3Config235B_A22B()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
