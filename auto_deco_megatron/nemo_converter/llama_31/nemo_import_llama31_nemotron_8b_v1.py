import tyro
from dataclasses import dataclass, field

from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import Llama31Config8B, LlamaModel


@dataclass
class Args:
    input: str
    output: str
    overwrite: bool = field(default=True)


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    llm.import_ckpt(
        model=LlamaModel(Llama31Config8B()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
