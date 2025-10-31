import tyro
from dataclasses import dataclass, field

from nemo.collections import llm
from nemo.collections.llm.gpt.model.gpt_oss import GPTOSSConfig20B, GPTOSSConfig120B, GPTOSSModel


@dataclass
class Args:
    input: str
    output: str
    overwrite: bool = field(default=True)


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    llm.import_ckpt(
        model=GPTOSSModel(GPTOSSConfig20B()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
