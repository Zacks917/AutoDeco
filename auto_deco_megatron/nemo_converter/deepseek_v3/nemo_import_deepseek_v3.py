import tyro
from dataclasses import dataclass, field

from nemo.collections import llm
from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV3Config


@dataclass
class Args:
    input: str
    output: str
    overwrite: bool = field(default=True)


if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    llm.import_ckpt(
        model=DeepSeekModel(DeepSeekV3Config()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
