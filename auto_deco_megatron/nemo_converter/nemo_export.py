import os
import tyro

from nemo.collections import llm

from dataclasses import dataclass, field


@dataclass
class Args:
    input: str
    output: str
    target: str = field(default="hf-peft")


if __name__ == '__main__':
    args: Args = tyro.cli(Args)

    llm.export_ckpt(
        path=args.input,
        target=args.target,
        output_path=args.output,
        overwrite=True
    )
    os.system(f"cp -r {os.path.join(args.input, 'weights')} {args.output}")
