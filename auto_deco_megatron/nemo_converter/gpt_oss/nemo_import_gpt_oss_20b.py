from nemo.collections import llm
from nemo.collections.llm.gpt.model.gpt_oss import GPTOSSConfig20B, GPTOSSConfig120B, GPTOSSModel


if __name__ == '__main__':
    llm.import_ckpt(
        model=GPTOSSModel(GPTOSSConfig20B()),
        source=f"hf://{args.input}",
        output_path=args.output,
        overwrite=args.overwrite,
    )
