import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.templlm_auto import AutoDecoModelForCausalLM, AutoDecoModelForCausalLMConfig
from transformers import AutoModelForCausalLM, AutoConfig
import argparse
# Add project root (parent of this script directory) to sys.path for imports

def main(base_model_name_or_path, output_dir):
    from transformers import AutoTokenizer
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    autodeco_config = AutoDecoModelForCausalLMConfig(
        base_model_name_or_path=base_model_name_or_path,
        enable_temperature_head=True,
        enable_top_p_head=True,
        use_enhanced_features=True, # TODO: make it a parameter
        **base_config.to_dict()
    )
    model = AutoDecoModelForCausalLM(autodeco_config, dtype=base_config.dtype)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"AutoDeco model saved to {output_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.base_model_name_or_path, args.output_dir)