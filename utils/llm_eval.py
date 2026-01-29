from vllm import LLM, SamplingParams
import json
from boxed_extract import *
import argparse
import os
from collections import OrderedDict
from boxed_extract import *
from transformers import AutoTokenizer
import math
# from model.templlm_auto import AutoDecoModelForCausalLM


def write_ascii_table(txt_path: str, dataset_name: str, avg_acc: float):
    headers = ["", "Acc"]
    row = [dataset_name, f"{avg_acc:.2f}"]
    col_widths = [max(len(headers[i]), len(row[i])) + 2 for i in range(2)]

    def make_border() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+\n"

    border = make_border()
    header_line = "|" + "|".join(headers[i].center(col_widths[i]) for i in range(2)) + "|\n"
    data_line = "|" + "|".join(row[i].center(col_widths[i]) for i in range(2)) + "|\n"

    table_str = border + header_line + border + data_line + border
    with open(txt_path, "w") as txt_file:
        txt_file.write(table_str)
if __name__ == "__main__":
    def extract_model_name(path):
        path = path.rstrip('/')
        parts = path.split('/')
        if len(parts) >= 2:
            return parts[-1]  
        elif len(parts) == 1:
            return parts[0]
        else:
            return 'unknown'
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--rp', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--model_name_or_path', type=str, default='/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='aime24')
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=32768)
    args = parser.parse_args()


    ckpt_name = extract_model_name(args.model_name_or_path)
    temp = args.temp
    k = args.k
    seed = args.seed

    with open(f'data/TempTest/{args.dataset}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    sampling_params = SamplingParams(temperature=temp, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, n=k, seed=seed, repetition_penalty=args.rp)

    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp_size)

    tokenizer = llm.get_tokenizer()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    problems = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item['problem'] + '\nMake sure you output the final answer within \\boxed{}.'}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) for item in data
    ]

    ground_truths = [
        item['gt'] for item in data
    ]

    if not os.path.exists(f'generation_log/{args.dataset}'):
        os.makedirs(f'generation_log/{args.dataset}')
    outputs = llm.generate(problems, sampling_params)

    with open(f'generation_log/{args.dataset}/{ckpt_name}-temp{temp}-top_p{args.top_p}-top_k{args.top_k}-rp{args.rp}-max_tokens{args.max_tokens}-seed{seed}.json', 'w') as f:
        all_acc = []
        for idx, output_group in enumerate(outputs):
            solutions = []
            temps = []
            gt = str(ground_truths[idx])
            scores = []
            logprobs = []
            top_ps = []
            for output in output_group.outputs:
                generated_text = output.text
                temp = getattr(output, 'temperatures', None)  # 使用当前循环的temp值作为默认值
                top_p = getattr(output, 'top_ps', None)
                score = compute_score(generated_text, gt)
                scores.append(score)
                solutions.append(generated_text)
                if temp is not None:
                    temps.append(temp)
                if top_p is not None:
                    top_ps.append(top_p)
                # logprobs.append(output.logprobs)
            all_acc.append(round(sum(scores)/len(scores)*100, 2))
            f.write(json.dumps({
                'problem': problems[idx],
                'ground_truth': ground_truths[idx], 
                'temp_acc': {args.temp: round(sum(scores)/len(scores)*100, 2)}, 
                'solutions': solutions,
                'temp': temps,
                'top_p': top_ps,
                # 'logprobs': logprobs
            }, ensure_ascii=False)+'\n')
        
        avg_acc = round(sum(all_acc)/len(all_acc), 2)
        print(f"Overall avg Acc: {avg_acc}%")

        txt_path = os.path.splitext(f.name)[0] + '.txt'
        write_ascii_table(txt_path, args.dataset, avg_acc)



