from vllm import LLM, SamplingParams
import json
from boxed_extract import *
import argparse
import os
from collections import OrderedDict
from boxed_extract import *
from transformers import AutoTokenizer
import math
from model.templlm_auto import AutoDecoModelForCausalLM

def compute_pass_at_k(scores, k):
    """
    计算pass@k指标
    Args:
        scores: 列表，包含每个样本的得分（0或1）
        k: 要计算的k值
    Returns:
        pass@k的值
    """
    if len(scores) < k:
        return 0.0
    
    # 对于pass@k，只要k个样本中至少有一个正确就算通过
    # 当k等于样本总数时，等价于检查是否有正确答案
    c = sum(scores)  # 正确答案数
    
    # 如果有至少一个正确答案，则通过
    return 1.0 if c > 0 else 0.0

def merge_baselines(dir_path):
    folder_path = dir_path
    from collections import defaultdict

    input_folder = folder_path
    output_file = os.path.join(folder_path, "merged_temp.jsonl")
    # 用 defaultdict 存储每条数据的 temp_acc（按 id 或 text 索引）
    merged_data = defaultdict(dict)

    # 遍历所有 JSONL 文件
    for filename in os.listdir(input_folder):
        if filename.startswith("qwen3-8B-think"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    # 假设用 "id" 或 "text" 作为唯一标识（根据你的数据调整）
                    key = data["problem"]
                    merged_data[key]["problem"] = data["problem"]
                    ground_truth = data["ground_truth"][1:] if data["ground_truth"].startswith('0') else data["ground_truth"]
                    merged_data[key]["ground_truth"] = ground_truth
                    temp = list(data["temp_acc"].keys())[0]
                    acc = data["temp_acc"][temp]
                    # acc = []
                    # for sol in data['solutions']:
                    #     acc.append(compute_score(sol, ground_truth))
                    # acc = round(sum(acc)/len(acc) * 100, 2)
                    # 合并到 merged_data
                    if 'temp_acc' not in merged_data[key]:
                        merged_data[key]["temp_acc"] = {temp: acc}
                    else:
                        merged_data[key]["temp_acc"].update({temp: acc})

    # 写入合并后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        for key, data in merged_data.items():
            sorted_items = sorted(
                data["temp_acc"].items(),
                key=lambda x: float(x[0])
            )
            data["temp_acc"] = OrderedDict(sorted_items)
            sorted_items = sorted(
                data["temp_acc"].items(),
                key=lambda x: float(x[1])
            )
            data["opt_temp"] = {sorted_items[-1][0]: sorted_items[-1][1]}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"合并完成，结果保存到 {output_file}")

def write_ascii_table(txt_path: str, dataset_name: str, avg_pass: float, avg_acc: float, k: int):
    """写入一行三列的 ASCII 表格。

    格式：顶行三列，左上角留空，其余为 Pass@k 与 Acc；
    第二行三列，第一列为数据集名，后两列依次为值。
    """
    headers = ["", f"Pass@{k}", "Acc"]
    row = [dataset_name, f"{avg_pass:.2f}", f"{avg_acc:.2f}"]
    # 根据 header 与数据行计算每列宽度（左右各留 1 空格）
    col_widths = [max(len(headers[i]), len(row[i])) + 2 for i in range(3)]

    def make_border() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+\n"

    border = make_border()
    header_line = "|" + "|".join(headers[i].center(col_widths[i]) for i in range(3)) + "|\n"
    data_line = "|" + "|".join(row[i].center(col_widths[i]) for i in range(3)) + "|\n"

    table_str = border + header_line + border + data_line + border
    with open(txt_path, "w") as txt_file:
        txt_file.write(table_str)
if __name__ == "__main__":

    # model_name_or_path = '/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/shortest_reasoning/TempLLM/DeepSeek-R1-Distill-Qwen-7B/checkpoint-7238/'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0605/checkpoint-41'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0613-16384-bs64/checkpoint-42'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0613-16384-bs64-5/checkpoint-6'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/shared/model/Qwen/Qwen3-8B'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0704-16384-bs64/checkpoint-42'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0704-16384-bs64-attention/checkpoint-42'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0704-16384-bs64-residual/checkpoint-42'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0706-16384-bs64-linear-rs/checkpoint-110'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0706-16384-bs64-linear-rs-test-swanlab/checkpoint-333'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0710-16384-bs64-linear-rs-epoch1/checkpoint-111'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0713-warmup-16384-bs64-linear/checkpoint-106'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0715-rs-16384-bs64-linear/checkpoint-111'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0721-rs-16384-bs64-linear/checkpoint-119'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0721-scaleloss-16384-bs64-linear/checkpoint-43'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0722-scaleloss-16384-bs64-linear-rs/checkpoint-119'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0723-ce-16384-bs64-linear-warmup/checkpoint-43'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0724-16384-bs64-linear-warmup-2.0/checkpoint-946'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/rs-5e-6LR-10Epochs-16384Tokens-1BS-4/checkpoint-238'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/warmup-all-5e-6LR-2Epochs-16384Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/warmup-all-5e-6LR-2Epochs-16384Tokens-1BS-4/checkpoint-123'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Combine-5e-6LR-1Epochs-16384Tokens-1BS-4'
    # # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/0724-16384-bs64-linear-warmup-2.0/checkpoint-946'
    # # model_name_or_path = 'ckpt/baseline-warmup-5e-6LR-2Epochs-16384Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Combine-5e-6LR-1Epochs-12000Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Top-p-5e-6LR-1Epochs-6000Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_fsgm/share_303843174/user/jettexu/zackszcwang/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    # python llm_eval.py --temp 1.0 --k 16
    # [42, 0, 119, 512, 32, 917, 3547, 2190, 6481, 1024]
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python llm_eval.py --temp 1.0 --k 16 --seed 6481
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_eval.py --temp 1.0 --k 16 --seed 1024
    # 从model_name_or_path中提取模型名称
    def extract_model_name(path):
        """从模型路径中提取模型名称"""
        # 移除末尾的斜杠
        path = path.rstrip('/')
        # 分割路径，取倒数第二个部分
        parts = path.split('/')
        if len(parts) >= 2:
            return parts[-1]  
        elif len(parts) == 1:
            return parts[0]   # 只有一个部分
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

    # model_name_or_path = args.model_name_or_path
    # # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Stage1-layer2-5e-6LR-1Epochs-12000Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Stage1-new-5e-6LR-1Epochs-12000Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/Stage1-DeepMath-R1-no-DFT-5e-6LR-1Epochs-12000Tokens-1BS-4'
    # model_name_or_path = '/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    # args.model_name_or_path = '/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/ckpt/ckpt/LLaMA-Stage1-DeepMath-R1-DFT-mask-5e-6LR-1Epochs-16384Tokens-1BS-4-5e-6LR-1Epochs-16384Tokens-1BS-4'
    args.model_name_or_path = 'ckpt/R1-no-DFT-End2End-5e-6LR-1Epochs-12000Tokens-1BS-4'
    ckpt_name = extract_model_name(args.model_name_or_path)

    temp = args.temp
    k = args.k
    seed = args.seed



    with open(f'data/TempTest/{args.dataset}.jsonl', 'r') as f:
    # with open('/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/data/shortest_reasoning/deepmath_processed_test.jsonl', 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    # with open('/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/data/shortest_reasoning/rs_preprocessed_test_0_16384_test.jsonl', 'r') as f:
    #     data = [json.loads(line) for line in f.readlines()]
    
    # for deepmath
    # problems = [
    #     item['prompt'][0]['content'] + 'output the final answer within \\boxed{}' for item in data
    # ]


    # ground_truths = [
    #     item['reward_model']['ground_truth'] for item in data
    # ]
    # # for qwen3
    # messages = [
    #     [{"role": "user", "content": item['problem'] + '\n Output the final answer within \\boxed{}'}] for item in data
    # ]
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # text = [tokenizer.apply_chat_template(
    #     message,
    #     tokenize=False,
    #     add_generation_problem=True,
    #     enable_thinking=True
    # ) for message in messages]


    sampling_params = SamplingParams(temperature=temp, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, n=k, seed=seed, repetition_penalty=args.rp)


    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp_size)

    tokenizer = llm.get_tokenizer()

    problems = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item['problem'] + '\nMake sure you output the final answer within \\boxed{}.'}],
            tokenize=False,
            add_generation_prompt=True,
        ) for item in data
    ]

    ground_truths = [
        item['gt'] for item in data
    ]


    if not os.path.exists(f'generation_log/{args.dataset}'):
        os.makedirs(f'generation_log/{args.dataset}')
    outputs = llm.generate(problems, sampling_params)

    # # for qwen3
    # outputs = llm.generate(text, sampling_params) 

    with open(f'generation_log/{args.dataset}/{ckpt_name}-temp{temp}-top_p{args.top_p}-top_k{args.top_k}-rp{args.rp}-pass@{k}-max_tokens{args.max_tokens}-seed{seed}.json', 'w') as f:
        all_scores = []  # 收集所有问题的得分用于计算pass@32
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
                # 安全地获取temp值，如果属性不存在则使用默认值
                temp = getattr(output, 'temps', None)  # 使用当前循环的temp值作为默认值
                top_p = getattr(output, 'top_p', None)
                score = compute_score(generated_text, gt)
                scores.append(score)
                solutions.append(generated_text)
                if temp is not None:
                    temps.append(temp)
                if top_p is not None:
                    top_ps.append(top_p)
                # logprobs.append(output.logprobs)
            # 计算当前问题的pass@32 (32个样本中有1个正确就算通过)
            pass_at_k = compute_pass_at_k(scores, k)
            all_scores.append(pass_at_k)  # 添加到全局得分列表
            all_acc.append(round(sum(scores)/len(scores)*100, 2))
            # f.write(json.dumps({'problem': data[idx]['problem'], 'ground_truth': ground_truths[idx], 'temp_acc': {args.temp: round(sum(scores)/len(scores)*100, 2)}, 'solutions': solutions, 'temp': temps}, ensure_ascii=False)+'\n')
            f.write(json.dumps({
                'problem': problems[idx],  # data[idx]['problem']
                'ground_truth': ground_truths[idx], 
                'temp_acc': {args.temp: round(sum(scores)/len(scores)*100, 2)}, 
                f'pass_at_{k}': round(pass_at_k * 100, 2),
                'solutions': solutions,
                'temp': temps,
                'top_p': top_ps,
                # 'logprobs': logprobs
            }, ensure_ascii=False)+'\n')
        
        # 计算整体的pass@32
        avg_pass_at_k = round(sum(all_scores)/len(all_scores)*100, 2)
        avg_acc = round(sum(all_acc)/len(all_acc), 2)
        print(f"Overall avg Pass@{k}: {avg_pass_at_k}%")
        print(f"Overall avg Acc: {avg_acc}%")

        # 保存 ASCII 表格到 .txt
        txt_path = os.path.splitext(f.name)[0] + '.txt'
        write_ascii_table(txt_path, 'AIME24', avg_pass_at_k, avg_acc, k)

    # with open(f'/apdcephfs_qy3/share_301812049/zackszcwang/TempLLM/generation_log/deepmath_deepseek-ai-temp0.6-pass@16.json', 'r') as f:
    #     data = [json.loads(line) for line in f.readlines()]
    #     avg_pass_at_16 = []  
    #     avg_success_16 = []
    #     for idx, item in enumerate(data):
    #         pass_at_16 = item[f'pass_at_16']
    #         success_16 = item['temp_acc'][f'{temp}']
    #         avg_pass_at_16.append(pass_at_16)
    #         avg_success_16.append(success_16)
    #     # 计算整体的pass@32
    #     avg_pass_at_16 = round(sum(avg_pass_at_16)/len(avg_pass_at_16), 2)
    #     avg_success_16 = round(sum(avg_success_16)/len(avg_success_16), 2)
    #     print(f"Overall avg Pass@{k}: {avg_pass_at_16}%")
    #     print(f"Overall avg Acc: {avg_success_16}%")

# CUDA_VISIBLE_DEVICES=0,1,2,3 python llm_eval.py --temp 0.6 --k 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python llm_eval.py --temp 1.0 --k 16 --top_p 1.0 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_eval.py --temp 1.0 --k 16 --top_p 0.95 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_eval.py --temp 0.6



