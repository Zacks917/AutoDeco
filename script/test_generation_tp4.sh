#!/bin/bash

# # 从命令行获取参数，如果没有提供则使用默认值
# DATASET=${1:-aime24}
# TEMP=${2:-1.0}
# TOP_P=${3:-1.0}
# TOP_K=${4:--1}
# RP=${5:-1.0}
# K=${6:-16}
# MODEL_NAME_OR_PATH=${7:-/apdcephfs_fsgm/share_303846923/user/zackszcwang/models/openai/gpt-oss-20b}

# echo "Using parameters:"
# echo "  DATASET: $DATASET"
# echo "  TEMP: $TEMP"
# echo "  TOP_P: $TOP_P"
# echo "  TOP_K: $TOP_K"
# echo "  RP: $RP"
# echo "  MODEL: $MODEL_NAME_OR_PATH"
# echo "  TP_SIZE: 8"
# echo ""

# # 生成9个随机seed (范围: 1-999999)
# echo "Generating 9 random seeds and running llm_eval with TP=4..."
# seeds=()
# for i in {1..8}; do
#     seed=$((RANDOM % 999999 + 1))
#     seeds+=($seed)
#     echo "Generated seed $i: $seed"
# done

# echo ""
# echo "All seeds: ${seeds[*]}"
# echo ""

# # 每次启动2个并行任务，每个任务占用4张卡，共两组GPU: [0,1,2,3] 和 [4,5,6,7]
# echo "Starting tasks."

# # 总共9个任务：每批次运行2个任务（8个），最后1个单独跑
# pids=()
# for i in {0..2}; do
#     seed=${seeds[$i]}
#     echo "Starting task $((i)) with seed $seed"
#     python llm_eval.py \
#         --k $K \
#         --temp $TEMP \
#         --top_p $TOP_P \
#         --top_k $TOP_K \
#         --rp $RP \
#         --seed $seed \
#         --dataset $DATASET \
#         --model_name_or_path $MODEL_NAME_OR_PATH \
#         --tp_size 8 &
#     pids+=($!)

#     wait ${pids[@]}
#     pids=()
#     echo "Batch completed."
# done


DATASET=${1:-aime24}
TEMP=${2:-1.0}
TOP_P=${3:-1.0}
TOP_K=${4:--1}
RP=${5:-1.0}
K=${6:-16}
MODEL_NAME_OR_PATH=${7:-/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}

echo "Using parameters:"
echo "  DATASET: $DATASET"
echo "  TEMP: $TEMP"
echo "  TOP_P: $TOP_P"
echo "  TOP_K: $TOP_K"
echo "  RP: $RP"
echo "  MODEL: $MODEL_NAME_OR_PATH"
echo "  TP_SIZE: 4"
echo ""

# 生成9个随机seed (范围: 1-999999)
echo "Generating 9 random seeds and running llm_eval with TP=4..."
seeds=()
for i in {1..9}; do
    seed=$((RANDOM % 999999 + 1))
    seeds+=($seed)
    echo "Generated seed $i: $seed"
done

echo ""
echo "All seeds: ${seeds[*]}"
echo ""

# 每次启动2个并行任务，每个任务占用4张卡，共两组GPU: [0,1,2,3] 和 [4,5,6,7]
echo "Starting tasks in batches of 2 (each uses 4 GPUs)..."

# 总共9个任务：每批次运行2个任务（8个），最后1个单独跑
pids=()
for i in {0..7}; do
    batch_slot=$((i % 2))
    seed=${seeds[$i]}
    if [ $batch_slot -eq 0 ]; then
        gpus="0,1,2,3"
    else
        gpus="4,5,6,7"
    fi
    echo "Starting task $((i+1)) with seed $seed on GPUs $gpus (TP=4)"
    CUDA_VISIBLE_DEVICES=$gpus python llm_eval.py \
        --k 16 \
        --temp $TEMP \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --rp $RP \
        --seed $seed \
        --dataset $DATASET \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tp_size 4 &
    pids+=($!)

    # 每启动两个任务就等待它们完成
    if [ $batch_slot -eq 1 ]; then
        wait ${pids[@]}
        pids=()
        echo "Batch completed."
    fi
done

# # 最后一个任务（第9个）单独运行，默认使用第一组GPU: [0,1,2,3]
# seed9=${seeds[8]}
# echo "Starting final task with seed $seed9 on GPUs 0,1,2,3 (TP=4)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python llm_eval.py \
#     --k 16 \
#     --temp $TEMP \
#     --top_p $TOP_P \
#     --seed $seed9 \
#     --dataset $DATASET \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --tp_size 4

# echo "All 9 runs completed!"

# ./script/test_generation_tp4.sh brumo25 0.9 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/
# ./script/test_generation_tp4.sh brumo25 0.8 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.7 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 1.0  -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.5 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.4 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.3 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.2 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.1 1.0 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/


# ./script/test_generation_tp4.sh brumo25 0.6 0.9 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.8 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.7 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.6  -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.5 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/

# ./script/test_generation_tp4.sh brumo25 0.6 0.4 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.3 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/

# ./script/test_generation_tp4.sh brumo25 0.6 0.2 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
# ./script/test_generation_tp4.sh brumo25 0.6 0.1 -1 1.0 16 ../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/




# ./script/test_generation_tp4.sh brumo25 0.9 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.7 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.6 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.5 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.4 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.3 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.2 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.1 1.0 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.9 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.8 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.7 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.6 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.5 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/

# ./script/test_generation_tp4.sh brumo25 0.8 0.4 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.3 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.2 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/
# ./script/test_generation_tp4.sh brumo25 0.8 0.1 -1 1.0 16 ../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/

# source /root/vllm/bin/activate
# cd ../benchmark/SuperGPQA
# ./gpqar1.sh 0.3 0.1 -1 1.0 1 ../../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/ 

# 29.213.104.242	vEK2TvVLk2MqOPK,
# 29.206.1.176
# 29.206.4.121
# 29.206.5.248
# 29.206.1.239
# 29.206.5.69
# 29.206.4.187
# 29.206.5.149


# llama brumo
# 57.89, 60.21, 60.42, 59.84, 60.27, 59.47, 59.59, 60.30, 60.23, 57.92, 56.67
# 60.42, 61.36, 59.79, 60.30, 60.36, 60.31, 59.48, 59.84, 60.24, 60.27

