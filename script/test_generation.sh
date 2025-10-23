#!/bin/bash

# 从命令行获取参数，如果没有提供则使用默认值
DATASET=${1:-aime24}
TEMP=${2:-1.0}
TOP_P=${3:-1.0}
TOP_K=${4:--1}
RP=${5:-1.0}
MODEL_NAME_OR_PATH=${6:-/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}

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


