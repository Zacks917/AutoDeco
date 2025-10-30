#!/bin/bash


DATASET=${1:-aime24}
TEMP=${2:-1.0}
TOP_P=${3:-1.0}
TOP_K=${4:--1}
RP=${5:-1.0}
K=${6:-16}
MODEL_NAME_OR_PATH=${7:-path_to_your_model}

echo "Using parameters:"
echo "  DATASET: $DATASET"
echo "  TEMP: $TEMP"
echo "  TOP_P: $TOP_P"
echo "  TOP_K: $TOP_K"
echo "  RP: $RP"
echo "  MODEL: $MODEL_NAME_OR_PATH"
echo "  TP_SIZE: 4"
echo ""

echo "Generating 8 random seeds and running llm_eval with TP=4..."
seeds=()
for i in {1..8}; do
    seed=$((RANDOM % 999999 + 1))
    seeds+=($seed)
    echo "Generated seed $i: $seed"
done

echo ""
echo "All seeds: ${seeds[*]}"
echo ""

echo "Starting tasks in batches of 2 (each uses 4 GPUs)..."

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

    if [ $batch_slot -eq 1 ]; then
        wait ${pids[@]}
        pids=()
        echo "Batch completed."
    fi
done

