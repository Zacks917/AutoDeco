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
echo "  TP_SIZE: 8"
echo ""

echo "Generating 9 random seeds and running llm_eval with TP=4..."
seeds=()
for i in {1..8}; do
    seed=$((RANDOM % 999999 + 1))
    seeds+=($seed)
    echo "Generated seed $i: $seed"
done

echo ""
echo "All seeds: ${seeds[*]}"
echo ""

echo "Starting tasks."

pids=()
for i in {0..7}; do
    seed=${seeds[$i]}
    echo "Starting task $((i)) with seed $seed"
    python llm_eval.py \
        --k $K \
        --temp $TEMP \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --rp $RP \
        --seed $seed \
        --dataset $DATASET \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tp_size 8 &
    pids+=($!)

    wait ${pids[@]}
    pids=()
    echo "Batch completed."
done

