#!/bin/bash

# Environment Setup
N_GPUS=8  # Adjust based on your available GPUs
N_NODES=1 # Number of machines to use
LOCAL_IP="127.0.0.1"  # Change this to your machine's IP if running distributed



MODEL_NAME_OR_PATH='test_qwen7b'
EXP_NAME='AutoDeco-R1-Distill-Qwen-7B'


MAX_LENGTH=12000
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=1


# Data Configuration
DATA_NAME=deepmath_30k_trl.json

for LEARNING_RATE in 5e-6; do \
    # export WANDB_RUN_ID="baseline-train2-${MODEL_BASE_NAME}-flash-attn2-lr${LEARNING_RATE}-${NUM_EPOCHS}Epochs-${MAX_LENGTH}Tokens-${BATCH_SIZE}BS-${GRADIENT_ACCUMULATION-step"}GA-think-step-byy
    # Config Output Directory
    OUTPUT_DIR="ckpt/${EXP_NAME}-${LEARNING_RATE}LR-${NUM_EPOCHS}Epochs-${MAX_LENGTH}Tokens-${BATCH_SIZE}BS-${GRADIENT_ACCUMULATION_STEPS}"
    # Launch training with DeepSpeed
    accelerate launch --config_file config/deepspeed/deepspeed_zero3_gradaccu4.yaml \
        --num_processes $N_GPUS \
        --num_machines $N_NODES \
        --main_process_ip $LOCAL_IP \
        trl_train.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset_name $DATA_NAME \
        --max_length $MAX_LENGTH \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --torch_dtype bfloat16 \
        --completion_only_loss true \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_EPOCHS \
        --gradient_checkpointing \
        --logging_steps 1 \
        --output_dir $OUTPUT_DIR \
        --save_strategy 'epoch' \
        --attn_implementation 'flash_attention_2' \
        --save_only_model \
        --train_temp true \
        --train_top_p true
done
