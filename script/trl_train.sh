#!/bin/bash

# set wandb
# export WANDB_PROJECT="AgentLLM"
# export WANDB_ENTITY="jiahao004"  # Replace with your wandb username
# export WANDB_API_KEY="f16981cf7a4f83646c11f243140aceeee6a153ce"


# # Environment Setup
# N_GPUS=8  # Adjust based on your available GPUs
# N_NODES=1  # Number of machines to use
# LOCAL_IP="127.0.0.1"  # Change this to your machine's IP if running distributed

# # Model and Data Configuration
# MODEL_NAME_OR_PATH='/apdcephfs_qy3/share_301812049/shared/model/Qwen/Qwen3-8B'
# MODEL_BASE_NAME=$(basename $MODEL_NAME_OR_PATH)

# # Training Hyperparameters
# N_GPUS=$HOST_GPU_NUM
# N_NODES=$HOST_NUM
# MAX_LENGTH=32768
# BATCH_SIZE=1
# GRADIENT_ACCUMULATION_STEPS=4
# NUM_EPOCHS=3

# # export WANDB_PROJECT="AgentLLM"

# # Data Configuration
# TRAIN_FILE=/apdcephfs_qy3/share_301812049/zackszcwang/agentLLM/data/agentllm_trainingset/deepmath.1k.train.action3_1_qwen.jsonl
# for LEARNING_RATE in 1e-5 5e-6 1e-6; do \
#     # export WANDB_RUN_ID="baseline-train2-${MODEL_BASE_NAME}-flash-attn2-lr${LEARNING_RATE}-${NUM_EPOCHS}Epochs-${MAX_LENGTH}Tokens-${BATCH_SIZE}BS-${GRADIENT_ACCUMULATION_STEPS}GA-think-step-by-step"
#     # Config Output Directory
#     OUTPUT_DIR="checkpoint/SFT-multi-action-train2-${MODEL_BASE_NAME}-${LEARNING_RATE}LR-${NUM_EPOCHS}Epochs-${MAX_LENGTH}Tokens-${BATCH_SIZE}BS-${GRADIENT_ACCUMULATION_STEPS}GA-think-step-by-step"
#     # Launch training with DeepSpeed
#     accelerate launch --config_file /apdcephfs_qy3/share_301812049/zackszcwang/agentLLM/scripts/accelerate_configs/deepspeed_zero3_gradaccu4.yaml \
#         --num_processes $N_GPUS \
#         --num_machines $N_NODES \
#         --main_process_ip $LOCAL_IP \
#         --main_process_port 29500 \
#         scripts/train_sft.py \
#         --model_name_or_path $MODEL_NAME_OR_PATH \
#         --dataset_name $TRAIN_FILE \
#         --max_length $MAX_LENGTH \
#         --per_device_train_batch_size $BATCH_SIZE \
#         --per_device_eval_batch_size $BATCH_SIZE \
#         --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#         --torch_dtype float16 \
#         --completion_only_loss true \
#         --learning_rate $LEARNING_RATE \
#         --num_train_epochs $NUM_EPOCHS \
#         --gradient_checkpointing \
#         --eos_token '<|im_end|>' \
#         --logging_steps 1 \
#         --report_to swanlab \
#         --output_dir $OUTPUT_DIR \
#         --save_strategy 'epoch' \
#         --attn_implementation 'flash_attention_2' \
#         --save_only_model
# done





#!/bin/bash

# set wandb
# export WANDB_PROJECT="AgentLLM"
# export WANDB_ENTITY="jiahao004"  # Replace with your wandb username
# export WANDB_API_KEY="f16981cf7a4f83646c11f243140aceeee6a153ce"
# Environment Setup
N_GPUS=8  # Adjust based on your available GPUs
N_NODES=1 # Number of machines to use
LOCAL_IP="127.0.0.1"  # Change this to your machine's IP if running distributed


# MODEL_NAME_OR_PATH='/apdcephfs_fsgm/share_303843174/user/jettexu/zackszcwang/models/Qwen/Qwen3-8B-Base'
# MODEL_NAME_OR_PATH='/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# MODEL_NAME_OR_PATH='/apdcephfs_fsgm/share_303843174/user/jettexu/zackszcwang/models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1'
# MODEL_NAME_OR_PATH='ckpt/Qwen3-8B-DM-SFT-5e-6LR-1Epochs-16384Tokens-1BS-4GA/'
# MODEL_NAME_OR_PATH='../models/openai/gpt-oss-20b/'
MODEL_NAME_OR_PATH='../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# MODEL_NAME_OR_PATH='../models/nvidia/Llama-3.1-Nemotron-Nano-8B-v1'
# MODEL_NAME_OR_PATH='/apdcephfs_fsgm/share_303846923/user/zackszcwang/temp/ckpt/TempR1-2hidden-layers-temp-5e-6LR-1Epochs-16384Tokens-1BS-4'
# MODEL_NAME_OR_PATH='ckpt/TempR1-Stage1-5e-6LR-1Epochs-16384Tokens-1BS-4'
EXP_NAME='R1-ODFT-End2End'
# EXP_NAME='Qwen3-8B-sft'
# EXP_NAME='Stage1-DeepMath-R1-DFT'

MAX_LENGTH=12000
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=1

# export WANDB_PROJECT="AgentLLM"

# Data Configuration
# DATA_NAME=deepmath_qwen3_sft_trl.jsonl
# DATA_NAME=qwen3-8b-trl.jsonl
# DATA_NAME=deepmath_30k_llama_trl.jsonl
DATA_NAME=deepmath_30k_trl.json
# DATA_NAME=deepmath_qwen3_deco_trl.jsonl
# data_name=rs
# TRAIN_FILE=data/agentllm/train/multi_think_speak_math.jsonl
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
        --report_to swanlab \
        --output_dir $OUTPUT_DIR \
        --save_strategy 'epoch' \
        --attn_implementation 'flash_attention_2' \
        --save_only_model \
        --train_temp true \
        --train_top_p true
done


