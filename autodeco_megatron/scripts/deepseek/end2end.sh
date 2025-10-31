export NCCL_IB_TIMEOUT=24
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_NVLS_ENABLE=0


NNODES=8
NPROC_PER_NODE=8


NEMO_MODEL_PATH="<the path of nemo checkpoint>"
HF_MODEL_PATH="<the path of huggingface checkpoint>"

LEARNING_RATE=5e-6
GLOBAL_BATCH_SIZE=32

RUN_NAME=""
TRAIN_FP=""
SAVE_DIR=""

mkdir -p $SAVE_DIR/$RUN_RUN_NAME

SCRIPT_PATH=$(readlink -f "$0")
cp $SCRIPT_PATH $SAVE_DIR/$RUN_NAME/
cp -r src $SAVE_DIR/$RUN_NAME/
cp -r nemo_trainer.py $SAVE_DIR/$RUN_NAME/

echo "LEARNING_RATE: $LEARNING_RATE"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "RUN_NAME: $RUN_NAME"

torchrun --nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --master-addr $MASTER_ADDR --node-rank $NODE_RANK --master-port 29500 nemo_trainer.py \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --model_type="deepseek-v3" \
  --nemo_model_path=$NEMO_MODEL_PATH \
  --hf_model_path=$HF_MODEL_PATH \
  --model_head_path="" \
  --train_temp=True \
  --train_top_p=True \
  --learning_rate=$LEARNING_RATE \
  --weight_decay=0.0 \
  --tensor_model_parallel_size=1 \
  --context_parallel_size=1 \
  --expert_model_parallel_size=8 \
  --pipeline_model_parallel_size=8 \
  --num_layers_in_first_pipeline_stage=7 \
  --num_layers_in_last_pipeline_stage=6 \
  --train_fp=$TRAIN_FP \
  --eval_fp="" \
  --max_length=6144 \
  --save_dir=$SAVE_DIR \
  --micro_batch_size=1 \
  --global_batch_size=$GLOBAL_BATCH_SIZE \
  --max_steps=-1 \
  --num_train_items=29844 \
  --max_epochs=1 \
  --warmup_steps=20 \
  --run_name=$RUN_NAME
