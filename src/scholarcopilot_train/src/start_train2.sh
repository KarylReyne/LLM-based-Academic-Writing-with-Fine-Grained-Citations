#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=2
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-60000}}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
export TORCHELASTIC_ERROR_FILE="start_train2_errorfile"

# conda activate scholar_copilot

output_dir="../../scholarcopilot_model_sc_train_data_500k/"
# model_dir="../../scholarcopilot_model_v1208"
model_dir="../../qwen2.5_7b_instruct"
# model_dir="../../qwen2.5_3b"
dataset_dir="../../scholarcopilot_data/scholar_copilot_train_data_500k.json"


torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 --max_restarts=2 \
 --rdzv-id=$SLURM_JOB_ID \
 --rdzv-backend=c10d \
 --rdzv-endpoint=$MASTER_ADDR \
 train.py \
 --deepspeed ds_zero3_config.json \
 --output_dir ${output_dir} \
 --model_name_or_path ${model_dir} \
 --save_steps 200 \
 --dataset_name json \
 --dataset_path ${dataset_dir} \
 --normalize true \
 --temperature 0.01 \
 --lora true \
 --lora_r 8 \
 --lora_alpha 64 \
 --lora_dropout 0.1 \
 --lora_target_modules "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj" \
 --lora_use_rslora true \
 --per_device_train_batch_size 1 \
 --gradient_checkpointing \
 --learning_rate 1e-5 \
 --query_max_len 16384 \
 --passage_max_len 16384 \
 --num_train_epochs 1 \
 --logging_steps 1 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 1

