#!/bin/bash

# 2-GPU 数据并行训练脚本
# 无需NVLink，自动使用NCCL进行梯度同步

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --config_file <(cat << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
) \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --save_freq=1000 --save_checkpoint=True \
  --output_dir=./outputs/ --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero --env.task=libero_10 \
  --steps=100000 \
  --batch_size=64 \
  --eval.batch_size=1 --eval.n_episodes=1 --eval_freq=1000 \
  --policy.train_delta_expert=True \
  --policy.pretrained_path=HuggingFaceVLA/smolvla_libero \
  --policy.expert_width_multiplier=0.5 \
  --dataset.root=/home/kemove/.cache/huggingface/smolvla_datasets/libero \
  --log_freq=100 \
  --num_workers=16 \
  --prefetch_factor=4

