#!/bin/bash
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 快速启动脚本 - Async Delta Inference with Libero
#
# 使用方法:
#   chmod +x examples/async_delta_inference/quick_start.sh
#   ./examples/async_delta_inference/quick_start.sh

set -e  # 遇到错误时退出

echo "============================================"
echo "Async Delta Inference - 快速启动"
echo "============================================"
echo ""

# 检查是否安装了 Libero
echo "1. 检查依赖..."
if ! python -c "import libero" 2>/dev/null; then
    echo "❌ Libero 未安装"
    echo "请运行: pip install -e \".[libero]\""
    exit 1
fi
echo "✅ Libero 已安装"
echo ""

# 配置参数
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8080"}
FPS=${FPS:-"30"}
ENV_TYPE=${ENV_TYPE:-"libero"}
ENV_TASK=${ENV_TASK:-"libero_object"}
POLICY_TYPE=${POLICY_TYPE:-"smolvla"}
MODEL_PATH=${MODEL_PATH:-"HuggingFaceVLA/smolvla_libero"}
DEVICE=${DEVICE:-"cuda"}
N_EPISODES=${N_EPISODES:-"5"}
ACTIONS_PER_CHUNK=${ACTIONS_PER_CHUNK:-"50"}

echo "2. 配置信息:"
echo "   服务器地址: $HOST:$PORT"
echo "   环境类型: $ENV_TYPE"
echo "   任务: $ENV_TASK"
echo "   策略: $POLICY_TYPE"
echo "   模型: $MODEL_PATH"
echo "   设备: $DEVICE"
echo "   评估集数: $N_EPISODES"
echo ""

# 创建日志目录
mkdir -p logs


# 定义清理函数
cleanup() {
    echo ""
    echo "============================================"
    echo "正在清理..."
    echo "============================================"
    if [ ! -z "$SERVER_PID" ]; then
        echo "停止 PolicyServer (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi
    echo "清理完成"
}

# 注册清理函数
trap cleanup EXIT INT TERM

# 启动 SimClient
echo "5. 启动 SimClient 进行评估..."
echo "============================================"
echo ""
echo "⚠️  重要提示: 如果 task 为空，模型无法理解要执行什么任务！"
echo "   请确保环境能够提供任务描述。"
echo ""

python -m lerobot.async_delta_inference.sim_client \
    --env.type=$ENV_TYPE \
    --env.task=$ENV_TASK \
    --env.task_ids=[3] \
    --policy_type=$POLICY_TYPE \
    --pretrained_name_or_path=$MODEL_PATH \
    --policy_device=$DEVICE \
    --actions_per_chunk=$ACTIONS_PER_CHUNK \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --n_episodes=$N_EPISODES \
    --fps=$FPS \
    --server_address=$HOST:$PORT

echo ""
echo "============================================"
echo "评估完成！"
echo "============================================"