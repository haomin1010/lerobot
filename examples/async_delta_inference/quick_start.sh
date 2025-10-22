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
echo "Async Delta Inference Server- 快速启动"
echo "============================================"
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

echo "配置信息:"
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

# 启动 PolicyServer
echo "启动 PolicyServer..."
python -m lerobot.async_inference.policy_server \
    --host=$HOST \
    --port=$PORT \
    --fps=$FPS \
    --inference_latency=0.033 \
    --obs_queue_timeout=2

SERVER_PID=$!
echo "✅ PolicyServer 已启动 (PID: $SERVER_PID)"
echo "   日志: logs/policy_server.log"
echo ""


