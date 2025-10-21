@echo off
REM Copyright 2025 The HuggingFace Inc. team. All rights reserved.
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM 快速启动脚本 - Async Delta Inference with Libero (Windows)
REM
REM 使用方法:
REM   .\examples\async_delta_inference\quick_start.bat

setlocal

echo ============================================
echo Async Delta Inference - 快速启动
echo ============================================
echo.

REM 检查是否安装了 Libero
echo 1. 检查依赖...
python -c "import libero" 2>nul
if errorlevel 1 (
    echo ❌ Libero 未安装
    echo 请运行: pip install -e ".[libero]"
    exit /b 1
)
echo ✅ Libero 已安装
echo.

REM 配置参数
if not defined HOST set HOST=127.0.0.1
if not defined PORT set PORT=8080
if not defined FPS set FPS=30
if not defined ENV_TYPE set ENV_TYPE=libero
if not defined ENV_TASK set ENV_TASK=libero_10
if not defined POLICY_TYPE set POLICY_TYPE=act
if not defined MODEL_PATH set MODEL_PATH=lerobot/act_libero_10
if not defined DEVICE set DEVICE=cuda
if not defined N_EPISODES set N_EPISODES=5
if not defined ACTIONS_PER_CHUNK set ACTIONS_PER_CHUNK=50

echo 2. 配置信息:
echo    服务器地址: %HOST%:%PORT%
echo    环境类型: %ENV_TYPE%
echo    任务: %ENV_TASK%
echo    策略: %POLICY_TYPE%
echo    模型: %MODEL_PATH%
echo    设备: %DEVICE%
echo    评估集数: %N_EPISODES%
echo.

REM 创建日志目录
if not exist logs mkdir logs

REM 启动 PolicyServer
echo 3. 启动 PolicyServer...
start /B python -m lerobot.async_inference.policy_server ^
    --host=%HOST% ^
    --port=%PORT% ^
    --fps=%FPS% ^
    --inference_latency=0.033 ^
    --obs_queue_timeout=2 ^
    > logs\policy_server.log 2>&1

echo ✅ PolicyServer 已启动
echo    日志: logs\policy_server.log
echo.

REM 等待服务器启动
echo 4. 等待服务器就绪...
timeout /t 3 /nobreak >nul
echo.

REM 启动 SimClient
echo 5. 启动 SimClient 进行评估...
echo ============================================
echo.

python src\lerobot\async_delta_inference\sim_client.py ^
    --env.type=%ENV_TYPE% ^
    --env.task=%ENV_TASK% ^
    --policy_type=%POLICY_TYPE% ^
    --pretrained_name_or_path=%MODEL_PATH% ^
    --policy_device=%DEVICE% ^
    --actions_per_chunk=%ACTIONS_PER_CHUNK% ^
    --chunk_size_threshold=0.5 ^
    --aggregate_fn_name=weighted_average ^
    --n_episodes=%N_EPISODES% ^
    --fps=%FPS% ^
    --server_address=%HOST%:%PORT%

echo.
echo ============================================
echo 评估完成！
echo ============================================
echo.
echo 正在停止 PolicyServer...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq PolicyServer*" 2>nul

endlocal

