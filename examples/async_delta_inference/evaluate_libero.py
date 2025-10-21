#!/usr/bin/env python3
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

"""
使用异步推理框架评估 Libero 环境的示例脚本。

使用步骤：
1. 在一个终端启动 PolicyServer:
   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30

2. 在另一个终端运行这个脚本:
   python examples/async_delta_inference/evaluate_libero.py

或者直接使用 sim_client:
   python src/lerobot/async_delta_inference/sim_client.py \
       --env.type=libero \
       --env.task=libero_10 \
       --policy_type=act \
       --pretrained_name_or_path=lerobot/act_libero_10 \
       --policy_device=cuda \
       --actions_per_chunk=50 \
       --n_episodes=10
"""

from lerobot.async_delta_inference.configs import SimClientConfig
from lerobot.async_delta_inference.sim_client import SimClient
from lerobot.envs.configs import LiberoEnv
import threading
import logging

def main():
    # 配置环境
    env_config = LiberoEnv(
        task="libero_10",
        fps=30,
        obs_type="pixels_agent_pos",
        observation_height=360,
        observation_width=360,
    )
    
    # 配置客户端
    config = SimClientConfig(
        # 环境配置
        env=env_config,
        
        # 策略配置
        policy_type="act",
        pretrained_name_or_path="lerobot/act_libero_10",  # 替换为你的模型路径
        policy_device="cuda",  # 或 "cpu"
        
        # 控制配置
        actions_per_chunk=50,
        chunk_size_threshold=0.5,
        aggregate_fn_name="weighted_average",
        fps=30,
        
        # 评估配置
        n_episodes=10,
        task="",  # Libero 任务从环境配置中获取
        
        # 网络配置
        server_address="127.0.0.1:8080",
        
        # 调试配置
        debug_visualize_queue_size=False,
    )
    
    # 创建客户端
    client = SimClient(config)
    
    # 连接到服务器
    if not client.start():
        logging.error("无法连接到策略服务器。请确保服务器正在运行。")
        return
    
    logging.info("成功连接到策略服务器！")
    logging.info("开始评估...")
    
    # 创建并启动动作接收线程
    action_receiver_thread = threading.Thread(
        target=client.receive_actions, 
        kwargs={"verbose": True},
        daemon=True
    )
    action_receiver_thread.start()
    
    try:
        # 运行控制循环
        client.control_loop(task=config.task, verbose=True)
        
    except KeyboardInterrupt:
        logging.info("\n用户中断评估")
        
    finally:
        # 清理
        client.stop()
        action_receiver_thread.join(timeout=5)
        
        if config.debug_visualize_queue_size:
            from lerobot.async_inference.helpers import visualize_action_queue_size
            visualize_action_queue_size(client.action_queue_size)
        
        logging.info("客户端已停止")
        
        # 打印最终统计
        if client.episode_rewards:
            avg_reward = sum(client.episode_rewards) / len(client.episode_rewards)
            success_rate = sum(client.episode_successes) / len(client.episode_successes)
            
            print(f"\n{'='*60}")
            print(f"评估结果:")
            print(f"  完成集数: {len(client.episode_rewards)}")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main()

