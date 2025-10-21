# Async Delta Inference - 支持仿真环境的异步推理框架

这个模块扩展了 `async_inference` 框架，使其支持仿真环境（如 Libero）。它最小化地改动了原始代码，主要是将真实机器人接口适配为仿真环境接口。

## 主要特性

- ✅ 支持 Libero 仿真环境
- ✅ 复用 `async_inference` 的 PolicyServer（无需修改）
- ✅ 支持动作聚合（action aggregation）
- ✅ 支持多集评估
- ✅ 自动计算成功率和平均奖励

## 架构

该框架采用客户端-服务器架构：

- **PolicyServer** (来自 `async_inference`): 运行策略推理，返回动作块
- **SimClient** (新增): 运行仿真环境，发送观察，接收动作

```
┌─────────────────┐        gRPC         ┌─────────────────┐
│   SimClient     │ ◄─────────────────► │  PolicyServer   │
│                 │                      │                 │
│ - Libero Env    │   Observations       │ - Policy Model  │
│ - Action Queue  │ ──────────────────►  │ - Preprocessor  │
│ - Observation   │                      │ - Postprocessor │
│   Sender        │ ◄──────────────────  │                 │
│                 │   Action Chunks      │                 │
└─────────────────┘                      └─────────────────┘
```

## 使用方法

### 1. 启动 PolicyServer

在一个终端中启动策略服务器：

```bash
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=2
```

### 2. 启动 SimClient (Libero)

在另一个终端中启动仿真客户端：

```bash
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=lerobot/act_libero_10 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --n_episodes=10 \
    --fps=30
```

### 参数说明

#### 环境配置
- `--env.type`: 环境类型（`libero`、`metaworld` 等）
- `--env.task`: 任务名称（如 `libero_10`、`libero_spatial` 等）
- `--env.obs_type`: 观察类型（`pixels_agent_pos` 或 `pixels`）

#### 策略配置
- `--policy_type`: 策略类型（`act`、`diffusion`、`smolvla` 等）
- `--pretrained_name_or_path`: 预训练模型路径或 HuggingFace Hub ID
- `--policy_device`: 推理设备（`cuda`、`cpu`、`mps`）

#### 控制配置
- `--actions_per_chunk`: 每个动作块的动作数量
- `--chunk_size_threshold`: 发送新观察的队列阈值（0.0-1.0）
- `--aggregate_fn_name`: 动作聚合函数（`weighted_average`、`latest_only`、`average`、`conservative`）

#### 评估配置
- `--n_episodes`: 评估的集数
- `--fps`: 控制频率（帧/秒）

#### 调试配置
- `--debug_visualize_queue_size`: 可视化动作队列大小

## 与 async_inference 的区别

### 相同点
- 使用相同的 PolicyServer
- 使用相同的 gRPC 通信协议
- 使用相同的动作聚合机制
- 使用相同的 helpers 工具

### 不同点
- **SimClient vs RobotClient**: SimClient 使用 Gym 环境而不是真实机器人
- **观察格式转换**: 将 Gym 观察格式转换为 LeRobot 格式
- **评估循环**: SimClient 运行固定数量的评估集
- **统计信息**: SimClient 自动计算成功率和平均奖励

## 代码结构

```
async_delta_inference/
├── __init__.py          # 模块导出
├── configs.py           # SimClientConfig 配置类
├── constants.py         # 常量定义
├── sim_client.py        # SimClient 主实现
└── README.md           # 本文档
```

## 扩展到其他仿真环境

要支持新的仿真环境（如 MetaWorld），只需：

1. 在 `constants.py` 中添加环境名称到 `SUPPORTED_ENVS`
2. 确保环境配置在 `lerobot/envs/configs.py` 中定义
3. 确保 `_format_env_observation` 函数正确转换观察格式

## 示例输出

```
[sim_client] Creating simulation environment: libero
[sim_client] Initializing client to connect to server at 127.0.0.1:8080
[sim_client] Simulation environment ready
[sim_client] Connected to policy server in 0.0023s
[sim_client] Sending policy instructions to policy server
[sim_client] Action receiving thread starting
[sim_client] Control loop thread starting
[sim_client] Episode 1/10 finished | Steps: 127 | Reward: 1.0000 | Success: True
[sim_client] Episode 2/10 finished | Steps: 134 | Reward: 1.0000 | Success: True
...
[sim_client] 
============================================================
Evaluation complete!
Episodes: 10
Average Reward: 0.9500
Success Rate: 95.00%
============================================================
```

## 常见问题

**Q: 为什么需要两个终端？**
A: PolicyServer 和 SimClient 是独立的进程，通过 gRPC 通信。这允许它们运行在不同的机器上（例如，将计算密集型的推理放在 GPU 服务器上）。

**Q: 可以同时连接多个客户端吗？**
A: PolicyServer 一次只处理一个客户端的连接。如果需要并行评估，应该启动多个 PolicyServer 实例。

**Q: chunk_size_threshold 参数的作用？**
A: 它控制何时发送新的观察。当队列大小 / chunk_size <= threshold 时，发送新观察。较小的值意味着更频繁地请求新动作。

**Q: 动作聚合函数如何工作？**
A: 当接收到新的动作块时，如果队列中已有相同时间步的动作，聚合函数决定如何合并它们。`weighted_average` 给新动作更高的权重。

## 贡献

如需添加新功能或修复 bug，请遵循以下原则：
- 最小化对 `async_inference` 的改动
- 保持与 RobotClient 的接口一致性
- 添加适当的文档和示例

