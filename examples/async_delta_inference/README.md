# Async Delta Inference 示例

这个目录包含使用异步推理框架在仿真环境中评估策略的示例。

## 文件说明

- `evaluate_libero.py`: 在 Libero 环境中评估策略的完整示例

## 快速开始

### 1. 安装依赖

确保已经安装了 Libero 环境：

```bash
pip install -e ".[libero]"
```

### 2. 启动 PolicyServer

在一个终端中：

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2
```

### 3. 运行评估

#### 选项 A: 使用示例脚本

在另一个终端中：

```bash
python examples/async_delta_inference/evaluate_libero.py
```

注意：你需要修改脚本中的 `pretrained_name_or_path` 为你的模型路径。

#### 选项 B: 直接使用 sim_client

```bash
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --policy_type=act \
    --pretrained_name_or_path=lerobot/act_libero_10 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --n_episodes=10 \
    --fps=30
```

## 配置选项

### 环境配置

通过 `--env.*` 参数配置环境：

```bash
--env.type=libero \
--env.task=libero_10 \
--env.obs_type=pixels_agent_pos \
--env.observation_height=360 \
--env.observation_width=360
```

支持的任务套件：
- `libero_10`: 10个长期任务
- `libero_90`: 90个长期任务
- `libero_spatial`: 空间推理任务
- `libero_object`: 物体操作任务
- `libero_goal`: 目标条件任务

### 策略配置

```bash
--policy_type=act \
--pretrained_name_or_path=path/to/checkpoint \
--policy_device=cuda \
--actions_per_chunk=50
```

支持的策略类型：
- `act`: Action Chunking Transformer
- `diffusion`: Diffusion Policy
- `smolvla`: SmolVLA (视觉语言动作模型)
- `pi0`: π0 策略
- `pi05`: π0.5 策略
- `tdmpc`: TD-MPC
- `vqbet`: VQ-BeT

### 控制配置

```bash
--chunk_size_threshold=0.5 \
--aggregate_fn_name=weighted_average \
--fps=30
```

动作聚合函数：
- `weighted_average`: 加权平均 (0.3*旧 + 0.7*新)
- `latest_only`: 只使用最新动作
- `average`: 平均 (0.5*旧 + 0.5*新)
- `conservative`: 保守策略 (0.7*旧 + 0.3*新)

## 预期输出

```
[sim_client] Creating simulation environment: libero
[sim_client] Simulation environment ready
[sim_client] Connected to policy server in 0.0023s
[sim_client] Action receiving thread starting
[sim_client] Control loop thread starting
[sim_client] Episode 1/10 finished | Steps: 127 | Reward: 1.0000 | Success: True
[sim_client] Episode 2/10 finished | Steps: 134 | Reward: 1.0000 | Success: True
...
============================================================
Evaluation complete!
Episodes: 10
Average Reward: 0.9500
Success Rate: 95.00%
============================================================
```

## 故障排查

### 连接失败

如果看到 "Failed to connect to policy server"：
1. 确保 PolicyServer 正在运行
2. 检查端口 8080 是否被占用
3. 确认防火墙允许本地连接

### 环境错误

如果看到环境相关错误：
1. 确保安装了 Libero: `pip install -e ".[libero]"`
2. 检查任务名称是否正确
3. 确认 GPU 可用（如果使用 `policy_device=cuda`）

### 推理错误

如果推理失败：
1. 检查模型路径是否正确
2. 确认策略类型与模型匹配
3. 检查 GPU 内存是否足够

## 高级用法

### 调试模式

启用详细日志和队列可视化：

```bash
python src/lerobot/async_delta_inference/sim_client.py \
    ... \
    --debug_visualize_queue_size=True
```

### 多集评估

评估更多集数以获得更可靠的统计：

```bash
--n_episodes=50
```

### 自定义控制频率

调整 FPS 以匹配训练时的频率：

```bash
--fps=15
```

## 扩展示例

要添加新的示例（如 MetaWorld），创建一个类似的脚本并修改环境配置：

```python
from lerobot.envs.configs import MetaWorldEnvConfig

env_config = MetaWorldEnvConfig(
    task="reach-v2",
    # ... 其他配置
)
```

