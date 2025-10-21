# Async Delta Inference - 支持仿真环境的异步推理框架

## 概述

本项目成功将 `async_inference` 框架扩展到仿真环境（特别是 Libero），最小化地改动了原有代码。新的 `async_delta_inference` 模块完全复用了 `async_inference` 的 PolicyServer，只是添加了一个适配仿真环境的客户端。

## 架构对比

### 原始 async_inference (真实机器人)
```
┌─────────────────┐        gRPC         ┌─────────────────┐
│  RobotClient    │ ◄─────────────────► │  PolicyServer   │
│                 │                      │                 │
│ - Real Robot    │   Observations       │ - Policy Model  │
│ - Cameras       │ ──────────────────►  │ - Preprocessor  │
│ - Motors        │                      │ - Postprocessor │
│                 │ ◄──────────────────  │                 │
│                 │   Action Chunks      │                 │
└─────────────────┘                      └─────────────────┘
```

### 新增 async_delta_inference (仿真环境)
```
┌─────────────────┐        gRPC         ┌─────────────────┐
│   SimClient     │ ◄─────────────────► │  PolicyServer   │
│                 │                      │   (复用原有)     │
│ - Gym Env       │   Observations       │ - Policy Model  │
│ - Libero/       │ ──────────────────►  │ - Preprocessor  │
│   MetaWorld     │                      │ - Postprocessor │
│                 │ ◄──────────────────  │                 │
│                 │   Action Chunks      │                 │
└─────────────────┘                      └─────────────────┘
```

## 文件结构

### 新增文件

```
src/lerobot/async_delta_inference/
├── __init__.py              # 模块初始化
├── configs.py              # SimClientConfig 配置类
├── constants.py            # 常量定义
├── sim_client.py           # 仿真客户端实现
└── README.md              # 详细使用文档

examples/async_delta_inference/
├── evaluate_libero.py      # Libero 评估示例
└── README.md              # 示例说明文档

ASYNC_DELTA_INFERENCE_GUIDE.md  # 本文档
```

### 复用的文件

从 `async_inference` 复用（无需修改）：
- `policy_server.py`: 策略服务器
- `helpers.py`: 辅助函数和数据类型
- `constants.py`: 部分常量

## 核心改动

### 1. SimClient vs RobotClient

| 特性 | RobotClient | SimClient |
|------|-------------|-----------|
| 硬件接口 | 真实机器人 | Gym 环境 |
| 观察获取 | `robot.get_observation()` | `env.step()` / `env.reset()` |
| 动作执行 | `robot.send_action()` | `env.step(action)` |
| 运行模式 | 持续运行 | 固定集数评估 |
| 统计信息 | 无 | 成功率、平均奖励 |

### 2. 观察格式转换

SimClient 包含 `_format_env_observation()` 函数，将 Gym 格式转换为 LeRobot 格式：

```python
# Gym 格式 (Libero)
{
    "pixels": {
        "agentview_image": np.ndarray,
        "robot0_eye_in_hand_image": np.ndarray
    },
    "agent_pos": np.ndarray
}

# 转换为 LeRobot 格式
{
    "observation.images.image": torch.Tensor,
    "observation.images.image2": torch.Tensor,
    "observation.state": torch.Tensor,
    "task": str
}
```

### 3. 配置系统

新增 `SimClientConfig` 类，与 `RobotClientConfig` 类似但适配仿真环境：

```python
@dataclass
class SimClientConfig:
    # 策略配置
    policy_type: str
    pretrained_name_or_path: str
    
    # 环境配置（新增）
    env: EnvConfig
    
    # 控制配置
    actions_per_chunk: int
    chunk_size_threshold: float
    aggregate_fn_name: str
    
    # 评估配置（新增）
    n_episodes: int
    
    # ... 其他配置
```

## 使用方法

### 基础用法

1. **启动 PolicyServer**（在终端 1）：
```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30
```

2. **启动 SimClient**（在终端 2）：
```bash
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --policy_type=act \
    --pretrained_name_or_path=lerobot/act_libero_10 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --n_episodes=10
```

### Python API 用法

```python
from lerobot.async_delta_inference import SimClient, SimClientConfig
from lerobot.envs.configs import LiberoEnv

# 配置环境
env_config = LiberoEnv(
    task="libero_10",
    fps=30,
    obs_type="pixels_agent_pos",
)

# 配置客户端
config = SimClientConfig(
    env=env_config,
    policy_type="act",
    pretrained_name_or_path="lerobot/act_libero_10",
    policy_device="cuda",
    actions_per_chunk=50,
    n_episodes=10,
)

# 创建并运行客户端
client = SimClient(config)
client.start()

# 启动动作接收线程
import threading
action_thread = threading.Thread(target=client.receive_actions, daemon=True)
action_thread.start()

# 运行评估
client.control_loop(task="")

# 清理
client.stop()
```

## 关键特性

### 1. 动作聚合

支持多种动作聚合策略，用于处理重叠的动作块：

- `weighted_average`: 0.3 * 旧动作 + 0.7 * 新动作
- `latest_only`: 只使用最新动作
- `average`: 0.5 * 旧动作 + 0.5 * 新动作
- `conservative`: 0.7 * 旧动作 + 0.3 * 新动作

### 2. 动态观察发送

使用 `chunk_size_threshold` 控制何时发送新观察：
- 当 `queue_size / chunk_size <= threshold` 时发送
- 较小的阈值 = 更频繁的推理请求
- 较大的阈值 = 更多地依赖缓存的动作

### 3. Must-Go 机制

当动作队列为空时，强制发送观察以确保系统不会停滞：
```python
observation.must_go = self.must_go.is_set() and self.action_queue.empty()
```

### 4. 评估统计

自动收集和报告：
- 每集的奖励
- 每集的成功率
- 平均性能指标

## 扩展到其他环境

要支持新的仿真环境（如 MetaWorld），只需：

1. **更新常量**：
```python
# src/lerobot/async_delta_inference/constants.py
SUPPORTED_ENVS = ["libero", "metaworld"]
```

2. **确保环境配置存在**：
```python
# 环境配置应该在 lerobot/envs/configs.py 中定义
```

3. **调整观察格式转换**（如果需要）：
```python
# 在 sim_client.py 的 _format_env_observation() 中
# 添加特定环境的观察格式转换逻辑
```

## 性能考虑

### 延迟优化

- **网络延迟**: 使用 gRPC 的流式传输减少往返次数
- **序列化**: 使用 pickle 快速序列化/反序列化
- **动作缓冲**: 预先计算多个动作减少推理频率

### 控制频率

推荐配置：
- Libero: 30 FPS (与训练数据匹配)
- 高精度任务: 15-20 FPS
- 快速评估: 50-60 FPS

### GPU 内存

- 策略服务器占用: ~2-8 GB (取决于模型)
- 仿真环境占用: ~500 MB - 1 GB
- 推荐: 至少 8 GB GPU 内存用于生产环境

## 故障排查

### 常见问题

1. **连接失败**
   - 检查 PolicyServer 是否运行
   - 验证端口未被占用
   - 确认防火墙设置

2. **环境错误**
   - 安装依赖: `pip install -e ".[libero]"`
   - 检查 CUDA 可用性
   - 验证任务名称正确

3. **推理缓慢**
   - 检查 GPU 利用率
   - 减少 `actions_per_chunk`
   - 增加 `chunk_size_threshold`

4. **动作不稳定**
   - 尝试不同的聚合函数
   - 调整 `chunk_size_threshold`
   - 检查模型是否适合任务

## 与原始框架的兼容性

### 完全兼容
- ✅ PolicyServer 无需任何修改
- ✅ 所有策略类型 (ACT, Diffusion, SmolVLA, 等)
- ✅ 预处理和后处理管道
- ✅ gRPC 通信协议

### 不适用
- ❌ 真实机器人特定的配置 (motors, cameras 等)
- ❌ 机器人安全限制 (仿真环境中不需要)

## 开发指南

### 添加新功能

1. 在 `sim_client.py` 中实现功能
2. 如果需要，在 `configs.py` 中添加配置选项
3. 更新 README.md 文档
4. 添加示例到 `examples/async_delta_inference/`

### 代码风格

- 遵循原始 `async_inference` 的命名约定
- 使用类型提示
- 添加详细的文档字符串
- 保持与 RobotClient 的接口一致性

### 测试

建议测试：
```bash
# 1. 启动服务器
python -m lerobot.async_inference.policy_server --port=8080

# 2. 运行测试（在另一个终端）
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --policy_type=act \
    --pretrained_name_or_path=your/model \
    --n_episodes=1  # 快速测试
```

## 总结

`async_delta_inference` 成功地：

1. ✅ **最小化改动**: 完全复用 PolicyServer，只添加了 SimClient
2. ✅ **支持 Libero**: 可以在 Libero 环境中运行异步推理
3. ✅ **保持一致性**: API 和配置与原始框架一致
4. ✅ **易于扩展**: 简单添加对其他仿真环境的支持
5. ✅ **完整文档**: 包含详细的使用说明和示例

## 下一步

可能的改进方向：

1. **支持更多环境**: MetaWorld, MuJoCo, Isaac Gym
2. **批量评估**: 并行运行多个环境实例
3. **可视化工具**: 实时显示评估进度和性能
4. **性能分析**: 详细的延迟和吞吐量分析
5. **测试套件**: 自动化测试覆盖各种场景

## 参考资料

- [LeRobot 文档](https://huggingface.co/docs/lerobot)
- [Libero 文档](https://lifelong-robot-learning.github.io/LIBERO/)
- [gRPC Python 指南](https://grpc.io/docs/languages/python/)

## 贡献

欢迎贡献！请确保：
- 最小化对现有代码的改动
- 添加适当的文档和示例
- 遵循代码风格指南
- 测试新功能

---

如有问题或建议，请在 GitHub 上提出 issue。

