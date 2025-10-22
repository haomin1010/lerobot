# Action Context 实现总结

## 概述
已成功将 delta_expert 的输入从可学习的 `learnable_noise` 改为从队列中获取的接下来5个actions（action_context）。

## 主要修改

### 1. modeling_smolvla.py - 移除 learnable_noise，添加 action_context_encoder

#### 移除的部分：
```python
# 已删除
self.learnable_noise = nn.Parameter(
    torch.randn(1, self.config.chunk_size, self.config.max_action_dim)
)
```

#### 新增的部分（第576-583行）：
```python
# Action context encoder: 将接下来的N个actions编码为delta_expert的输入
self.action_context_encoder = nn.Sequential(
    nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size),
    nn.ReLU(),
    nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
)
```

**作用**：
- 输入：`(batch_size, num_context_actions, action_dim)` - 例如接下来的5个actions
- 输出：`(batch_size, chunk_size, action_dim)` - 扩展到与chunk_size匹配的形状

### 2. modeling_smolvla.py - 新增 encode_action_context 方法（第1046-1079行）

```python
def encode_action_context(self, action_context: torch.Tensor) -> torch.Tensor:
    """将action context编码并扩展到chunk_size.
    
    策略：
    - 如果 num_context_actions < chunk_size: 重复填充
    - 如果 num_context_actions > chunk_size: 线性插值降采样
    - 如果 num_context_actions == chunk_size: 直接使用
    """
```

### 3. modeling_smolvla.py - 修改 generate_delta_actions（第1081-1123行）

**新增参数**：
```python
action_context: torch.Tensor = None  # (batch_size, num_actions, action_dim)
```

**逻辑**：
```python
if action_context is not None:
    # 编码并使用action_context
    x_t = self.encode_action_context(action_context)
else:
    # 回退方案：使用零张量
    x_t = torch.zeros(batch_size, chunk_size, max_action_dim, device=device)
```

### 4. modeling_smolvla.py - 更新 sample_actions 接口（第858-913行）

**新增参数**：
```python
action_context: torch.Tensor = None  # 从队列传入的action context
```

**调用链**：
```
sample_actions(action_context=...)
  -> generate_delta_actions(action_context=...)
    -> encode_action_context(action_context)
      -> denoise_step_delta(x_t=encoded_context)
```

### 5. modeling_smolvla.py - 更新 Policy 接口

#### SmolVLAPolicy._get_action_chunk（第248-250行）
```python
def _get_action_chunk(..., action_context: Tensor | None = None)
```

#### SmolVLAPolicy.predict_action_chunk（第293-295行）
```python
def predict_action_chunk(..., action_context: Tensor | None = None)
```

### 6. sim_client.py - 添加从队列获取actions的功能（第587-610行）

```python
def _get_next_n_actions_from_queue(self, n: int = 5) -> list[np.ndarray] | None:
    """从队列中获取接下来的N个actions（不移除）"""
    with self.action_queue_lock:
        if self.action_queue.qsize() < n:
            return None
        
        actions = []
        for i, timed_action in enumerate(self.action_queue.queue):
            if i >= n:
                break
            action_tensor = timed_action.get_action()
            action_array = action_tensor.cpu().numpy() if isinstance(action_tensor, torch.Tensor) else action_tensor
            actions.append(action_array)
        
        return actions if len(actions) == n else None
```

### 7. sim_client.py - 在发送observation时附加action_context（第627-633行）

```python
# 获取接下来5个actions作为context
action_context = self._get_next_n_actions_from_queue(n=5)
if action_context is not None:
    # 添加到raw_observation中
    raw_observation["action_context"] = np.array(action_context)  # Shape: (5, action_dim)
```

### 8. policy_server.py - 提取并传递action_context（第338-360行）

```python
# 1. 从raw observation中提取action_context
raw_obs = observation_t.get_observation()
action_context = None
if "action_context" in raw_obs:
    action_context_np = raw_obs["action_context"]
    # 转换为torch tensor: (num_actions, action_dim) -> (1, num_actions, action_dim)
    action_context = torch.from_numpy(action_context_np).unsqueeze(0).float()
    action_context = action_context.to(self.device)

# 2. 传递给policy
past_key_values, prefix_pad_masks, delta_actions = self.policy.predict_action_chunk(
    observation, cal_cache=True, action_context=action_context
)

# 3. 使用delta_actions作为返回的actions
action_tensor = delta_actions
```

## 数据流程

```
sim_client.control_loop_observation
  │
  ├─→ _get_next_n_actions_from_queue(n=5)
  │     └─→ 从action_queue获取接下来5个actions
  │
  ├─→ raw_observation["action_context"] = actions  # (5, action_dim)
  │
  └─→ send_observation(observation)
        │
        ▼
policy_server._precompute_actions
  │
  ├─→ 提取 action_context from raw_observation
  │     └─→ 转换为 torch.Tensor (1, 5, action_dim)
  │
  └─→ policy.predict_action_chunk(obs, cal_cache=True, action_context=action_context)
        │
        ▼
SmolVLAPolicy._get_action_chunk(action_context=...)
  │
  └─→ model.sample_actions(action_context=...)
        │
        ▼
VLAFlowMatching.sample_actions(action_context=...)
  │
  ├─→ 计算 prefix (image + language + state)
  ├─→ 缓存 past_key_values, prefix_pad_masks
  │
  └─→ generate_delta_actions(action_context=...)
        │
        ├─→ encode_action_context(action_context)
        │     └─→ (1, 5, action_dim) -> (1, chunk_size, action_dim)
        │
        └─→ denoise_step_delta(x_t=encoded_context)
              └─→ 使用 delta_expert 生成 delta_actions
```

## 关键特性

### 1. Action Context 编码策略
- **输入长度 < chunk_size**: 重复填充到 chunk_size
- **输入长度 > chunk_size**: 线性插值降采样
- **输入长度 == chunk_size**: 直接使用

### 2. 回退机制
当队列中actions不足5个时：
- `sim_client`: 不添加 `action_context` 字段
- `policy_server`: `action_context = None`
- `VLAFlowMatching`: 使用零张量作为 fallback

### 3. 非阻塞获取
`_get_next_n_actions_from_queue` 只读取队列，不移除actions，不影响正常的action执行流程。

## 训练 delta_expert

### 方式 1：使用ground truth actions作为context
```python
# 在训练时，可以从数据集中获取接下来的actions作为context
action_context = batch["future_actions"]  # (B, 5, action_dim)
loss, loss_dict = policy.forward_delta_expert(batch, action_context=action_context)
```

### 方式 2：训练时不使用context（零输入）
```python
# 让模型学习从零输入生成有意义的delta actions
loss, loss_dict = policy.forward_delta_expert(batch)  # action_context=None, 使用零张量
```

## 使用示例

### 推理（仿真环境）
```bash
# 启动 policy_server
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080

# 启动 sim_client (自动从队列获取5个actions并发送)
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --policy_type=smolvla \
    --pretrained_name_or_path=lerobot/smolvla_base \
    --n_episodes=10
```

### 训练
```python
from lerobot.policies.smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# 训练 delta_expert
for batch in dataloader:
    # 选项1：使用ground truth future actions
    action_context = batch["future_actions"]  # (B, 5, action_dim)
    
    # 选项2：使用零输入（让模型学习无条件生成）
    # action_context = None
    
    loss, loss_dict = policy.forward_delta_expert(
        batch, 
        action_context=action_context  # 可选
    )
    
    loss.backward()
    optimizer.step()
```

## 可配置参数

### action_context数量（默认5个）
在 `sim_client.py` 第628行修改：
```python
action_context = self._get_next_n_actions_from_queue(n=5)  # 修改n值
```

### 时间步（timestep）
在 `modeling_smolvla.py` 第1113行修改：
```python
timestep = torch.tensor(0.5, ...)  # 修改固定时间步
```

### encoder架构
在 `modeling_smolvla.py` 第579-583行修改：
```python
self.action_context_encoder = nn.Sequential(
    # 自定义网络结构
)
```

## 修改的文件清单

1. ✅ `src/lerobot/policies/smolvla/modeling_smolvla.py`
   - 移除 `learnable_noise`
   - 添加 `action_context_encoder`
   - 添加 `encode_action_context` 方法
   - 修改 `generate_delta_actions` 接受 `action_context`
   - 更新所有相关接口传递 `action_context`

2. ✅ `src/lerobot/async_delta_inference/sim_client.py`
   - 添加 `_get_next_n_actions_from_queue` 方法
   - 修改 `control_loop_observation` 附加 `action_context`

3. ✅ `src/lerobot/async_delta_inference/policy_server.py`
   - 修改 `_precompute_actions` 提取和传递 `action_context`
   - 正确接收三个返回值：`(past_key_values, prefix_pad_masks, delta_actions)`

## 测试建议

### 1. 单元测试
```python
def test_encode_action_context():
    # 测试不同长度的action_context编码
    model = VLAFlowMatching(config)
    
    # 测试少于chunk_size
    context = torch.randn(1, 3, action_dim)
    encoded = model.encode_action_context(context)
    assert encoded.shape == (1, chunk_size, action_dim)
    
    # 测试多于chunk_size
    context = torch.randn(1, 100, action_dim)
    encoded = model.encode_action_context(context)
    assert encoded.shape == (1, chunk_size, action_dim)
```

### 2. 集成测试
```bash
# 测试完整流程
python src/lerobot/async_delta_inference/sim_client.py \
    --env.type=libero \
    --env.task=libero_10 \
    --policy_type=smolvla \
    --n_episodes=1
```

## 优势

1. **动态输入**：不再依赖固定的可学习参数，而是使用实时的action context
2. **自适应**：delta_actions可以根据当前队列中的actions进行调整
3. **可解释性**：action_context提供了明确的输入来源
4. **灵活性**：可以很容易地修改context的数量和来源
5. **回退机制**：当没有足够的actions时，使用零张量作为fallback

## 注意事项

1. **队列长度**：确保队列中至少有5个actions，否则不会发送action_context
2. **维度匹配**：action_context的action_dim必须与policy的action_dim匹配（会自动pad到max_action_dim）
3. **设备一致性**：action_context会自动移到policy所在的device
4. **训练一致性**：训练时建议使用与推理相似的action_context分布

