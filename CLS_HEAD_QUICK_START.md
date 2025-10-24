# CLS Head 快速开始指南

## 概述

本指南快速介绍如何使用 CLS head 机制进行训练和推理。

## 安装

确保已安装 LeRobot:
```bash
pip install -e ".[smolvla]"
```

## 训练

### 基础训练命令

```bash
# 训练 delta_expert (启用 CLS head)
lerobot-train \
  --policy.type=smolvla \
  --policy.train_delta_expert=True \
  --policy.use_cls_head=True \
  --dataset.repo_id=your_dataset \
  --batch_size=64 \
  --steps=200000
```

### Python 训练代码

```python
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

# 创建配置
config = SmolVLAConfig(
    train_delta_expert=True,  # 训练 delta expert
    use_cls_head=True,        # 启用 CLS head
)

# 创建 policy
policy = SmolVLAPolicy(config)
policy.train()

# 训练循环
for batch in dataloader:
    loss, loss_dict = policy.forward(batch)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")
```

## 推理

### 基础推理 (不使用 CLS 判断)

```python
from lerobot.policies.smolvla import SmolVLAPolicy

# 加载模型
policy = SmolVLAPolicy.from_pretrained("your_model")
policy.eval()

# 环境循环
obs = env.reset()
policy.reset()  # 重要!

for step in range(100):
    batch = prepare_batch(obs)
    actions = policy.predict_action_chunk(batch)
    obs, reward, done, _ = env.step(actions[0])
```

### 使用 CLS 相似度判断

```python
from lerobot.policies.smolvla import SmolVLAPolicy

# 加载模型 (确保启用 CLS head)
policy = SmolVLAPolicy.from_pretrained("your_model")
policy.config.use_cls_head = True
policy.eval()

# 环境循环
obs = env.reset()
policy.reset()

for step in range(100):
    batch = prepare_batch(obs)
    
    # 获取动作和 CLS 信息
    result = policy.predict_action_chunk(
        batch,
        compute_cls_similarity=True  # 启用 CLS 判断
    )
    
    if isinstance(result, tuple):
        actions, obs_cls_out, should_cache = result
        
        if should_cache:
            print(f"Step {step}: Scene stable (can cache)")
        else:
            print(f"Step {step}: Scene changed (recompute needed)")
    else:
        actions = result
    
    obs, reward, done, _ = env.step(actions[0])
```

## 在 Policy Server 中使用

```python
class PolicyServer:
    def __init__(self, policy):
        self.policy = policy
        self.cache_queue = []
        
    def predict(self, observation):
        batch = prepare_batch(observation)
        
        # 计算动作并获取缓存判断
        result = self.policy.predict_action_chunk(
            batch,
            compute_cls_similarity=True
        )
        
        if isinstance(result, tuple):
            actions, obs_cls_out, should_cache = result
            
            if should_cache:
                # 场景稳定,入队 (实际应用中可以缓存 KV cache)
                self.cache_queue.append({
                    'obs_cls_out': obs_cls_out.cpu(),
                    'timestamp': time.time(),
                })
                print(f"✓ Cached (queue: {len(self.cache_queue)})")
            else:
                # 场景变化,清空队列
                self.cache_queue.clear()
                print("✗ Scene changed, cleared cache")
        else:
            actions = result
        
        return actions

# 使用
policy = SmolVLAPolicy.from_pretrained("your_model")
policy.config.use_cls_head = True
server = PolicyServer(policy)

while True:
    obs = get_observation()
    actions = server.predict(obs)
    execute_actions(actions)
```

## 配置调优

### 调整相似度阈值

```python
# 默认配置 (平衡)
config = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.5,  # 默认
)

# 严格模式 (场景稍变就重新计算)
config_strict = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.3,
)

# 宽松模式 (更多使用缓存)
config_relaxed = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.7,
)
```

### 调整 VICReg 损失权重

```python
config = SmolVLAConfig(
    train_delta_expert=True,
    use_cls_head=True,
    
    # VICReg 损失权重
    vicreg_weight=0.01,    # 总权重 (推荐 0.001-0.1)
    vicreg_lambda=25.0,    # 不变性损失
    vicreg_mu=25.0,        # 方差损失
    vicreg_nu=1.0,         # 协方差损失
    
    # 噪声尺度
    cls_noise_scale=0.01,  # 推荐 0.001-0.1
)
```

## 监控和调试

### 监控相似度

```python
similarity_history = []

for step in range(100):
    result = policy.predict_action_chunk(
        batch,
        compute_cls_similarity=True
    )
    
    if isinstance(result, tuple):
        actions, obs_cls_out, should_cache = result
        
        # 手动计算相似度
        if policy._prev_obs_cls_out is not None:
            import torch.nn.functional as F
            current = obs_cls_out[:, 0, :]
            prev = policy._prev_obs_cls_out[:, 1, :]
            
            similarity = torch.mean(
                torch.sum(
                    F.normalize(current, p=2, dim=-1) * 
                    F.normalize(prev, p=2, dim=-1),
                    dim=-1
                )
            )
            similarity_history.append(similarity.item())
            print(f"Step {step}: Similarity={similarity:.3f}, Cache={should_cache}")

# 可视化
import matplotlib.pyplot as plt
plt.plot(similarity_history)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
plt.xlabel('Step')
plt.ylabel('CLS Similarity')
plt.legend()
plt.savefig('similarity_history.png')
```

### 统计缓存效率

```python
stats = {'cached': 0, 'not_cached': 0}

for step in range(1000):
    result = policy.predict_action_chunk(
        batch,
        compute_cls_similarity=True
    )
    
    if isinstance(result, tuple):
        _, _, should_cache = result
        if should_cache:
            stats['cached'] += 1
        else:
            stats['not_cached'] += 1

total = sum(stats.values())
print(f"Cache hit rate: {stats['cached']/total:.1%}")
print(f"  Cached: {stats['cached']}")
print(f"  Not cached: {stats['not_cached']}")
```

## 常见问题

### Q: 如何禁用 CLS head?
```python
config = SmolVLAConfig(use_cls_head=False)
```

### Q: 环境 reset 后需要做什么?
```python
env.reset()
policy.reset()  # 必须调用!清空 _prev_obs_cls_out
```

### Q: 如何查看当前是否启用了 CLS head?
```python
print(f"CLS head enabled: {policy.config.use_cls_head}")
```

### Q: 损失不下降怎么办?
尝试调整:
- 降低 `vicreg_weight` (例如从 0.01 到 0.001)
- 降低 `cls_noise_scale` (例如从 0.01 到 0.005)
- 提高学习率

### Q: 缓存命中率太低?
尝试:
- 提高 `cls_similarity_threshold` (例如从 0.5 到 0.6-0.7)
- 检查场景是否确实在频繁变化

### Q: 缓存命中率太高?
尝试:
- 降低 `cls_similarity_threshold` (例如从 0.5 到 0.3-0.4)
- 可能阈值过于宽松

## 完整示例

### 训练脚本

```python
import torch
from torch.utils.data import DataLoader
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

# 配置
config = SmolVLAConfig(
    train_delta_expert=True,
    use_cls_head=True,
    vicreg_weight=0.01,
)

# 模型
policy = SmolVLAPolicy(config)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

# 数据
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练
policy.train()
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 保存
policy.save_pretrained("my_model")
```

### 推理脚本

```python
import torch
from lerobot.policies.smolvla import SmolVLAPolicy

# 加载
policy = SmolVLAPolicy.from_pretrained("my_model")
policy.config.use_cls_head = True
policy.eval()

# 推理
env.reset()
policy.reset()

cache_stats = {'cached': 0, 'not_cached': 0}

for episode in range(10):
    obs = env.reset()
    policy.reset()
    
    for step in range(100):
        batch = prepare_batch(obs)
        
        result = policy.predict_action_chunk(
            batch,
            compute_cls_similarity=True
        )
        
        if isinstance(result, tuple):
            actions, _, should_cache = result
            cache_stats['cached' if should_cache else 'not_cached'] += 1
        else:
            actions = result
        
        obs, _, done, _ = env.step(actions[0])
        if done:
            break
    
    print(f"Episode {episode} complete")

# 统计
total = sum(cache_stats.values())
print(f"\nCache Statistics:")
print(f"  Hit rate: {cache_stats['cached']/total:.1%}")
print(f"  Cached: {cache_stats['cached']}")
print(f"  Not cached: {cache_stats['not_cached']}")
```

## 下一步

- 查看 `CLS_HEAD_USAGE_EXAMPLE.md` 了解更多训练细节
- 查看 `CLS_HEAD_INFERENCE_USAGE.md` 了解更多推理细节
- 查看 `CLS_HEAD_MIGRATION_SUMMARY.md` 了解技术实现

## 支持

如有问题,请参考:
1. 相关文档
2. 原始实现 (`pi05/pi0.py`)
3. 代码实现 (`src/lerobot/policies/smolvla/modeling_smolvla.py`)

