# CLS Head 推理流程使用指南

## 概述

本文档说明如何在推理时使用 CLS head 机制,特别是如何根据场景相似度来决定是否缓存 KV cache。

## 核心逻辑

### 相似度计算
- 使用当前轮的 `obs_cls_out[0]` (第一个 CLS token)
- 与上一轮的 `obs_cls_out[1]` (第二个 CLS token) 计算余弦相似度
- 如果相似度 > 阈值 (默认 0.5): 场景未变化,缓存 KV cache
- 如果相似度 <= 阈值: 场景变化大,不缓存 (强制重新计算)

### 存储机制
- `_prev_obs_cls_out` 存储在 Policy 对象中
- 每次推理后自动更新
- 环境 reset 时自动清空

## 基本使用

### 1. 单次动作预测 (不使用 CLS 判断)

```python
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

# 创建 policy
config = SmolVLAConfig(use_cls_head=True)
policy = SmolVLAPolicy(config)
policy.eval()

# 准备观测数据
batch = {
    "obs.images.camera0": image,
    "obs.state": state,
    "obs.language.tokens": lang_tokens,
    "obs.language.attention_mask": lang_mask,
}

# 预测动作 (标准方式)
actions = policy.predict_action_chunk(batch)
print(f"Actions shape: {actions.shape}")  # [batch, chunk_size, action_dim]
```

### 2. 带 CLS 相似度判断的推理

```python
# 第一次调用 (没有历史 CLS)
result = policy.predict_action_chunk(
    batch,
    compute_cls_similarity=True  # 启用 CLS 相似度计算
)

if isinstance(result, tuple):
    actions, obs_cls_out, should_cache = result
    print(f"Actions: {actions.shape}")
    print(f"obs_cls_out: {obs_cls_out.shape}")  # [batch, num_cls_prefix, hidden_dim]
    print(f"should_cache: {should_cache}")  # True/False
else:
    # CLS head 被禁用,只返回 actions
    actions = result

# 第二次调用 (有历史 CLS,会计算相似度)
result = policy.predict_action_chunk(
    batch,
    compute_cls_similarity=True
)
actions, obs_cls_out, should_cache = result

if should_cache:
    print("场景相似,可以缓存 KV cache")
else:
    print("场景变化大,需要重新计算")
```

### 3. 在循环中使用

```python
# 重置环境
env.reset()
policy.reset()  # 重要: 清空 _prev_obs_cls_out

for step in range(1000):
    # 获取观测
    obs = env.get_obs()
    batch = prepare_batch(obs)
    
    # 预测动作并获取 CLS 信息
    result = policy.predict_action_chunk(
        batch,
        compute_cls_similarity=True
    )
    
    if isinstance(result, tuple):
        actions, obs_cls_out, should_cache = result
        
        # 根据 should_cache 决定是否执行某些操作
        if should_cache:
            # 场景稳定,可以使用快速路径
            pass
        else:
            # 场景变化,可能需要重新规划
            print(f"Step {step}: Scene changed!")
    else:
        actions = result
    
    # 执行动作
    env.step(actions[0])  # 取第一个动作
```

## Policy Server 集成示例

### 在 policy_server 中使用

```python
class PolicyServer:
    def __init__(self, policy):
        self.policy = policy
        self.kv_cache_queue = []  # KV cache 队列
        self.max_queue_size = 10
        
    def predict(self, observation):
        """预测动作并管理 KV cache."""
        batch = self.prepare_batch(observation)
        
        # 计算动作和 CLS 相似度
        result = self.policy.predict_action_chunk(
            batch,
            compute_cls_similarity=True
        )
        
        if isinstance(result, tuple):
            actions, obs_cls_out, should_cache = result
            
            # 根据相似度决定是否入队
            if should_cache:
                # 场景相似,可以缓存 past_key_values
                # 注意: 这里需要在实际使用时获取 past_key_values
                # 可以通过先调用 cal_cache=True 来获取
                cache_item = {
                    'obs_cls_out': obs_cls_out.cpu(),
                    'timestamp': time.time(),
                    # 'past_key_values': past_key_values,  # 需要额外获取
                }
                
                # 入队
                self.kv_cache_queue.append(cache_item)
                
                # 限制队列长度
                if len(self.kv_cache_queue) > self.max_queue_size:
                    self.kv_cache_queue.pop(0)
                
                print(f"✓ Cached KV (queue size: {len(self.kv_cache_queue)})")
            else:
                # 场景变化,清空队列
                self.kv_cache_queue.clear()
                print("✗ Scene changed, cleared cache queue")
        else:
            actions = result
        
        return actions

# 使用示例
server = PolicyServer(policy)

while True:
    obs = get_observation()
    actions = server.predict(obs)
    execute_actions(actions)
```

### 更完整的 Policy Server 示例

```python
class AdvancedPolicyServer:
    def __init__(self, policy):
        self.policy = policy
        self.kv_cache_queue = []
        self.past_key_values = None
        self.prefix_pad_masks = None
        
    def predict_with_cache(self, observation):
        """使用缓存的 KV values 进行预测."""
        batch = self.prepare_batch(observation)
        
        # 第一步: 计算 prefix 并获取 KV cache
        past_key_values, prefix_pad_masks, _ = self.policy.predict_action_chunk(
            batch,
            cal_cache=True,
            action_context=None  # 可以传入 action context
        )
        
        # 第二步: 使用 KV cache 预测动作并计算相似度
        result = self.policy.predict_action_chunk(
            batch,
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            compute_cls_similarity=True
        )
        
        if isinstance(result, tuple):
            actions, obs_cls_out, should_cache = result
            
            if should_cache:
                # 缓存 KV values
                self.past_key_values = past_key_values
                self.prefix_pad_masks = prefix_pad_masks
                
                # 入队
                cache_item = {
                    'past_key_values': past_key_values,
                    'prefix_pad_masks': prefix_pad_masks,
                    'obs_cls_out': obs_cls_out,
                }
                self.kv_cache_queue.append(cache_item)
                
                print(f"✓ Cached (similarity above threshold)")
            else:
                # 清空缓存
                self.past_key_values = None
                self.prefix_pad_masks = None
                self.kv_cache_queue.clear()
                
                print(f"✗ Not cached (similarity below threshold)")
        else:
            actions = result
        
        return actions

# 使用
server = AdvancedPolicyServer(policy)
actions = server.predict_with_cache(obs)
```

## 配置参数

### 相似度阈值调整

```python
# 创建配置时设置
config = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.5,  # 默认值
)

# 不同场景的建议值:
# - 0.3-0.4: 严格模式,场景稍有变化就重新计算
# - 0.5-0.6: 平衡模式 (推荐)
# - 0.7-0.8: 宽松模式,更多使用缓存

# 示例: 严格模式
config_strict = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.3,  # 更严格
)

# 示例: 宽松模式
config_relaxed = SmolVLAConfig(
    use_cls_head=True,
    cls_similarity_threshold=0.7,  # 更宽松
)
```

## 性能考虑

### 1. 计算开销
- 启用 `compute_cls_similarity=True` 会增加轻微开销
- CLS 相似度计算很快 (只是余弦相似度)
- 主要开销在于是否重新计算 prefix embeddings

### 2. 内存使用
- `_prev_obs_cls_out`: 很小 (batch_size × num_cls_prefix × hidden_dim)
- 典型: (1 × 2 × 2048) = 4KB (float32)

### 3. 缓存策略
- 如果 `should_cache=True`: 可以复用 past_key_values,节省计算
- 如果 `should_cache=False`: 必须重新计算,但保证准确性

## 实际应用场景

### 场景 1: 静态环境
```python
# 机器人在固定工作站工作
# 场景很少变化,可以使用高阈值
config = SmolVLAConfig(
    cls_similarity_threshold=0.7  # 更多使用缓存
)
```

### 场景 2: 动态环境
```python
# 机器人在移动或环境经常变化
# 需要更频繁地重新计算
config = SmolVLAConfig(
    cls_similarity_threshold=0.3  # 更少使用缓存
)
```

### 场景 3: 自适应阈值
```python
class AdaptiveThresholdPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.cache_hit_rate = 0.0
        self.window_size = 100
        
    def adjust_threshold(self):
        """根据缓存命中率自适应调整阈值."""
        if self.cache_hit_rate < 0.3:
            # 命中率太低,提高阈值
            self.policy.config.cls_similarity_threshold += 0.05
        elif self.cache_hit_rate > 0.8:
            # 命中率太高,可能过于保守,降低阈值
            self.policy.config.cls_similarity_threshold -= 0.05
        
        # 限制范围
        self.policy.config.cls_similarity_threshold = max(0.2, min(0.8, 
            self.policy.config.cls_similarity_threshold))
```

## 调试和监控

### 1. 记录相似度值

```python
similarity_history = []

result = policy.predict_action_chunk(
    batch,
    compute_cls_similarity=True
)

if isinstance(result, tuple):
    actions, obs_cls_out, should_cache = result
    
    # 手动计算相似度以记录
    if policy._prev_obs_cls_out is not None:
        current_cls = obs_cls_out[:, 0, :]
        prev_cls = policy._prev_obs_cls_out[:, 1, :]
        
        current_norm = F.normalize(current_cls, p=2, dim=-1)
        prev_norm = F.normalize(prev_cls, p=2, dim=-1)
        
        similarity = torch.mean(torch.sum(current_norm * prev_norm, dim=-1))
        similarity_history.append(similarity.item())
        
        print(f"Similarity: {similarity.item():.4f}, Cache: {should_cache}")

# 绘制相似度曲线
import matplotlib.pyplot as plt
plt.plot(similarity_history)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
plt.xlabel('Step')
plt.ylabel('CLS Similarity')
plt.legend()
plt.show()
```

### 2. 统计缓存效果

```python
cache_stats = {
    'total': 0,
    'cached': 0,
    'not_cached': 0,
}

for step in range(1000):
    result = policy.predict_action_chunk(
        batch,
        compute_cls_similarity=True
    )
    
    if isinstance(result, tuple):
        actions, obs_cls_out, should_cache = result
        cache_stats['total'] += 1
        if should_cache:
            cache_stats['cached'] += 1
        else:
            cache_stats['not_cached'] += 1

print(f"Cache hit rate: {cache_stats['cached']/cache_stats['total']:.2%}")
```

## 注意事项

### 1. 环境重置
```python
# ⚠️ 重要: 环境重置时必须调用 policy.reset()
env.reset()
policy.reset()  # 清空 _prev_obs_cls_out
```

### 2. 批处理
```python
# CLS 相似度在批次内取平均
# 如果批次中某些样本场景变化大,某些小
# 最终的 should_cache 是基于平均相似度
# 可能需要单独处理每个样本
```

### 3. 多线程/多进程
```python
# 每个 policy 实例有自己的 _prev_obs_cls_out
# 多线程/多进程环境下需要注意同步
```

## 完整示例: 机器人控制循环

```python
import torch
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

def robot_control_loop():
    # 初始化
    config = SmolVLAConfig(
        use_cls_head=True,
        cls_similarity_threshold=0.5,
    )
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.config.use_cls_head = True
    policy.eval()
    
    # 重置
    env.reset()
    policy.reset()
    
    # 统计
    stats = {'cached': 0, 'not_cached': 0}
    
    # 控制循环
    for episode in range(10):
        obs = env.reset()
        policy.reset()  # 新 episode 需要重置
        
        for step in range(100):
            # 准备批次
            batch = prepare_batch(obs)
            
            # 预测动作
            result = policy.predict_action_chunk(
                batch,
                compute_cls_similarity=True
            )
            
            if isinstance(result, tuple):
                actions, obs_cls_out, should_cache = result
                
                # 统计
                if should_cache:
                    stats['cached'] += 1
                else:
                    stats['not_cached'] += 1
                    print(f"Ep {episode}, Step {step}: Scene changed!")
            else:
                actions = result
            
            # 执行动作
            obs, reward, done, info = env.step(actions[0])
            
            if done:
                break
        
        print(f"Episode {episode} finished")
    
    # 输出统计
    total = stats['cached'] + stats['not_cached']
    print(f"\nCache Statistics:")
    print(f"  Cached: {stats['cached']} ({stats['cached']/total:.1%})")
    print(f"  Not Cached: {stats['not_cached']} ({stats['not_cached']/total:.1%})")

if __name__ == "__main__":
    robot_control_loop()
```

## 参考

- `src/lerobot/policies/smolvla/modeling_smolvla.py` - 实现代码
- `src/lerobot/policies/smolvla/configuration_smolvla.py` - 配置参数
- `pi05/pi0.py` - 原始实现参考 (line 398)

