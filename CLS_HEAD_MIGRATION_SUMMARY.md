# CLS Head 机制迁移总结

## 概述
本文档总结了将 pi05 中的 CLS head 机制迁移到 SmolVLA policy 的实现。

## 迁移的核心功能

### 1. VICReg 损失函数
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: 第 72-126 行
- **功能**: 实现了 Variance-Invariance-Covariance Regularization 损失,用于对比学习

```python
def vicreg_loss(z1, z2, lambda_param=25.0, mu_param=25.0, nu_param=1.0, gamma=1.0, eps=1e-4):
    """VICReg loss with Variance-Invariance-Covariance Regularization."""
    # 计算不变性损失、方差损失和协方差损失
```

### 2. 可学习的 CLS Token 参数
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: `VLAFlowMatching.__init__` 方法中
- **新增参数**:
  - `self.pre_cls_param`: 在图像/指令网络开头添加的 2 个可学习 token (形状: [1, 2, vlm_hidden_size])
  - `self.suf_cls_param`: 在动作网络末尾添加的 2 个可学习 token (形状: [1, 2, expert_hidden_size])

### 3. MLP 投影层
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: `VLAFlowMatching.__init__` 方法中
- **新增模块**:
  - `self.obs_cls_proj`: 将 prefix CLS tokens 投影到统一维度 (MLP: vlm_hidden → vlm_hidden → expert_hidden)
  - `self.act_cls_proj`: 将 suffix CLS tokens 投影到统一维度 (MLP: expert_hidden → vlm_hidden → expert_hidden)

### 4. embed_prefix 方法修改
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: 第 836-866 行
- **修改内容**:
  - 在所有图像/语言/状态 embeddings 之前添加 `pre_cls_param` tokens
  - 更新 padding masks 和 attention masks 以包含 CLS tokens
  - CLS tokens 的 attention mask 设置为 0 (可以看到所有图像/指令 tokens)

### 5. embed_suffix_delta 方法修改
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: 第 1177-1233 行
- **修改内容**:
  - 在所有动作 tokens 之后添加 `suf_cls_param` tokens
  - 更新 padding masks 和 attention masks 以包含 CLS tokens
  - Suffix CLS tokens 可以看到动作 tokens (注意力模式可进一步定制)

### 6. forward_delta_expert 方法增强
- **文件**: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- **位置**: 第 1028-1158 行
- **修改内容**:
  1. 通过 VLM 前向传播获取 prefix 输出 (包含 CLS tokens)
  2. 提取前 2 个 prefix CLS tokens
  3. 使用主 expert 进行去噪得到 x_t
  4. 对 x_t 添加噪声得到 x_t_noisy
  5. 通过 delta_expert 前向传播处理 x_t_noisy
  6. 提取后 2 个 suffix CLS tokens
  7. 通过 MLP 投影 CLS tokens
  8. 计算 VICReg 损失
  9. 结合去噪损失和 VICReg 损失 (权重为 0.01)

## 训练流程

### Delta Expert 训练模式
当 `config.train_delta_expert=True` 时:

1. **输入**: 图像、语言指令、状态、ground truth actions
2. **Prefix 处理**:
   - 添加 pre_cls_param tokens
   - 通过 VLM 获取 prefix 输出和 KV cache
   - 提取 prefix CLS tokens
3. **主 Expert 去噪**:
   - 使用主 expert 进行完整的去噪流程得到 x_t
4. **Delta Expert 处理**:
   - 对 x_t 添加小噪声得到 x_t_noisy
   - 添加 suf_cls_param tokens
   - 通过 delta_expert 预测去噪方向 v_t
   - 提取 suffix CLS tokens
5. **损失计算**:
   - 去噪损失: MSE(v_t, x_t - x_t_noisy)
   - VICReg 损失: vicreg_loss(obs_cls_proj, act_cls_proj)
   - 总损失: denoising_loss + 0.01 * vicreg_loss

## 注意力模式

### Prefix CLS Tokens
- **位置**: 序列的最开始 (索引 0-1)
- **注意力**: 可以看到所有图像、语言、状态 tokens
- **目的**: 编码观察信息的全局表示

### Suffix CLS Tokens
- **位置**: 序列的最末尾 (最后 2 个 tokens)
- **注意力**: 当前实现可以看到所有动作 tokens
- **可扩展**: 可以通过 `make_att_2d_masks_with_cls` 函数进一步定制注意力模式
  - 例如: 只看前 n 个和后 n 个动作 tokens

## 配置参数

### CLS Head 设置 (在 SmolVLAConfig 中)
- `use_cls_head`: bool = True (启用 CLS head 机制)
- `num_cls_prefix`: int = 2 (prefix 开头的 CLS token 数量)
- `num_cls_suffix`: int = 2 (suffix 末尾的 CLS token 数量)
- `vicreg_weight`: float = 0.01 (VICReg 损失在总损失中的权重)
- `vicreg_lambda`: float = 25.0 (VICReg 不变性损失权重)
- `vicreg_mu`: float = 25.0 (VICReg 方差损失权重)
- `vicreg_nu`: float = 1.0 (VICReg 协方差损失权重)
- `cls_noise_scale`: float = 0.01 (添加到 x_t 的噪声尺度)

### 使用示例
```python
from lerobot.policies.smolvla import SmolVLAConfig

# 启用 CLS head
config = SmolVLAConfig(
    train_delta_expert=True,
    use_cls_head=True,
    num_cls_prefix=2,
    num_cls_suffix=2,
    vicreg_weight=0.01,
    vicreg_lambda=25.0,
    vicreg_mu=25.0,
    vicreg_nu=1.0,
    cls_noise_scale=0.01,
)

# 禁用 CLS head (只使用去噪损失)
config = SmolVLAConfig(
    train_delta_expert=True,
    use_cls_head=False,
)
```

## 推理流程 (待实现)

推理时的代码逻辑需要根据具体需求实现,可能包括:
1. 使用 obs_cls_head 判断是否需要重新采样动作
2. 类似 pi0.py 中的 `should_sample` 逻辑

## 文件修改清单

1. **src/lerobot/policies/smolvla/modeling_smolvla.py**
   - 新增: `vicreg_loss` 函数
   - 新增: `make_att_2d_masks_with_cls` 函数 (框架)
   - 修改: `VLAFlowMatching.__init__` - 添加 CLS 参数和投影层
   - 修改: `VLAFlowMatching.embed_prefix` - 添加 prefix CLS tokens
   - 修改: `VLAFlowMatching.embed_suffix_delta` - 添加 suffix CLS tokens
   - 修改: `VLAFlowMatching.forward_delta_expert` - 添加 VICReg 损失

## 使用方法

### 训练 Delta Expert
```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.train_delta_expert=True \
  --dataset.repo_id=your_dataset \
  --batch_size=64 \
  --steps=200000
```

### 代码示例
```python
# 创建 policy
config = SmolVLAConfig(train_delta_expert=True)
policy = SmolVLAPolicy(config)

# 训练时会自动使用 forward_delta_expert
loss, loss_dict = policy.forward(batch)

# loss_dict 包含:
# - 'delta_loss': 总损失
# - 'delta_losses_after_forward': 原始损失
# - 'delta_losses_after_in_ep_bound': 应用 episode 边界后
# - 'delta_losses_after_rm_padding': 移除 padding 后
```

## 未来改进方向

1. **注意力模式优化**: 
   - 实现 `make_att_2d_masks_with_cls` 的完整逻辑
   - 让 suffix CLS tokens 只关注前 n 个和后 n 个动作 tokens

2. **推理逻辑**: 
   - 实现基于 obs_cls_head 的重采样判断
   - 添加 CLS head 的缓存机制

3. **超参数调优**:
   - VICReg 损失权重
   - 噪声尺度
   - CLS tokens 的初始化策略

4. **可视化和分析**:
   - CLS tokens 的表示分布
   - VICReg 损失的各个组件
   - Suffix CLS tokens 的注意力热图

## 参考

- 原始实现: `pi05/pi0.py` 和 `pi05/gemma.py`
- VICReg 论文: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"

