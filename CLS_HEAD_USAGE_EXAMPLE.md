# CLS Head 机制使用示例

## 快速开始

### 1. 训练 Delta Expert (启用 CLS Head)

```bash
# 使用命令行训练
lerobot-train \
  --policy.type=smolvla \
  --policy.train_delta_expert=True \
  --policy.use_cls_head=True \
  --policy.num_cls_prefix=2 \
  --policy.num_cls_suffix=2 \
  --policy.vicreg_weight=0.01 \
  --policy.vicreg_lambda=25.0 \
  --policy.vicreg_mu=25.0 \
  --policy.vicreg_nu=1.0 \
  --policy.cls_noise_scale=0.01 \
  --dataset.repo_id=your_dataset \
  --batch_size=64 \
  --steps=200000
```

### 2. Python 代码示例

```python
import torch
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

# 创建配置
config = SmolVLAConfig(
    # Delta expert 训练设置
    train_delta_expert=True,
    
    # CLS head 设置
    use_cls_head=True,
    num_cls_prefix=2,
    num_cls_suffix=2,
    
    # VICReg 损失权重
    vicreg_weight=0.01,
    vicreg_lambda=25.0,
    vicreg_mu=25.0,
    vicreg_nu=1.0,
    
    # 噪声设置
    cls_noise_scale=0.01,
    
    # 其他设置
    chunk_size=50,
    n_action_steps=50,
    max_state_dim=32,
    max_action_dim=32,
)

# 创建 policy
policy = SmolVLAPolicy(config)
policy.train()

# 准备批次数据
batch = {
    "obs.images.camera0": torch.randn(4, 3, 480, 640),  # 4 个样本
    "obs.state": torch.randn(4, 14),  # 状态
    "obs.language.tokens": torch.randint(0, 1000, (4, 48)),  # 语言 tokens
    "obs.language.attention_mask": torch.ones(4, 48, dtype=torch.bool),
    "action": torch.randn(4, 50, 14),  # ground truth actions
}

# 前向传播 (训练)
loss, loss_dict = policy.forward(batch)

print(f"Total loss: {loss.item()}")
print(f"Loss dict: {loss_dict}")

# 反向传播
loss.backward()

# 检查参数是否可训练
for name, param in policy.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}, shape: {param.shape}")
```

### 3. 禁用 CLS Head (仅使用去噪损失)

```python
config = SmolVLAConfig(
    train_delta_expert=True,
    use_cls_head=False,  # 禁用 CLS head
)

policy = SmolVLAPolicy(config)
```

## 损失组成

### 启用 CLS Head 时
```
总损失 = 去噪损失 + vicreg_weight * VICReg损失
```

其中:
- **去噪损失**: MSE(v_t, x_t - x_t_noisy)
  - v_t: delta_expert 预测的去噪方向
  - x_t: 主 expert 去噪后的动作
  - x_t_noisy: 添加噪声后的动作

- **VICReg 损失**: vicreg_loss(obs_cls_proj, act_cls_proj)
  - obs_cls_proj: 图像/指令 CLS tokens 的投影
  - act_cls_proj: 动作 CLS tokens 的投影
  - 包含三个组件:
    - 不变性损失 (Invariance): 让两个表示相似
    - 方差损失 (Variance): 保持表示的多样性
    - 协方差损失 (Covariance): 减少特征之间的冗余

### 禁用 CLS Head 时
```
总损失 = 去噪损失
```

## 训练流程详解

### Delta Expert 训练 (启用 CLS Head)

1. **Prefix 处理**
   ```
   输入: 图像 + 语言指令 + 状态
   ↓
   添加 2 个 pre_cls_param tokens (在最开始)
   ↓
   通过 VLM 获取 prefix 输出
   ↓
   提取前 2 个 CLS tokens 作为 obs_cls_out
   ```

2. **主 Expert 去噪**
   ```
   初始化: x_t = 零向量 (或噪声)
   ↓
   多步去噪 (使用主 expert)
   ↓
   得到去噪后的 x_t
   ```

3. **Delta Expert 处理**
   ```
   x_t
   ↓
   添加小噪声 → x_t_noisy
   ↓
   添加 2 个 suf_cls_param tokens (在最末尾)
   ↓
   通过 delta_expert 预测 v_t
   ↓
   提取后 2 个 CLS tokens 作为 act_cls_out
   ```

4. **损失计算**
   ```
   去噪损失 = MSE(v_t, x_t - x_t_noisy)
   ↓
   obs_cls_proj = MLP(obs_cls_out)
   act_cls_proj = MLP(act_cls_out)
   ↓
   VICReg损失 = vicreg_loss(obs_cls_proj, act_cls_proj)
   ↓
   总损失 = 去噪损失 + 0.01 * VICReg损失
   ```

## 参数冻结

当 `train_delta_expert=True` 时,以下参数被冻结:
- VLM 的所有参数
- 主 expert 的所有参数
- 图像编码器的所有参数

仅以下参数可训练:
- delta_expert 的所有参数
- delta_action_in_proj
- delta_action_out_proj
- delta_action_time_mlp_in
- delta_action_time_mlp_out
- action_context_encoder
- pre_cls_param (如果启用 CLS head)
- suf_cls_param (如果启用 CLS head)
- obs_cls_proj (如果启用 CLS head)
- act_cls_proj (如果启用 CLS head)

## 超参数调优建议

### VICReg 损失权重
- `vicreg_weight`: 0.001 - 0.1 (推荐从 0.01 开始)
  - 太大: VICReg 损失主导,可能影响去噪性能
  - 太小: CLS tokens 学不到有用的表示

### VICReg 内部权重
- `vicreg_lambda`: 25.0 (不变性损失)
- `vicreg_mu`: 25.0 (方差损失)
- `vicreg_nu`: 1.0 (协方差损失)
- 这些权重来自原始 VICReg 论文,一般不需要修改

### 噪声尺度
- `cls_noise_scale`: 0.001 - 0.1 (推荐从 0.01 开始)
  - 太大: 去噪任务太难
  - 太小: 去噪任务太简单,模型学不到有用信息

### CLS Token 数量
- `num_cls_prefix`: 1 - 4 (推荐 2)
- `num_cls_suffix`: 1 - 4 (推荐 2)
  - 更多 tokens: 可能学到更丰富的表示,但计算成本更高
  - 更少 tokens: 计算效率高,但表示能力有限

## 监控指标

训练时应监控以下指标:
1. `delta_loss`: 总损失
2. `delta_losses_after_forward`: 原始损失 (包含 padding)
3. `delta_losses_after_in_ep_bound`: 应用 episode 边界后
4. `delta_losses_after_rm_padding`: 移除 padding 后

如果启用 CLS head,还应监控:
- VICReg 不变性损失
- VICReg 方差损失
- VICReg 协方差损失
- obs_cls 和 act_cls 的余弦相似度

## 常见问题

### Q1: 损失不下降?
**可能原因**:
- vicreg_weight 设置太大
- cls_noise_scale 设置太大
- 学习率太小

**解决方案**:
- 降低 vicreg_weight (例如从 0.01 降到 0.001)
- 降低 cls_noise_scale (例如从 0.01 降到 0.005)
- 提高学习率

### Q2: VICReg 损失为 NaN?
**可能原因**:
- 批次大小太小 (VICReg 需要足够的样本计算协方差)
- 梯度爆炸

**解决方案**:
- 增大批次大小 (至少 32)
- 启用梯度裁剪
- 降低学习率

### Q3: 如何验证 CLS tokens 是否学到有用的表示?
**验证方法**:
1. 可视化 obs_cls 和 act_cls 的余弦相似度
2. 在测试集上计算 VICReg 损失
3. 使用 t-SNE 可视化 CLS token 的分布
4. 检查不同任务/场景下 CLS token 的区分度

## 进阶使用

### 自定义 VICReg 损失
```python
from lerobot.policies.smolvla.modeling_smolvla import vicreg_loss

# 使用自定义权重
loss = vicreg_loss(
    z1=obs_cls_proj,
    z2=act_cls_proj,
    lambda_param=50.0,  # 提高不变性损失权重
    mu_param=10.0,      # 降低方差损失权重
    nu_param=2.0,       # 提高协方差损失权重
    gamma=0.5,          # 降低目标标准差
)
```

### 自定义注意力模式
```python
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks_with_cls

# 使用自定义注意力掩码
att_2d_masks = make_att_2d_masks_with_cls(
    pad_masks=pad_masks,
    att_masks=att_masks,
    num_cls_prefix=2,
    num_cls_suffix=2,
    num_action_context=5,  # suffix CLS 只看前后 5 个动作 tokens
)
```

## 参考文献

1. VICReg: Bardes, A., Ponce, J., & LeCun, Y. (2021). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
2. SmolVLA: 原始 SmolVLA 论文
3. Pi0/Pi0.5: 原始 Pi0 实现 (`pi05/pi0.py` 和 `pi05/gemma.py`)

