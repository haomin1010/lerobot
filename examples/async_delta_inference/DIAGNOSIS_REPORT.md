# Async Delta Inference 成功率低问题诊断报告

## 问题描述
- **lerobot-eval**: 成功率特别高
- **async (quick_start.sh)**: 成功率特别低
- 两种方法使用相同的策略和参数

## 根本原因：图像分辨率不匹配

### 1. 问题发现
诊断工具检测到两种方法传给策略的图像分辨率不同：

```
lerobot-eval: torch.Size([1, 3, 256, 256])
async:        torch.Size([1, 3, 360, 360])
```

### 2. 图像处理流程分析

#### 完整流程：
```
环境输出 → Preprocessor → Policy内部处理 → 推理
```

#### LiberoEnv 环境配置

**LiberoEnv 类默认参数** (`src/lerobot/envs/libero.py:110-111`):
```python
observation_width: int = 256,   # 默认 256
observation_height: int = 256,
```

**LiberoEnv 配置类** (`src/lerobot/envs/configs.py:239-240`):
```python
observation_height: int = 360,  # 配置默认 360
observation_width: int = 360,
```

**gym_kwargs 传递** (`src/lerobot/envs/configs.py:275-280`):
```python
@property
def gym_kwargs(self) -> dict:
    return {
        "obs_type": self.obs_type,
        "render_mode": self.render_mode,
        "task_ids": [0],
    }
    # ⚠️ 注意：没有传递 observation_width 和 observation_height！
```

#### SmolVLA 策略内部处理

**配置** (`src/lerobot/policies/smolvla/configuration_smolvla.py:47`):
```python
resize_imgs_with_padding: tuple[int, int] = (512, 512)
```

**实际处理** (`src/lerobot/policies/smolvla/modeling_smolvla.py:360-361`):
```python
if self.config.resize_imgs_with_padding is not None:
    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
```

### 3. 两种方法的完整流程对比

#### lerobot-eval 流程：
```
1. make_env(LiberoEnv配置)
   └→ gym_kwargs 不含分辨率参数
   └→ LiberoEnv(gym.Env) 使用类默认值 256x256

2. 环境输出: 256x256 ✅

3. preprocess_observation()
   └→ 只转换格式，不resize
   └→ torch.Size([1, 3, 256, 256])

4. Preprocessor Pipeline
   └→ 保持 256x256（无额外resize）

5. Policy内部 prepare_images()
   └→ resize_with_pad() 到 512x512
   └→ 归一化到 [-1, 1]

6. 推理 ✅ (正确的输入分布)
```

#### async 方法流程：
```
1. make_env(LiberoEnv配置)
   └→ gym_kwargs 不含分辨率参数
   └→ LiberoEnv(gym.Env) 使用类默认值 256x256

2. 环境实际输出: 256x256

3. _format_env_observation()
   └→ 保持 256x256

4. raw_observation_to_observation()
   └→ prepare_raw_observation()
   └→ resize_robot_observation_image()
      └→ ⚠️ 关键问题！使用 policy_image_features.shape
      └→ policy_image_features 来自 env_config.features
      └→ env_config.features 使用配置值 (360, 360, 3)
      └→ 将 256x256 resize 到 360x360 ❌

5. Preprocessor Pipeline
   └→ 保持 360x360

6. Policy内部 prepare_images()
   └→ resize_with_pad() 从 360x360 到 512x512
   └→ 归一化到 [-1, 1]

7. 推理 ❌ (错误的输入分布 - 多了一次 resize)
```

### 4. 问题的影响

**为什么分辨率不同会导致成功率低？**

1. **特征分布差异**
   - 256x256 → 512x512: 2倍放大，使用双线性插值
   - 360x360 → 512x512: 1.42倍放大，插值方式相同
   - 但**起始分辨率不同导致最终特征图内容不同**

2. **Padding 位置不同**
   - 256x256: 较多 padding
   - 360x360: 较少 padding
   - Padding 影响视觉编码器的注意力分布

3. **训练与推理分布不匹配**
   - SmolVLA 在特定分辨率上训练
   - 如果训练时用 256x256，推理时用 360x360 会导致性能下降

### 5. 需要定位的关键问题

**为什么 async 方法实际输出是 360x360？**

需要检查：

1. **SimClient 环境创建**
   ```python
   # src/lerobot/async_delta_inference/sim_client.py:192
   env_dict = make_env(config.env, n_envs=1, use_async_envs=False)
   ```
   检查 `config.env` 的实际配置值

2. **可能的覆盖点**
   - SimClientConfig 初始化时可能覆盖了 observation_height/width
   - 环境配置传递过程中可能有参数注入

3. **policy_image_features 的来源**
   ```python
   # 在 _env_obs_to_lerobot_features 中
   # features["pixels/image"] = PolicyFeature(shape=(360, 360, 3))
   ```
   这里使用了 `env_config.observation_height/width`

### 6. 解决方案方向（不修改）

有两个可能的解决方向：

**方案 A: 统一为 256x256** ✅ 推荐
- 修改 `LiberoEnv` 配置的默认值从 360 改为 256
- 或在 `gym_kwargs` 中显式传递 256
- 优点：与 lerobot-eval 完全一致
- 缺点：需要修改配置

**方案 B: 统一为 360x360**
- 修改 lerobot-eval 的环境配置
- 缺点：可能影响其他使用场景
- 不推荐

### 7. 验证步骤

修改后需要验证：

1. 环境实际输出分辨率
   ```python
   obs = env.reset()
   print(obs['pixels']['image'].shape)  # 应该是 (256, 256, 3)
   ```

2. Preprocessor 输入
   ```python
   # 在 prepare_raw_observation 前打印
   print(raw_obs['image'].shape)  # 应该是 torch.Size([256, 256, 3])
   ```

3. Policy 输入
   ```python
   # 在 prepare_images 前打印
   print(batch['observation.images.image'].shape)  # 应该是 [B, 3, 256, 256]
   ```

4. Policy 内部处理后
   ```python
   # 在 resize_with_pad 后
   print(img.shape)  # 应该是 [B, 3, 512, 512]
   ```

### 8. 其他可能的成功率低的原因

虽然图像分辨率是主要问题，但也需要检查：

1. **动作队列管理差异**
   - lerobot-eval: 直接执行每个动作
   - async: 使用动作队列和聚合
   - 可能影响动作的时序和一致性

2. **策略状态管理**
   - lerobot-eval: 每个 episode 调用 policy.reset()
   - async: 通过 reset_policy 标志
   - 确认状态正确重置

3. **观测值时序**
   - lerobot-eval: 同步获取观测和执行动作
   - async: 异步处理可能导致观测-动作不匹配

4. **FPS 和时序控制**
   - 两种方法的实际 FPS 是否一致
   - 动作执行的时机是否对齐

## 结论

**主要问题：图像分辨率不匹配**
- lerobot-eval: 256x256
- async: 360x360

这会导致策略看到的特征分布不同，严重影响推理准确性。

**建议优先解决分辨率问题**，然后再检查其他可能的差异。

---
生成时间: 2025-10-21

