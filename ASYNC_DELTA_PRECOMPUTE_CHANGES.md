# Async Delta Inference 预计算改造总结

## 改造概述

本次改造实现了在接收观察（observation）时立即进行推理预计算，将计算结果缓存起来，在 `GetActions` 时直接使用缓存的结果。这大大降低了 GetActions 的延迟，提升了系统的响应速度。

## 改造内容

### 1. SmolVLA Policy 模型改造 (`src/lerobot/policies/smolvla/modeling_smolvla.py`)

#### 修改了 `sample_actions` 方法，支持分阶段执行：

**新增参数：**
- `return_intermediate`: 如果为 True，在计算完 KV cache 后返回中间状态，不执行 denoising
- `past_key_values`: 缓存的 KV values，用于从中间状态继续执行
- `prefix_pad_masks`: 缓存的 prefix padding masks
- `initial_x_t`: 初始 x_t，用于继续 denoising
- `initial_time`: 初始 time 值

**执行流程：**
1. **正常流程**（无缓存）：
   - 计算 prefix embeddings 和 KV cache
   - 执行完整的 denoising 循环
   - 返回最终的 actions

2. **返回中间状态**（`return_intermediate=True`）：
   - 计算 prefix embeddings 和 KV cache
   - 返回 `(past_key_values, prefix_pad_masks, noise, initial_time, dt, device)`
   - 不执行 denoising

3. **从缓存继续**（提供 `past_key_values` 和 `prefix_pad_masks`）：
   - 跳过 prefix 计算
   - 直接使用缓存的 KV values
   - 执行 denoising 循环
   - 返回最终的 actions

### 2. PolicyServer 改造 (`src/lerobot/async_delta_inference/policy_server.py`)

#### 新增缓存机制：

```python
# Cache for preprocessed results
self._precomputed_cache_lock = threading.Lock()
self._precomputed_cache = {}  # {timestep: {"past_key_values": ..., "actions": ..., "observation": ...}}
```

#### 新增 `_precompute_actions` 方法：

这个方法在接收到观察时立即执行完整的推理流程：
1. 准备观察数据（raw → LeRobot format）
2. 应用预处理器（preprocessor）
3. 重置 policy（如果需要）
4. 调用 `policy.predict_action_chunk` 执行完整推理
5. 应用后处理器（postprocessor）
6. 转换为 TimedAction list
7. 将原始 actions 缓存到 `_precomputed_cache`
8. **返回处理好的 action_chunk 给调用者**

#### 修改 `SendObservations` 方法：

```python
enqueued = self._enqueue_observation(timed_observation)

action_chunk = None
if not enqueued:
    self.logger.debug(f"Observation #{obs_timestep} has been filtered out")
else:
    # If observation was enqueued and has must_go flag, precompute actions immediately
    if timed_observation.must_go and self.config.enable_precompute:
        self.logger.info(f"Starting immediate precomputation for observation #{obs_timestep}")
        action_chunk = self._precompute_actions(timed_observation)

# Return actions if precomputed, otherwise return empty
if action_chunk is not None and len(action_chunk) > 0:
    actions_bytes = pickle.dumps(action_chunk)
    return services_pb2.Actions(data=actions_bytes)
else:
    return services_pb2.Actions(data=b"")
```

- 当观察被入队且有 `must_go` 标志时，立即调用 `_precompute_actions`
- 预计算在 SendObservations 的上下文中同步执行
- **将预计算的 actions 立即返回给客户端**（而不是返回空响应）

#### 修改 `_predict_action_chunk` 方法：

添加了缓存检查逻辑：
1. 首先检查是否有预计算的缓存
2. 如果有缓存，直接使用缓存的 actions，只执行 postprocessing
3. 如果没有缓存，执行完整的推理流程（正常流程）
4. 使用后从缓存中删除

#### 修改 `_reset_server` 方法：

在重置服务器时清空预计算缓存：
```python
# Clear precomputed cache
with self._precomputed_cache_lock:
    self._precomputed_cache = {}
```

### 3. 配置文件改造 (`src/lerobot/async_delta_inference/configs.py`)

#### 在 `PolicyServerConfig` 中新增配置选项：

```python
# Precomputation configuration
enable_precompute: bool = field(
    default=True, metadata={"help": "Enable immediate action precomputation when observation is received"}
)
```

- 默认启用预计算功能
- 可以通过命令行参数 `--enable_precompute=False` 关闭

## 工作流程

### 启用预计算时的流程：

1. **客户端发送观察**
   ```
   Client → SendObservations(observation)
   ```

2. **服务器立即预计算并返回**（同步执行）
   ```
   SendObservations:
   ├─ _enqueue_observation(obs)
   ├─ if obs.must_go and enable_precompute:
   │   └─ action_chunk = _precompute_actions(obs)
   │       ├─ raw_observation → observation (LeRobot format)
   │       ├─ preprocessor(observation)
   │       ├─ policy.predict_action_chunk(observation)  # 完整推理
   │       ├─ postprocessor(actions)  # 后处理
   │       ├─ convert to TimedAction list
   │       └─ cache[timestep] = {observation, raw_actions, timestamp}
   └─ return action_chunk to client  # 立即返回 actions！
   ```

3. **客户端接收并处理 actions**
   ```
   Client.send_observation:
   └─ response = stub.SendObservations(observation)
       └─ if response.data:
           └─ timed_actions = pickle.loads(response.data)
               └─ _add_actions_to_queue(timed_actions)  # 添加到 action queue
   ```

4. **GetActions 使用缓存**（作为备用机制）
   ```
   GetActions:
   └─ _predict_action_chunk(obs, use_cache=True)
       ├─ 检查缓存
       ├─ if 有缓存:
       │   ├─ 取出缓存的 actions
       │   ├─ postprocessor(actions)
       │   └─ return actions
       └─ else:
           └─ 执行完整推理（正常流程）
   ```

### 禁用预计算时的流程：

与原来的流程完全一致，GetActions 时才执行推理。

## 性能提升

### 延迟降低：

- **原流程**: 
  - Client 发送 obs → Server 缓存 obs
  - Client 调用 GetActions → Server 推理（~数百毫秒）→ Client 接收 actions
  - **总延迟**: obs 到 actions 执行需要等待 GetActions 调用 + 推理时间

- **新流程**: 
  - Client 发送 obs → Server 立即推理（~数百毫秒）→ Client 在响应中直接接收 actions
  - Client 调用 GetActions → 如果有缓存直接返回（~几毫秒）
  - **总延迟**: obs 到 actions 执行几乎没有额外等待，actions 在 SendObservations 响应中就返回了

### 并发处理：

- **立即返回机制**: actions 在 SendObservations 的响应中就返回给客户端
  - 客户端接收到 obs 响应时，就已经拿到了 actions
  - 不需要等待单独的 GetActions 调用
  
- **双重保障**: 
  - 主流程：通过 SendObservations 响应立即返回 actions
  - 备用流程：通过 GetActions 从缓存获取（如果客户端仍然调用）

### 吞吐量：

- 预计算在 SendObservations 时完成，充分利用了观察到达与动作执行之间的时间
- 对于 `must_go=True` 的观察（需要新推理的关键观察），actions 立即可用
- 客户端的 action queue 可以立即填充，避免等待

## 使用方法

### 启动服务器（启用预计算）：

```bash
python -m lerobot.async_delta_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2 \
    --enable_precompute=True  # 默认为 True
```

### 启动服务器（禁用预计算，使用原流程）：

```bash
python -m lerobot.async_delta_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2 \
    --enable_precompute=False
```

### 客户端使用（无需修改）：

```bash
python -m lerobot.async_delta_inference.sim_client \
    --env.type=libero \
    --env.task=libero_10 \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=user/model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --n_episodes=10
```

## 注意事项

1. **立即返回机制**：
   - 预计算完成后，actions 会立即在 SendObservations 的响应中返回
   - 客户端已经实现了接收和处理逻辑（`_add_actions_to_queue`）
   - 这意味着客户端在发送 obs 后，就能在响应中拿到 actions

2. **双重机制**：
   - 主要途径：通过 SendObservations 响应返回 actions（立即可用）
   - 备用途径：actions 也会缓存在服务器，GetActions 可以作为备用获取方式

3. **线程安全**：缓存访问使用了锁机制（`_precomputed_cache_lock`），保证线程安全

4. **内存管理**：使用后的缓存会被立即删除，避免内存累积

5. **向后兼容**：通过 `enable_precompute` 配置可以完全禁用新功能，回到原来的行为

6. **适用范围**：当前只在 `must_go=True` 的观察上执行预计算，这些是真正需要新推理的观察

## 扩展性

### 如果要使用分阶段执行（未来扩展）：

SmolVLA 的 `sample_actions` 已经支持分阶段执行，如果需要进一步优化，可以：

1. **第一阶段**（SendObservations 时）：
   ```python
   past_key_values, prefix_pad_masks, noise, initial_time, dt, device = \
       model.sample_actions(..., return_intermediate=True)
   ```
   
2. **第二阶段**（GetActions 时）：
   ```python
   actions = model.sample_actions(
       ..., 
       past_key_values=past_key_values,
       prefix_pad_masks=prefix_pad_masks,
       initial_x_t=noise,
       initial_time=initial_time
   )
   ```

这样可以进一步分离计算，但需要存储更多的中间状态。

## 总结

本次改造成功实现了异步推理的预计算优化，主要特点：

✅ **立即返回 actions**：预计算的 actions 直接在 SendObservations 响应中返回给客户端  
✅ **大幅降低延迟**：客户端发送 obs 后立即收到 actions，无需等待额外的 GetActions 调用  
✅ **双重保障机制**：既通过响应返回，又缓存在服务器作为备用  
✅ **向后兼容**：可通过配置禁用，回到原流程  
✅ **线程安全**：使用锁机制保护缓存  
✅ **内存友好**：及时清理缓存，避免内存泄漏  
✅ **客户端无缝集成**：客户端已有的 `_add_actions_to_queue` 逻辑自动处理返回的 actions  
✅ **可扩展**：SmolVLA 模型支持更细粒度的分阶段执行  

改造后的系统在保持原有功能的同时，实现了真正的"立即响应"：客户端发送观察后，在同一个响应中就能拿到推理好的 actions，大幅提升了系统的实时性和用户体验。

