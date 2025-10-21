#!/usr/bin/env python3
"""诊断脚本：对比 lerobot-eval 和 async 方法的差异

这个脚本将帮助定位为什么 async 方法成功率低的问题。
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.async_delta_inference.sim_client import _format_env_observation, _env_obs_to_lerobot_features
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.configs.types import PolicyFeature, FeatureType

print("=" * 70)
print("诊断报告：lerobot-eval vs async 方法差异分析")
print("=" * 70)
print()

# 创建环境
env_config = LiberoEnv(task='libero_object')
print("1. 环境配置")
print(f"   - 任务: {env_config.task}")
print(f"   - 观测类型: {env_config.obs_type}")
print(f"   - 图像尺寸: {env_config.observation_height}x{env_config.observation_width}")
print()

# 创建环境实例
env_dict = make_env(env_config, n_envs=1, use_async_envs=False)
suite_name = list(env_dict.keys())[0]
task_id = list(env_dict[suite_name].keys())[0]
vec_env = env_dict[suite_name][task_id]

# 重置环境获取观测
obs_raw, info = vec_env.reset()
obs_single = {
    'pixels': {k: v[0] for k, v in obs_raw['pixels'].items()},
    'agent_pos': obs_raw['agent_pos'][0]
}

print("2. 环境返回的原始观测")
print(f"   - pixels keys: {list(obs_single['pixels'].keys())}")
print(f"   - agent_pos shape: {obs_single['agent_pos'].shape}")
for key, img in obs_single['pixels'].items():
    print(f"   - pixels[{key}] shape: {img.shape}, dtype: {img.dtype}")
print()

# ============================================================================
# lerobot-eval 方法
# ============================================================================
print("3. lerobot-eval 方法的观测处理")
print("-" * 70)

obs_eval = preprocess_observation(obs_raw)
obs_eval = add_envs_task(vec_env, obs_eval)

print("   处理后的观测值键:")
for key in sorted(obs_eval.keys()):
    val = obs_eval[key]
    if isinstance(val, torch.Tensor):
        print(f"   - {key}: shape={val.shape}, dtype={val.dtype}, "
              f"range=[{val.min():.3f}, {val.max():.3f}]")
    elif isinstance(val, list):
        print(f"   - {key}: {val}")
print()

# ============================================================================
# async 方法
# ============================================================================
print("4. async 方法的观测处理")
print("-" * 70)

lerobot_features = _env_obs_to_lerobot_features(env_config)
print("   lerobot_features:")
for key, val in lerobot_features.items():
    print(f"   - {key}: {val}")
print()

raw_obs = _format_env_observation(obs_single, env_config, task="test task")
print("   raw_obs keys:", list(raw_obs.keys()))
print()

# 模拟 policy_image_features
policy_image_features = {
    'observation.images.image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 360)),
    'observation.images.image2': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 360)),
}

obs_async = raw_observation_to_observation(raw_obs, lerobot_features, policy_image_features)

print("   处理后的观测值键:")
for key in sorted(obs_async.keys()):
    val = obs_async[key]
    if isinstance(val, torch.Tensor):
        print(f"   - {key}: shape={val.shape}, dtype={val.dtype}, "
              f"range=[{val.min():.3f}, {val.max():.3f}]")
    else:
        print(f"   - {key}: {repr(val)}")
print()

# ============================================================================
# 关键差异对比
# ============================================================================
print("5. 关键差异对比")
print("=" * 70)

# 图像键名
eval_image_keys = sorted([k for k in obs_eval.keys() if 'image' in k])
async_image_keys = sorted([k for k in obs_async.keys() if 'image' in k])

print("   图像键名对比:")
print(f"   - lerobot-eval: {eval_image_keys}")
print(f"   - async:        {async_image_keys}")
if eval_image_keys != async_image_keys:
    print("   ⚠️  警告: 图像键名不一致!")
else:
    print("   ✅ 图像键名一致")
print()

# 图像值对比
print("   图像值对比:")
for eval_key in eval_image_keys:
    # 找到对应的 async key
    async_key = eval_key  # 现在应该是一样的了
    if async_key in obs_async:
        eval_img = obs_eval[eval_key]
        async_img = obs_async[async_key]
        
        # 对比形状
        if eval_img.shape != async_img.shape:
            print(f"   ⚠️  {eval_key} 形状不一致:")
            print(f"       lerobot-eval: {eval_img.shape}")
            print(f"       async:        {async_img.shape}")
        else:
            # 对比值
            diff = torch.abs(eval_img - async_img).max()
            print(f"   - {eval_key}: 形状一致 {eval_img.shape}, 最大差值={diff:.6f}")
            if diff > 0.001:
                print(f"     ⚠️  警告: 图像值差异较大!")
    else:
        print(f"   ⚠️  {eval_key} 在 async 观测中缺失!")
print()

# state 对比
print("   State 对比:")
eval_state = obs_eval.get('observation.state')
async_state = obs_async.get('observation.state')
if eval_state is not None and async_state is not None:
    print(f"   - 形状: eval={eval_state.shape}, async={async_state.shape}")
    if eval_state.shape == async_state.shape:
        diff = torch.abs(eval_state - async_state).max()
        print(f"   - 最大差值: {diff:.6f}")
        if diff > 0.001:
            print(f"     ⚠️  警告: State 值差异较大!")
        else:
            print("   ✅ State 值一致")
    else:
        print("   ⚠️  警告: State 形状不一致!")
else:
    print("   ⚠️  警告: State 在某个方法中缺失!")
print()

# Task 对比
print("   Task 对比:")
eval_task = obs_eval.get('task')
async_task = obs_async.get('task')
print(f"   - lerobot-eval: {eval_task}")
print(f"   - async:        {async_task}")
if eval_task != async_task:
    print("   ⚠️  警告: Task 不一致!")
else:
    print("   ✅ Task 一致")
print()

# ============================================================================
# 动作处理差异分析
# ============================================================================
print("6. 动作处理差异分析")
print("=" * 70)

print("   lerobot-eval 方法:")
print("   - 每次循环调用 policy.select_action()")
print("   - select_action 内部管理动作队列")
print("   - 队列为空时才调用 predict_action_chunk()")
print("   - 每次只执行 1 个动作")
print()

print("   async 方法:")
print("   - 客户端维护动作队列")
print("   - 服务器异步推理")
print("   - 可能存在的问题:")
print("     1. 动作执行延迟（网络+推理时间）")
print("     2. 动作聚合策略（weighted_average）")
print("     3. 动作替换策略（replace_actions_on_new）")
print("     4. 队列管理逻辑（request_new_at, max_actions_to_use）")
print()

# ============================================================================
# 建议的调查方向
# ============================================================================
print("7. 建议的调查方向")
print("=" * 70)

suggestions = [
    "1. 检查动作值分布",
    "   - 记录 lerobot-eval 的动作序列",
    "   - 记录 async 的动作序列",
    "   - 对比动作值的统计特性（均值、方差、范围）",
    "",
    "2. 检查动作执行时机",
    "   - async 的第一个动作何时执行？",
    "   - 是否存在明显的延迟导致错过关键时刻？",
    "",
    "3. 检查策略状态",
    "   - reset_policy 是否在正确的时机触发？",
    "   - 策略的内部状态是否正确维护？",
    "",
    "4. 检查动作聚合效果",
    "   - weighted_average (0.3*old + 0.7*new) 是否合适？",
    "   - 尝试使用 latest_only（不聚合）对比",
    "",
    "5. 检查动作队列参数",
    "   - max_actions_to_use=10 vs actions_per_chunk=50",
    "   - request_new_at=5 的时机是否合适？",
    "   - replace_actions_on_new=false 是否应改为 true？",
    "",
    "6. 添加详细日志",
    "   - 记录每步的观测值",
    "   - 记录每步的动作值",
    "   - 记录队列状态变化",
    "   - 记录成功/失败的 episode 的完整轨迹",
]

for line in suggestions:
    print(line)

print()
print("=" * 70)
print("诊断完成！")
print("=" * 70)

vec_env.close()

