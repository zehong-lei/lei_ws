# src/G1/ppo.py

# ── 0. 必须最先导入 isaacgym，确保 gymapi/gymtorch 在 torch 之前初始化 ──
import isaacgym
from isaacgym import gymapi, gymtorch

# ── 1. 再导入其余库 ────────────────────────────────────────
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from G1.envs.gym_wrapper import G1GymEnv

def make_env(headless: bool):
    """
    创建并返回 G1GymEnv 实例的工厂函数。
    headless=True 时不弹出渲染窗口，加速训练。
    """
    def _init():
        return G1GymEnv(headless=headless)
    return _init

def main():
    # 1) 只用 1 个并行环境，避免重复初始化
    env_fns = [ make_env(headless=True) ]   # 这里列表长度 = 1 即可
    vec_env = DummyVecEnv(env_fns)

    # 2) 归一化观测值和奖励
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3) 检查点回调：每 50k 步保存一次模型
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ppo_g1"
    )

    # 4) 构建并训练 PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        batch_size=1024,       # 原来 2048，可根据资源再调
        n_epochs=20,
        learning_rate=1e-4,    # 比 3e-4 更保守
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,        # 更小的 clip_range
        ent_coef=0.0,
        max_grad_norm=0.5,     # 梯度裁剪
        target_kl=0.03,        # KL 散度目标，超过就停止当前 epoch
        tensorboard_log="./tb_logs/"
    )
    model.learn(
        total_timesteps=2_000_000,
        callback=[ckpt_cb]
    )

    # 5) 保存最终模型和归一化参数
    model.save("ppo_g1_final")
    vec_env.save("vec_normalize.pkl")

    # 6) 关闭环境
    vec_env.close()

if __name__ == "__main__":
    main()
