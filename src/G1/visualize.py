# visualize.py

# 1) 最先导入 Isaac Gym，完成底层初始化
import isaacgym
from isaacgym import gymapi, gymtorch

# 2) 导入环境包装器
from G1.envs.gym_wrapper import G1GymEnv

# 3) 其余依赖
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def main():
    # 4) 创建原始环境（用于渲染）
    raw_env = G1GymEnv(headless=False)

    # 5) 用 DummyVecEnv 包装成单进程向量环境
    vec_env = DummyVecEnv([lambda: raw_env])

    # 6) 加载训练时保存的归一化参数
    norm_path = "vec_normalize.pkl"
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(f"找不到归一化文件: {norm_path}")
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # 7) 自动选择要加载的模型文件
    candidates = [
        "best_model/best_model.zip",  # EvalCallback 保存的最优模型
        "ppo_g1_final.zip"            # 最终模型
    ]
    model_path = next((p for p in candidates if os.path.isfile(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            f"在当前目录下找不到模型文件，请先训练并保存模型：\n" + "\n".join(candidates)
        )
    print(f"加载模型：{model_path}")

    # 8) 加载模型
    model = PPO.load(model_path, env=vec_env)

    # 9) 运行并渲染，同时记录 reward
    obs = vec_env.reset()
    rewards = []
    max_steps = 10000
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        raw_env.render()

        # reward, done 都是长度为 1 的数组
        rewards.append(float(reward[0]))
        if done[0]:
            obs = vec_env.reset()

    # 10) 关闭环境
    raw_env.close()
    vec_env.close()

    # 11) 保存并绘制 reward 曲线
    np.savetxt("rewards.csv", rewards, delimiter=",")
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.title("Reward over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("rewards.png")
    print("Rewards 已保存到 rewards.csv 和 rewards.png")
    plt.show()

if __name__ == "__main__":
    main()
