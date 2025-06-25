# visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from G1.envs.gym_wrapper import G1GymEnv

def main():
    # 1) 创建一个单环境（开启渲染窗口）
    env = G1GymEnv(headless=False)

    # 2) 载入归一化参数，如果不存在就退出提示
    norm_path = "vec_normalize.pkl"
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(f"找不到归一化文件: {norm_path}")
    env = VecNormalize.load(norm_path, env)
    env.training = False      # 切换到评估模式
    env.norm_reward = False   # 在渲染时使用真实 reward

    # 3) 载入模型
    model_path = "best_model/best_model.zip"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    model = PPO.load(model_path, env=env)

    # 4) 运行并渲染，同时收集 reward
    obs = env.reset()
    rewards = []
    max_steps = 10000
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        rewards.append(reward)
        if done:
            obs = env.reset()

    # 5) 关闭环境
    env.close()

    # 6) 保存并绘制 reward 曲线
    np.savetxt("rewards.csv", rewards, delimiter=",")
    print(f"Reward 曲线已保存到 rewards.csv，共 {len(rewards)} 步。")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Reward over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("rewards.png")
    print("Reward 曲线图已保存到 rewards.png。")
    plt.show()

if __name__ == "__main__":
    main()
