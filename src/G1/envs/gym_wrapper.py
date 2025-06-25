# src/G1/envs/gym_wrapper.py

from G1.envs.env import G1Env
import gym
from gym import spaces
import numpy as np
from G1.envs.env import G1Env
from G1.config    import SIM
import torch

class G1GymEnv(gym.Env):
    """
    将多并行 Isaac-Gym 环境 G1Env 包装成单实例的 gym.Env，
    方便用 Stable-Baselines3、RLlib、rl-games 等库直接训练。
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, headless: bool = True):
        super().__init__()
        # 强制内部只用 1 个并行实例
        SIM['num_envs'] = 1

        # 创建底层环境
        self.env = G1Env(headless=headless)
        self.env.reset()

        # 缓存维度
        self.num_envs = 1
        self.num_dof  = self.env.num_dof

        # 动作空间：每个关节输出扭矩（连续）
        max_torque = SIM.get('max_action', 200.0)
        self.action_space = spaces.Box(
            low   = -max_torque,
            high  =  max_torque,
            shape = (self.num_dof,),
            dtype = np.float32
        )

        # 观测空间：关节位置 + 速度 + 基座四元数 + 基座角速度
        obs_dim = self.num_dof * 2 + 4 + 3
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32
        )

    def reset(self):
        """重置底层环境，返回初始观测（1, obs_dim）→(obs_dim,)"""
        self.env.reset()
        return self._get_obs()

    def step(self, action):
        """
        action: np.ndarray, shape=(num_dof,)
        返回：obs(np.ndarray)、reward(float)、done(bool)、info(dict)
        """
        # 转成 (1, num_dof) 的 torch Tensor 并步进
        a = torch.tensor(action, device=self.env.device, dtype=torch.float32)
        a = a.unsqueeze(0)  # [1, num_dof]
        self.env.step(a)

        obs    = self._get_obs()
        reward = self._compute_reward(obs, action)
        done   = self._check_done(obs)
        info   = {}

        return obs, reward, done, info

    def _get_obs(self):
        # dof_pos, dof_vel: [num_envs, num_dof]
        pos  = self.env.dof_pos[0].cpu().numpy()
        vel  = self.env.dof_vel[0].cpu().numpy()
        # base_quat: [num_envs, 1, 4] → [4,]
        quat = self.env.base_quat[0,0].cpu().numpy()
        # base_ang_vel: [num_envs, 1, 3] → [3,]
        angv = self.env.base_ang_vel[0,0].cpu().numpy()

        return np.concatenate([pos, vel, quat, angv]).astype(np.float32)

    def _compute_reward(self, obs, action):
        """
        简单奖励：基座直立程度 - 力矩惩罚
        quat 保存在 obs[num_dof*2 : num_dof*2+4]
        """
        idx0 = self.num_dof * 2
        w = obs[idx0]
        upright = float(abs(w))               # w 越接近 1 越直立
        torque_pen = 1e-3 * np.sum(action**2)
        return upright - torque_pen

    def _check_done(self, obs):
        """
        如果机器人俯仰或横滚角过大，则判断为摔倒 → done
        通过四元数计算 roll/pitch
        """
        idx0 = self.num_dof * 2
        w,x,y,z = obs[idx0:idx0+4]
        # 俯仰（pitch）和横滚（roll）公式
        pitch = np.arctan2(2*(w*y - z*x), 1 - 2*(y*y + x*x))
        roll  = np.arctan2(2*(w*x - y*z), 1 - 2*(x*x + z*z))
        return bool(abs(pitch) > 0.5 or abs(roll) > 0.5)

    def render(self, mode="human"):
        """调用底层 render"""
        return self.env.render()

    def close(self):
        """销毁底层资源"""
        self.env.close()
