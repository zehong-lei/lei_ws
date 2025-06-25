# src/G1/controllers/pd_cbf.py

from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import osqp
from scipy import sparse

from G1.config import PD, CBF
from G1.envs.env import G1Env

class PDControllerCBF:
    """
    低级跟踪层：PD 控制 + CBF 约束 + 速率限幅 + 低通滤波
    """
    def __init__(self, env: G1Env):
        # assert isinstance(env, G1Env), "需要把 G1Env 实例传入"
        self.env = env
        self.device = env.device
        self.num_dof = env.num_dof

        # —— 1) PD 增益 —— 
        # env.dof_names 示例： ['hip_roll_joint', 'hip_pitch_joint', 'knee_joint', …]
        # key 要与 config.py 中 PD['stiffness'] 的键一致
        kp_list = []
        kd_list = []
        # —— 正确地用 env.dof_names（我们在 G1Env.__init__ 中解析 URDF 挂上去的）——
        for name in env.dof_names:
            # name 已经去掉了 "_joint" 后缀
            kp_list.append(PD['stiffness'].get(name, 0.0))
            kd_list.append(PD['damping']  .get(name, 0.0))
        self.Kp = torch.tensor(kp_list, device=self.device)
        self.Kd = torch.tensor(kd_list, device=self.device)

        # —— 2) CBF 参数 —— 
        self.alpha    = CBF['alpha']
        self.max_tau  = CBF['max_tau']
        self.max_rate = CBF['max_rate']

        # —— 3) 滤波状态 —— 
        self.tau_prev = torch.zeros((env.num_envs, self.num_dof), device=self.device)
        self.lpf_alpha = 0.3

        # OSQP 求解器初始化，先给一个“空约束” (0×D) 矩阵
        self.prob = osqp.OSQP()
        P = sparse.eye(self.num_dof)
        q = np.zeros(self.num_dof)
        # 空约束矩阵 A (0 rows × num_dof cols)
        A_empty = sparse.csc_matrix((0, self.num_dof))
        l_empty = np.zeros(0)
        u_empty = np.zeros(0)
        self.prob.setup(
            P=P, q=q,
            A=A_empty, l=l_empty, u=u_empty,
            warm_start=True, verbose=False
        )


    def compute_pd(self):
        """
        计算原始 PD 力矩
        输出 shape [num_envs, num_dof]
        """
        q    = self.env.dof_pos      # [N, D]
        dq   = self.env.dof_vel      # [N, D]
        # 目标位置默认就是 URDF 里的初始 pos
        # 用 env.default_dof_pos 作为目标位置 (shape [N, D])
        q_des = self.env.default_dof_pos
        dq_des= torch.zeros_like(dq)

        tau_pd = self.Kp*(q_des - q) + self.Kd*(dq_des - dq)
        return tau_pd

    def apply_cbf(self, tau_pd: torch.Tensor):
        """
        对每个 env 都用 QP 加 CBF 约束：
           h_low  = q - q_min >=0,   h_high = q_max - q >=0
           ∂h/∂q u >= -α h
        并加上 |u| <= max_tau
        返回滤波后 u_safe
        """
        q_np    = self.env.dof_pos.cpu().numpy()        # [N, D]
        q_min, q_max = (self.env.dof_props['lower'], 
                        self.env.dof_props['upper'])    # each [D]

        N, D = q_np.shape
        u_safe = np.zeros_like(q_np)

        # 构造 G, h （|u|<=max_tau）常量
        G = sparse.vstack([sparse.eye(D), -sparse.eye(D)])
        h = np.hstack([np.ones(D)*self.max_tau,
                       np.ones(D)*self.max_tau])

        for i in range(N):
            # h_low, h_high
            h_low  = q_np[i] - q_min
            h_high = q_max - q_np[i]
            # 构造 CBF 约束 A u >= b
            A_cbf = np.vstack([ np.eye(D), -np.eye(D) ])  # reuse shape
            b_cbf = np.hstack([ self.alpha*h_low, self.alpha*h_high ])

            # 合并不等式： G u <= h   and  -A_cbf u <= -b_cbf
            A = sparse.vstack([G, -sparse.csc_matrix(A_cbf)])
            u_bound = np.hstack([h, -b_cbf])

            # QP 参数更新
            u_pd = tau_pd[i].cpu().numpy()
            self.prob.update(q=-u_pd, A=A.tocsc(), u=u_bound)

            res = self.prob.solve()
            if res.info.status_val != osqp.constant('OSQP_SOLVED'):
                # 降级：直接用原始 PD
                u_safe[i] = u_pd
            else:
                u_safe[i] = res.x

        return torch.tensor(u_safe, device=self.device, dtype=torch.float32)

    def filter_rate_lpf(self, u: torch.Tensor):
        """
        速率限幅 + 一阶低通
        """
        delta = u - self.tau_prev
        delta = torch.clamp(delta, -self.max_rate, self.max_rate)
        u_limited = self.tau_prev + delta
        # LPF
        u_f = self.lpf_alpha * u_limited + (1 - self.lpf_alpha) * self.tau_prev
        self.tau_prev = u_f.clone()
        return u_f

    def get_action(self):
        """
        完整控制管道：PD -> CBF -> 限幅滤波 -> 返回动作张量
        """
        tau_pd  = self.compute_pd()                      # [N,D]
        tau_cbf = self.apply_cbf(tau_pd)                 # [N,D]
        tau     = self.filter_rate_lpf(tau_cbf)          # [N,D]
        # IsaacGym 接受的 actions 就是力矩/驱动命令
        return tau

