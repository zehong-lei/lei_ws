import os
import numpy as np
import xml.etree.ElementTree as ET

# 必须先导入 isaacgym，然后再导入 torch
from isaacgym import gymapi, gymtorch

# —— 单例化 Gym 实例 ——
# 如果首次导入，则调用 acquire_gym()；否则复用已有实例
if not hasattr(gymapi, '_gym_instance'):
    _gym = gymapi.acquire_gym()
    gymapi._gym_instance = _gym
else:
    _gym = gymapi._gym_instance

import torch
from G1.config import URDF_PATH, SIM, DOF_NAMES, INIT_STATE, PD, CBF

class G1Env:
    def __init__(self, headless: bool = False):
        # ─── 1) 使用单例 Gym 实例 ──────────────────────────────
        self.gym = _gym
        physics_engine = gymapi.SIM_PHYSX

        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt               = SIM['dt']
        sim_params.substeps         = SIM.get('substeps', 2)
        sim_params.up_axis          = gymapi.UP_AXIS_Z
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu     = True
        sim_params.gravity          = gymapi.Vec3(0, 0, -9.81)

        # 创建仿真
        self.sim = self.gym.create_sim(0, 0, physics_engine, sim_params)

        # ─── 2) 添加地面 ─────────────────────────────────────
        plane_params = gymapi.PlaneParams()
        plane_params.normal           = gymapi.Vec3(0., 0., 1.)
        plane_params.distance         = 0.0
        plane_params.static_friction  = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

        # ─── 3) 可视化窗口 ───────────────────────────────────
        if not headless:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
        else:
            self.viewer = None

        # ─── 4) 加载 URDF ─────────────────────────────────────
        asset_root, asset_file = os.path.split(URDF_PATH)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link          = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials     = True
        self.asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # ─── 5) 解析 URDF 关节名 ────────────────────────────────
        tree = ET.parse(URDF_PATH)
        root = tree.getroot()
        dof_names = []
        for joint in root.findall('joint'):
            if joint.get('type') in ('revolute', 'prismatic'):
                name = joint.get('name')
                if name.endswith('_joint'):
                    name = name[:-6]
                dof_names.append(name)
        self.dof_names = dof_names

        # ─── 6) 创建并行 env & actor ───────────────────────────
        spacing = SIM.get('env_spacing', 2.0)
        lower   = gymapi.Vec3(-spacing, 0.0, 0.0)
        upper   = gymapi.Vec3( spacing, spacing, spacing)
        self.envs, self.actors = [], []
        num_envs = SIM['num_envs']
        for i in range(num_envs):
            env = self.gym.create_env(
                self.sim, lower, upper,
                int(np.ceil(np.sqrt(num_envs)))
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.8)
            actor_handle = self.gym.create_actor(
                env, self.asset, pose, f"g1_{i}", i, 1
            )
            self.envs.append(env)
            self.actors.append(actor_handle)

        # ─── 7) 设置 DOF 属性 ──────────────────────────────────
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        for env, actor in zip(self.envs, self.actors):
            self.gym.set_actor_dof_properties(env, actor, self.dof_props)

        # ─── 8) GPU pipeline 准备 ──────────────────────────────
        self.gym.prepare_sim(self.sim)

        # ─── 9) 相机对准第一个 env 的机器人 ─────────────────────
        if self.viewer is not None:
            cam_pos    = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[0], cam_pos, cam_target
            )

        # ─── 10) wrap 张量 ──────────────────────────────────────
        self.dof_state_tensor        = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_force_tensor        = self.gym.acquire_dof_force_tensor(self.sim)
        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.dof_state        = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.actions          = gymtorch.wrap_tensor(self.dof_force_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state_tensor)

        # —— 设备 & 维度绑定 ——
        self.device  = self.dof_state.device
        self.num_envs = num_envs
        self.num_dof  = len(self.dof_props['lower'])

        # —— 初始化站立姿态 ——
        init_pos = torch.tensor(INIT_STATE['dof_pos'], device=self.device)
        self.default_dof_pos = init_pos.unsqueeze(0).repeat(self.num_envs, 1)

    def reset(self):
        """重置所有 env 到默认站立姿态"""
        default_vel = torch.zeros_like(self.default_dof_pos)
        state = torch.zeros(self.num_envs, self.num_dof, 2, device=self.device)
        state[:, :, 0] = self.default_dof_pos
        state[:, :, 1] = default_vel
        self.gym.set_dof_state_tensor(
            self.sim, gymtorch.unwrap_tensor(state)
        )
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def step(self, actions: torch.Tensor):
        """一步仿真，输入扭矩 actions(shape=[num_envs, num_dof])"""
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(actions)
        )
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def render(self):
        """渲染一帧到窗口"""
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def close(self):
        """销毁 Viewer 与 Sim"""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    # ———— 常用属性  ——————————————————————————————
    @property
    def dof_pos(self):
        return self.dof_state.view(
            self.num_envs, self.num_dof, 2
        )[..., 0]

    @property
    def dof_vel(self):
        return self.dof_state.view(
            self.num_envs, self.num_dof, 2
        )[..., 1]

    @property
    def base_pos(self):
        return self.rigid_body_state.view(
            self.num_envs, -1, 13
        )[:, :, 0:3]

    @property
    def base_quat(self):
        return self.rigid_body_state.view(
            self.num_envs, -1, 13
        )[:, :, 3:7]

    @property
    def base_lin_vel(self):
        return self.rigid_body_state.view(
            self.num_envs, -1, 13
        )[:, :, 7:10]

    @property
    def base_ang_vel(self):
        return self.rigid_body_state.view(
            self.num_envs, -1, 13
        )[:, :, 10:13]
