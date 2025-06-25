# src/G1/config.py

import os

# ———— 1) 仿真参数 ——————————————————————————————————
SIM = {
    'dt'         : 0.005,   # 时间步长
    'substeps'   : 2,       # PhysX 子步数
    'num_envs'   : 1,       # 并行环境数量
    'env_spacing': 2.0,     # 多 env 时的间隔
    'use_gpu'    : True,    # 是否启用 GPU pipeline
}

# ———— 2) URDF 路径 ——————————————————————————————————
ROOT      = os.path.dirname(__file__)
URDF_PATH = os.path.join(ROOT, 'g1_29dof.urdf')

# ———— 3) 关节名称列表 (29 DoF) ————————————————————————
# 顺序必须与 URDF 中 revolute/prismatic joint 的顺序一致
DOF_NAMES = [
    'left_hip_pitch',  'left_hip_roll',   'left_hip_yaw',
    'left_knee',
    'left_ankle_pitch','left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll',  'right_hip_yaw',
    'right_knee',
    'right_ankle_pitch','right_ankle_roll',
    'waist_yaw',       'waist_roll',      'waist_pitch',
    'left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw',
    'left_elbow',
    'left_wrist_roll','left_wrist_pitch','left_wrist_yaw',
    'right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw',
    'right_elbow',
    'right_wrist_roll','right_wrist_pitch','right_wrist_yaw',
]

# ———— 4) 重置时的初始姿态与速度 ————————————————————————
# 列表长度必须为 29
INIT_STATE = {
    'dof_pos': [
        # 左髋
        0.0,  0.6,  -0.3,
        # 左膝
        1.2,
        # 左踝
        -0.8, 0.0,
        # 右髋
        0.0,  0.6,  -0.3,
        # 右膝
        1.2,
        # 右踝
        -0.8, 0.0,
        # 腰
        0.0,  0.0,  0.0,
        # 左臂
        0.3,  0.0,  0.0,
        0.5,
        0.0,  0.0,  0.0,
        # 右臂
        0.3,  0.0,  0.0,
        0.5,
        0.0,  0.0,  0.0,
    ],
    'dof_vel': [0.0] * len(DOF_NAMES)
}

# ———— 5) PD 控制增益 ——————————————————————————————————
# 针对不同关节设置不同刚度/阻尼
PD = {
    'stiffness': {
        # 腿部
        'left_hip_pitch' : 80.0,  'left_hip_roll' : 80.0,  'left_hip_yaw' : 30.0,
        'left_knee'      :150.0,
        'left_ankle_pitch':200.0,  'left_ankle_roll':200.0,
        'right_hip_pitch': 80.0,  'right_hip_roll': 80.0,  'right_hip_yaw': 30.0,
        'right_knee'     :150.0,
        'right_ankle_pitch':200.0, 'right_ankle_roll':200.0,
        # 腰
        'waist_yaw'      : 30.0,  'waist_roll'    : 30.0,  'waist_pitch' : 30.0,
        # 臂部
        'left_shoulder_pitch':50.0, 'left_shoulder_roll':50.0, 'left_shoulder_yaw':50.0,
        'left_elbow'         :30.0,
        'left_wrist_roll'    :15.0, 'left_wrist_pitch'  :15.0, 'left_wrist_yaw'  :15.0,
        'right_shoulder_pitch':50.0,'right_shoulder_roll':50.0,'right_shoulder_yaw':50.0,
        'right_elbow'         :30.0,
        'right_wrist_roll'    :15.0, 'right_wrist_pitch'  :15.0, 'right_wrist_yaw'  :15.0,
    },
    'damping': {
        # 腿部
        'left_hip_pitch' :  5.0,  'left_hip_roll' :  5.0,  'left_hip_yaw' :  2.0,
        'left_knee'      : 10.0,
        'left_ankle_pitch':3.0,   'left_ankle_roll':3.0,
        'right_hip_pitch':  5.0,  'right_hip_roll':  5.0,  'right_hip_yaw':  2.0,
        'right_knee'     : 10.0,
        'right_ankle_pitch':3.0,  'right_ankle_roll':3.0,
        # 腰
        'waist_yaw'      :  2.0,  'waist_roll'    :  2.0,  'waist_pitch' :  2.0,
        # 臂部
        'left_shoulder_pitch': 3.0,'left_shoulder_roll': 3.0,'left_shoulder_yaw': 3.0,
        'left_elbow'         : 1.0,
        'left_wrist_roll'    : 0.5,'left_wrist_pitch'  : 0.5,'left_wrist_yaw'  : 0.5,
        'right_shoulder_pitch':3.0,'right_shoulder_roll':3.0,'right_shoulder_yaw':3.0,
        'right_elbow'         :1.0,
        'right_wrist_roll'    :0.5,'right_wrist_pitch'  :0.5,'right_wrist_yaw'  :0.5,
    }
}

# ———— 6) CBF 安全过滤参数 ——————————————————————————————
CBF = {
    'alpha'    : 10.0,   # Barrier 增益
    'max_tau'  :200.0,   # 最大扭矩
    'max_rate' :20.0,    # 最大扭矩变化
}
