from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# sparse reward structure
def reached_goal_position(
    env: ManagerBasedRLEnv,
    goal_position_tolerance: float,
    goal_angle_tolerance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target") 

):
    
    robot = env.scene[robot_cfg.name] 
    goal = env.scene[target_cfg.name]

    robot_poses = robot.data.root_state_w # (num_envs, 13)
    goal_pose = goal.data.root_state_w # (num_envs, 13)

    pose_diffs, orientation_diffs = math_utils.subtract_frame_transforms(
        t01=robot_poses[:, :3],  # Shape: (num_envs, 3)
        q01=robot_poses[:, 3:7],  # Shape: (num_envs, 4)
        t02=goal_pose[:, :3],  # Shape: (num_envs, 3)
        q02=goal_pose[:, 3:7]  # Shape: (num_envs, 4) 
    )

    _, _, angle_diffs = math_utils.euler_xyz_from_quat(orientation_diffs)
    angle_diffs = angle_diffs.unsqueeze(1) # (num_envs, 1)

    dist_diff = torch.sqrt(torch.sum(torch.square(pose_diffs), dim=1)) # (num_envs,)
    dist_diff = dist_diff.unsqueeze(1) # (num_envs, 1)

    distance_reward_factor = torch.where(dist_diff[:, 0] < goal_position_tolerance, torch.tensor(50.0), torch.tensor(-50.0))
    distance_reward_factor.unsqueeze(1)

    return distance_reward_factor 