from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, frame_transformer_data

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def relative_pose_to_goal(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg=SceneEntityCfg("robot"),
    target_pose_cfg: SceneEntityCfg=SceneEntityCfg("target") 
) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    target_pose = env.scene[target_pose_cfg.name]
    
    robot_poses = robot.data.root_state_w
    target_pose = target_pose.data.root_state_w

    relative_positions, relative_orientations = math_utils.subtract_frame_transforms(
        t01=robot_poses[:, :3],  # Shape: (num_envs, 3)
        q01=robot_poses[:, 3:7],  # Shape: (num_envs, 4)
        t02=target_pose[:, :3],  # Shape: (num_envs, 3) 
        q02=target_pose[:, 3:7]  # Shape: (num_envs, 4) 
    )

    _, _, relative_yaws = math_utils.euler_xyz_from_quat(relative_orientations)
    relative_yaws = relative_yaws.unsqueeze(1)
    relative_goal_pose = torch.cat((relative_positions, relative_yaws), dim=1)
    
    return relative_goal_pose

def distance_to_goal(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg=SceneEntityCfg("robot"),
    target_pose_cfg: SceneEntityCfg=SceneEntityCfg("target")
):
    
    goal_pose_relative = relative_pose_to_goal(env=env, robot_cfg=robot_cfg, target_pose_cfg=target_pose_cfg) # (num_envs, 4)
    pos_diff = goal_pose_relative[:, :3]
 
    dists_to_goal = torch.sqrt(torch.sum(torch.square(pos_diff), dim=1)) # (num_envs,)
    dists_to_goal = dists_to_goal.unsqueeze(1) # (num_envs, 1)
    return dists_to_goal
    
# lidar plugins of the isaaclab still not fully developed properly (ver 4.2.0)
def distance_to_obstacle():
    pass

