from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, frame_transformer_data
import omni.isaac.lab.envs.mdp as mdp_utils


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

class Queue: # makeshift for now (can handle only 1 env for now)
    def __init__(self, max_capacity):
        self.buff = []
        self.maxlen = max_capacity

    def add(self, data):
        self.buff.append(data)
        if len(self.buff) > self.maxlen:
            self.buff.pop(0) # pop from front
    
    def get(self):
        return self.buff[0] # get from the front FIFO


previous_dists = Queue(max_capacity=2)

def relative_pose_to_goal(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg=SceneEntityCfg("robot"),
    target_pose_cfg: SceneEntityCfg=SceneEntityCfg("target") 
) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    target_pose = env.scene[target_pose_cfg.name]
    
    robot_poses = robot.data.root_state_w
    target_poses = target_pose.data.root_state_w

    relative_positions, relative_orientations = math_utils.subtract_frame_transforms(
        t01=robot_poses[:, :3],  # Shape: (num_envs, 3)
        q01=robot_poses[:, 3:7],  # Shape: (num_envs, 4)
        t02=target_poses[:, :3],  # Shape: (num_envs, 3) 
        q02=target_poses[:, 3:7]  # Shape: (num_envs, 4) 
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

    if previous_dists is None:
        return
    
    goal_pose_relative = relative_pose_to_goal(env=env, robot_cfg=robot_cfg, target_pose_cfg=target_pose_cfg) # (num_envs, 4)
    pos_diff = goal_pose_relative[:, :2] # (num_envs, 2)

    dists_to_goal = torch.sqrt(torch.sum(torch.square(pos_diff), dim=1)) # (num_envs,)
    dists_to_goal = dists_to_goal.unsqueeze(1) # (num_envs, 1)
    
    previous_dists.add(dists_to_goal)
    
    previous_distances = torch.tensor(previous_dists.get(), device=env.device, dtype=torch.float32).unsqueeze(1)
    ddists = torch.sub(dists_to_goal, previous_distances).squeeze(1)
    distance_data = torch.cat((dists_to_goal, ddists), dim=1)
    # print(f"Distance Data: {distance_data}")

    return distance_data


def orientation_to_goal(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg=SceneEntityCfg("robot"),
    target_pose_cfg: SceneEntityCfg=SceneEntityCfg("target")
):
    
    goal_pose_relative = relative_pose_to_goal(env=env, robot_cfg=robot_cfg, target_pose_cfg=target_pose_cfg) # (num_envs, 4)
    angles_diff = goal_pose_relative[:, 3]
    angles_diff = angles_diff.unsqueeze(1)
    angles_diff = math_utils.wrap_to_pi(angles_diff)
    
    return angles_diff


def bot_velocities(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    
    bot_lin_velocity = mdp_utils.base_lin_vel(env=env, asset_cfg=robot_cfg)
    bot_ang_velocity = mdp_utils.base_ang_vel(env=env, asset_cfg=robot_cfg)

    bot_x_velocity = bot_lin_velocity[:, 0] # (num_envs,)
    bot_x_velocity = bot_x_velocity.unsqueeze(1) # (num_envs, 1)
    bot_yaw_velocity = bot_ang_velocity[:, 2] # (num_envs,)
    bot_yaw_velocity = bot_yaw_velocity.unsqueeze(1) # (num_envs, 1)

    bot_velocity_filt = torch.cat((bot_x_velocity, bot_yaw_velocity), dim=1) # (num_envs, 3)
    
    return bot_velocity_filt  