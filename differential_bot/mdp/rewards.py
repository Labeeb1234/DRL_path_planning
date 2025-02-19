from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster
import omni.isaac.lab_tasks.manager_based.classic.differential_bot.mdp as cmdp_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.classic.differential_bot.mdp.observations import Queue

# creating a queue ds for storage of previous distance and angle datas
previous_distances = Queue(2)
previous_yaws = Queue(2) 

# sparse reward structure
def reached_goal_position(
    env: ManagerBasedRLEnv,
    goal_distance_tolerance: float,
    goal_angle_tolerance: float,
    distance_scale: float,
    reduction_scale: float,
    yaw_alignment_radius: float,
    yaw_scale: float,
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

    _, _, angle_diffs = math_utils.euler_xyz_from_quat(orientation_diffs) # (each value is --> (num_envs,))
    angle_diffs = angle_diffs.unsqueeze(1) # (num_envs, 1)
    angle_diffs = math_utils.wrap_to_pi(angle_diffs) # (yaw range => (-3.14, 3.14))
    previous_yaws.add(angle_diffs)

    rel_pos = pose_diffs[:, :2]

    dist_diff = torch.sqrt(torch.sum(torch.square(rel_pos), dim=1)) # (num_envs,)
    dist_diff = dist_diff.unsqueeze(1) # (num_envs, 1)
    previous_distances.add(dist_diff)

    # distance progression reward/ distance reduction reward
    dist_reduc = torch.tensor(dist_diff[:, 0]-previous_distances.get(), device=env.device).squeeze(1) # (num_envs,)
    reduction_reward = (reduction_scale)*(-1.0*dist_reduc)

    # distance-progression reward
    distance_reward = -distance_scale*(torch.sign(dist_reduc)/(1+dist_diff)) # using inverse hyperbole function (smoother)
    distance_reward = distance_reward.squeeze(1)

    # Calculate proximity mask (1.0 when close to goal, 0.0 when far)
    proximity_mask = torch.where(
        dist_diff[:, 0] < yaw_alignment_radius,
        torch.ones_like(dist_diff[:, 0]),
        torch.zeros_like(dist_diff[:, 0])
    )

    # new yaw penalty/reward (to be tested)
    yaw_change_reward = yaw_scale*(1-2.0*(torch.abs(angle_diffs[:, 0]))/(3.14/2))
    # yaw_change_reward = yaw_scale*torch.cos(angle_diffs.squeeze(1))
    yaw_change_reward = yaw_change_reward * proximity_mask
    # print(f"yaw reward: {yaw_change_reward}")

    # Additional bonus for reaching the goal
    goal_reached_bonus = torch.where(
        (dist_diff[:, 0] < goal_distance_tolerance),
        torch.tensor(15.0),
        torch.tensor(0.0)
    )

    total_task_reward = distance_reward + reduction_reward + yaw_change_reward + goal_reached_bonus

    return total_task_reward

def bot_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot_vel = cmdp_utils.bot_velocities(env=env, robot_cfg=robot_cfg) # (num_envs, 2) (bot_xvel, bot_yawvel)
    return torch.sum(torch.square(robot_vel), dim=1)

