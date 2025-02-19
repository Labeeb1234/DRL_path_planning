import argparse
import os
import torch
import numpy as np
import random
import logging
import time

import omni.isaac.core.utils.prims as prim_utils # type: ignore
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.datasets import HDF5DatasetFileHandler

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# ----------------- mdp cfg modules ------------------------------
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg, RecorderManagerBaseCfg, DatasetExportMode
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
#-----------------------------------------------------------



from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.devices import Se2Keyboard

# custom modules
from omni.isaac.lab_tasks.manager_based.classic.differential_bot.diff_bot_env_cfg import DiffBotSceneCfg
import omni.isaac.lab_tasks.manager_based.classic.differential_bot.mdp as cmdp

@configclass
class ActionCfg:
    bot_vel = cmdp.DifferentialControllerActionCfg(
        asset_name="robot",
        body_name="base_link",
        joint_names=["Revolute_1", "Revolute_2", "Revolute_3_01", "Revolute_4_01"],
        wheel_diameter=0.4,
        wheel_separation=0.5778, 
        scale=(0.5, 1.0)
    )
    
# observation space configuration
@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # (relative) distance to goal
        dists_to_goal = ObsTerm(func=cmdp.distance_to_goal)
        # (relative) orientation to goal
        angles_to_goal = ObsTerm(func=cmdp.orientation_to_goal)
        # rel_pose_to_goal = ObsTerm(
        #     func=cmdp.relative_pose_to_goal
        # ) # from custom observation space module (only x,y,z)
        # ---------------------------------------------------
        # distance to obstacles (need to find an alternative)
        # ---------------------------------------------------
        # current bot pose and target pose
        # bot_pos = ObsTerm(
        #     func=mdp.root_pos_w,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot")
        #     }
        # )
        # target_pos = ObsTerm(
        #     func=mdp.root_pos_w,
        #     params={
        #         "asset_cfg": SceneEntityCfg("target")
        #     }
        # )
        # bot frame velocities
        bot_frame_velocity = ObsTerm(func=cmdp.bot_velocities)

        # history_length = 5
        # flatten_history_dim = False 

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


# event configuration
@configclass
class EventCfg:
    # events on reset
    # reset bot active joint states
    joint_states_reset = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1", "Revolute_2", "Revolute_3_01", "Revolute_4_01"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0)
        }
    )

    # randomize the bot pose
    randomize_bot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-0.01, 0.01)},
            "velocity_range": {}
        }
    )

    # randomize goal poses
    # randomize_goal_poses = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("target"),
    #         "pose_range": {"x": (-3.0, 3.0), "y": (3.0, 4.0)},
    #         "velocity_range": {}
    #     }
    # )

@configclass
class TerminationsCfg:
    # timeout termination
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    # terminate if bot reaches the goal within certain tolerance limit
    done_goal = DoneTerm(
        func=cmdp.reached_goal_termination,
        params={
            "goal_distance_tolerance": 0.25,
            "goal_angle_tolerance": 0.3
        },
        time_out=True
    )    
    # terminate if bot goes out of certain specified environment bounds
    out_of_bounds = DoneTerm(
        func=cmdp.out_of_bounds,
        params={
            "distance_bound": 5.0 # radius of 5[m]
        }
    )
    

@configclass
class RewardCfg:
    # penalty for just existing
    alive_reward = RewTerm(
        func=mdp.is_alive,
        weight=-0.1
    )

    # waypoint_reward_structure
    goal_reward = RewTerm(
        func=cmdp.reached_goal_position,
        weight=1.0,
        params={
            "goal_distance_tolerance": 0.36, # 36cm xy_accuracy
            "goal_angle_tolerance": 0.3, # 0.3rads yaw_accuracy  
            "distance_scale": 10.0,
            "reduction_scale": 2.0,
            "yaw_alignment_radius": 1.0,
            "yaw_scale": 0.5
        }
    )
    # joint angular velocity penalty
    # joint_vel_penalty = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=0.005,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1", "Revolute_2", "Revolute_3_01", "Revolute_4_01"])
    #     }
    # )

# for data recording in HDF5 format
@configclass
class RecordCfg(RecorderManagerBaseCfg):
    dataset_file_handler_class_type: type = HDF5DatasetFileHandler
    dataset_export_dir_path: str = "/home/inlabust/labeeb/logs"
    dataset_filename: str = "dataset"
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    pre_step_obsrecorder = mdp.PreStepFlatPolicyObservationsRecorderCfg()

# creating the manager based env
@configclass
class DiffBotRLEnvCfg(ManagerBasedRLEnvCfg):
    # scene setting
    scene: DiffBotSceneCfg = DiffBotSceneCfg(num_envs=2, env_spacing=1.5)
    # base settings
    observations = ObservationCfg()
    actions = ActionCfg()
    events = EventCfg()
    # MDP settings
    # no curriculum for now
    rewards = RewardCfg()
    terminations = TerminationsCfg()

    # to record episodic and steps_per_episode data
    # recorders = RecordCfg()
    
    def __post_init__(self):
        # viewport view settings
        self.viewer.eye = [2.5, 0.0, 4.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]

        self.episode_length_s = 10 # in [s] (10s)
        # env_step update rate/period settings  ==> sim_dt/decimation = step_dt
        self.decimation = 4 # 100Hz 
        # simulation update rate/period settings
        self.sim.dt = 1/200 # 5ms: 200Hz
        self.sim.render_interval = self.decimation
    
        
        
    


