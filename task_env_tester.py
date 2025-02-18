import argparse
import torch
import numpy as np
import random

np.random.seed(1)
random.seed(1)

from omni.isaac.lab.app import AppLauncher
# create argparser
parser = argparse.ArgumentParser(description="Example on creating an empty stage.")
parser.add_argument(
    "--num_envs", type=int, default=2, help="Number of environments to spawn for simulation"
)

# Appending AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# ----------------- mdp cfg modules ------------------------------
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
#---------------------------------------------------------------

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.devices import Se2Keyboard

# custom modules
from differential_bot import DIFF_BOT_CFG
import differential_bot.mdp as cmdp
from differential_bot.diff_bot_env_cfg import DiffBotSceneCfg


@configclass
class ActionCfg:
    bot_vel = cmdp.DifferentialControllerActionCfg(
        asset_name="robot",
        body_name="base_link",
        joint_names=["Revolute_1", "Revolute_2", "Revolute_3_01", "Revolute_4_01"],
        wheel_diameter=0.4,
        wheel_separation=0.5778,
        scale=(1.0, 1.0)
    )
    
# observation space configuration
@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # distance to goal (need fixing ---> dimensional incompatibility)
        dists_to_goal = ObsTerm(func=cmdp.distance_to_goal)
        # relative position and orientation of goal wrt bot frame
        rel_pose_to_goal = ObsTerm(
            func=cmdp.relative_pose_to_goal
        ) # from custom observation space module (only x,y,z)

        # ---------------------------------------------------
        # distance to obstacles (need to find an alternative)
        # ---------------------------------------------------
        
        # current bot pose
        bot_pos = ObsTerm(func=mdp.root_pos_w)
        # bot frame velocites
        bot_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        bot_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False
    
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
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-0.01, 0.01)},
            "velocity_range": {}
        }
    )

    # randomize goal poses
    randomize_goal_poses = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
            "velocity_range": {}
        }
    )

@configclass
class TerminationsCfg:
    # timeout termination
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )

    done_goal = DoneTerm(
        func=cmdp.reached_goal_termination,
        params={
            "goal_position_tolerance": 0.2,
            "goal_angle_tolerance": 0.2
        }
    )    
    # for now only time_out and goal_terminations termination

@configclass
class RewardCfg:
    # penalty for just existing
    alive_reward = RewTerm(
        func=mdp.is_alive,
        weight=-1.0
    )
    # sparse reward structure for the task (just for env testing)
    goal_reward = RewTerm(
        func=cmdp.reached_goal_position,
        weight=2.0,
        params={
            "goal_position_tolerance": 0.3,
            "goal_angle_tolerance": 0.3
        }
    )

# creating the manager based env
@configclass
class DiffBotRLEnvCfg(ManagerBasedRLEnvCfg):
    # scene setting
    scene: DiffBotSceneCfg = DiffBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # base settings
    observations = ObservationCfg()
    actions = ActionCfg()
    events = EventCfg()
    # MDP settings
    # no curriculum for now
    rewards = RewardCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        # viewport view settings
        self.viewer.eye = [2.5, 0.0, 4.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]

        self.episode_length_s = 5 # in [s] (5s)
        # env_step update rate/period settings  ==> sim_dt/decimation = step_dt
        self.decimation = 4 # 100Hz 
        # simulation update rate/period settings
        self.sim.dt = 0.005 # 5ms: 200Hz
        self.sim.render_interval = self.decimation


def run_simulator():
    env_cfg = DiffBotRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Add debugging prints
    print(f"Env numbers: {env.num_envs}")
    obs, info  = env.reset()
    state = obs["policy"]
    steps_count = 0
    while simulation_app.is_running():
        steps_count += 1
        with torch.inference_mode():
            if steps_count % 500 == 0:
                steps_count = 0
                obs, info = env.reset()
                state = obs["policy"]
                print(f"="*80)
                print(f"[INFO]: Resetting Env. ...")
                print(f"-"*80)
                print(f"Initial state:\n{state}")
                print(f"-"*80)
                            
            bot_velocity = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            # bot_velocity = torch.rand_like(env.action_manager.action)
            obs, reward, terminated, trunc, info = env.step(action=bot_velocity)
            new_state = obs["policy"]

            print(f"------------------ ACTION -----------------------")
            print(f"taken action: {bot_velocity}")
            print(f"------------------ {steps_count} ----------------------") 

            print(f"------------------- STATE' FEEDBACK --------------------")
            print(f"new state:\n{new_state}")
            print(f"------------------ {steps_count} ----------------------\n")

            print(f"------------------- REWARD --------------------")
            print(f"reward:\n{reward}")
            print(f"------------------ {steps_count} ----------------------\n")

            print(f"------------------- END INFO --------------------")
            print(f"(terminated, truncated): ({terminated[0]}, {trunc[0]})")
            print(f"------------------ {steps_count} ----------------------\n")
       
    env.close()


def main():
    run_simulator()

if __name__ == "__main__":
    main()