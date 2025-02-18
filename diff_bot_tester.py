import argparse
import os
import torch
import numpy as np
import random
import logging

np.random.seed(1)

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s]: %(message)s', 
)

from omni.isaac.lab.app import AppLauncher
# create argparser
parser = argparse.ArgumentParser(description="Example on creating an empty stage.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn for simulation"
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
from omni.isaac.lab.sensors import ImuCfg, FrameTransformerData
from omni.isaac.lab.sensors.ray_caster import RayCasterCfg, patterns
from omni.isaac.lab.markers import VisualizationMarkersCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.devices import Se2Keyboard


from differential_bot import DIFF_BOT_CFG


def wheel_config_ik(wheel_separation_distance, wheel_diameter, bot_frame_vel):
    a = wheel_diameter/2
    w = wheel_separation_distance/2
    W_pinv = np.array([
        [1/a, 0.0, (-w)/a],
        [1/a, 0.0,(w)/a],
        [1/a, 0.0, (-w)/a],
        [1/a, 0.0, (w)/a]
    ])
    wheel_velocities = np.dot(W_pinv, np.array([bot_frame_vel]).transpose())
    wheel_velocities = torch.tensor(wheel_velocities, device=args_cli.device)
    return wheel_velocities.transpose(0,1)


@configclass
class DiffBotSceneCfg(InteractiveSceneCfg):
    # default ground plane
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)) # size in square metres
    )

    lights = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # goal position marker (waypoint)
    goal_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/TargetPosition",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, 0.12), 
            rot=(1.0, 0.0, 0.0, 0.0) # in quaternion [w,x,y,z]
        )
    )

    # rigid cube prop model (for the environment) (sample obstacle)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            mass_props=sim_utils.MassPropertiesCfg(
                mass=100.0
            ),
            size=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, 0.5), 
            rot=(1.0, 0.0, 0.0, 0.0) # in quaternion [w,x,y,z]
        )
    )

    # adding robot model
    robot: ArticulationCfg = DIFF_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # adding sensors to the bot
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/diff_bot/imu_link",
        debug_vis=False
    )

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/diff_bot/lidar_link",
        update_period=1/60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.55)),
        mesh_prim_paths=["/World/GroundPlane"],
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
        ),
        debug_vis=False,
    )

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    target_pose = scene["goal_marker"]
    # cube_obstacle = scene["cube"]
    sim_dt = sim.get_physics_dt()
    
    count = 0
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = scene.env_origins
    initial_states = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    joint_pos, joint_vel = initial_states[0], initial_states[1]
    logging.info(f"{joint_pos}, {joint_vel}")

    teleop_interface = Se2Keyboard(
        v_x_sensitivity=0.8,
        v_y_sensitivity=0.4,
        omega_z_sensitivity=1.0
    ) # with default settings
    logging.info(f"Keyboard teleop settings loaded!")
    logging.info(f"{teleop_interface}")

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0

            # randomizing goal positions
            goal_marker_state = target_pose.data.default_root_state.clone()
            goal_marker_state[:, :2] = torch.rand_like(goal_marker_state[:, :2])*4.0
            target_pose.write_root_pose_to_sim(goal_marker_state[:, :7])
            _, _, goal_angle = math_utils.euler_xyz_from_quat(goal_marker_state[:, 3:7])
            logging.info(f"New Goal Position(x,y,yaw): {goal_marker_state[:, :2]}, {goal_angle}")

            # randomizing bot postions (withing a 3 metre radius around the centre of the world frame)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            initial_states = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos, joint_vel = initial_states[0], initial_states[1]
            logging.info(f"initial joint positions: {joint_pos}, initial joint velocities: {joint_vel}")
            
            scene.reset()
            # teleop_interface.reset()
            # logging.info(f"{teleop_interface}")
            logging.info("Simulation Reset Completed... Physics Scene Initialized!")

        se2_commands = teleop_interface.advance()
        print(f"---------- Keyboard(Se2)Commands -------------")
        logging.info(f"commands: {se2_commands}")
        print(f"---------- {count} ---------------------------")
        body_velx, body_vely, body_velyaw =  se2_commands[0], se2_commands[1] , se2_commands[2]
        target_joint_vel_cmd = wheel_config_ik(wheel_separation_distance=0.5778, wheel_diameter=0.3, bot_frame_vel=[0.5, 0.0, 0.0])
        jacobian = robot.root_physx_view.get_jacobians()

        robot.set_joint_velocity_target(target_joint_vel_cmd)

        scene.write_data_to_sim()
        sim.step()
        count+=1
        scene.update(dt=sim_dt)

        print("-------------IMU DATA------------------")
        print("Recieved angular position: ",{scene["imu"].data.quat_w})
        print("Received linear velocity: ",{scene["imu"].data.lin_vel_b})
        print("Received angular velocity: ",{scene["imu"].data.ang_vel_b})
        print("Received linear acceleration: ",{scene["imu"].data.lin_acc_b})
        print("Received angular acceleration: ",{scene["imu"].data.ang_acc_b})
        print(f"--------------{count}--------------------------")
        # joint data
        joint_vel = robot.data.joint_vel
        joint_pos = robot.data.joint_pos
        # odom data
        robot_pos = robot.data.root_pos_w # [x, y, z] position wrt world frame
        robot_orientation = robot.data.root_quat_w # [w, x, y, z] wrt world frame
        robot_yaw = robot.data.heading_w # theta or phi value
        print(f"------------------BOT ODOMETRY-----------------")
        print(f"Robot Position: {robot_pos}")
        print(f"Robot yaw: {robot_yaw}")
        print(f"---------------{count}----------------------")

        # print("--------Relative Transform to Cube(ref.bot)------")
        # robot_poses = robot.data.root_state_w
        # for robot_pose in robot_poses:
        #     cube_transform = math_utils.subtract_frame_transforms(
        #         t01=robot_pose[:3].reshape(1,3),
        #         q01=robot_pose[3:7].reshape(1,4),
        #         t02=cube_obstacle.data.root_state_w[:, :3],
        #         q02=cube_obstacle.data.root_state_w[:, 3:7]
        #     )
        # print("relative transforms:", cube_transform)
        # print(f"---------------{count}----------------------")
        
        print(f"----------------LIDAR DATA-------------------")
        print(scene["lidar"])
        print("Ray cast hit results: ", scene["lidar"].data.ray_hits_w)
        print(f"---------------{count}----------------------")


def main():
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = DiffBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Play the simulator
    # Now we are ready!
    logging.info("Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()