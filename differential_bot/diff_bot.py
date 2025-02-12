import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import os

DIFF_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/labeeb/diff_bot_isaac/diff_bot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force"
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            enabled_self_collisions=False
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12), 
        joint_pos={"Revolute_1": 0.0, "Revolute_2": 0.0, "Revolute_3_01": 0.0, "Revolute_4_01": 0.0},
        joint_vel={"Revolute_1": 0.0, "Revolute_2": 0.0, "Revolute_3_01": 0.0, "Revolute_4_01": 0.0},
    ),
    actuators={
        "Revolute_1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            effort_limit=500.0,
            velocity_limit=100.0, # in deg/s (same for all the revolute active joints in this bot),
            stiffness=0.0,
            damping=10.0 # starting small
        ),
        "Revolute_2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            effort_limit=500.0,
            velocity_limit=100.0, # in deg/s (same for all the revolute active joints in this bot),
            stiffness=0.0,
            damping=10.0 # starting small
        ),
        "Revolute_3_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_3_01"],
            effort_limit=500.0,
            velocity_limit=100.0, # in deg/s (same for all the revolute active joints in this bot),
            stiffness=0.0,
            damping=10.0 # starting small
        ),
        "Revolute_4_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_4_01"],
            effort_limit=500.0,
            velocity_limit=100.0, # in deg/s (same for all the revolute active joints in this bot),
            stiffness=0.0,
            damping=10.0 # starting small
        )
    }
)