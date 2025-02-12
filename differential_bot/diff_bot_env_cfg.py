import omni.isaac.core.utils.prims as prim_utils # type: ignore
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors import ImuCfg
from omni.isaac.lab.sensors.ray_caster import RayCasterCfg, patterns

# mdp cfg modules
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
#---------------------------------------------------------------

from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

# from differential_bot import DIFF_BOT_CFG
from omni.isaac.lab_tasks.manager_based.classic.differential_bot import DIFF_BOT_CFG

# custom mdp modules
# import differential_bot.mdp as cmdp

# Bot scene configuration
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

    # rigid cube prop model (for the environment)
    cube = AssetBaseCfg(
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

    # ===============================================================================
    # Scene entities
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPosition",
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

    # adding the bot model
    robot: ArticulationCfg = DIFF_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # adding sensors to the bot
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/diff_bot/imu_link",
        debug_vis=True
    )

    
    # lidar sensor to the bot 
    # lidar: RayCasterCfg = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/diff_bot/lidar_link",
    #     update_period=1/60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.55)),
    #     mesh_prim_paths=["/World/GroundPlane"],
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
    #     ),
    #     debug_vis=True,
    # )





