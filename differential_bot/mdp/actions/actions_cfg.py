from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from . import differential_controller


@configclass
class DifferentialControllerActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = differential_controller.DifferentialController

    body_name: str = MISSING
    joint_names: list[str] = MISSING

    wheel_separation: float = MISSING
    wheel_diameter: float = MISSING

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0)."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0)."""
