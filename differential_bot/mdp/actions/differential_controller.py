from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class DifferentialController(ActionTerm):
    cfg: actions_cfg.DifferentialControllerActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""


    def __init__(self, cfg: actions_cfg.DifferentialControllerActionCfg, env: ManagerBasedEnv): 
        super().__init__(cfg, env)

        # get bot body id and name
        self.body_idx, self.body_name = self._asset.find_bodies(self.cfg.body_name)
        # if self.body_idx != 1:
        #     raise ValueError(f"Found more than one body match for the body name: {self.cfg.body_name}")

        # get joint_names and ids
        self.joint_idxs, self.joint_names = self._asset.find_joints(self.cfg.joint_names)
        if len(self.joint_idxs) > 4:
            raise ValueError(f"Yea nice joke fukker!")
        
        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self.joint_names} [{self.joint_idxs}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self.body_name} [{self.body_idx}]"
        )

        # getting the bot physical properties (kinematic properties)
        self.wheel_radius = self.cfg.wheel_diameter/2
        self.wheel_separation = self.cfg.wheel_separation

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, len(self.joint_names), device=self.device)

        # save the scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
    
    # specifying the properties of this class
    @property 
    def action_dim(self) -> int:
        return 2 # actions ----> linear forward motion(body_x_vel) and omega_vel (yaw_vel) 

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    # operations
    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
    
    def apply_actions(self):
        a = self.wheel_radius
        w_2 = self.wheel_separation
        w = w_2/2
        # wheel configuration matrix
        W_pinv = torch.tensor([
            [1/a, 0.0, (-w)/a],
            [1/a, 0.0,(w)/a],
            [1/a, 0.0, (-w)/a],
            [1/a, 0.0, (w)/a]
        ], device=self.device)

        # processed action
        body_velx = self._processed_actions[:, 0]
        body_velyaw = self._processed_actions[:, 1]
        command = torch.stack([body_velx, torch.zeros_like(body_velx), body_velyaw], dim=1).unsqueeze(2)

        # Kinemtic Model
        # going simple for version 0.1
        joint_vel_targets = torch.matmul(W_pinv, command).transpose(1, 2)
        joint_vel_targets = joint_vel_targets.squeeze(1)
        # setting joint velocity targets
        self._asset.set_joint_velocity_target(joint_vel_targets, self.joint_idxs)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        print(f"{env_ids}")
        self._raw_actions[env_ids] = 0.0
