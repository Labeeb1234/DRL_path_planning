import os
import sys
from math import pi, atan2, sin, cos, sqrt
import numpy as np
import time


def euler_to_quaternion(roll, pitch, yaw, torch=False, device='cpu'):
    """
    Convert Euler angles to quaternion. in XYZ convention. Brytan's convention.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
        torch (bool): Whether to return a torch tensor.
        device (str): Device for torch tensor. (only required if torch=True)
    Returns:
        np.array: Quaternion [w, x, y, z].
    """
    qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
    qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
    qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
    qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)

    if torch:
        import torch
        return torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=device) # w, x, y, z ---> shape (4,)

    return np.array([qw, qx, qy, qz])  # w, x, y, z ---> shape (4,)

def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert quaternion to Euler angles. in XYZ convention. Brytan's convention.

    Args:
        qx (float): Quaternion x component.
        qy (float): Quaternion y component.
        qz (float): Quaternion z component.
        qw (float): Quaternion w component.

    Returns:
        tuple: Euler angles (roll, pitch, yaw) in radians.
    """
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = pi/2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # in radians


def get_transform(state): # state -> [x, y, theta] 3x1 vector

    rotation = np.array([
        [cos(state[2, 0]), -sin(state[2, 0])],
        [sin(state[2, 0]), cos(state[2, 0])],
    ])
    translation = state[0:2]
    return translation, rotation

def transform_point_with_state(point, state):
    """
    Transform a point using a state.

    Args:
        point (np.array): Point [x, y] (2x1).
        state (np.array): State [x, y, theta] (3x1).

    Returns:
        np.array: Transformed point (2x1).
    """
    trans, rot = get_transform(state)
    new_point = rot @ point[0:2] + trans
    return new_point


def WrapToPi(rad, positive=False):
    """The function `WrapToPi` transforms an angle in radians to the range [-pi, pi].

    Args:

        rad (float): Angle in radians.
            The `rad` parameter in the `WrapToPi` function represents an angle in radians that you want to
        transform to the range [-π, π]. The function ensures that the angle is within this range by wrapping
        it around if it exceeds the bounds.

        positive (bool): Whether to return the positive value of the angle. Useful for angles difference.

    Returns:
        The function `WrapToPi(rad)` returns the angle `rad` wrapped to the range [-pi, pi].

    """
    while rad > pi:
        rad = rad - 2 * pi
    while rad < -pi:
        rad = rad + 2 * pi

    return rad if not positive else abs(rad)
