from dataclasses import dataclass

import numpy as np


@dataclass
class SruRobotState:
    linear_vel_b: np.ndarray
    angular_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    robot_pos_w: np.ndarray
    robot_quat_wxyz: np.ndarray
