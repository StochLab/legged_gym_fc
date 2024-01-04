import numpy as np
import torch
from scipy import linalg

from legged_gym.go1_scripts.Parameters import Parameters
from legged_gym.go1_scripts.common.Quadruped import Quadruped
from legged_gym.go1_scripts.math_utils.moving_window_filter import MovingWindowFilter
from legged_gym.go1_scripts.go1_utils import DTYPE
from isaacgym.torch_utils import *
from legged_gym.go1_scripts.math_utils.orientation_tools import quat_to_rot, quat_to_rpy,rot_to_rpy, get_rot_from_normals, rpy_to_rot, Quaternion



class StateEstimate:
    def __init__(self, batch_size):
        self.position = np.zeros((batch_size, 3), dtype=DTYPE)
        self.vWorld = np.zeros((batch_size, 3), dtype=DTYPE)
        self.omegaWorld = np.zeros((batch_size, 3), dtype=DTYPE)
        self.orientation = np.zeros((batch_size, 4), dtype=DTYPE)

        self.rBody = np.zeros((batch_size, 3,3), dtype=DTYPE)
        self.rpy = np.zeros((batch_size, 3), dtype=DTYPE)
        self.rpyBody = np.zeros((batch_size, 3), dtype=DTYPE)

        self.ground_normal_world = np.array([0,0,1], dtype=DTYPE)
        self.ground_normal_yaw = np.array([0,0,1], dtype=DTYPE)

        self.vBody = np.zeros((batch_size, 3), dtype=DTYPE)
        self.omegaBody = np.zeros((batch_size, 3), dtype=DTYPE)


class StateEstimator:

    def __init__(self, quadruped:Quadruped, batch_size):
        self.batch_size = batch_size
        self.result = StateEstimate(batch_size)
        self._quadruped = quadruped
        # self._phase = np.zeros((batch_size, 4), dtype=DTYPE)
        self._phase = np.zeros((4, 1), dtype=DTYPE)
        self._contactPhase = self._phase
        self._foot_contact_history: np.ndarray = None
        self.ground_R_body_frame: np.ndarray = None
        self.body_height: np.array = np.array([self._quadruped._bodyHeight] * batch_size)
        self.result.position[:, 2] = self.body_height

    def reset(self):
        self.result = StateEstimate(self.batch_size)
        # self._phase = np.zeros((self.batch_size, 4), dtype=DTYPE)
        self._phase = np.zeros((4, 1), dtype=DTYPE)
        self._contactPhase = self._phase
        self._foot_contact_history:np.ndarray = None
        self.ground_R_body_frame:np.ndarray = None
        self.body_height = np.array([self._quadruped._bodyHeight] * self.batch_size)
        self.result.position[:, 2] = self.body_height

    def getContactPhase(self):
        return self._contactPhase.squeeze()

    def setContactPhase(self, phase:np.ndarray):
        self._contactPhase = phase

    def getResult(self):
        return self.result

    def update(self, root_states):
        self.result.vWorld = root_states[:, 7:10]
        self.result.omegaWorld = root_states[:, 10:13]
        self.result.orientation = root_states[:, 3:7]

        self.result.vBody = quat_rotate_inverse(self.result.orientation, self.result.vWorld).numpy()
        self.result.omegaBody = quat_rotate_inverse(self.result.orientation, self.result.omegaWorld).numpy()

        # RPY of body in the world frame
        r, p, y = get_euler_xyz(self.result.orientation)
        r[r > np.pi] = r[r > np.pi] - 2 * np.pi
        p[p > np.pi] = p[p > np.pi] - 2 * np.pi
        y[y > np.pi] = y[y > np.pi] - 2 * np.pi

        self.result.rpy = np.vstack((r, p, y)).T
        self.result.rBody = rpy_to_rot_batch(torch.tensor(self.result.rpy)).numpy()

        self.ground_R_body_frame = []
        self.rpyBody = []

        # Make it batch operational later
        for i in range(self.batch_size):
            world_R_yaw_frame = rpy_to_rot([0, 0, y[i]])
            yaw_R_ground_frame = get_rot_from_normals(np.array([0, 0, 1], dtype=DTYPE),
                                                      self.result.ground_normal_yaw)
            ground_R_body_frame = self.result.rBody[i].T @ world_R_yaw_frame.T @ yaw_R_ground_frame.T
            rpyBody = rot_to_rpy(ground_R_body_frame)
            self.ground_R_body_frame.append(ground_R_body_frame)
            self.rpyBody.append(rpyBody.flatten())

        self.result.ground_R_body_frame = np.concatenate([ele[np.newaxis, ...]
                                                          for ele in self.ground_R_body_frame])
        self.result.rpyBody = np.vstack(self.rpyBody)

        # print("Parallel: ")
        # print("vBody: ", self.result.vBody[0])
        # print('rBody: ', self.result.rBody[0])
        # print("omegaBody: ", self.result.omegaBody[0])
        # print("rpy: ", self.result.rpy)
        # print('ground R body frame: ', self.ground_R_body_frame[0])
        # print('rpyBody: ', self.result.rpyBody[0])

    def _init_contact_history(self, foot_positions:np.ndarray):
        self._foot_contact_history = foot_positions.copy()
        for leg in range(4):
            self._foot_contact_history[:, leg, 2] = -self.body_height

    def _update_com_position_ground_frame(self, foot_positions: np.ndarray):
        foot_contacts = np.tile(self._contactPhase.flatten(), (self.batch_size, 1))
        sum_foot_contacts = np.sum(foot_contacts, axis=1)
        # TO-DO: take care if sum of foot contacts is zero
        if np.sum(self._contactPhase.flatten()) == 0:
            self.result.position[:, 2] = self.body_height
            return
        foot_positions_ground_frame = np.einsum('bij,bjk->bik', foot_positions,
                                                np.transpose(self.ground_R_body_frame, (0, 2, 1)))
        foot_heights = -foot_positions_ground_frame[:, :, 2]
        height_in_ground_frame = np.sum(foot_heights * foot_contacts, axis=1) / sum_foot_contacts
        self.result.position[:, 2] = height_in_ground_frame


    def _update_contact_history(self, foot_positions:np.ndarray):
        foot_positions_ = foot_positions.copy()
        self._foot_contact_history[self._contactPhase.unsqueeze(-1)] \
            = foot_positions_[self._contactPhase.unsqueeze(-1)]

        # for leg_id in range(4):
        #     self._foot_contact_history[self._contactPhase[:, leg_id], leg_id, :] \
        #         = foot_positions_[self._contactPhase[:, leg_id], leg_id, :]



@torch.jit.script
def rpy_to_rot_batch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of roll-pitch-yaw angles to rotation matrices.

    Args:
        angles: Batch of roll-pitch-yaw angles (shape: batch_size x 3).

    Returns:
        Batch of rotation matrices.
    """
    batch_size = angles.size(0)

    # Extract roll, pitch, and yaw from the input
    roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]

    # Calculate trigonometric functions
    sr, cr = roll.sin(), roll.cos()
    sp, cp = pitch.sin(), pitch.cos()
    sy, cy = yaw.sin(), yaw.cos()

    # Calculate elements of the rotation matrix
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr

    # Create the rotation matrices
    rotation_matrices = torch.stack([
        torch.stack([r11, r12, r13], dim=-1),
        torch.stack([r21, r22, r23], dim=-1),
        torch.stack([r31, r32, r33], dim=-1)
    ], dim=-2)

    return rotation_matrices.view(batch_size, 3, 3)

import torch
import math

@torch.jit.script
def rot_to_quat_batch(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of coordinate transformation matrices to orientation quaternions.
    """
    batch_size = rot.size(0)
    quat = torch.empty((batch_size, 4), dtype=rot.dtype, device=rot.device)

    r = rot.transpose(-1, -2).clone()  # Transpose to match batch dimensions

    tr = r.diagonal(dim1=-2, dim2=-1).sum(-1)
    mask = tr > 0.0

    S = torch.sqrt(tr[mask] + 1.0) * 2.0
    quat[mask, 0] = 0.25 * S
    quat[mask, 1] = (r[mask, 2, 1] - r[mask, 1, 2]) / S
    quat[mask, 2] = (r[mask, 0, 2] - r[mask, 2, 0]) / S
    quat[mask, 3] = (r[mask, 1, 0] - r[mask, 0, 1]) / S

    cond1 = (r[:, 0, 0] > r[:, 1, 1]) & (r[:, 0, 0] > r[:, 2, 2])
    mask1 = mask & cond1

    S = torch.sqrt(1.0 + r[mask1, 0, 0] - r[mask1, 1, 1] - r[mask1, 2, 2]) * 2.0
    quat[mask1, 0] = (r[mask1, 2, 1] - r[mask1, 1, 2]) / S
    quat[mask1, 1] = 0.25 * S
    quat[mask1, 2] = (r[mask1, 0, 1] + r[mask1, 1, 0]) / S
    quat[mask1, 3] = (r[mask1, 0, 2] + r[mask1, 2, 0]) / S

    cond2 = ~cond1 & (r[:, 1, 1] > r[:, 2, 2])
    mask2 = mask & cond2

    S = torch.sqrt(1.0 + r[mask2, 1, 1] - r[mask2, 0, 0] - r[mask2, 2, 2]) * 2.0
    quat[mask2, 0] = (r[mask2, 0, 2] - r[mask2, 2, 0]) / S
    quat[mask2, 1] = (r[mask2, 0, 1] + r[mask2, 1, 0]) / S
    quat[mask2, 2] = 0.25 * S
    quat[mask2, 3] = (r[mask2, 1, 2] + r[mask2, 2, 1]) / S

    mask3 = ~cond1 & ~cond2

    S = torch.sqrt(1.0 + r[mask3, 2, 2] - r[mask3, 0, 0] - r[mask3, 1, 1]) * 2.0
    quat[mask3, 0] = (r[mask3, 1, 0] - r[mask3, 0, 1]) / S
    quat[mask3, 1] = (r[mask3, 0, 2] + r[mask3, 2, 0]) / S
    quat[mask3, 2] = (r[mask3, 1, 2] + r[mask3, 2, 1]) / S
    quat[mask3, 3] = 0.25 * S

    return quat






