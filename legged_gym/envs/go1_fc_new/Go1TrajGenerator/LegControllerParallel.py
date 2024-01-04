import numpy as np
import torch
from legged_gym.go1_scripts.common.Quadruped import Quadruped

DTYPE = np.float32
CASTING = "same_kind"
SIDE_SIGN = [1, -1, 1, -1]

def getSideSign(leg:int):
    """
    Get if the i-th leg is on the left (+) or right (-) of the robot
    """
    assert leg >= 0 and leg < 4
    return SIDE_SIGN[leg]

class LegControllerCommand:
    def __init__(self, batchsize):
        self.tauFeedForward = np.zeros((batchsize, 3), dtype=DTYPE)
        self.forceFeedForward = np.zeros((batchsize, 3), dtype=DTYPE)

        self.qDes = np.zeros((batchsize, 3), dtype=DTYPE)
        self.qdDes = np.zeros((batchsize, 3), dtype=DTYPE)
        self.pDes = np.zeros((batchsize, 3), dtype=DTYPE)
        self.vDes = np.zeros((batchsize, 3), dtype=DTYPE)

        self.kpCartesian = np.zeros((3,3), dtype=DTYPE)
        self.kdCartesian = np.zeros((3,3), dtype=DTYPE)
        self.kpJoint = np.zeros((3,3), dtype=DTYPE)
        self.kdJoint = np.zeros((3,3), dtype=DTYPE)

    def zero(self):
        """
        Zero the leg command so the leg will not output torque
        """
        self.tauFeedForward.fill(0)
        self.forceFeedForward.fill(0)

        self.qDes.fill(0)
        self.qdDes.fill(0)
        self.pDes.fill(0)
        self.vDes.fill(0)

        self.kpCartesian.fill(0)
        self.kdCartesian.fill(0)
        self.kpJoint.fill(0)
        self.kdJoint.fill(0)

class LegControllerData:
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.q = np.zeros((batchsize, 3), dtype=DTYPE)
        self.qd = np.zeros((batchsize, 3), dtype=DTYPE)
        self.p = np.zeros((batchsize, 3), dtype=DTYPE)
        self.v = np.zeros((batchsize, 3), dtype=DTYPE)
        self.J = np.zeros((batchsize, 3, 3), dtype=DTYPE)

    def zero(self):
        self.q.fill(0)
        self.qd.fill(0)
        self.p.fill(0)
        self.v.fill(0)
        self.J.fill(0)

    def setQuadruped(self, quad:Quadruped):
        self.quadruped = quad

class LegController:

    def __init__(self, quad:Quadruped, batchsize):
        self.batchsize = batchsize
        self.datas = [LegControllerData(batchsize) for _ in range(4)]
        self.commands = [LegControllerCommand(batchsize) for _ in range(4)]
        self.IKModule = Serial3RKinematics([0.08, 0.213, 0.213])
        self.leg_names = ["fl", "fr", "bl", "br"]

        self._quadruped = quad
        for data in self.datas:
            data.setQuadruped(self._quadruped)

    def zeroCommand(self):
        """
        Zero all leg commands.  This should be run *before* any control code, so if
        the control code is confused and doesn't change the leg command, the legs
        won't remember the last command.
        """
        for cmd in self.commands:
            cmd.zero()

    def updateData(self, dof_pos, dof_vel):
        """
        update leg data from simulator
        """
        # ! update q, qd, J, p and v here
        for leg in range(4):
            self.datas[leg].q = dof_pos[:, 3 * leg:3 * (leg + 1)]
            self.datas[leg].qd = dof_vel[:, 3 * leg:3 * (leg + 1)]
            self.computeLegJacobian(leg)
            self.computeLegPosition(leg)
            self.datas[leg].v = np.einsum('bij,bj->bi', self.datas[leg].J, self.datas[leg].qd)


    def computeLegPosition(self, leg: int):
        dy = self._quadruped._abadLinkLength * getSideSign(leg)
        dz1 = -self._quadruped._hipLinkLength
        dz2 = -self._quadruped._kneeLinkLength

        q = self.datas[leg].q

        s1 = np.sin(q[:, 0])
        s2 = np.sin(q[:, 1])
        s3 = np.sin(q[:, 2])

        c1 = np.cos(q[:, 0])
        c2 = np.cos(q[:, 1])
        c3 = np.cos(q[:, 2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        self.datas[leg].p[:, 0] = dz2 * s23 + dz1 * s2
        self.datas[leg].p[:, 1] = dy * c1 - dz1 * c2 * s1 - dz2 * s1 * c23
        self.datas[leg].p[:, 2] = dy * s1 + dz1 * c1 * c2 + dz2 * c1 * c23

    def computeLegJacobian(self, leg: int):
        """
        return J and p
        """

        dy = self._quadruped._abadLinkLength * getSideSign(leg)
        dz1 = -self._quadruped._hipLinkLength
        dz2 = -self._quadruped._kneeLinkLength

        q = self.datas[leg].q

        s1 = np.sin(q[:, 0])
        s2 = np.sin(q[:, 1])
        s3 = np.sin(q[:, 2])

        c1 = np.cos(q[:, 0])
        c2 = np.cos(q[:, 1])
        c3 = np.cos(q[:, 2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        self.datas[leg].J[:, 0, 0] = 0.0
        self.datas[leg].J[:, 1, 0] = - dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2
        self.datas[leg].J[:, 2, 0] = - dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1

        self.datas[leg].J[:, 0, 1] = dz2 * c23 + dz1 * c2
        self.datas[leg].J[:, 1, 1] = dz2 * s1 * s23 + dz1 * s1 * s2
        self.datas[leg].J[:, 2, 1] = - dz2 * c1 * s23 - dz1 * c1 * s2

        self.datas[leg].J[:, 0, 2] = dz2 * c23
        self.datas[leg].J[:, 1, 2] = dz2 * s1 * s23
        self.datas[leg].J[:, 2, 2] = - dz2 * c1 * s23

    def computeLegVelocity(self, leg: int):
        self.computeLegJacobian(leg)
        self.datas[leg].v = np.einsum('bij,bj->bi', self.datas[leg].J, self.datas[leg].qd)

    def computeLegAngles(self, leg: int):
        _, qDes = self.IKModule.inverseKinematics(self.leg_names[leg], self.commands[leg].pDes)
        self.commands[leg].qDes = qDes


from typing import Tuple
PI = np.pi

class Serial2RKinematics:
    def __init__(self, link_lengths: list = [1.0, 1.0]):
        self.link_lengths = link_lengths

    def cosineRule(self, a: np.ndarray, b: np.ndarray, c: np.ndarray)-> np.ndarray:
        """
        Cosine Rule implementation for triangle sides a, b, c.
        Each have shape (batch_size, 1) cos A
        """
        return np.arccos(np.clip((c**2 + b**2 - a**2)/(2*b*c), -1, 1))

    def inWorkSpace(self, a: np.ndarray) -> np.ndarray:
        """
        Checks if the given points lies inside the workspace space of the serial 2r chain
        Args: a: Points to be checked
        Returns: mask: A mask specifying which points are inside the workspace
        """
        [l1, l2] = self.link_lengths
        r = np.linalg.norm(a, axis=1)
        return (r ** 2 > (l1 - l2) ** 2) & (r ** 2 < (l1 + l2) ** 2)

    def searchSafePositions(self, des_pos: np.ndarray) -> np.ndarray:
        """
        Function to search for the closest end-effector point within the workspace.
        This uses the bisection method to search for a feasible point on the boundary of the workspace.
        Args: des_pos : desired position of the end-effector
        Return: des_pos | p_in : Valid position inside the workspace
        """
        [l1, l2] = self.link_lengths
        rd = (l1 + l2) / 4
        delta_max = 0.001
        max_iter = 20
        valid_mask = self.inWorkSpace(des_pos)

        if not valid_mask.any():
            # Bisection method to find the closest point on the boundary
            # Initialize the search space
            p_out = des_pos.copy()
            unit_vec = p_out / np.linalg.norm(p_out, axis=1)[:, np.newaxis]
            p_in = rd * unit_vec
            # Bisection method
            n = 0
            while (np.linalg.norm(p_out - p_in, axis=1) > delta_max).any() and (n < max_iter):
                p = (p_in + p_out) / 2
                valid_mask = self.inWorkSpace(p)
                p_in[valid_mask] = p[valid_mask]
                p_out[~valid_mask] = p[~valid_mask]
                n += 1
            return p_in
        else:
            return des_pos

    def inverseKinematics(self, ee_pos: np.ndarray, branch=">") -> Tuple[np.ndarray]:
        """
        Inverse kinematics of a serial 2-R manipulator
        Note - Leg is in x-z plane, rotation about y. And ee_pos has shape (batch_size, 2)
        Args: ee_pos: position of the end-effector [x, y] (Cartesian co-ordinates)
              branch: Specify the branch of the leg
        Output: valid_mask : A mask specifying which angles are valid
                q : The joint angles of the manipulator [q_hip, q_knee], where the angle q_knee is
                specified relative to the thigh link
        """
        Nb, dim = ee_pos.shape
        q = np.zeros((Nb, dim), float)

        [l1, l2] = self.link_lengths
        r = np.linalg.norm(ee_pos, axis=1)

        # Check if the end-effector point lies in the workspace of the manipulator
        valid_mask = self.inWorkSpace(ee_pos)
        invalid_mask = ~valid_mask
        ee_pos[invalid_mask] = self.searchSafePositions(ee_pos[invalid_mask])

        t1 = np.arctan2(-ee_pos[:, 0], -ee_pos[:, 1])

        q[:, 0] = t1 + self.cosineRule(l2, r, l1)
        q[:, 1] = self.cosineRule(r, l1, l2) - PI

        if branch == "<":
            q[:, 0] = t1 - self.cosineRule(l2, r, l1)
            q[:, 1] = q[:, 1] * -1

        return valid_mask, q


class Serial3RKinematics:
    def __init__(self, link_lengths: list = [0.5, 1.0, 1.0]):
        self.link_lengths = link_lengths
        self.serial_2R = Serial2RKinematics([link_lengths[1], link_lengths[2]])

    def inWorkSpace(self, a: np.ndarray) -> np.ndarray:
        """
        Checks if the given points lies inside the workspace space of the serial 3r chain
        Args: a: Points to be checked
        Returns: mask: A mask specifying which points are inside the workspace
        """
        l1= self.link_lengths[0]
        r = np.linalg.norm(a[:, 1:3], axis=1)
        ee_pos_2r = np.zeros((a.shape[0], 2), float)
        valid1 = (r >= l1)
        ee_pos_2r[:, 0] = a[:, 0]
        ee_pos_2r[valid1, 1] = -np.sqrt(r[valid1] ** 2 - l1 ** 2)
        ee_pos_2r[~valid1, 1] = -0
        valid2 = self.serial_2R.inWorkSpace(ee_pos_2r)

        return valid1 & valid2

    def searchSafePositions(self, des_pos: np.ndarray) -> np.ndarray:
        """
        Function to search for the closest end-effector point within the workspace.
        This uses the bisection method to search for a feasible point on the boundary of the workspace.
        Args: des_pos : desired position of the end-effector
        Return: des_pos | p_in : Valid position inside the workspace
        """
        [l1, l2, l3] = self.link_lengths
        rd = (l1 + l2 + l3) / 6
        delta_max = 0.001
        max_iter = 20
        valid_mask = self.inWorkSpace(des_pos)
        if not valid_mask.any():
            # Bisection method to find the closest point on the boundary
            # Initialize the search space
            p_out = des_pos.copy()
            unit_vec = p_out / np.linalg.norm(p_out, axis=1)[:, np.newaxis]
            p_in = rd * unit_vec
            # Bisection method
            n = 0
            while (np.linalg.norm(p_out - p_in, axis=1) > delta_max).any() and (n < max_iter):
                p = (p_in + p_out) / 2
                valid_mask = self.inWorkSpace(p)
                p_in[valid_mask] = p[valid_mask]
                p_out[~valid_mask] = p[~valid_mask]
                n += 1
            return p_in
        else:
            return des_pos

    def inverseKinematics(self, leg_name: str, ee_pos: np.ndarray, branch: str = ">") -> Tuple[np.ndarray]:
        """
        Inverse kinematics of a serial 3-R manipulator
        Note:
            - Leg is in x-z plane, rotation about y. And ee_pos has shape (batch_size, 3)
            - Note the hip is taken with respective to the negative z axis
            - The solution can be in 2 forms, based on the branch selected
        Args:
            ee_pos: position of the end-effector [x, y, z] (Cartesian co-ordinates)
            branch: Specify the branch of the leg
        Output:
            valid_mask : A mask specifying which angles are valid
            q : The joint angles of the manipulator [q_abd, q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        """

        Nb, dim = ee_pos.shape
        q = np.zeros((Nb, dim), float)

        abd_link = self.link_lengths[0]
        valid_mask = self.inWorkSpace(ee_pos)
        ee_pos[~valid_mask] = self.searchSafePositions(ee_pos[~valid_mask])
        l = np.linalg.norm(ee_pos[:, 1:3], axis=1)

        safety_mask = l >= abd_link
        z_prime = np.zeros(safety_mask.shape, dtype=float)
        z_prime[safety_mask] = -np.sqrt(l[safety_mask] ** 2 - abd_link ** 2)
        z_prime[~safety_mask] = -0.0

        t1 = np.arctan2(-z_prime, abd_link)

        if leg_name == "FR" or leg_name == "fr" or leg_name == "BR" or leg_name == "br":
            t2 = np.arctan2(-ee_pos[:, 1], -ee_pos[:, 2])
            q[:, 0] = PI / 2 - t1 - t2
        else:
            t2 = np.arctan2(ee_pos[:, 1], -ee_pos[:, 2])
            q[:, 0] = t1 + t2 - PI / 2

        x_prime = ee_pos[:, 0]
        ee_pos_2r = np.concatenate([x_prime.reshape(-1, 1), z_prime.reshape(-1, 1)], axis=1)
        valid_mask_2r, q_2r = self.serial_2R.inverseKinematics(ee_pos_2r, branch)
        q[:, 1:3] = q_2r

        return valid_mask & valid_mask_2r, q


if __name__ == '__main__':
    leg_controller = LegController(Quadruped(), batchsize=1)
    dof_pos = torch.tensor([[ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269]])
    dof_vel = torch.tensor([[ 5.5077,  0.3963, -1.6515,  5.3137,  1.3622, -5.0699,
                         4.8765,  1.6945, -5.6742,  5.6938, -1.7153,  2.6550],
                        [ 5.5077,  0.3963, -1.6513,  5.3137,  1.3623, -5.0699,
                          4.8765,  1.6945, -5.6742,  5.6936, -1.7159,  2.6551],
                        [ 5.5077,  0.3962, -1.6511,  5.3137,  1.3623, -5.0699,
                          4.8765,  1.6946, -5.6742,  5.6935, -1.7153,  2.6556],
                        [ 5.5080,  0.3964, -1.6513,  5.3137,  1.3622, -5.0699,
                          4.8765,  1.6945, -5.6742,  5.6941, -1.7158,  2.6548]])
    leg_controller.updateData(dof_pos[0].reshape(-1, 12), dof_vel[0].reshape(-1, 12))

    leg_names = ["fl", "fr", "bl", "br"]
    serial_3r_ik = Serial3RKinematics([0.08, 0.213, 0.213])

    for leg in range(4):
        print('actual angles:', dof_pos[0][3*leg:3*(leg+1)])
        print('predicted angles: ', serial_3r_ik.inverseKinematics(leg_names[leg], leg_controller.datas[leg].p))







