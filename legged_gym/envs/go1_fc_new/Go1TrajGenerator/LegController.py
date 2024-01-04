import math
import numpy as np
from math import sin, cos
# from legged_gym.go1_scripts.Parameters import Parameters
from legged_gym.go1_scripts.common.Quadruped import Quadruped
# from legged_gym.go1_scripts.go1_utils import DTYPE, getSideSign

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
    def __init__(self):
        self.tauFeedForward = np.zeros((3,1), dtype=DTYPE)
        self.forceFeedForward = np.zeros((3,1), dtype=DTYPE)

        self.qDes = np.zeros((3,1), dtype=DTYPE)
        self.qdDes = np.zeros((3,1), dtype=DTYPE)
        self.pDes = np.zeros((3,1), dtype=DTYPE)
        self.vDes = np.zeros((3,1), dtype=DTYPE)

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
    def __init__(self):
        self.q = np.zeros((3,1), dtype=DTYPE)
        self.qd = np.zeros((3,1), dtype=DTYPE)
        self.p = np.zeros((3,1), dtype=DTYPE)
        self.v = np.zeros((3,1), dtype=DTYPE)
        self.J = np.zeros((3,3), dtype=DTYPE)

    def zero(self):
        self.q.fill(0)
        self.qd.fill(0)
        self.p.fill(0)
        self.v.fill(0)
        self.J.fill(0)

    def setQuadruped(self, quad:Quadruped):
        self.quadruped = quad


class LegController:

    def __init__(self, quad:Quadruped):
        self.datas = [LegControllerData() for _ in range(4)]
        self.commands = [LegControllerCommand() for _ in range(4)]

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

    def updateData(self, dof_states):
        """
        update leg data from simulator
        """
        # ! update q, qd, J, p and v here
        for leg in range(4):
            self.datas[leg].q[:, 0] = dof_states["pos"][3*leg:3*(leg+1)]
            self.datas[leg].qd[:, 0] = dof_states["vel"][3*leg:3*(leg+1)]

            self.computeLegJacobian(leg)
            self.computeLegPosition(leg)
            self.datas[leg].v = self.datas[leg].J @ self.datas[leg].qd

    def computeLegJacobian(self, leg:int):
        """
        return J and p
        """
        
        dy = self._quadruped._abadLinkLength * getSideSign(leg)
        dz1 = -self._quadruped._hipLinkLength
        dz2 = -self._quadruped._kneeLinkLength

        q = self.datas[leg].q

        s1 = sin(q[0])
        s2 = sin(q[1])
        s3 = sin(q[2])

        c1 = cos(q[0])
        c2 = cos(q[1])
        c3 = cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        self.datas[leg].J[0, 0] = 0.0
        self.datas[leg].J[1, 0] = - dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2
        self.datas[leg].J[2, 0] = - dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1

        self.datas[leg].J[0, 1] = dz2 * c23 + dz1 * c2
        self.datas[leg].J[1, 1] = dz2 * s1 * s23 + dz1 * s1 * s2
        self.datas[leg].J[2, 1] = - dz2 * c1 * s23 - dz1 * c1 * s2

        self.datas[leg].J[0, 2] = dz2 * c23
        self.datas[leg].J[1, 2] = dz2 * s1 * s23
        self.datas[leg].J[2, 2] = - dz2 * c1 * s23

    def computeLegPosition(self, leg: int):
        dy = self._quadruped._abadLinkLength * getSideSign(leg)
        dz1 = -self._quadruped._hipLinkLength
        dz2 = -self._quadruped._kneeLinkLength

        q = self.datas[leg].q

        s1 = sin(q[0])
        s2 = sin(q[1])
        s3 = sin(q[2])

        c1 = cos(q[0])
        c2 = cos(q[1])
        c3 = cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        self.datas[leg].p[0] = dz2 * s23 + dz1 * s2
        self.datas[leg].p[1] = dy * c1 - dz1 * c2 * s1 - dz2 * s1 * c23
        self.datas[leg].p[2] = dy * s1 + dz1 * c1 * c2 + dz2 * c1 * c23

    def computeLegVelocity(self, leg: int):
        self.computeLegJacobian(leg)
        self.datas[leg].v = self.datas[leg].J @ self.datas[leg].qd

if __name__ == '__main__':
    leg_controller = LegController(Quadruped())
    dof_pos = np.array([[ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269],
                        [ 0.0380,  0.7797, -1.5595,  0.0396,  0.8184, -1.6593,
                          0.0386,  0.8207, -1.6636,  0.0467,  0.7640, -1.5269]])
    dof_vel = np.array([[ 5.5077,  0.3963, -1.6515,  5.3137,  1.3622, -5.0699,
                         4.8765,  1.6945, -5.6742,  5.6938, -1.7153,  2.6550],
                        [ 5.5077,  0.3963, -1.6513,  5.3137,  1.3623, -5.0699,
                          4.8765,  1.6945, -5.6742,  5.6936, -1.7159,  2.6551],
                        [ 5.5077,  0.3962, -1.6511,  5.3137,  1.3623, -5.0699,
                          4.8765,  1.6946, -5.6742,  5.6935, -1.7153,  2.6556],
                        [ 5.5080,  0.3964, -1.6513,  5.3137,  1.3622, -5.0699,
                          4.8765,  1.6945, -5.6742,  5.6941, -1.7158,  2.6548]])

    dof_states =dict()
    dof_states["pos"] = dof_pos[0]
    dof_states["vel"] = dof_vel[0]

    leg_controller.updateData(dof_states)

    for leg in range(4):
        print('leg position:', leg_controller.datas[leg].p)
        print('jacobian: ', leg_controller.datas[leg].J)


