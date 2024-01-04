from enum import Enum, auto
import numpy as np
DTYPE = np.float32
CASTING = "same_kind"
SIDE_SIGN = [1, -1, 1, -1]

# Data structure containing parameters for quadruped robot
class Quadruped:

    def __init__(self):
        self._abadLinkLength = 0.08
        self._hipLinkLength = 0.213
        self._kneeLinkLength = 0.213
        self._kneeLinkY_offset = 0.0
        self._abadLocation = np.array([0.1881, 0.04675, 0], dtype=DTYPE).reshape((3,1))
        self._bodyName = "trunk"
        self._footNames = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._bodyMass = 5.204 * 2
        self._bodyInertia = np.array([0.0168128557, 0, 0,
                                  0, 0.063009565, 0,
                                  0, 0, 0.0716547275]) * 5
        self._bodyHeight = 0.30
        self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
        # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
        self._mpc_weights = np.array([1.0, 1.5, 0.0,
                             0.0, 0.0, 50,
                             0.0, 0.0, 0.1,
                             1.0, 1.0, 0.1,
                             0.0], dtype=DTYPE) * 10

    def getHipLocation(self, leg:int):
        """
        Get location of the hip for the given leg in robot frame
        """
        assert leg >= 0 and leg < 4
        pHip = np.array([
            self._abadLocation[0] if (leg == 0 or leg == 1) else -self._abadLocation[0],
            self._abadLocation[1] if (leg == 0 or leg == 2) else -self._abadLocation[1],
            self._abadLocation[2]
            ], dtype=DTYPE).reshape((3,1))

        return pHip

def getSideSign(leg:int):
    """
    Get if the i-th leg is on the left (+) or right (-) of the robot
    """
    assert leg >= 0 and leg < 4
    return SIDE_SIGN[leg]

    