# Compute foot swing trajectory with a bezier curve
# phase : How far along we are in the swing (0 to 1)
# swingTime : How long the swing should take (seconds)
import numpy as np

class FootSwingTrajectory:
    def __init__(self, batchsize):
        # vec3 (3,1)
        self.batchsize = batchsize
        self._p0: np.ndarray = np.zeros((batchsize, 3), dtype=np.float32)
        self._pf: np.ndarray = np.zeros((batchsize, 3), dtype=np.float32)
        self._p: np.ndarray = np.zeros((batchsize, 3), dtype=np.float32)
        self._v: np.ndarray = np.zeros((batchsize, 3), dtype=np.float32)
        self._a: np.ndarray = np.zeros((batchsize, 3), dtype=np.float32)
        # float or int
        self._height = 0.0

    def setInitialPosition(self, env_ids, p0: np.ndarray):
        """
        Set the starting location of the foot
        """
        self._p0[env_ids] = p0.copy()

    def setFinalPosition(self, env_ids, pf: np.ndarray):
        """
        Set the desired final position of the foot
        """
        self._pf[env_ids] = pf.copy()

    def setHeight(self, h: float):
        """
        Set the maximum height of the swing
        """
        self._height = h

    def getPosition(self):
        """
        Get the foot position at the current point along the swing
        """
        return self._p

    def getVelocity(self):
        """
        Get the foot velocity at the current point along the swing
        """
        return self._v

    def getAcceleration(self):
        """
        Get the foot acceleration at the current point along the swing
        """
        return self._a

    def computeSwingTrajectoryBezier(self, env_ids, phase, swingTime: float):
        self._p[env_ids] = cubicBezier(self._p0[env_ids], self._pf[env_ids], phase)
        self._v[env_ids] = cubicBezierFirstDerivative(self._p0[env_ids], self._pf[env_ids], phase) / swingTime
        self._a[env_ids] = cubicBezierSecondDerivative(self._p0[env_ids], self._pf[env_ids], phase) / (swingTime * swingTime)

        indices = np.asarray(np.where(phase < 0.5)[0])
        if len(indices) > 0:
            sub_env_ids = env_ids[indices]
            self._p[sub_env_ids, 2] = cubicBezier(self._p0[sub_env_ids, 2], self._p0[sub_env_ids, 2] + self._height, phase[indices] * 2)
            self._v[sub_env_ids, 2] = cubicBezierFirstDerivative(self._p0[sub_env_ids, 2], self._p0[sub_env_ids, 2] + self._height,
                                            phase[indices] * 2) * 2 / swingTime
            self._a[sub_env_ids, 2] = (cubicBezierSecondDerivative(self._p0[sub_env_ids, 2], self._p0[sub_env_ids, 2] + self._height, phase[indices] * 2)
                  * 4 / (swingTime * swingTime))

        indices = np.asarray(np.where(phase >= 0.5)[0])
        if len(indices) > 0:
            sub_env_ids = env_ids[indices]
            self._p[sub_env_ids, 2] = cubicBezier(self._p0[sub_env_ids, 2], self._pf[sub_env_ids, 2],
                                                  phase[indices] * 2 - 1)
            self._p[sub_env_ids, 2] = cubicBezier(self._p0[sub_env_ids, 2] + self._height, self._pf[sub_env_ids, 2], phase[indices] * 2 - 1)
            self._v[sub_env_ids, 2] = cubicBezierFirstDerivative(self._p0[sub_env_ids, 2] + self._height, self._pf[sub_env_ids, 2],
                                            phase[indices] * 2 - 1) * 2 / swingTime
            self._a[sub_env_ids, 2] = (cubicBezierSecondDerivative(self._p0[sub_env_ids, 2] + self._height, self._pf[sub_env_ids, 2], phase[indices] * 2 - 1)
                  * 4 / (swingTime * swingTime))


# Interpolation
def cubicBezier(y0:np.ndarray, yf:np.ndarray, x:np.ndarray):
    """
    Cubic bezier interpolation between y0 and yf.  x is between 0 and 1
    """
    assert np.all(x >= 0) and np.all(x <= 1)
    yDiff = yf - y0
    bezier = x * x * x + 3.0 * (x * x * (1.0 - x))
    if x.ndim != y0.ndim:
        return y0 + bezier[:, np.newaxis] * yDiff
    else:
        return y0 + bezier * yDiff

def cubicBezierFirstDerivative(y0:np.ndarray, yf:np.ndarray, x:np.ndarray):
    """
    Cubic bezier interpolation derivative between y0 and yf.  x is between 0 and 1
    """
    assert np.all(x >= 0) and np.all(x <= 1)
    yDiff = yf - y0
    bezier = 6.0 * x * (1.0 - x)
    if x.ndim != y0.ndim:
        return bezier[:, np.newaxis] * yDiff
    else:
        return bezier * yDiff

def cubicBezierSecondDerivative(y0:np.ndarray, yf:np.ndarray, x:np.ndarray):
    """Cubic bezier interpolation derivative between y0 and yf.  x is between 0 and 1"""
    assert np.all(x >= 0) and np.all(x <= 1)
    yDiff = yf - y0
    bezier = 6.0 - 12.0 * x
    if x.ndim != y0.ndim:
        return bezier[:, np.newaxis] * yDiff
    else:
        return bezier * yDiff
